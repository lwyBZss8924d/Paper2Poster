from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
import json
import random
import asyncio
import os
from datetime import datetime

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from PosterAgent.mistral_ocr import (
    MistralOCRConfig,
    process_pdf_with_mistral,
    check_mistral_output_quality,
    MistralAPIError,
    save_assets_to_files,
    process_markdown_tables_to_images,
)
from mistralai import Mistral

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path
import PIL

from utils.wei_utils import account_token, get_agent_config
# Note: These star imports are kept for compatibility with original code
# They may contain functions used elsewhere in the codebase
from utils.pptx_utils import *  # noqa: F401, F403
from utils.critic_utils import *  # noqa: F401, F403
from jinja2 import Template
import re
import argparse

load_dotenv()
IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


async def parse_raw_async(args, actor_config, version=2):
    """Async version of parse_raw with Mistral OCR as primary processor."""
    raw_source = args.poster_path
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")

    text_content = ""
    raw_result = None

    config = MistralOCRConfig()

    # Create debug directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Phase 1: Try Mistral OCR (Primary)
    try:
        client = Mistral(api_key=config.api_key)
        mistral_resp, text_content, images, tables = (
            await process_pdf_with_mistral(
                pdf_path=raw_source,
                client=client,
                config=config,
            )
        )

        ok, reasons = check_mistral_output_quality(
            mistral_resp, text_content, config
        )

        if not ok:
            raise MistralAPIError(";".join(reasons))

        # Save OCR markdown output for debugging
        ocr_output_file = debug_dir / f"ocr_output_{timestamp}.md"
        with open(ocr_output_file, 'w', encoding='utf-8') as f:
            f.write("# OCR Output from Mistral\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"PDF: {raw_source}\n\n")
            f.write(text_content)
        print(f"💾 Saved OCR output to: {ocr_output_file}")

        # Save assets to files for compatibility
        processed_images, processed_tables = save_assets_to_files(
            images, tables,
            f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
            f'{args.poster_name}',
            args.poster_name
        )

        # Process any markdown tables to images
        output_dir = (
            f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
            f'{args.poster_name}'
        )
        md_tables = process_markdown_tables_to_images(
            text_content, output_dir, args.poster_name
        )

        # Combine all tables
        all_tables = processed_tables + md_tables

        # Create result structure compatible with original pipeline
        raw_result = {
            "source": "mistral",
            "pages": mistral_resp.get("pages", []),
            "images": processed_images,
            "tables": all_tables,
            "markdown": text_content
        }

        print("✅ Mistral OCR processing successful")

    except Exception as e:
        print(f"\n⚠️ Mistral OCR failed: {e}")
        print("🔄 Using Docling fallback\n")

        # Phase 2: Fallback to DOCLING (Original Pipeline)
        raw_result = doc_converter.convert(raw_source)
        raw_markdown = raw_result.document.export_to_markdown()
        text_content = markdown_clean_pattern.sub("", raw_markdown)

        # Save DOCLING output for debugging
        ocr_output_file = debug_dir / f"ocr_output_docling_{timestamp}.md"
        with open(ocr_output_file, 'w', encoding='utf-8') as f:
            f.write("# OCR Output from DOCLING (Fallback)\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"PDF: {raw_source}\n\n")
            f.write(text_content)
        print(f"💾 Saved DOCLING output to: {ocr_output_file}")

        # Check if DOCLING content is sufficient
        if len(text_content) < 500:
            print('\n⚠️ Docling content too short, using marker fallback\n')
            try:
                from marker.models import create_model_dict
                import torch
                from utils.src.model_utils import parse_pdf

                parser_model = create_model_dict(
                    device='cuda', dtype=torch.float16
                )
                text_content, rendered = parse_pdf(
                    raw_source, model_lst=parser_model, save_file=False
                )
            except Exception as marker_error:
                print(f"⚠️ Marker fallback also failed: {marker_error}")
                # Continue with whatever content we have

    # Continue with existing LLM processing logic
    return await _process_with_llm(
        args, actor_config, version, text_content, raw_result, timestamp
    )


async def _process_with_llm(
    args, actor_config, version, text_content, raw_result, timestamp=None
):
    """Process text content with LLM to generate structured JSON."""
    print(f"\n📝 Using prompt version: {version}")

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)

    if version == 1:
        template = Template(
            open("utils/prompts/gen_poster_raw_content.txt").read()
        )
    elif version == 2:
        template = Template(
            open("utils/prompts/gen_poster_raw_content_v2.txt").read()
        )

    if args.model_name_t.startswith('vllm_qwen'):
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
        # Add max_tokens to model config if not already set
        model_config = actor_config.get('model_config', {})
        if model_config is None:
            model_config = {}
        if 'max_tokens' not in model_config:
            # Use Gemini 2.5 Pro's max output tokens
            model_config['max_tokens'] = 65535

        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=model_config,
        )

    actor_sys_msg = (
        'You are the author of the paper, and you will create a poster '
        'for the paper.'
    )

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
        token_limit=actor_config.get('token_limit', None)
    )

    while True:
        prompt = template.render(
            markdown_document=text_content,
        )

        actor_agent.reset()
        response = actor_agent.step(prompt)
        input_token, output_token = account_token(response)

        # Save conversation history using CAMEL's memory
        try:
            # Get full conversation context from agent memory
            openai_messages, total_tokens = actor_agent.memory.get_context()

            # Save conversation history
            history_file = debug_dir / f"llm_conversation_{timestamp}.json"
            conversation_data = {
                "timestamp": timestamp,
                "model": args.model_name_t,
                "prompt_version": version,
                "total_tokens": total_tokens,
                "input_tokens": input_token,
                "output_tokens": output_token,
                "messages": [
                    {
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "name": msg.get("name", None)
                    }
                    for msg in openai_messages
                ]
            }

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved conversation history to: {history_file}")

        except Exception as e:
            print(f"⚠️ Failed to save conversation history: {e}")

        content_json = get_json_from_response(response.msgs[0].content)

        # Debug: Print LLM response
        print("\n🔍 Debug - LLM Response Preview (first 500 chars):")
        print(response.msgs[0].content[:500])
        print("\n🔍 Debug - Parsed JSON keys: "
              f"{list(content_json.keys()) if content_json else 'None'}")

        if len(content_json) > 0:
            break
        print('Error: Empty response, retrying...')
        if args.model_name_t.startswith('vllm_qwen'):
            text_content = text_content[:80000]

    # Debug: Check if 'sections' key exists
    if 'sections' not in content_json:
        print(f"❌ Error: 'sections' key not found in JSON. Keys found: "
              f"{list(content_json.keys())}")
        print(f"❌ JSON content: "
              f"{json.dumps(content_json, indent=2)[:1000]}...")

        # Save error details
        error_file = debug_dir / f"llm_error_{timestamp}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                "error": "Missing 'sections' key",
                "keys_found": list(content_json.keys()),
                "response_preview": response.msgs[0].content[:2000],
                "parsed_json": content_json
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved error details to: {error_file}")

        raise ValueError("JSON response missing 'sections' key")

    if len(content_json['sections']) > 9:
        # First 2 sections + randomly select 5 + last 2 sections
        # Ensure we have enough sections to sample from
        middle_sections = content_json['sections'][2:-2]
        sample_size = min(5, len(middle_sections))

        selected_sections = (
            content_json['sections'][:2] +
            random.sample(middle_sections, sample_size) +
            content_json['sections'][-2:]
        )
        content_json['sections'] = selected_sections

    has_title = False

    for section in content_json['sections']:
        if not isinstance(section, dict) or 'title' not in section:
            print(
                "Ouch! The response is invalid, the LLM is not "
                "following the format :("
            )
            print('Trying again...')
            raise ValueError(
                "LLM response format invalid - missing required fields"
            )

        # Version 1 expects subsections, Version 2 expects content
        if version == 1:
            if 'subsections' not in section:
                print(
                    f"Error: Section '{section.get('title', 'unknown')}' "
                    f"missing 'subsections' field for version 1 format"
                )
                raise ValueError(
                    "LLM response format invalid - missing subsections"
                )
        elif version == 2:
            if 'content' not in section:
                print(
                    f"Error: Section '{section.get('title', 'unknown')}' "
                    f"missing 'content' field for version 2 format"
                )
                raise ValueError(
                    "LLM response format invalid - missing content"
                )

        # Check for title section
        if 'title' in section['title'].lower():
            has_title = True

    if not has_title and version == 2:
        # Version 2 requires a title section
        print(
            'Ouch! The response is invalid, the LLM is not '
            'following the format - missing title section'
        )
        raise ValueError("LLM response missing title section")

    os.makedirs('contents', exist_ok=True)
    output_file = (
        f'contents/<{args.model_name_t}_{args.model_name_v}>_'
        f'{args.poster_name}_raw_content.json'
    )
    json.dump(content_json, open(output_file, 'w'), indent=4)

    return input_token, output_token, raw_result


@retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, version=2):
    """Sync wrapper for backward compatibility."""
    return asyncio.run(parse_raw_async(args, actor_config, version))


def gen_image_and_table(args, conv_res):
    """
    Enhanced image and table generation with full Mistral and DOCLING support.
    """
    input_token, output_token = 0, 0

    output_dir = Path(
        f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
        f'{args.poster_name}'
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.poster_name

    tables = {}
    images = {}

    # Handle Mistral OCR format (new primary path)
    if isinstance(conv_res, dict) and conv_res.get("source") == "mistral":
        print("📊 Processing Mistral OCR assets...")

        # Process Mistral images - already saved to files
        for i, img in enumerate(conv_res.get("images", [])):
            images[str(i+1)] = {
                'caption': img.get('caption', ''),
                'image_path': img.get('image_path', ''),
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'figure_size': img.get('figure_size', 0),
                'figure_aspect': img.get('figure_aspect', 1.0),
                'page_number': img.get('page_number', 0),
                'bbox': img.get('bbox', [0, 0, 0, 0]),
                'source': 'mistral'
            }

        # Process Mistral tables - already saved to files
        for i, tbl in enumerate(conv_res.get("tables", [])):
            tables[str(i+1)] = {
                'caption': tbl.get('caption', ''),
                'table_path': tbl.get('table_path', ''),
                'width': tbl.get('width', 0),
                'height': tbl.get('height', 0),
                'figure_size': tbl.get('figure_size', 0),
                'figure_aspect': tbl.get('figure_aspect', 1.0),
                'page_number': tbl.get('page_number', 0),
                'bbox': tbl.get('bbox', [0, 0, 0, 0]),
                'source': tbl.get('source', 'mistral')
            }

        print(f"✅ Processed {len(images)} images and "
              f"{len(tables)} tables from Mistral OCR")

    # Handle legacy formats (DOCLING/MARKER fallback)
    else:
        print("📊 Processing DOCLING/legacy assets...")

        # Original DOCLING processing logic
        if hasattr(conv_res, 'document'):
            # Save page images
            for page_no, page in conv_res.document.pages.items():
                page_no = page.page_no
                page_image_filename = (
                    output_dir / f"{doc_filename}-{page_no}.png"
                )
                with page_image_filename.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")

            # Save images of figures and tables
            table_counter = 0
            picture_counter = 0
            for element, _level in conv_res.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    element_image_filename = (
                        output_dir /
                        f"{doc_filename}-table-{table_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")

                if isinstance(element, PictureItem):
                    picture_counter += 1
                    element_image_filename = (
                        output_dir /
                        f"{doc_filename}-picture-{picture_counter}.png"
                    )
                    with element_image_filename.open("wb") as fp:
                        element.get_image(conv_res.document).save(fp, "PNG")

            # Save markdown and HTML exports
            md_filename = output_dir / f"{doc_filename}-with-images.md"
            conv_res.document.save_as_markdown(
                md_filename, image_mode=ImageRefMode.EMBEDDED
            )

            md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
            conv_res.document.save_as_markdown(
                md_filename, image_mode=ImageRefMode.REFERENCED
            )

            html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
            conv_res.document.save_as_html(
                html_filename, image_mode=ImageRefMode.REFERENCED
            )

            # Process DOCLING tables
            table_index = 1
            for table in conv_res.document.tables:
                caption = table.caption_text(conv_res.document)
                if len(caption) > 0:
                    table_img_path = (
                        f'<{args.model_name_t}_{args.model_name_v}>'
                        f'_images_and_tables/{args.poster_name}/'
                        f'{args.poster_name}-table-{table_index}.png'
                    )
                    try:
                        table_img = PIL.Image.open(table_img_path)
                        tables[str(table_index)] = {
                            'caption': caption,
                            'table_path': table_img_path,
                            'width': table_img.width,
                            'height': table_img.height,
                            'figure_size': (
                                table_img.width * table_img.height
                            ),
                            'figure_aspect': (
                                table_img.width / table_img.height
                            ),
                            'source': 'docling'
                        }
                    except Exception as e:
                        print(f"⚠️ Failed to load table image "
                              f"{table_img_path}: {e}")

                table_index += 1

            # Process DOCLING images
            image_index = 1
            for image in conv_res.document.pictures:
                caption = image.caption_text(conv_res.document)
                if len(caption) > 0:
                    image_img_path = (
                        f'<{args.model_name_t}_{args.model_name_v}>'
                        f'_images_and_tables/{args.poster_name}/'
                        f'{args.poster_name}-picture-{image_index}.png'
                    )
                    try:
                        image_img = PIL.Image.open(image_img_path)
                        images[str(image_index)] = {
                            'caption': caption,
                            'image_path': image_img_path,
                            'width': image_img.width,
                            'height': image_img.height,
                            'figure_size': (
                                image_img.width * image_img.height
                            ),
                            'figure_aspect': (
                                image_img.width / image_img.height
                            ),
                            'source': 'docling'
                        }
                    except Exception as e:
                        print(f"⚠️ Failed to load image {image_img_path}: {e}")
                image_index += 1

        print(f"✅ Processed {len(images)} images and "
              f"{len(tables)} tables from DOCLING")

    # Save JSON outputs (same format as original)
    images_file = (
        f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
        f'{args.poster_name}_images.json'
    )
    tables_file = (
        f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/'
        f'{args.poster_name}_tables.json'
    )

    json.dump(images, open(images_file, 'w'), indent=4)
    json.dump(tables, open(tables_file, 'w'), indent=4)

    print(f"💾 Saved assets: {images_file}, {tables_file}")

    return input_token, output_token, images, tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='4o')
    parser.add_argument('--poster_path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    agent_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = (
            args.poster_path.split('/')[-1]
            .replace('.pdf', '').replace(' ', '_')
        )

    # Parse raw content
    input_token, output_token, raw_res = parse_raw(args, agent_config)

    # Generate images and tables
    _, _, _, _ = gen_image_and_table(args, raw_res)

    print(f'Token consumption: {input_token} -> {output_token}')
