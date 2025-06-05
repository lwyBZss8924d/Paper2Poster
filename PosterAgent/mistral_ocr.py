import os
import logging
import base64
import re
import mimetypes
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.models import response_format_from_pydantic_model
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type
)


@dataclass
class MistralOCRConfig:
    """Configuration for Mistral OCR processing."""

    api_key: str = field(
        default_factory=lambda: os.getenv("MISTRAL_API_KEY", "")
    )
    model: str = "mistral-ocr-latest"
    max_pages: int = 1000
    timeout: int = 300
    include_image_base64: bool = True
    retry_attempts: int = 3
    fallback_to_docling: bool = True
    min_markdown_length: int = 500
    min_token_ratio: float = 0.05


class ImageAnnotation(BaseModel):
    """Schema for image annotations from Mistral OCR."""
    image_type: str = Field(description="figure|chart|diagram|photo")
    caption: str | None = Field(
        default=None, description="Image caption text"
    )
    figure_number: str | None = Field(
        default=None, description="Figure reference number"
    )
    semantic_context: str = Field(description="Context description")


class TableAnnotation(BaseModel):
    """Schema for table annotations from Mistral OCR."""
    caption: str | None = Field(
        default=None, description="Table caption text"
    )
    table_number: str | None = Field(
        default=None, description="Table reference number"
    )
    column_headers: List[str] = Field(
        default_factory=list, description="Table column headers"
    )
    data_type: str = Field(
        description="results|comparison|statistics|data"
    )


logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Custom exception for Mistral API errors."""
    pass


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


@retry(
    wait=wait_exponential(min=4, max=60),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((MistralAPIError, ConnectionError))
)
async def process_pdf_with_mistral(
    pdf_path: Optional[str] = None,
    pdf_base64: Optional[str] = None,
    client: Optional[Mistral] = None,
    config: Optional[MistralOCRConfig] = None,
    filename_for_logging: Optional[str] = "unnamed_pdf"
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process PDF using Mistral OCR API with chunking support.

    Args:
        pdf_path: Path to the PDF file
        pdf_base64: Base64 encoded PDF string
        client: Optional Mistral client instance
        config: Optional MistralOCRConfig instance
        filename_for_logging: Filename for logging purposes

    Returns:
        Tuple containing:
            - raw_response: The raw JSON response from Mistral API
            - markdown_text: Combined markdown from all pages
            - images_list: List of extracted image metadata
            - tables_list: List of extracted table metadata

    Raises:
        ValueError: If neither pdf_path nor pdf_base64 is provided
        PDFProcessingError: If PDF processing fails after retries
    """
    if not config:
        config = MistralOCRConfig()

    if not client:
        if not config.api_key:
            raise ValueError(
                "MISTRAL_API_KEY must be set in MistralOCRConfig "
                "or environment if client is not provided."
            )
        client = Mistral(api_key=config.api_key)

    if not pdf_path and not pdf_base64:
        raise ValueError("Either pdf_path or pdf_base64 must be provided.")

    pdf_doc: Optional[fitz.Document] = None
    try:
        if pdf_path:
            filename_for_logging = pdf_path
            pdf_doc = fitz.open(pdf_path)
        elif pdf_base64:
            filename_for_logging = filename_for_logging or "base64_pdf"
            pdf_doc = fitz.open(
                stream=base64.b64decode(pdf_base64), filetype="pdf"
            )
        else:
            raise ValueError("Either pdf_path or pdf_base64 must be provided.")

        page_count = len(pdf_doc)
        logger.info(
            f"PDF '{filename_for_logging}' has {page_count} pages. "
            f"Max pages per chunk: {config.max_pages}."
        )

        if page_count > config.max_pages:
            return await _process_large_pdf_chunks(
                pdf_doc, page_count, client, config, filename_for_logging
            )
        else:
            return await _process_single_pdf(
                pdf_doc, pdf_path, pdf_base64, client, config,
                filename_for_logging
            )

    except fitz.fitz.PyMuPDFError as f_err:
        logger.error(
            f"PyMuPDF (fitz) error while processing "
            f"'{filename_for_logging}': {f_err}"
        )
        raise PDFProcessingError(
            f"PyMuPDF error processing '{filename_for_logging}': {f_err}"
        )
    except Exception as e:
        logger.error(
            f"Error during Mistral OCR processing "
            f"for '{filename_for_logging}': {e}"
        )
        if "timeout" in str(e).lower():
            raise ConnectionError(
                f"Mistral API request timed out "
                f"for '{filename_for_logging}': {e}"
            )
        elif not isinstance(e, (
            MistralAPIError, PDFProcessingError, ConnectionError, ValueError
        )):
            raise MistralAPIError(
                f"Mistral API call failed for '{filename_for_logging}': {e}"
            )
        raise
    finally:
        if pdf_doc:
            pdf_doc.close()


async def _process_large_pdf_chunks(
    pdf_doc: fitz.Document,
    page_count: int,
    client: Mistral,
    config: MistralOCRConfig,
    filename_for_logging: str
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process large PDF by splitting into chunks."""
    logger.info(
        f"PDF '{filename_for_logging}' exceeds max_pages "
        f"({config.max_pages}). Chunking required."
    )

    all_raw_responses = []
    all_markdown_parts = []
    all_images_lists = []
    all_tables_lists = []

    for i in range(0, page_count, config.max_pages):
        start_page = i
        end_page = min(i + config.max_pages - 1, page_count - 1)
        logger.info(
            f"Processing chunk for '{filename_for_logging}': "
            f"pages {start_page + 1}-{end_page + 1}"
        )

        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page)

        chunk_base64 = base64.b64encode(chunk_doc.tobytes()).decode('utf-8')
        chunk_doc.close()

        chunk_filename = (
            f"{filename_for_logging}_chunk_{start_page+1}-{end_page+1}"
        )
        raw_res_chunk, md_chunk, img_chunk, tbl_chunk = (
            await process_pdf_with_mistral(
                pdf_base64=chunk_base64,
                client=client,
                config=config,
                filename_for_logging=chunk_filename
            )
        )

        all_raw_responses.append(raw_res_chunk)
        all_markdown_parts.append(md_chunk)

        # Adjust page numbers for images and tables
        adjusted_images = _adjust_asset_page_numbers(
            img_chunk, start_page
        )
        adjusted_tables = _adjust_asset_page_numbers(
            tbl_chunk, start_page
        )

        all_images_lists.extend(adjusted_images)
        all_tables_lists.extend(adjusted_tables)

    combined_markdown = "\n\n".join(all_markdown_parts)
    logger.info(
        f"Finished processing all chunks for '{filename_for_logging}'."
    )

    return (
        {"chunks": all_raw_responses},
        combined_markdown,
        all_images_lists,
        all_tables_lists
    )


async def _process_single_pdf(
    pdf_doc: fitz.Document,
    pdf_path: Optional[str],
    pdf_base64: Optional[str],
    client: Mistral,
    config: MistralOCRConfig,
    filename_for_logging: str
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single PDF without chunking."""
    current_pdf_base64: str
    current_content_type: str

    if pdf_base64:
        logger.info(
            f"Processing PDF '{filename_for_logging}' "
            f"from base64 encoded string."
        )
        current_pdf_base64 = pdf_base64
        current_content_type = "application/pdf"
    elif pdf_path:
        logger.info(f"Processing PDF from path: {pdf_path}")
        pdf_bytes = pdf_doc.tobytes()
        current_pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        current_content_type = (
            mimetypes.guess_type(pdf_path)[0] or "application/pdf"
        )
    else:
        raise ValueError("No PDF input available for single chunk processing.")

    logger.info(
        f"Sending request to Mistral OCR API for '{filename_for_logging}' "
        f"(model: {config.model}, timeout: {config.timeout}s)"
    )

    response = await client.ocr.process(
        model=config.model,
        document_base64=current_pdf_base64,
        content_type=current_content_type,
        include_image_base64=config.include_image_base64,
        bbox_annotation_format=response_format_from_pydantic_model(
            ImageAnnotation
        ),
        timeout=config.timeout,
    )

    if not response or not response.pages:
        logger.error(
            f"Mistral OCR response is empty or invalid "
            f"for '{filename_for_logging}'."
        )
        raise MistralAPIError("Mistral OCR response is empty or invalid.")

    logger.info(
        f"Successfully received response from Mistral OCR "
        f"for '{filename_for_logging}'."
    )

    markdown_text = "\n\n".join([
        page.markdown for page in response.pages if page.markdown
    ])

    # Parse images and tables from response
    images_list, tables_list = parse_mistral_response(response)

    if not _is_mistral_response_valid(
        response, markdown_text, config, filename_for_logging
    ):
        raise MistralAPIError(
            f"Mistral OCR output failed validation "
            f"for '{filename_for_logging}'."
        )

    return response.dict(), markdown_text, images_list, tables_list


def _adjust_asset_page_numbers(
    assets: List[Dict[str, Any]],
    page_offset: int
) -> List[Dict[str, Any]]:
    """Adjust page numbers in asset metadata for chunked processing."""
    adjusted_assets = []
    for asset in assets:
        adjusted_asset = asset.copy()
        if 'page_number' in adjusted_asset:
            adjusted_asset['page_number'] += page_offset
        adjusted_assets.append(adjusted_asset)
    return adjusted_assets


def parse_mistral_response(
    resp: Any
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert Mistral response to image and table lists compatible with
    Paper2Poster.
    """
    images = []
    tables = []
    fig_count = 1
    table_count = 1

    for page_idx, page in enumerate(resp.pages):
        for img in page.images:
            caption = img.image_annotation or ""
            width = img.bottom_right_x - img.top_left_x
            height = img.bottom_right_y - img.top_left_y

            # Decode base64 image data
            try:
                data = base64.b64decode(img.image_base64 or "")
            except Exception as e:
                logger.warning(f"Failed to decode image base64: {e}")
                data = b""

            entry = {
                "caption": caption,
                "width": width,
                "height": height,
                "data": data,
                "page_number": page_idx,
                "bbox": [
                    img.top_left_x,
                    img.top_left_y,
                    img.bottom_right_x,
                    img.bottom_right_y
                ]
            }

            # Parse figure/table identification
            fig_match = re.match(
                r"^(Figure\s*\d+):?\s*(.*)", caption, re.IGNORECASE
            )
            tbl_match = re.match(
                r"^(Table\s*\d+):?\s*(.*)", caption, re.IGNORECASE
            )

            if tbl_match:
                entry["id"] = tbl_match.group(1)
                entry["caption"] = tbl_match.group(2)
                entry["index"] = table_count
                tables.append(entry)
                table_count += 1
            else:
                if fig_match:
                    entry["id"] = fig_match.group(1)
                    entry["caption"] = fig_match.group(2)
                else:
                    entry["id"] = f"Image {fig_count}"
                entry["index"] = fig_count
                images.append(entry)
                fig_count += 1

    return images, tables


def _is_mistral_response_valid(
    response: Any,
    extracted_markdown: str,
    config: MistralOCRConfig,
    file_description: str = "processed PDF"
) -> bool:
    """Validates the Mistral OCR API response based on thresholds."""
    if not response or not hasattr(response, 'pages') or not response.pages:
        logger.warning(
            f"Validation failed for {file_description}: "
            f"Response is empty or has no pages."
        )
        return False

    if len(extracted_markdown) < config.min_markdown_length:
        logger.warning(
            f"Validation failed for {file_description}: "
            f"Markdown length ({len(extracted_markdown)}) is less than "
            f"min_markdown_length ({config.min_markdown_length})."
        )
        return False

    # Token ratio validation (simplified)
    num_chars = len(extracted_markdown)
    if num_chars > 0:
        num_pseudo_tokens = num_chars / 4.5
        token_char_ratio = num_pseudo_tokens / num_chars
        logger.debug(
            f"For {file_description}: Pseudo-token count: "
            f"{num_pseudo_tokens:.0f}, Char count: {num_chars}, "
            f"Pseudo-token/char ratio: {token_char_ratio:.4f}"
        )

    logger.info(
        f"Mistral OCR response for {file_description} passed basic validation."
    )
    return True


def check_mistral_output_quality(
    api_response: Dict[str, Any],
    markdown_text: str,
    config: MistralOCRConfig,
) -> Tuple[bool, List[str]]:
    """Check whether the OCR result meets quality thresholds."""
    reasons: List[str] = []

    if len(markdown_text) < config.min_markdown_length:
        reasons.append("markdown_length")

    if not api_response.get("pages"):
        reasons.append("no_pages")

    token_count = api_response.get("usage", {}).get("total_tokens", 0)
    if token_count:
        ratio = len(markdown_text) / token_count
        if ratio < config.min_token_ratio:
            reasons.append("low_token_ratio")

    return len(reasons) == 0, reasons


def extract_assets_from_mistral_response(
    api_response: Dict[str, Any],
    output_dir: str
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Extract markdown text, images, and tables from Mistral response.

    Returns:
        - markdown_text: Combined markdown content
        - images_meta: List of image metadata dicts
        - tables_meta: List of table metadata dicts
    """
    pages = api_response.get("pages", [])
    markdown_text = "\n\n".join(p.get("markdown", "") for p in pages)

    images_meta = []
    tables_meta = []
    fig_count = 1
    table_count = 1

    for page_idx, page in enumerate(pages):
        for img in page.get("images", []):
            # Extract image metadata
            caption = img.get("image_annotation", "")
            width = img.get("bottom_right_x", 0) - img.get("top_left_x", 0)
            height = img.get("bottom_right_y", 0) - img.get("top_left_y", 0)

            # Save image to file
            image_id = f"img_{page_idx}_{len(images_meta)}"
            image_filename = f"{image_id}.png"
            image_path = os.path.join(output_dir, image_filename)

            # Decode and save base64 image
            if img.get("image_base64"):
                image_data = base64.b64decode(img["image_base64"])
                os.makedirs(output_dir, exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(image_data)

            # Determine if it's a table or figure
            fig_match = re.match(
                r"^(Figure\s*\d+):?\s*(.*)", caption, re.IGNORECASE
            )
            tbl_match = re.match(
                r"^(Table\s*\d+):?\s*(.*)", caption, re.IGNORECASE
            )

            entry = {
                "id": image_id,
                "caption": caption,
                "width": width,
                "height": height,
                "file_path": image_path,
                "page_number": page_idx,
                "bbox": [
                    img.get("top_left_x", 0),
                    img.get("top_left_y", 0),
                    img.get("bottom_right_x", 0),
                    img.get("bottom_right_y", 0)
                ],
                "data": image_data if img.get("image_base64") else None
            }

            if tbl_match:
                entry["table_id"] = tbl_match.group(1)
                entry["clean_caption"] = tbl_match.group(2)
                entry["index"] = table_count
                tables_meta.append(entry)
                table_count += 1
            else:
                if fig_match:
                    entry["figure_id"] = fig_match.group(1)
                    entry["clean_caption"] = fig_match.group(2)
                else:
                    entry["figure_id"] = f"Figure {fig_count}"
                    entry["clean_caption"] = caption
                entry["index"] = fig_count
                images_meta.append(entry)
                fig_count += 1

    return markdown_text, images_meta, tables_meta


def generate_unique_asset_ids(
    images: List[Dict],
    tables: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Generate unique IDs for all assets across chunks."""
    # Ensure unique IDs across multiple chunks
    for i, img in enumerate(images):
        img["unique_id"] = f"figure_{i+1:03d}"

    for i, tbl in enumerate(tables):
        tbl["unique_id"] = f"table_{i+1:03d}"

    return images, tables


def convert_markdown_table_to_image(
    markdown_table: str,
    output_path: str,
    options: Dict[str, Any] = None
) -> bool:
    """Convert Markdown table to PNG image using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        # Parse markdown table
        lines = markdown_table.strip().split('\n')
        header_line = lines[0]
        data_lines = lines[2:] if len(lines) > 2 else []

        # Extract headers
        headers = [h.strip() for h in header_line.split('|') if h.strip()]

        # Extract data rows
        data = []
        for line in data_lines:
            row = [cell.strip() for cell in line.split('|') if cell.strip()]
            if row:  # Skip empty rows
                data.append(row)

        if not data:
            logger.warning("No data found in markdown table")
            return False

        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, len(data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Save to file
        plt.savefig(
            output_path, bbox_inches='tight', dpi=300,
            facecolor='white', edgecolor='none'
        )
        plt.close()

        logger.info(
            f"Successfully converted markdown table to image: {output_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to convert markdown table to image: {e}")
        return False


def extract_markdown_tables(markdown_text: str) -> List[Dict[str, str]]:
    """Extract table structures from markdown text."""
    tables = []
    lines = markdown_text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for table headers (lines with |)
        if '|' in line and line.startswith('|') and line.endswith('|'):
            table_start = i
            table_lines = [line]

            # Check if next line is separator
            if (i + 1 < len(lines) and
                    '|' in lines[i + 1] and
                    '-' in lines[i + 1]):
                i += 1
                table_lines.append(lines[i])

                # Collect remaining table rows
                i += 1
                while (i < len(lines) and
                       lines[i].strip() and
                       '|' in lines[i]):
                    table_lines.append(lines[i].strip())
                    i += 1

                # Extract caption (look before and after table)
                caption = ""
                if (table_start > 0 and
                        "table" in lines[table_start - 1].lower() and
                        '|' in lines[table_start - 1]):
                    caption = lines[table_start - 1].strip()

                tables.append({
                    "markdown": '\n'.join(table_lines),
                    "caption": caption,
                    "start_line": table_start,
                    "end_line": i - 1
                })
            else:
                i += 1
        else:
            i += 1

    return tables


def save_assets_to_files(
    images: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    output_dir: str,
    doc_filename: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Save image and table data to files and return file paths."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process images
    processed_images = []
    for i, img in enumerate(images):
        img_filename = f"{doc_filename}-picture-{i+1}.png"
        img_path = output_path / img_filename

        # Save image data to file
        if img.get("data"):
            try:
                with img_path.open("wb") as f:
                    f.write(img["data"])

                # Create compatible entry for Paper2Poster
                processed_img = {
                    "id": img.get("id", f"Image {i+1}"),
                    "caption": img.get("caption", ""),
                    "image_path": str(img_path),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "figure_size": img.get("width", 0) * img.get("height", 0),
                    "figure_aspect": (
                        img.get("width", 1) / max(img.get("height", 1), 1)
                    ),
                    "page_number": img.get("page_number", 0),
                    "bbox": img.get("bbox", [0, 0, 0, 0])
                }
                processed_images.append(processed_img)

            except Exception as e:
                logger.error(f"Failed to save image {img_filename}: {e}")

    # Process tables
    processed_tables = []
    for i, tbl in enumerate(tables):
        tbl_filename = f"{doc_filename}-table-{i+1}.png"
        tbl_path = output_path / tbl_filename

        # Save table data to file
        if tbl.get("data"):
            try:
                with tbl_path.open("wb") as f:
                    f.write(tbl["data"])

                # Create compatible entry for Paper2Poster
                processed_tbl = {
                    "id": tbl.get("id", f"Table {i+1}"),
                    "caption": tbl.get("caption", ""),
                    "table_path": str(tbl_path),
                    "width": tbl.get("width", 0),
                    "height": tbl.get("height", 0),
                    "figure_size": tbl.get("width", 0) * tbl.get("height", 0),
                    "figure_aspect": (
                        tbl.get("width", 1) / max(tbl.get("height", 1), 1)
                    ),
                    "page_number": tbl.get("page_number", 0),
                    "bbox": tbl.get("bbox", [0, 0, 0, 0])
                }
                processed_tables.append(processed_tbl)

            except Exception as e:
                logger.error(f"Failed to save table {tbl_filename}: {e}")

    return processed_images, processed_tables


def process_markdown_tables_to_images(
    markdown_text: str,
    output_dir: str,
    doc_filename: str
) -> List[Dict[str, Any]]:
    """Extract markdown tables and convert them to images."""
    markdown_tables = extract_markdown_tables(markdown_text)
    table_images = []

    for i, table_info in enumerate(markdown_tables):
        table_filename = f"{doc_filename}-md-table-{i+1}.png"
        table_path = os.path.join(output_dir, table_filename)

        # Convert markdown table to image
        success = convert_markdown_table_to_image(
            table_info["markdown"], table_path
        )

        if success:
            # Get image dimensions
            try:
                import PIL.Image
                with PIL.Image.open(table_path) as img:
                    width, height = img.size
            except Exception:
                width, height = 800, 600  # Default dimensions

            table_entry = {
                "id": f"Markdown Table {i+1}",
                "caption": table_info.get("caption", ""),
                "table_path": table_path,
                "width": width,
                "height": height,
                "figure_size": width * height,
                "figure_aspect": width / max(height, 1),
                "source": "markdown",
                "start_line": table_info.get("start_line", 0),
                "end_line": table_info.get("end_line", 0)
            }
            table_images.append(table_entry)

    return table_images
