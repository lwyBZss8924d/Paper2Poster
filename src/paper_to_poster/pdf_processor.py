import logging
from typing import Dict, List, Tuple, Any, Optional
import base64
import mimetypes
import io

import fitz  # PyMuPDF
from mistralai import Mistral
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.paper_to_poster.config import MistralOCRConfig

from pydantic import BaseModel, Field
from mistralai.extra import response_format_from_pydantic_model

# Issue #3: Structured Annotation Schemas
class ImageAnnotation(BaseModel):
    image_type: str = Field(description="Type of image, e.g., figure, chart, diagram, photo, equation")
    caption: Optional[str] = Field(default=None, description="Image caption text")
    figure_number: Optional[str] = Field(default=None, description="Figure reference number, e.g., Fig. 1, Figure 2a")
    semantic_context: str = Field(description="Brief description of the image content and its context within the document")

class TableAnnotation(BaseModel):
    caption: Optional[str] = Field(default=None, description="Table caption text")
    table_number: Optional[str] = Field(default=None, description="Table reference number, e.g., Table 1, Table II")
    column_headers: List[str] = Field(default_factory=list, description="Detected column headers of the table")
    data_type: str = Field(default="generic", description="Type of data in table, e.g., results, comparison, statistics, data")


# Configure logging
logger = logging.getLogger(__name__)

class MistralAPIError(Exception):
    """Custom exception for Mistral API errors."""
    pass

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


@retry(
    wait=wait_exponential(min=4, max=60),  # Exponential backoff: 4s, 8s, 16s, 32s, 60s, 60s...
    stop=stop_after_attempt(3),  # Max 3 attempts
    retry=retry_if_exception_type((MistralAPIError, ConnectionError)) # Retry on specific errors
)
async def process_pdf_with_mistral(
    pdf_path: Optional[str] = None,
    pdf_base64: Optional[str] = None,
    client: Optional[Mistral] = None,
    config: Optional[MistralOCRConfig] = None,
    filename_for_logging: Optional[str] = "unnamed_pdf" # Added for better logging in chunking
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process PDF using Mistral OCR API. Handles chunking for large PDFs.
    Handles file path or base64 input, includes retry logic, and basic error handling.

    Args:
        pdf_path: Path to the PDF file.
        pdf_base64: Base64 encoded string of the PDF file.
        client: Optional Mistral client instance.
        config: Optional MistralOCRConfig instance.

    Returns:
        Tuple containing:
            - raw_response: The raw JSON response from Mistral API.
            - markdown_text: Combined markdown from all pages.
            - images_list: List of extracted image metadata. (Placeholder)
            - tables_list: List of extracted table metadata. (Placeholder)

    Raises:
        ValueError: If neither pdf_path nor pdf_base64 is provided.
        PDFProcessingError: If PDF processing fails after retries.
    """
    if not config:
        config = MistralOCRConfig() # Use default config if not provided

    if not client:
        if not config.api_key:
            raise ValueError("MISTRAL_API_KEY must be set in MistralOCRConfig or environment if client is not provided.")
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
            pdf_doc = fitz.open(stream=base64.b64decode(pdf_base64), filetype="pdf")
        else:
            raise ValueError("Either pdf_path or pdf_base64 must be provided.")

        page_count = len(pdf_doc)
        logger.info(f"PDF '{filename_for_logging}' has {page_count} pages. Max pages per chunk: {config.max_pages}.")

        if page_count > config.max_pages:
            logger.info(f"PDF '{filename_for_logging}' exceeds max_pages ({config.max_pages}). Chunking required.")
            all_raw_responses = []
            all_markdown_parts = []
            all_images_lists = [] # TODO: Adjust image/table metadata for combined output
            all_tables_lists = [] # TODO: Adjust image/table metadata for combined output

            for i in range(0, page_count, config.max_pages):
                start_page = i
                end_page = min(i + config.max_pages -1 , page_count - 1)
                logger.info(f"Processing chunk for '{filename_for_logging}': pages {start_page + 1}-{end_page + 1}")

                chunk_doc = fitz.open() # Create a new empty PDF for the chunk
                chunk_doc.insert_pdf(pdf_doc, from_page=start_page, to_page=end_page)

                chunk_base64 = base64.b64encode(chunk_doc.tobytes()).decode('utf-8')
                chunk_doc.close()

                # Recursive call to process this chunk (won't re-chunk if small enough)
                # Pass along client and config.
                # We expect this inner call NOT to chunk further.
                chunk_filename_for_logging = f"{filename_for_logging}_chunk_{start_page+1}-{end_page+1}"
                raw_res_chunk, md_chunk, img_chunk, tbl_chunk = await process_pdf_with_mistral(
                    pdf_base64=chunk_base64,
                    client=client,
                    config=config,
                    filename_for_logging=chunk_filename_for_logging
                )
                all_raw_responses.append(raw_res_chunk) # Storing list of raw responses
                all_markdown_parts.append(md_chunk)

                # TODO: Adjust page numbers in img_chunk and tbl_chunk before appending
                # For now, just appending them. This will be an issue for Issue #4.
                all_images_lists.extend(img_chunk)
                all_tables_lists.extend(tbl_chunk)

            combined_markdown = "\n\n".join(all_markdown_parts)
            # For now, returning list of raw responses. A combined view might be complex.
            # Image and table lists are naively combined; page numbers will be wrong.
            logger.info(f"Finished processing all chunks for '{filename_for_logging}'.")
            return {"chunks": all_raw_responses}, combined_markdown, all_images_lists, all_tables_lists
        else:
            # Process the whole PDF as one chunk (original logic)
            current_pdf_base64: str
            current_content_type: str

            if pdf_base64:
                logger.info(f"Processing PDF '{filename_for_logging}' from base64 encoded string.")
                current_pdf_base64 = pdf_base64
                current_content_type = "application/pdf"
            elif pdf_path: # This implies pdf_doc is already opened from pdf_path
                logger.info(f"Processing PDF from path: {pdf_path}")
                pdf_bytes = pdf_doc.tobytes()
                current_pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                current_content_type = mimetypes.guess_type(pdf_path)[0] or "application/pdf"
            else: # Should not happen due to initial checks
                 raise ValueError("No PDF input available for single chunk processing.")

            document_payload = {
                "type": "document_base64",
                "document_base64": current_pdf_base64,
                "content_type": current_content_type
            }

            logger.info(f"Sending request to Mistral OCR API for '{filename_for_logging}' (model: {config.model}, timeout: {config.timeout}s)")
            response = await client.ocr.process(
                model=config.model,
                document_base64=document_payload["document_base64"],
                content_type=document_payload["content_type"],
                include_image_base64=config.include_image_base64,
                bbox_annotation_format=response_format_from_pydantic_model(ImageAnnotation), # Added Issue #3
                timeout=config.timeout,
            )

            if not response or not response.pages:
                logger.error(f"Mistral OCR response is empty or invalid for '{filename_for_logging}'.")
                raise MistralAPIError("Mistral OCR response is empty or invalid.")

            logger.info(f"Successfully received response from Mistral OCR for '{filename_for_logging}'.")
            markdown_text = "\n\n".join([page.markdown for page in response.pages if page.markdown])

            images_list: List[Dict[str, Any]] = [] # Placeholder for Issue #4
            tables_list: List[Dict[str, Any]] = [] # Placeholder for Issue #4

            if not _is_mistral_response_valid(response, markdown_text, config, filename_for_logging):
                raise MistralAPIError(f"Mistral OCR output failed validation for '{filename_for_logging}'.")

            return response.dict(), markdown_text, images_list, tables_list

    except fitz.fitz.PyMuPDFError as f_err:
        logger.error(f"PyMuPDF (fitz) error while processing '{filename_for_logging}': {f_err}")
        raise PDFProcessingError(f"PyMuPDF error processing '{filename_for_logging}': {f_err}")
    except Exception as e:
        logger.error(f"Error during Mistral OCR processing for '{filename_for_logging}': {e}")
        if "timeout" in str(e).lower():
            raise ConnectionError(f"Mistral API request timed out for '{filename_for_logging}': {e}")
        elif not isinstance(e, (MistralAPIError, PDFProcessingError, ConnectionError, ValueError)):
            raise MistralAPIError(f"Mistral API call failed for '{filename_for_logging}': {e}")
        raise
    finally:
        if pdf_doc:
            pdf_doc.close()


def _is_mistral_response_valid(
    response: Any,
    extracted_markdown: str,
    config: MistralOCRConfig,
    file_description: str = "processed PDF"
) -> bool:
    """
    Validates the Mistral OCR API response based on configuration thresholds.
    """
    if not response or not hasattr(response, 'pages') or not response.pages:
        logger.warning(f"Validation failed for {file_description}: Response is empty or has no pages.")
        return False

    if len(extracted_markdown) < config.min_markdown_length:
        logger.warning(
            f"Validation failed for {file_description}: "
            f"Markdown length ({len(extracted_markdown)}) is less than "
            f"min_markdown_length ({config.min_markdown_length})."
        )
        return False

    # Validate token ratio (simple character-based proxy for now)
    # A more sophisticated token count would require a tokenizer
    num_chars = len(extracted_markdown)
    # Approximating tokens; a real token count would be better.
    # Assuming an average of 4-5 chars per token for English text.
    # This is a very rough estimate.
    num_pseudo_tokens = num_chars / 4.5
    if num_chars > 0: # Avoid division by zero for empty markdown
        token_char_ratio = num_pseudo_tokens / num_chars
        # This ratio (tokens/characters) doesn't directly map to min_token_ratio (tokens/page or similar)
        # The PRD's min_token_ratio (0.05) likely implies something like (actual_tokens / max_possible_tokens_for_pages).
        # For now, this check is a placeholder and needs refinement based on how min_token_ratio is defined.
        # Let's assume min_token_ratio refers to a minimum density of meaningful content.
        # A low char count for a high page count might be a better indicator here if we had page count.
        # For now, we'll log it. A direct check against config.min_token_ratio is not yet meaningful.
        logger.debug(f"For {file_description}: Pseudo-token count: {num_pseudo_tokens:.0f}, Char count: {num_chars}, Pseudo-token/char ratio: {token_char_ratio:.4f}")

    # Add more validation checks as needed:
    # - Number of pages vs expected (if available)
    # - Presence of images/tables if expected
    # - Sanity check bbox coordinates if structured extraction is used later

    logger.info(f"Mistral OCR response for {file_description} passed basic validation.")
    return True


if __name__ == "__main__":
    # Example of how to run this async function (basic setup)
    import asyncio

    async def main():
        # This is a basic example. In a real application, you'd get the API key securely.
        # And you would have a PDF file to test with.
        # Ensure MISTRAL_API_KEY is set in your environment or .env file

        # Create a dummy config for testing
        test_config = MistralOCRConfig()
        if not test_config.api_key:
            print("MISTRAL_API_KEY not found. Please set it in your environment or a .env file.")
            print("Skipping example run of process_pdf_with_mistral.")
            return

        # Create a dummy PDF file (base64 encoded) for testing
        # This is a very small, valid PDF saying "Hello World"
        dummy_pdf_base64 = "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVGl0bGUgKP7/KQovQ3JlYXRvciAo/v8pCi9Qcm9kdWNlciAoU2tpYS9QREYgbTExNikKL0NyZWF0aW9uRGF0ZSAoRDoyMDI0MDgwNzE1MzAyN1opCi9Nb2REYXRlIChEOjIwMjQwODA3MTUzMDI3WikKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL1R5cGUgL0NhdGFsb2cKL1BhZ2VzIDIgMCBSCj4+CmVuZG9iago0IDAgb2JqCjw8Ci9UeXBlIC9QYWdlcwovQ291bnQgMQovS2lkcyBbIDUgMCBSIF0KPj4KZW5kb2JqCjUgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1BhcmVudCA0IDAgUgovUmVzb3VyY2VzCjw8Ci9YT2JqZWN0IDw8Cj4+Ci9Gb250IDw8Ci9GMSA2IDAgUgo+PgovUHJvY1NldCBbIC9QREYgL1RleHQgL0ltYWdlQiAvSW1hZ2VDIC9JbWFnZUkgXQo+PgovTWVkaWFCb3ggWyAwIDAgNjEyIDc5MiBdCi9Db250ZW50cyA3IDAgUgovR3JvdXAgPDwKL1R5cGUgL0dyb3VwCi9TIC9UcmFuc3BhcmVuY3kKL0NTIC9EZXZpY2VSR0IKPj4KL1RhYnMgL1MKL1N0cnVjdFBhcmVudHMgMAo+PgplbmRvYmoKNiAwIG9iago8PAovVHlwZSAvRm9udAovU3VidHlwZSAvVHlwZTEKL0Jhc2VGb250IC9IZWx2ZXRpY2EKL0VuY29kaW5nIC9XaW5BbnNpRW5jb2RpbmcKPj4KZW5kb2JqCjcgMCBvYmoKPDwKL0xlbmd0aCA0NAo+Pgphc3RyZWFtCkJUCjcwIDcwMCBUZAovRjEgMTIgVGYKKCBIZWxsbyBXb3JsZCApIFRqCkVUCmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDgKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAwMDAwIG4gCjAwMDAwMDAyNTAgMDAwMDAgbiAKMDAwMDAwMDMwMyAwMDAwMCBuIAowMDAwMDAwMzgyIDAwMDAwIG4gCjAwMDAwMDA1NzAgMDAwMDAgbiAKMDAwMDAwMDc3MyAwMDAwMCBuIAp0cmFpbGVyCjw8Ci9TaXplIDgKL1Jvb3QgMyAwIFIKL0luZm8gMSAwIFIKL0lEIFsgPDVDNTJBNjk5ODY4RDA5NDdCM0NEM0Q1RjgzNEM4QTYzPiA8NUM1MkE2OTk4NjhEMDk0N0IzQ0QzRDVGODM0QzhBNjM+IF0KPj4Kc3RhcnR4cmVmCjg4OQolJUVPRgo="

        print(f"Attempting to process dummy PDF with Mistral (API Key: {'*' * 10 if test_config.api_key else 'Not Set'})...")
        try:
            raw_response, markdown, images, tables = await process_pdf_with_mistral(
                pdf_base64=dummy_pdf_base64,
                config=test_config
            )
            print("\n--- Mistral OCR Result ---")
            print("Raw Response ( первые 200 символов ):", str(raw_response)[:200])
            print("\nMarkdown Output ( первые 200 символов ):")
            print(markdown[:200])
            print(f"\nImages Extracted: {len(images)}")
            print(f"Tables Extracted: {len(tables)}")
            print("--------------------------")

            # Example with a non-existent file path
            # print("\nAttempting to process non-existent PDF path...")
            # await process_pdf_with_mistral(pdf_path="non_existent_file.pdf", config=test_config)

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except PDFProcessingError as pe:
            print(f"PDFProcessingError: {pe}")
        except MistralAPIError as mae:
            print(f"MistralAPIError: {mae}")
        except ConnectionError as ce:
            print(f"ConnectionError: {ce}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    asyncio.run(main())
