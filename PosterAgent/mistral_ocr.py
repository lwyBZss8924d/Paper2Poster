import os
import base64
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field
from mistralai import Mistral
from tenacity import retry, wait_exponential, stop_after_attempt


@dataclass
class MistralOCRConfig:
    """Configuration for Mistral OCR processing."""

    api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    model: str = "mistral-ocr-latest"
    max_pages: int = 1000
    timeout: int = 300
    include_image_base64: bool = True
    retry_attempts: int = 3
    fallback_to_docling: bool = True
    min_markdown_length: int = 500
    min_token_ratio: float = 0.05


class ImageAnnotation(BaseModel):
    image_type: str = Field(description="figure|chart|diagram|photo")
    caption: str | None = Field(default=None, description="Image caption text")
    figure_number: str | None = Field(default=None, description="Figure reference number")
    semantic_context: str = Field(description="Context description")


class TableAnnotation(BaseModel):
    caption: str | None = Field(default=None, description="Table caption text")
    table_number: str | None = Field(default=None, description="Table reference number")
    column_headers: List[str] = Field(default_factory=list, description="Table column headers")
    data_type: str = Field(description="results|comparison|statistics|data")


class MistralError(Exception):
    """Raised when Mistral OCR fails."""


def call_mistral_ocr(pdf_path: str, config: MistralOCRConfig | None = None) -> dict:
    """Call Mistral OCR API and return the JSON response."""
    config = config or MistralOCRConfig()
    if not config.api_key:
        raise MistralError("MISTRAL_API_KEY not configured")

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = Mistral(api_key=config.api_key)
    resp = client.ocr.process(
        model=config.model,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        include_image_base64=config.include_image_base64,
        timeout=config.timeout,
    )
    return resp if isinstance(resp, dict) else resp.to_dict()


@retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(3))
def process_pdf_with_mistral(
    pdf_path: str,
    client: Mistral,
    config: MistralOCRConfig,
) -> Tuple[Dict, str, List[Dict], List[Dict]]:
    """Process PDF using Mistral OCR and parse the response."""

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = client.ocr.process(
        model=config.model,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        include_image_base64=config.include_image_base64,
        timeout=config.timeout,
    )
    resp_dict = resp if isinstance(resp, dict) else resp.to_dict()
    markdown, images, tables = parse_mistral_response(resp_dict)
    return resp_dict, markdown, images, tables


def parse_mistral_response(resp: dict):
    """Convert Mistral response to markdown string and asset lists."""
    pages = resp.get("pages", [])
    markdown = "\n".join(p.get("markdown", "") for p in pages)
    images = []
    tables = []
    fig_count = 1
    table_count = 1
    for page in pages:
        for img in page.get("images", []):
            caption = img.get("image_annotation", "")
            width = img.get("bottom_right_x", 0) - img.get("top_left_x", 0)
            height = img.get("bottom_right_y", 0) - img.get("top_left_y", 0)
            data = base64.b64decode(img.get("image_base64", ""))
            entry = {
                "caption": caption,
                "width": width,
                "height": height,
                "data": data,
            }
            fig_match = re.match(r"^(Figure\s*\d+):?\s*(.*)", caption, re.IGNORECASE)
            tbl_match = re.match(r"^(Table\s*\d+):?\s*(.*)", caption, re.IGNORECASE)
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
    return markdown, images, tables


def is_output_complete(markdown: str, images, tables) -> bool:
    """Simple heuristic to check Mistral OCR output completeness."""
    if len(markdown) < 500:
        return False
    if not images and not tables:
        return False
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
