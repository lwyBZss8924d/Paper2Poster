import os
import base64
import re
from mistralai import Mistral

class MistralError(Exception):
    """Raised when Mistral OCR fails."""


def call_mistral_ocr(pdf_path: str) -> dict:
    """Call Mistral OCR API and return the JSON response."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise MistralError("MISTRAL_API_KEY not configured")

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = Mistral(api_key=api_key)
    resp = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_b64}",
        },
        include_image_base64=True,
    )
    return resp if isinstance(resp, dict) else resp.to_dict()


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
