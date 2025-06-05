import os
import base64
import requests
import re

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"

class MistralError(Exception):
    """Raised when Mistral OCR fails."""


def call_mistral_ocr(pdf_path: str) -> dict:
    """Call Mistral OCR API and return the JSON response."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise MistralError("MISTRAL_API_KEY not configured")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    payload = {
        "model": "mistral-ocr-latest",
        "document": {"document_base64": base64.b64encode(pdf_bytes).decode("utf-8")},
        "include_image_base64": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(MISTRAL_OCR_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


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
