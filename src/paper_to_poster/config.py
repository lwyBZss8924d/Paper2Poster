import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

class MistralOCRConfig(BaseModel):
    api_key: str = Field(default=MISTRAL_API_KEY, description="Mistral API Key")
    model: str = Field(default="mistral-ocr-latest", description="Mistral OCR model")
    max_pages: int = Field(default=1000, description="Maximum number of pages to process")
    timeout: int = Field(default=300, description="API request timeout in seconds")
    include_image_base64: bool = Field(default=True, description="Include base64 encoded images in the output")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for API calls")
    fallback_to_docling: bool = Field(default=True, description="Fallback to DOCLING if Mistral OCR fails or quality is low")
    min_markdown_length: int = Field(default=500, description="Minimum markdown length for quality check")
    min_token_ratio: float = Field(default=0.05, description="Minimum token ratio for quality check (tokens/characters)")

# Example usage (optional, for testing)
if __name__ == "__main__":
    if MISTRAL_API_KEY is None:
        print("Warning: MISTRAL_API_KEY environment variable not set.")
    config = MistralOCRConfig()
    print("Mistral OCR Configuration:")
    print(f"  API Key: {'*' * 10 if config.api_key else 'Not Set'}")
    print(f"  Model: {config.model}")
    print(f"  Max Pages: {config.max_pages}")
    print(f"  Timeout: {config.timeout}")
    print(f"  Include Image Base64: {config.include_image_base64}")
    print(f"  Retry Attempts: {config.retry_attempts}")
    print(f"  Fallback to DOCLING: {config.fallback_to_docling}")
    print(f"  Min Markdown Length: {config.min_markdown_length}")
    print(f"  Min Token Ratio: {config.min_token_ratio}")
