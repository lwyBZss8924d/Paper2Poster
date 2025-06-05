# NewFeatures: Revised PDF-to-Markdown Parsing Pipeline(2)

issue tag: #NewFeatures #PDF2MD-Pipeline

Based on comprehensive research of the Paper2Poster architecture, Mistral Document AI API capabilities, and system requirements, this implementation plan provides a detailed roadmap for integrating Mistral OCR as the primary PDF processor while maintaining full compatibility with the existing pipeline.

## Current Paper2Poster Architecture Analysis

**Paper2Poster** implements a sophisticated 3-stage multi-agent pipeline that transforms academic papers into structured posters:

1. **Parser**: Distills papers into structured asset libraries containing section-level text summaries and extracted visual elements
2. **Planner**: Semantically matches text-visual pairs into binary-tree layouts with preserved reading order
3. **Painter-Commenter**: Refines panels through executable rendering code and VLM feedback loops

The Parser component currently leverages **DOCLING** for advanced PDF processing and **MARKER** for high-speed PDF-to-markdown conversion, producing structured JSON outputs that feed the downstream planning and rendering stages.

## Mistral OCR API Capabilities Assessment

**Mistral Document AI** offers compelling advantages for Paper2Poster integration:

- **Superior Accuracy**: 94.89% accuracy rate (vs 83.4% Google Document AI, 89.5% Azure OCR)
- **High Throughput**: 2,000 pages/minute processing speed
- **Academic Paper Optimization**: Excellent performance on scientific documents with LaTeX, equations, and complex layouts
- **Structured Extraction**: Built-in support for figures, tables, and mathematical expressions
- **Cost Efficiency**: $1/1000 pages basic OCR, $3/1000 pages with structured annotations
- **Multimodal Output**: Returns Markdown text with bounding box coordinates for images and tables

## Required Data Structure Compatibility

The integration must preserve Paper2Poster's existing JSON schema:

### _raw_content.json Structure
```json
{
  "sections": [
    {
      "section_id": "string",
      "section_title": "string",
      "section_type": "abstract|introduction|methods|results|conclusion",
      "content": "string",
      "summary": "string",
      "order": "integer"
    }
  ],
  "metadata": {
    "paper_title": "string",
    "authors": ["string"],
    "abstract": "string",
    "total_sections": "integer"
  }
}
```

### _images.json Structure
```json
{
  "images": [
    {
      "image_id": "string",
      "file_path": "string",
      "caption": "string",
      "figure_number": "string",
      "image_type": "figure|chart|diagram|photo",
      "dimensions": {"width": "integer", "height": "integer"},
      "extraction_metadata": {
        "page_number": "integer",
        "bbox": [x1, y1, x2, y2],
        "dpi": "integer"
      },
      "semantic_tags": ["string"]
    }
  ]
}
```

### _tables.json Structure
```json
{
  "tables": [
    {
      "table_id": "string",
      "caption": "string",
      "table_number": "string",
      "headers": ["string"],
      "data": [["cell_value", "cell_value", "..."]],
      "table_type": "results|comparison|statistics|data",
      "extraction_metadata": {
        "page_number": "integer",
        "bbox": [x1, y1, x2, y2]
      },
      "rendering_info": {
        "has_complex_formatting": "boolean",
        "requires_image_render": "boolean",
        "image_path": "string"
      }
    }
  ]
}
```

## Implementation Plan: Modified parse_raw.py

### Phase 1: Core Infrastructure Setup

#### 1.1 Dependencies and Configuration
```python
import os
import json
import logging
import base64
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel
import docling  # Fallback option
# Remove marker imports entirely

class MistralOCRConfig:
    def __init__(self):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.model = "mistral-ocr-latest"
        self.max_pages = 1000
        self.timeout = 300
        self.include_image_base64 = True
        self.retry_attempts = 3
        self.fallback_to_docling = True
```

#### 1.2 Structured Annotation Schema Design
```python
class ImageAnnotation(BaseModel):
    description: str
    content_type: str  # "figure", "chart", "diagram", "table", "equation"
    caption: Optional[str] = None
    figure_number: Optional[str] = None
    semantic_context: str

class TableAnnotation(BaseModel):
    description: str
    caption: Optional[str] = None
    table_number: Optional[str] = None
    column_headers: List[str]
    data_type: str  # "results", "comparison", "statistics", "data"
```

### Phase 2: Primary Mistral OCR Implementation

#### 2.1 Core OCR Processing Function
```python
class MistralPDFProcessor:
    def __init__(self, config: MistralOCRConfig):
        self.config = config
        self.client = Mistral(api_key=config.api_key)
        self.logger = logging.getLogger(__name__)

    def process_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Primary Mistral OCR processing with structured annotations."""
        try:
            # Configure structured annotations
            bbox_format = response_format_from_pydantic_model(ImageAnnotation)

            response = self.client.ocr.process(
                model=self.config.model,
                document={
                    "type": "document_url",
                    "document_url": f"file://{pdf_path}"
                },
                include_image_base64=self.config.include_image_base64,
                bbox_annotation_format=bbox_format,
                timeout=self.config.timeout
            )

            return self._validate_mistral_output(response)

        except Exception as e:
            self.logger.error(f"Mistral OCR processing failed: {str(e)}")
            return None

    def _validate_mistral_output(self, response) -> Optional[Dict[str, Any]]:
        """Validate Mistral output quality and completeness."""
        if not response or not response.pages:
            return None

        # Quality checks
        total_text_length = sum(len(page.markdown) for page in response.pages)
        image_count = sum(len(page.images) for page in response.pages)

        # Minimum thresholds for academic papers
        if total_text_length < 1000:  # Too little text extracted
            self.logger.warning("Mistral output below quality threshold: insufficient text")
            return None

        return {
            "pages": response.pages,
            "usage_info": response.usage_info,
            "model": response.model,
            "total_text_length": total_text_length,
            "image_count": image_count
        }
```

#### 2.2 Content Structuring and JSON Generation
```python
def _extract_sections_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
    """Extract paper sections from Mistral's markdown output."""
    import re

    # Pattern matching for academic paper sections
    section_patterns = {
        'abstract': r'(?i)#+\s*abstract[^\n]*\n(.*?)(?=\n#+|\Z)',
        'introduction': r'(?i)#+\s*(?:introduction|intro)[^\n]*\n(.*?)(?=\n#+|\Z)',
        'methods': r'(?i)#+\s*(?:methods?|methodology|approach)[^\n]*\n(.*?)(?=\n#+|\Z)',
        'results': r'(?i)#+\s*(?:results?|findings?|experiments?)[^\n]*\n(.*?)(?=\n#+|\Z)',
        'conclusion': r'(?i)#+\s*(?:conclusion|discussion|summary)[^\n]*\n(.*?)(?=\n#+|\Z)'
    }

    sections = []
    for section_type, pattern in section_patterns.items():
        matches = re.findall(pattern, markdown_content, re.DOTALL)
        if matches:
            sections.append({
                "section_id": f"{section_type}_{len(sections)}",
                "section_title": section_type.title(),
                "section_type": section_type,
                "content": matches[0].strip(),
                "summary": self._generate_section_summary(matches[0].strip()),
                "order": len(sections)
            })

    return sections

def _process_images_from_mistral(self, pages) -> List[Dict[str, Any]]:
    """Convert Mistral image data to Paper2Poster format."""
    images = []

    for page_idx, page in enumerate(pages):
        for img_idx, image in enumerate(page.images):
            # Extract image to file
            image_filename = f"img_{page_idx}_{img_idx}.{image.id.split('.')[-1]}"
            image_path = self._save_image_from_base64(
                image.image_base64, image_filename
            )

            # Parse annotation if available
            annotation = getattr(image, 'image_annotation', {})

            images.append({
                "image_id": image.id,
                "file_path": image_path,
                "caption": annotation.get('caption', ''),
                "figure_number": annotation.get('figure_number', ''),
                "image_type": annotation.get('content_type', 'figure'),
                "dimensions": {
                    "width": image.bottom_right_x - image.top_left_x,
                    "height": image.bottom_right_y - image.top_left_y
                },
                "extraction_metadata": {
                    "page_number": page_idx,
                    "bbox": [
                        image.top_left_x, image.top_left_y,
                        image.bottom_right_x, image.bottom_right_y
                    ],
                    "dpi": getattr(page, 'dimensions', {}).get('dpi', 200)
                },
                "semantic_tags": self._extract_semantic_tags(annotation)
            })

    return images
```

### Phase 3: DOCLING Fallback Implementation

#### 3.1 Fallback Logic with Quality Assessment
```python
class DoclingFallbackProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_pdf_fallback(self, pdf_path: str, mistral_failure_reason: str = None) -> Dict[str, Any]:
        """DOCLING fallback processing when Mistral fails or produces low-quality output."""
        try:
            from docling.document_converter import DocumentConverter

            self.logger.info(f"Falling back to DOCLING. Reason: {mistral_failure_reason}")

            converter = DocumentConverter()
            result = converter.convert(pdf_path)

            return self._convert_docling_to_standard_format(result)

        except Exception as e:
            self.logger.error(f"DOCLING fallback also failed: {str(e)}")
            raise Exception(f"Both Mistral and DOCLING processing failed: {str(e)}")

    def _convert_docling_to_standard_format(self, docling_result) -> Dict[str, Any]:
        """Convert DOCLING output to Paper2Poster-compatible format."""
        # Implementation to convert DOCLING's DoclingDocument format
        # to the same JSON structure expected by downstream processes
        pass
```

### Phase 4: Integrated parse_raw.py Main Function

#### 4.1 Primary Processing Pipeline
```python
def parse_raw(pdf_path: str, output_dir: str, config: Optional[MistralOCRConfig] = None) -> bool:
    """
    Main function to parse PDF using Mistral OCR with DOCLING fallback.

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save output JSON files
        config: Mistral OCR configuration

    Returns:
        bool: Success status
    """
    if config is None:
        config = MistralOCRConfig()

    logger = logging.getLogger(__name__)

    # Initialize processors
    mistral_processor = MistralPDFProcessor(config)
    docling_processor = DoclingFallbackProcessor()

    # Phase 1: Try Mistral OCR
    logger.info("Starting PDF processing with Mistral OCR")
    mistral_result = mistral_processor.process_pdf(pdf_path)

    if mistral_result:
        try:
            # Process Mistral output into Paper2Poster format
            processed_data = _process_mistral_output(mistral_result)

            # Validate output quality
            if _validate_output_quality(processed_data):
                # Save JSON files
                _save_json_outputs(processed_data, output_dir)
                logger.info("Successfully processed PDF with Mistral OCR")
                return True
            else:
                logger.warning("Mistral output failed quality validation")

        except Exception as e:
            logger.error(f"Error processing Mistral output: {str(e)}")

    # Phase 2: Fallback to DOCLING
    if config.fallback_to_docling:
        logger.info("Falling back to DOCLING processing")
        try:
            docling_result = docling_processor.process_pdf_fallback(
                pdf_path, "Mistral processing failed or low quality"
            )

            _save_json_outputs(docling_result, output_dir)
            logger.info("Successfully processed PDF with DOCLING fallback")
            return True

        except Exception as e:
            logger.error(f"Both Mistral and DOCLING processing failed: {str(e)}")
            return False

    logger.error("PDF processing failed - no fallback configured")
    return False

def _process_mistral_output(mistral_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Mistral OCR output to Paper2Poster JSON format."""

    # Combine all page markdown content
    full_markdown = "\n\n".join(page.markdown for page in mistral_result["pages"])

    # Extract structured data
    sections = _extract_sections_from_markdown(full_markdown)
    images = _process_images_from_mistral(mistral_result["pages"])
    tables = _process_tables_from_mistral(mistral_result["pages"])

    # Generate metadata
    metadata = _extract_paper_metadata(full_markdown, sections)

    return {
        "raw_content": {
            "sections": sections,
            "metadata": metadata
        },
        "images": {"images": images},
        "tables": {"tables": tables}
    }

def _save_json_outputs(processed_data: Dict[str, Any], output_dir: str):
    """Save the three required JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save _raw_content.json
    with open(output_path / "_raw_content.json", "w", encoding="utf-8") as f:
        json.dump(processed_data["raw_content"], f, indent=2, ensure_ascii=False)

    # Save _images.json
    with open(output_path / "_images.json", "w", encoding="utf-8") as f:
        json.dump(processed_data["images"], f, indent=2, ensure_ascii=False)

    # Save _tables.json
    with open(output_path / "_tables.json", "w", encoding="utf-8") as f:
        json.dump(processed_data["tables"], f, indent=2, ensure_ascii=False)
```

### Phase 5: Error Handling and Quality Assurance

#### 5.1 Comprehensive Error Handling
```python
def _validate_output_quality(processed_data: Dict[str, Any]) -> bool:
    """Validate processed data meets Paper2Poster quality standards."""

    # Check for required sections
    sections = processed_data["raw_content"]["sections"]
    if len(sections) < 3:  # Minimum sections expected
        return False

    # Check content length
    total_content_length = sum(len(section["content"]) for section in sections)
    if total_content_length < 2000:  # Minimum content threshold
        return False

    # Check for images and tables
    images = processed_data["images"]["images"]
    tables = processed_data["tables"]["tables"]

    # Validate image extraction
    for image in images:
        if not image.get("file_path") or not Path(image["file_path"]).exists():
            return False

    # Validate table structure
    for table in tables:
        if not table.get("headers") or not table.get("data"):
            return False

    return True

class ProcessingTimeoutError(Exception):
    pass

class InsufficientContentError(Exception):
    pass

class APIRateLimitError(Exception):
    pass
```

### Phase 6: Configuration and Monitoring

#### 6.1 Advanced Configuration Options
```python
class AdvancedMistralConfig(MistralOCRConfig):
    def __init__(self):
        super().__init__()
        self.enable_quality_scoring = True
        self.minimum_content_threshold = 2000
        self.maximum_api_retries = 3
        self.rate_limit_backoff = True
        self.cost_monitoring = True
        self.parallel_processing = False  # For batch processing
        self.custom_annotation_schema = None

    def estimate_processing_cost(self, pdf_path: str) -> float:
        """Estimate API cost for processing given PDF."""
        # Rough page count estimation
        import fitz  # PyMuPDF for page counting
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()

        base_cost = (page_count / 1000) * 1.0  # $1 per 1000 pages
        annotation_cost = (page_count / 1000) * 3.0 if self.enable_annotations else 0

        return base_cost + annotation_cost
```

## Integration Deployment Strategy

### Phase 1: Gradual Rollout
1. **Testing Phase**: Deploy alongside existing MARKER/DOCLING pipeline for A/B testing
2. **Quality Comparison**: Run parallel processing to compare output quality metrics
3. **Performance Benchmarking**: Measure throughput, accuracy, and cost effectiveness

### Phase 2: Primary Deployment
1. **Mistral as Primary**: Switch to Mistral OCR as the default processor
2. **DOCLING Fallback**: Maintain DOCLING as reliable fallback option
3. **MARKER Removal**: Complete removal of MARKER dependencies

### Phase 3: Optimization
1. **Cost Monitoring**: Implement usage tracking and budget alerts
2. **Quality Metrics**: Monitor paper processing success rates and poster generation quality
3. **Performance Tuning**: Optimize API calls, caching, and batch processing

## Expected Benefits and Performance Improvements

### Accuracy Improvements
- **Text Extraction**: 94.89% accuracy vs current tools (83-89%)
- **Academic Content**: Optimized for scientific papers with LaTeX and equations
- **Structured Data**: Superior figure and table extraction with semantic annotations

### Throughput Enhancement
- **Processing Speed**: 2,000 pages/minute potential throughput
- **Batch Processing**: Support for up to 1,000 pages per document
- **API Efficiency**: Single API call vs multiple tool chain

### Cost Optimization
- **Predictable Pricing**: $1/1000 pages base rate ($3/1000 with annotations)
- **Reduced Infrastructure**: Eliminate MARKER processing overhead
- **Scalable Costs**: Pay-per-use model aligned with processing volume

This comprehensive integration plan provides a robust foundation for transitioning Paper2Poster to Mistral OCR while maintaining full compatibility with existing workflows and data structures. The phased approach ensures reliable fallback options and gradual optimization of the new processing pipeline.
