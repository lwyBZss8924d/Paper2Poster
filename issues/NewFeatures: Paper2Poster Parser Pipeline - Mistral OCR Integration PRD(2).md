# NewFeatures: Paper2Poster Parser Pipeline - Mistral OCR Integration PRD(2)

issue tag: #NewFeatures #PDF2MD-Pipeline

## 🎯 Epic: Integrate Mistral Document AI as Primary PDF Processor

Replace MARKER with Mistral Document AI API (`mistral-ocr-latest`) as the primary PDF-to-Markdown processor while maintaining DOCLING as fallback, ensuring full compatibility with existing Paper2Poster pipeline.

---

## 📋 Issue #1: Core Infrastructure Setup

**Title:** Setup Mistral OCR Infrastructure and Dependencies

**Description:**
Establish the foundational infrastructure for Mistral Document AI integration, including SDK setup, configuration management, and dependency updates.

**Acceptance Criteria:**
- [ ] Install and configure `mistralai` Python SDK
- [ ] Set up environment variable management for `MISTRAL_API_KEY`
- [ ] Create `MistralOCRConfig` class with all necessary parameters
- [ ] Remove all MARKER-related imports and dependencies
- [ ] Update `requirements.txt` with new dependencies
- [ ] Create configuration file for quality thresholds and API settings

**Technical Details:**
```python
# Configuration structure
class MistralOCRConfig:
    api_key: str
    model: str = "mistral-ocr-latest"
    max_pages: int = 1000
    timeout: int = 300
    include_image_base64: bool = True
    retry_attempts: int = 3
    fallback_to_docling: bool = True
    min_markdown_length: int = 500
    min_token_ratio: float = 0.05
```

**Priority:** P0 (Critical)
**Size:** M
**Labels:** `infrastructure`, `setup`, `breaking-change`

---

## 📋 Issue #2: Implement Mistral PDF Processing

**Title:** Create Primary Mistral OCR Processing Pipeline

**Description:**
Implement the main PDF processing function using Mistral Document AI API with proper error handling and response validation.

**Acceptance Criteria:**
- [ ] Implement `process_pdf_with_mistral()` function
- [ ] Add retry logic with exponential backoff using `tenacity`
- [ ] Handle both file path and base64 input methods
- [ ] Implement proper timeout handling
- [ ] Create response validation logic
- [ ] Support PDF chunking for documents > 1000 pages

**Technical Details:**
```python
@retry(wait=wait_exponential(min=4, max=60), stop=stop_after_attempt(3))
async def process_pdf_with_mistral(
    pdf_path: str,
    client: Mistral,
    config: MistralOCRConfig
) -> Tuple[Dict, str, List[Dict], List[Dict]]:
    """
    Process PDF using Mistral OCR API
    Returns: (raw_response, markdown_text, images_list, tables_list)
    """
```

**Priority:** P0 (Critical)
**Size:** L
**Labels:** `core-feature`, `api-integration`

---

## 📋 Issue #3: Structured Annotation Schema

**Title:** Define and Implement Bbox Annotation Schemas

**Description:**
Create Pydantic models for structured annotations to enhance image and table extraction accuracy.

**Acceptance Criteria:**
- [ ] Define `ImageAnnotation` Pydantic model
- [ ] Define `TableAnnotation` Pydantic model
- [ ] Integrate with `bbox_annotation_format` parameter
- [ ] Test annotation extraction accuracy
- [ ] Document annotation schema usage

**Technical Details:**
```python
class ImageAnnotation(BaseModel):
    image_type: str = Field(description="figure|chart|diagram|photo")
    caption: Optional[str] = Field(description="Image caption text")
    figure_number: Optional[str] = Field(description="Figure reference number")
    semantic_context: str = Field(description="Context description")

class TableAnnotation(BaseModel):
    caption: Optional[str] = Field(description="Table caption text")
    table_number: Optional[str] = Field(description="Table reference number")
    column_headers: List[str] = Field(description="Table column headers")
    data_type: str = Field(description="results|comparison|statistics|data")
```

**Priority:** P1 (High)
**Size:** M
**Labels:** `enhancement`, `data-extraction`

---

## 📋 Issue #4: Asset Extraction and Transformation

**Title:** Implement Asset Extraction from Mistral Response

**Description:**
Create functions to extract and transform images, tables, and text from Mistral API response into Paper2Poster-compatible formats.

**Acceptance Criteria:**
- [ ] Implement `extract_assets_from_mistral_response()` function
- [ ] Extract and save base64 images with proper metadata
- [ ] Handle table extraction for all three scenarios (image, markdown+bbox, markdown only)
- [ ] Generate unique IDs for all assets
- [ ] Ensure bounding box coordinate normalization
- [ ] Implement caption extraction logic (direct + heuristic)

**Technical Details:**
```python
def extract_assets_from_mistral_response(
    api_response: Dict[str, Any],
    output_dir: str
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Extract markdown text, images, and tables from Mistral response

    Returns:
        - markdown_text: Combined markdown content
        - images_meta: List of image metadata dicts
        - tables_meta: List of table metadata dicts
    """
```

**Priority:** P0 (Critical)
**Size:** XL
**Labels:** `core-feature`, `data-transformation`

---

## 📋 Issue #5: Markdown Table to Image Conversion

**Title:** Implement Table Image Generation from Markdown

**Description:**
Create functionality to convert Markdown tables to images when Mistral returns tables as text rather than images.

**Acceptance Criteria:**
- [ ] Evaluate and select rendering library (imgkit/playwright/markdown2image)
- [ ] Implement `convert_markdown_table_to_image()` function
- [ ] Handle various table formats and sizes
- [ ] Ensure high-quality image output
- [ ] Add error handling for conversion failures
- [ ] Optimize for performance

**Technical Details:**
```python
def convert_markdown_table_to_image(
    markdown_table: str,
    output_path: str,
    options: Dict[str, Any] = None
) -> bool:
    """Convert Markdown table to PNG image"""
```

**Priority:** P1 (High)
**Size:** L
**Labels:** `feature`, `table-processing`

---

## 📋 Issue #6: Quality Assessment and Fallback Logic

**Title:** Implement Quality Checks and DOCLING Fallback

**Description:**
Create comprehensive quality assessment for Mistral output and seamless fallback to DOCLING when needed.

**Acceptance Criteria:**
- [ ] Implement `check_mistral_output_quality()` function
- [ ] Define quality thresholds (text length, token ratio, asset count)
- [ ] Create fallback trigger conditions
- [ ] Implement smooth transition to DOCLING
- [ ] Add detailed logging for fallback events
- [ ] Create metrics collection for quality monitoring

**Technical Details:**
```python
def check_mistral_output_quality(
    api_response: Dict,
    markdown_text: str,
    pdf_metadata: Dict
) -> Tuple[bool, List[str]]:
    """
    Returns: (is_quality_acceptable, list_of_failure_reasons)
    """
```

**Priority:** P0 (Critical)
**Size:** L
**Labels:** `reliability`, `fallback-mechanism`

---

## 📋 Issue #7: DOCLING Fallback Enhancement

**Title:** Enhance DOCLING Fallback Processing

**Description:**
Upgrade DOCLING fallback to ensure it produces output matching Mistral's format and quality.

**Acceptance Criteria:**
- [ ] Update `process_pdf_with_docling()` to match new output format
- [ ] Enable DOCLING enrichment features (picture classification/description)
- [ ] Implement table image extraction using DOCLING's capabilities
- [ ] Ensure identical output schema with Mistral path
- [ ] Add DOCLING-specific error handling

**Priority:** P1 (High)
**Size:** M
**Labels:** `fallback`, `compatibility`

---

## 📋 Issue #8: Main Processing Pipeline Integration

**Title:** Integrate New Processing Pipeline into parse_raw.py

**Description:**
Replace existing parse_raw.py logic with new Mistral-primary, DOCLING-fallback pipeline.

**Acceptance Criteria:**
- [ ] Implement new `parse_paper_to_assets()` main function
- [ ] Ensure backward compatibility with output formats
- [ ] Maintain existing file naming conventions
- [ ] Preserve all metadata fields required by downstream agents
- [ ] Add comprehensive error handling and logging
- [ ] Create unified output generation for both paths

**Technical Details:**
```python
def parse_paper_to_assets(
    pdf_path: str,
    output_dir: str,
    config: Optional[MistralOCRConfig] = None
) -> bool:
    """Main entry point for PDF processing"""
```

**Priority:** P0 (Critical)
**Size:** L
**Labels:** `integration`, `core-feature`

---

## 📋 Issue #9: Performance Optimization

**Title:** Optimize Processing Performance and Resource Usage

**Description:**
Implement performance optimizations for large PDFs and batch processing scenarios.

**Acceptance Criteria:**
- [ ] Implement PDF chunking for documents > 1000 pages
- [ ] Add memory-efficient streaming for large responses
- [ ] Optimize image processing and saving
- [ ] Implement parallel processing where applicable
- [ ] Add progress tracking for long-running operations
- [ ] Create performance benchmarks

**Priority:** P2 (Medium)
**Size:** L
**Labels:** `performance`, `optimization`

---

## 📋 Issue #10: Testing Suite

**Title:** Create Comprehensive Test Suite

**Description:**
Develop unit, integration, and end-to-end tests for the new pipeline.

**Acceptance Criteria:**
- [ ] Unit tests for all major functions
- [ ] Integration tests for Mistral API interaction
- [ ] Integration tests for DOCLING fallback
- [ ] End-to-end tests with sample PDFs
- [ ] Mock API responses for offline testing
- [ ] Test output compatibility with downstream agents
- [ ] Performance regression tests

**Test Categories:**
- API error handling
- Quality check triggers
- Output format validation
- Large PDF handling
- Edge cases (empty PDFs, corrupted files)

**Priority:** P1 (High)
**Size:** XL
**Labels:** `testing`, `quality-assurance`

---

## 📋 Issue #11: Monitoring and Observability

**Title:** Implement Monitoring and Analytics

**Description:**
Add comprehensive monitoring to track API usage, fallback rates, and processing quality.

**Acceptance Criteria:**
- [ ] Implement structured logging for all operations
- [ ] Track Mistral API usage and costs
- [ ] Monitor fallback trigger rates and reasons
- [ ] Create quality metrics dashboard
- [ ] Set up alerts for high failure rates
- [ ] Add performance metrics collection

**Metrics to Track:**
- API success/failure rates
- Average processing time per page
- Fallback trigger frequency by reason
- Output quality scores
- Cost per document processed

**Priority:** P2 (Medium)
**Size:** M
**Labels:** `monitoring`, `analytics`

---

## 📋 Issue #12: Documentation and Migration Guide

**Title:** Create Documentation and Migration Guide

**Description:**
Comprehensive documentation for the new pipeline and migration guide from MARKER-based system.

**Acceptance Criteria:**
- [ ] API integration documentation
- [ ] Configuration guide with examples
- [ ] Migration guide from old pipeline
- [ ] Troubleshooting guide
- [ ] Performance tuning recommendations
- [ ] Update README with new dependencies and setup

**Priority:** P1 (High)
**Size:** M
**Labels:** `documentation`, `migration`

---

## 🚀 Implementation Phases

### Phase 1: Foundation (Issues #1, #2, #3)
- Set up infrastructure
- Basic Mistral integration
- Initial testing

### Phase 2: Core Features (Issues #4, #5, #6, #7, #8)
- Asset extraction
- Table processing
- Fallback mechanism
- Pipeline integration

### Phase 3: Production Ready (Issues #9, #10, #11)
- Performance optimization
- Comprehensive testing
- Monitoring setup

### Phase 4: Documentation (Issue #12)
- Complete documentation
- Migration support

---

## 📊 Success Metrics

1. **Quality Metrics:**
   - OCR accuracy > 94% (Mistral's benchmark)
   - Successful extraction rate > 95% for figures/tables
   - Fallback rate < 10% for standard academic PDFs

2. **Performance Metrics:**
   - Average processing time < 2s per page
   - Memory usage < 2GB for typical papers
   - API cost < $0.003 per paper

3. **Reliability Metrics:**
   - System uptime > 99.9%
   - Zero data loss during processing
   - 100% output format compatibility

---

## 🔗 Dependencies

- **External APIs:** Mistral Document AI API
- **Python Libraries:** mistralai, docling, tenacity, pydantic
- **Optional Libraries:** imgkit/playwright (for table rendering)
- **System Dependencies:** wkhtmltoimage (if using imgkit)

---

## ⚠️ Risks and Mitigations

1. **Risk:** Mistral API downtime
   - **Mitigation:** Robust DOCLING fallback

2. **Risk:** Table image generation performance
   - **Mitigation:** Caching, parallel processing, optimized rendering

3. **Risk:** Breaking changes for downstream agents
   - **Mitigation:** Strict output format validation, comprehensive testing
