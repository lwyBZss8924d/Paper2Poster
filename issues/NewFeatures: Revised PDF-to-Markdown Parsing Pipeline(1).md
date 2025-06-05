# NewFeatures: Revised PDF-to-Markdown Parsing Pipeline(1)

issue tag: #NewFeatures #PDF2MD-Pipeline

Current Pipeline Overview (Marker + Docling + LLM)

The existing Paper2Poster parser first converts the input PDF into Markdown using document parsing tools (MARKER and DOCLING) . Marker is an AI-powered pipeline capable of converting PDFs (including scanned pages) into Markdown format, extracting images, formatting tables, and even converting equations . Docling is an efficient open-source PDF parser that extracts text, layout coordinates, and images from programmatic (digitally generated) PDFs . In the current system, these tools preserve the paper’s structure (sections, headings, lists, etc.) in Markdown . The full Markdown text is then processed by a Large Language Model (LLM) to generate a structured JSON outline of the paper . This structured asset library includes distilled text (key sections with summaries) as well as lists of extracted figures and tables with their captions . The output of the Parser stage is therefore:
	•	A JSON outline of the paper’s content (sections titles + brief descriptions).
	•	An _images.json listing figures (image_information) with captions (and possibly size info).
	•	A _tables.json listing tables (table_information) with captions (and size info)  .

The Planner and subsequent agents use this structured data to lay out the poster. The goal of the revised pipeline is to replace Marker with a more robust OCR-based parser (Mistral API) while preserving the output format so that downstream LLM stages require no changes.

## Step 1: Integrate Mistral OCR API for PDF Processing

We use the Mistral OCR model (version mistral-ocr-latest) as the primary document processor. Mistral’s API accepts PDFs either by direct upload (as base64 content) or by a URL, and returns recognized text per page in Markdown format, along with extracted images in base64 form. The API call is configured to include images by setting include_image_base64=true 【49†L365, L367-L370】. We can optionally request structured bounding-box annotations via bbox_annotation_format for more fine-grained data (e.g. image type or coordinates), though the default output already provides bounding boxes for each image. Below is a high-level example of calling Mistral and processing its response:

Example:
```python
import base64, requests, json

def call_mistral_ocr(pdf_path):
    # Read PDF and encode in base64 (if using direct upload)
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    # Prepare API request payload
    payload = {
        "model": "mistral-ocr-latest",
        "document": {"document_base64": pdf_b64},  # or {"document_url": "..."}
        "include_image_base64": True,
        # Optionally request additional annotation metadata:
        # "bbox_annotation_format": { ... JSON schema for image annotations ... }
    }
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post("https://api.mistral.ai/v1/ocr", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()
```

Mistral returns a JSON with an entry for each processed page. Each page object contains the recognized Markdown text for that page ("markdown" field) and a list of extracted "images" (if any) with their bounding boxes and base64 data  . For example, the response structure is (abbreviated):

Example:
```JSON
{
  "pages": [
    {
      "index": 0,
      "markdown": "## Introduction\nThis paper proposes ...",
      "images": [
        {
          "id": "image_0_1",
          "top_left_x": 100, "top_left_y": 200,
          "bottom_right_x": 300, "bottom_right_y": 400,
          "image_base64": "<base64-string>",
          "image_annotation": "Figure 1: Overview of the system."
        },
        { ... next image ... }
      ],
      "dimensions": {"width": 612, "height": 792, "dpi": 72}
    },
    { "index": 1, "markdown": "...", "images": [ ... ] },
    ...
  ],
  "model": "mistral-ocr-latest",
  "usage_info": { "pages_processed": 10, "doc_size_bytes": 500000 }
}
```

Using this output, our code will assemble the full document content and assets:
	•	Combine page texts: Concatenate the markdown from all pages in order to produce the full paper content in Markdown. (Minor post-processing can be done here, e.g., handling page-break hyphenations or removing repeated headers/footers, though Mistral often already removes common artifacts ).
	•	Extract image data: Gather all entries in the "images" lists across pages. For each image, record its base64 data and metadata. We enable include_image_base64 so we directly get a base64 string for each image  . We will map Mistral’s image fields to our schema:
	•	ID: We assign an identifier used in the image_information. If the image_annotation (Mistral’s caption or description) starts with “Figure ” or “Table ”, we use that as the ID (e.g. "Figure 1" or "Table 2"). Otherwise, we generate a generic ID like "Image1", "Image2", etc.
	•	Caption: We take image_annotation as the caption text, but strip any leading label (e.g., remove the leading “Figure 1:” from the caption since the ID already captures the figure number). This preserves the original caption content describing the figure/table.
	•	Size (width/height): Calculate the image’s width and height from the bounding box coordinates. For example, width = bottom_right_x - top_left_x and height = bottom_right_y - top_left_y. These correspond to the image’s size in the source PDF coordinates (often in points or pixels). We include these as the “size constraints” for the image  so the Planner can account for approximate aspect ratio or relative sizing.
	•	Base64 data: Use the image_base64 string as-is (this is the raw image data encoded in base64).

Using the above, we populate two lists: one for figures and one for tables. We classify each Mistral image entry by inspecting its caption or other cues:

Example:
```python
figures = []
tables = []
for img in all_images:  # collected from each page in resp["pages"]
    caption = img["image_annotation"] or ""
    # Determine ID and cleaned caption
    id = None; clean_caption = caption
    # e.g. caption = "Figure 3: Results chart ..." -> id="Figure 3", clean_caption="Results chart ..."
    fig_match = re.match(r'^(Figure\s+\d+):\s*(.*)$', caption, re.IGNORECASE)
    table_match = re.match(r'^(Table\s+\d+):\s*(.*)$', caption, re.IGNORECASE)
    if fig_match:
        id = fig_match.group(1)  # "Figure X"
        clean_caption = fig_match.group(2)
    elif table_match:
        id = table_match.group(1)  # "Table Y"
        clean_caption = table_match.group(2)
    else:
        # No explicit label – assign a new ID
        if looks_like_table(caption) or some_table_heuristic(img):
            id = f"Table {len(tables)+1}"
        else:
            id = f"Image {len(figures)+1}"
    # Compute size
    width = img["bottom_right_x"] - img["top_left_x"]
    height = img["bottom_right_y"] - img["top_left_y"]
    entry = {"id": id, "caption": clean_caption, "width": width, "height": height, "base64": img["image_base64"]}
    # Sort into figure or table list
    if id.lower().startswith("table"):
        tables.append(entry)
    else:
        figures.append(entry)
```

In most cases, academic papers label figures and tables in captions, so this reliably separates figures vs tables. (If needed, additional heuristics can be applied – e.g., detecting many numbers in a grid might indicate an unlabeled table.)

Optional – BBox Annotations: We can enhance the above by using bbox_annotation_format in the Mistral request to get structured info for each image. For instance, we could define a Pydantic schema with an image_type field (to classify image vs table) and perhaps ask for a short auto-generated summary  . This would leverage Mistral’s multimodal abilities to classify or describe each image region. While optional, this could provide more reliable identification of tables. For example:

Example:
```python
# Define desired fields for each image in annotations
ImageSchema = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "properties": {
                "image_type": {"type": "string", "title": "Type", "description": "Either 'Figure' or 'Table'"},
                "caption_text": {"type": "string", "title": "Caption"}
            },
            "required": ["image_type", "caption_text"],
            "additionalProperties": false,
            "title": "ImageAnnotation",
            "type": "object"
        },
        "name": "image_annotation",
        "strict": True
    }
}
payload["bbox_annotation_format"] = ImageSchema
```

In the response, each image’s image_annotation would then be a JSON object with our specified fields (e.g. "image_type": "Table"). For simplicity, the above heuristic approach (using caption text) is usually sufficient, but this shows how the Mistral API can be customized further if needed  .

After processing Mistral’s output, we have:
	•	full_markdown_text: the complete paper content in Markdown.
	•	figures list and tables list: each entry containing id, caption, width, height, base64 (matching the schema of image_information and table_information in the original system  ).

## Step 2: LLM-based Content Distillation to JSON

With the full Markdown text extracted, the next step in the Parser is unchanged: we feed this text into the LLM to produce the JSON outline of sections. The revised pipeline will reuse the existing prompt and LLM calls as in the original system, so that the structure of the output JSON remains identical. For example, the LLM (e.g., GPT-4 or an open-source model) will be prompted to output a JSON with each top-level section’s title and a concise summary of that section . This yields the raw_content.json (structured outline) that includes the paper’s key textual content. We ensure the keys and format mirror the original: e.g., a top-level JSON object with a list of sections (often under a "sections" or similar field), where each section has "title" and a short "description" (or "summary"). The paper title and other metadata can also be included if that was part of the original schema (for instance, we could add the paper title as the poster title in the JSON if required).

Why use the LLM after OCR? The Mistral OCR gives us a faithful Markdown of the paper, which is likely very long. The LLM is used to distill and organize this content. This step remains the same as before – we do not alter prompts or parsing logic. By preserving the output format (JSON outline + images list + tables list), we ensure the Planner agent can consume the data without any modifications.

## Step 3: Fallback to Docling on Failure or Low-Quality Output

While Mistral OCR is powerful, we include a fallback mechanism using DOCLING for robustness. The code will attempt the Mistral OCR call first, but if it fails or the output is deemed insufficient, we switch to using Docling’s parser:

Example:
```python
try:
    result = call_mistral_ocr(pdf_path)
    # ... process result as above ...
    if not full_markdown_text or len(full_markdown_text) < MIN_LENGTH_THRESHOLD:
        raise ValueError("Mistral output too short")
    # Example heuristic: if output text is empty or far shorter than expected.
    if len(full_markdown_text.split()) < 0.5 * estimated_word_count(pdf_path):
        raise ValueError("Mistral output incomplete")
    # (Additional heuristic: if no images/tables were found but the PDF likely contains them, etc.)
except Exception as e:
    print("Mistral OCR failed or output incomplete, falling back to Docling:", e)
    figures, tables = []; full_markdown_text = ""
    # --- Use Docling to parse PDF ---
    from docling_parse.pdf_parser import DoclingPdfParser
    parser = DoclingPdfParser()
    pdf_doc = parser.load(pdf_path)
    # The PdfDocument object contains pages with text elements and images
    for page in pdf_doc:
        # Extract text from page (docling provides text boxes or lines in reading order)
        page_text = "".join([t.text for t in page.text_cells])  # pseudo-code
        full_markdown_text += page_text + "\n"
        for img_obj in page.images:
            img_data = img_obj.to_image_bytes()  # get raw image bytes
            img_b64 = base64.b64encode(img_data).decode('utf-8')
            # We don't have caption directly; could attempt to find nearby text for caption.
            entry = {
                "id": f"Image {len(figures)+1}",
                "caption": "",  # caption extraction from Docling would require extra logic
                "width": img_obj.width, "height": img_obj.height,
                "base64": img_b64
            }
            figures.append(entry)
    # (Optional: attempt to identify captions for Docling images by spatially finding text below the image's bounding box that matches figure/table caption patterns.)
```

Fallback heuristics: We trigger the above if:
	•	The Mistral API call returns an error/exception (e.g. network issues or API error).
	•	The returned content is suspiciously empty or incomplete. For example, if the PDF is 20 pages but Mistral returns only a few lines of text, it likely failed to extract properly. We can estimate expected length by PDF page count (if text-based) or file size.
	•	The output is missing major components. For instance, if we know the PDF has figures/tables but Mistral’s result contains none, it might indicate a parsing failure (though this is rare). In such cases, Docling (which directly extracts embedded text and images) might still recover content.

Docling’s parser will extract text and images from the PDF’s internal structure. It works well for digitally-generated PDFs (text is embedded) and can extract images (both raster figures and vector graphics as bitmaps)  . However, Docling does not perform OCR – if the PDF is purely scanned images, Docling would yield little text (hence, in those cases Mistral is the only solution).

In the fallback, we use Docling to gather text (concatenating text boxes in reading order for each page) and to extract images. We then format this extracted content into the same schema:
	•	The full_markdown_text from Docling may lack Markdown formatting (it’s mostly plain text). We can still attempt to preserve basic structure: for example, Docling can provide font size or style information that we might use to infer headings vs body text (not shown above for brevity). In practice, since the LLM will summarize the content, having plain text with section titles still recognizable (e.g., by all-caps or line breaks) might suffice.
	•	Each image from Docling is added to the figures list. (Docling doesn’t inherently classify images vs tables or extract captions; implementing caption association in fallback may involve searching the text for lines like “Figure X” near the image’s position. Depending on complexity, this can be an extended feature. For now, fallback images may have empty captions or we attempt a simple regex in the text to find caption text.)
	•	We do not use Marker at all in this new pipeline. Marker is fully removed, simplifying the dependency chain. Mistral OCR covers both OCR and layout understanding, and Docling covers the pure parsing fallback. (Marker’s multi-model pipeline is effectively supplanted by Mistral’s single-model API which yields comparable or better results in one step.)

After Docling fallback extraction, we proceed to the same LLM summarization step to produce the JSON outline. The downstream steps don’t need to know whether Mistral or Docling was used – they receive the final _raw_content.json, _images.json, _tables.json in the expected format.

### Step 3 (continued): Removing Marker and Ensuring Compatibility

Marker is entirely removed from parse_raw.py. All calls or imports related to Marker are deleted. This reduces complexity (no need to manage Marker’s GPU models or deal with its output format). Mistral’s OCR provides the Markdown and images directly  , and our code maps it to the same JSON schema that Marker/Docling provided. The output JSON files conform exactly to the current schema:
	•	_raw_content.json: Contains the JSON outline of the paper’s content. (In the original pipeline this was described as a JSON-like outline with sections, which we preserve .) For example, it might look like:
Example:
```JSON
{
  "sections": [
    {
      "title": "Introduction",
      "description": "A brief overview of the problem statement and approach..."
    },
    {
      "title": "Methodology",
      "description": "Key details of the proposed multi-agent pipeline, including Parser, Planner, etc..."
    },
    ...
  ]
}
```

All section titles and summaries come from the LLM, as before. If the original schema used different keys or nested structure, we output the same (our LLM prompt can be the same to ensure consistency).
	•	_images.json: Contains the list of figure entries (what the paper refers to as image_information). Each entry includes:
	•	id: the figure identifier (e.g., "Figure 1"), which will be used by the Planner or the asset-matching agent to refer to this image  .
	•	caption: the figure’s caption text (without the leading label, just the descriptive part).
	•	width and height: the original size of the figure in the PDF (in points or pixels). These serve as size constraints for layout planning , ensuring the poster design knows the relative sizing (for example, a large chart versus a small icon).
	•	base64: the image data, base64-encoded. Downstream, the Painter agent can decode this to actually render the image into the poster file. (We do not include any image file paths – everything is self-contained in base64, as was done in the original pipeline to maintain portability.)
	•	_tables.json: Similarly, a list of table entries. Schema is the same as images, but these are tables (with id like "Table 1"). For tables, caption holds the table caption text. The base64 would typically be an image of the table. In cases where the table was originally text and we didn’t convert it to an image in this stage, one could either: (a) generate an image from it later in the Painter stage, or (b) include the data in a structured form. The original Paper2Poster appears to treat tables as visual blocks as well, with captions and size info , so we follow that. If Mistral identified a table as an image (e.g., a screenshot or figure of a table), it will already be in the images list with a caption starting “Table X: …”, and we will place it in tables.json. If the table was textual and not captured as an image by Mistral, our pipeline currently would leave it in the Markdown text. (However, the LLM summary likely will not list every table cell; it might just mention the presence of a table in the summary. The Planner’s asset-matching step can only assign images/tables that exist in the lists  . Thus, for completeness, one could extend the parser to render textual tables to an image in the future. In this design, we focus on using Mistral’s outputs directly. Mistral’s table formatting in Markdown could be preserved in the text summary if needed.)

By replicating the original keys and structure, we ensure compatibility. For example, the Asset Matching agent (as described in the paper) expects json_content (the outline), image_information, and table_information as inputs . Our output provides those same objects (the JSON outline and the two lists) so that the agent can map sections to images/tables seamlessly, just as before.

Step 4: Performance and Batching Considerations

Batch processing: Mistral’s OCR API can handle up to 1000 pages per document in a single request (a limit noted in the service documentation). If a PDF exceeds this (very rare for academic papers), our implementation will batch the processing. We can split the page range into chunks of at most 1000 pages. For example, a 1200-page document would be processed in two calls: pages 0-999 and 1000-1199. After each call, we combine the markdown and images outputs. This requires adjusting image IDs to remain unique across batches (for instance, continue numbering images sequentially). We also have to be mindful of ordering – we append the markdown in the correct sequence so that the text content isn’t jumbled. The overhead for two batches is minimal besides two HTTP calls, and since this is an extreme case, it’s acceptable.

Memory and speed: Including image_base64 means the API response can be large for image-heavy papers. We must ensure to handle the JSON parsing efficiently and perhaps drop unnecessary fields. For instance, we don’t actually need the per-page "dimensions" or the original "document_annotation" (if not used) – we can ignore those to save memory. It’s also wise to set a reasonable timeout for the API call (e.g., 60 seconds or more for very long documents) and possibly stream the response if supported. In practice, for typical papers (5-20 pages with a handful of figures), Mistral’s response arrives in a few seconds and is manageable in size.

Asynchronous processing: For now, we ignore advanced async concurrency. The pipeline can be synchronous: call Mistral, get JSON, then proceed. This is simpler and aligns with the original single-threaded parsing approach (which waited for Marker/Docling and the LLM sequentially). In the future, if needed, one could overlap the LLM processing with ongoing OCR of later pages, but this adds complexity and is not required for correctness.

Poppler dependency: We keep the poppler library installed (as per the original requirements) . In our new design, Poppler could be used if we decide to render table regions or images in fallback scenarios. For example, if Docling identifies a table’s bounding box and text, we could use Poppler (via pdftoppm or similar) to crop that region into an image for consistency. This is an optional enhancement; our current primary path (Mistral) already extracts images without needing Poppler. Still, having Poppler ensures we can handle edge cases (and it was likely needed by Marker or other PDF tools previously).

Summary of the Revised parse_raw.py
	1.	Primary OCR (Mistral): The PDF is processed through the Mistral OCR API. We send the PDF (base64 or URL) with include_image_base64=True to get Markdown text and images in one step. We optionally leverage bbox_annotation_format for richer metadata, but primarily rely on the default output. The result is parsed into a combined Markdown string (text_content) and two asset lists (images.json and tables.json) containing base64 images and captions.
	2.	Quality Check & Fallback: If the Mistral output is empty, incomplete, or an error occurs, we invoke Docling as a fallback. Docling extracts text and images from the PDF’s internal structure. We then format that output in the same way (though Docling may not provide nice Markdown, we still obtain the raw text and images to work with). Marker is not used at all in the new pipeline.
	3.	LLM Distillation: We feed the extracted Markdown text into the existing LLM prompt to produce the JSON outline of the paper’s content (sections and summaries). This yields the _raw_content.json in the same format as before , ensuring the Planner agent receives the structured asset library it expects (text sections + images + tables).
	4.	Output Files: The code writes out _raw_content.json, _images.json, and _tables.json (and any other necessary intermediate files as in the original design) with the same schema as the original Marker/Docling pipeline. For example, the Planner will load the image_information from _images.json and table_information from _tables.json and find entries with id and caption exactly as before . No downstream code changes are needed.
	5.	Performance: We handle large PDFs by batching the OCR calls (max 1000 pages per batch) and ensure memory/timeouts are managed. Asynchronous operation is not required initially, but the design doesn’t preclude adding it later.

By implementing these changes in PosterAgent/parse_raw.py, the parser becomes both simpler and more robust. We rely on a state-of-the-art OCR service to do in one stage what previously required multiple tools. We maintain all necessary information (text content, structure, figures, tables) in the output. This aligns with the original system’s goal of distilling a long paper into a structured, multimodal content library  , ready for automated poster generation. The new pipeline will thus achieve the same end result (a JSON content outline plus assets) with a cleaner implementation, paving the way for improved accuracy and easier maintenance.
