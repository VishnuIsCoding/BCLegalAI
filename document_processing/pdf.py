"""
PDF Processing with PyMuPDF (fitz) and OCR Fallback for Scanned Pages
Extracts structured per-page metadata:
{
    "document": "filename.pdf",
    "page": 1,
    "text": "..."
}
If text extraction yields empty or very low text, runs Tesseract OCR on the page image.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
import pymupdf as fitz  # PyMuPDF
from PIL import Image
import io
import logging
from .ocr import perform_ocr

logger = logging.getLogger(__name__)

# Minimum characters on a page to consider text extraction successful without OCR
MIN_TEXT_CHARS = 35


def process_pdf(file_data: Union[str, bytes, Path], filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Process PDF and extract page-by-page text and metadata.
    Automatically applies OCR on scanned pages with little or no digital text.
    """
    results: List[Dict[str, Any]] = []

    try:
        if isinstance(file_data, (str, Path)):
            doc = fitz.open(file_data)
            doc_name = Path(file_data).name
        else:
            doc = fitz.open(stream=file_data, filetype="pdf")
            doc_name = filename

        total_pages = len(doc)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Attempt standard digital text extraction
            extracted_text = page.get_text("text").strip()

            # 2. Check if the page is a scan (empty or very sparse text)
            is_scanned = len(extracted_text) < MIN_TEXT_CHARS

            if is_scanned:
                logger.info(f"Page {page_num} of {doc_name} has low text ({len(extracted_text)} chars). Performing OCR...")
                # Render page to high-res pixmap (2x scale / 144 dpi for clear OCR)
                matrix = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                ocr_res = perform_ocr(img)
                if ocr_res["text"]:
                    extracted_text = f"[OCR Extracted]\n{ocr_res['text']}"
                elif not extracted_text:
                    extracted_text = "[No readable text found on this page]"

            results.append({
                "document": doc_name,
                "page": page_num,
                "text": extracted_text,
                "is_ocr": is_scanned,
            })

        doc.close()

    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        results.append({
            "document": filename,
            "page": 1,
            "text": f"Error extracting PDF: {str(e)}",
            "is_ocr": False,
        })

    return results
