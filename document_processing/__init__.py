"""
Document Processing Package
Unified interface for legal documents: PDF, DOCX, TXT, and Images (OCR).
"""

from pathlib import Path
from typing import List, Dict, Any, Union
from .pdf import process_pdf
from .docx import process_docx
from .txt import process_txt
from .image import process_image
from .ocr import perform_ocr


def process_document(file_data: Union[str, bytes, Path], filename: str) -> List[Dict[str, Any]]:
    """
    Dispatcher to parse any supported document or image into a list of page/chunk dictionaries:
    [
        {
            "document": "...",
            "page": int,
            "text": "...",
            "is_ocr": bool,
        }
    ]
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return process_pdf(file_data, filename)
    elif ext in [".docx", ".doc"]:
        return process_docx(file_data, filename)
    elif ext in [".txt", ".md"]:
        return process_txt(file_data, filename)
    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        res = process_image(file_data, filename)
        return [res]
    else:
        # Fallback to text reading
        return process_txt(file_data, filename)


__all__ = [
    "process_document",
    "process_pdf",
    "process_docx",
    "process_txt",
    "process_image",
    "perform_ocr",
]
