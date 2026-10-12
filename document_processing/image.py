"""
Image Processing for Legal Documents
Supports PNG, JPG, JPEG with local OCR.
"""

import os
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import io
from .ocr import perform_ocr


def process_image(file_data: Union[str, bytes, Path], filename: str = "document_image") -> Dict[str, Any]:
    """
    Process an uploaded legal document image (PNG/JPG/JPEG).
    Returns structured text and metadata.
    """
    try:
        if isinstance(file_data, (str, Path)):
            img = Image.open(file_data)
            doc_name = Path(file_data).name
        else:
            img = Image.open(io.BytesIO(file_data))
            doc_name = filename

        ocr_result = perform_ocr(img)

        return {
            "document": doc_name,
            "type": "image",
            "page": 1,
            "text": ocr_result["text"],
            "status": ocr_result["status"],
            "message": ocr_result["message"],
        }
    except Exception as e:
        return {
            "document": filename,
            "type": "image",
            "page": 1,
            "text": "",
            "status": "error",
            "message": f"Failed to process image: {str(e)}",
        }
