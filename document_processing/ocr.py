"""
Local Tesseract OCR Engine
Processes images of legal documents, notices, agreements, and scans.
Runs strictly on CPU to avoid competing with Qwen for GPU VRAM.
"""

import logging
from typing import Dict, Any, Union
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

logger = logging.getLogger(__name__)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for optimal Tesseract OCR accuracy on scanned legal text.
    Converts to grayscale and enhances contrast.
    """
    try:
        # Convert to grayscale
        gray = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        # Apply slight sharpening
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        return sharpened
    except Exception as e:
        logger.warning(f"Image preprocessing failed, using original: {e}")
        return image


def perform_ocr(image: Union[Image.Image, bytes, str]) -> Dict[str, Any]:
    """
    Perform local OCR using Tesseract.
    Returns extracted text and quality status.
    """
    try:
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            import io
            img = Image.open(io.BytesIO(image))
        else:
            img = image

        processed = preprocess_image(img)
        # Configure Tesseract for document OCR
        config = "--oem 3 --psm 3"
        text = pytesseract.image_to_string(processed, config=config)
        cleaned_text = text.strip()

        if not cleaned_text or len(cleaned_text) < 15:
            # Try alternate PSM (e.g., PSM 6: Assume a single uniform block of text)
            alt_text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6").strip()
            if len(alt_text) > len(cleaned_text):
                cleaned_text = alt_text

        if not cleaned_text:
            return {
                "text": "",
                "status": "warning",
                "message": "The image text could not be read reliably. Please upload a clearer image.",
            }

        return {
            "text": cleaned_text,
            "status": "success",
            "message": "OCR completed successfully.",
        }

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return {
            "text": "",
            "status": "error",
            "message": f"OCR processing failed: {str(e)}",
        }
