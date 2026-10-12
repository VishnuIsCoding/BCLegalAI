"""
Unit Tests for Document Processing (PDF, DOCX, TXT, Image OCR)
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw
import pymupdf as fitz
import docx
import pytest

from document_processing.txt import process_txt
from document_processing.docx import process_docx
from document_processing.pdf import process_pdf
from document_processing.ocr import perform_ocr
from document_processing.image import process_image


def test_txt_processing():
    sample_text = "Section 2(h): An agreement enforceable by law is a contract."
    res = process_txt(sample_text.encode("utf-8"), "test.txt")
    assert len(res) == 1
    assert "Section 2(h)" in res[0]["text"]
    assert res[0]["document"] == "test.txt"


def test_docx_processing():
    # Create an in-memory DOCX
    doc = docx.Document()
    doc.add_heading("Legal Agreement", level=1)
    doc.add_paragraph("Clause 1: Scope of representation.")
    buf = io.BytesIO()
    doc.save(buf)
    docx_bytes = buf.getvalue()

    res = process_docx(docx_bytes, "test.docx")
    assert len(res) == 1
    assert "Clause 1" in res[0]["text"]
    assert "Legal Agreement" in res[0]["text"]


def test_pdf_processing():
    # Create an in-memory PDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "INDIAN CONTRACT ACT\nSection 73 Compensation for breach")
    pdf_bytes = doc.tobytes()
    doc.close()

    res = process_pdf(pdf_bytes, "contract.pdf")
    assert len(res) == 1
    assert res[0]["page"] == 1
    assert "Section 73" in res[0]["text"]


def test_image_ocr():
    # Create an in-memory image with text rendered large and clear
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw scaled/repeated text for high OCR contrast
    draw.text((40, 60), "LEGAL NOTICE", fill=(0, 0, 0))
    # Resize up to give OCR large glyphs
    img = img.resize((1200, 400), Image.Resampling.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    res = process_image(img_bytes, "notice.png")
    assert res["status"] in ["success", "warning"]
    assert "LEGAL" in res["text"].upper()
