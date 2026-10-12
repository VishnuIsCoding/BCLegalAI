"""
DOCX Document Processing using python-docx
Extracts headings, paragraphs, and tables while preserving document structure.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Union
import docx


def process_docx(file_data: Union[str, bytes, Path], filename: str = "document.docx") -> List[Dict[str, Any]]:
    """
    Process DOCX file and extract structured text preserving headings, paragraphs, and tables.
    Returns list of page/section dictionaries.
    """
    results: List[Dict[str, Any]] = []

    try:
        if isinstance(file_data, (str, Path)):
            doc = docx.Document(str(file_data))
            doc_name = Path(file_data).name
        else:
            doc = docx.Document(io.BytesIO(file_data))
            doc_name = filename

        text_lines = []

        # 1. Extract paragraphs and headings
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style_name = para.style.name.lower() if para.style else ""
            if "heading" in style_name or "title" in style_name:
                text_lines.append(f"\n### {text}\n")
            else:
                text_lines.append(text)

        # 2. Extract tables
        for t_idx, table in enumerate(doc.tables):
            table_lines = [f"\n[Table {t_idx + 1}]"]
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # filter duplicates caused by merged cells
                cleaned_cells = []
                for c in cells:
                    if not cleaned_cells or c != cleaned_cells[-1]:
                        cleaned_cells.append(c)
                if any(cleaned_cells):
                    table_lines.append(" | ".join(cleaned_cells))
            if len(table_lines) > 1:
                text_lines.append("\n".join(table_lines) + "\n")

        full_text = "\n\n".join(text_lines).strip()
        if not full_text:
            full_text = "[Empty DOCX document]"

        results.append({
            "document": doc_name,
            "page": 1,
            "text": full_text,
            "is_ocr": False,
        })

    except Exception as e:
        results.append({
            "document": filename,
            "page": 1,
            "text": f"Error extracting DOCX: {str(e)}",
            "is_ocr": False,
        })

    return results
