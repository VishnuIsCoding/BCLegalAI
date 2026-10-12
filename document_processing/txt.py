"""
Plain Text (TXT) Document Processing
Reads UTF-8 with fallback encodings to handle encoding errors gracefully.
"""

from pathlib import Path
from typing import List, Dict, Any, Union


def process_txt(file_data: Union[str, bytes, Path], filename: str = "document.txt") -> List[Dict[str, Any]]:
    """
    Process plain text file with graceful encoding fallback (UTF-8, Latin-1, CP1252).
    """
    results: List[Dict[str, Any]] = []

    try:
        if isinstance(file_data, (str, Path)):
            doc_name = Path(file_data).name
            with open(file_data, "rb") as f:
                raw_bytes = f.read()
        else:
            doc_name = filename
            raw_bytes = file_data

        # Attempt decoding with UTF-8 first, fallback to common alternatives
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            # Fallback with replacement
            text = raw_bytes.decode("utf-8", errors="replace")

        cleaned = text.strip()
        if not cleaned:
            cleaned = "[Empty text document]"

        results.append({
            "document": doc_name,
            "page": 1,
            "text": cleaned,
            "is_ocr": False,
        })

    except Exception as e:
        results.append({
            "document": filename,
            "page": 1,
            "text": f"Error extracting TXT: {str(e)}",
            "is_ocr": False,
        })

    return results
