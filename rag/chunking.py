"""
Text Chunking for Indian Legal Documents
Splits documents by sections, paragraphs, and clauses while retaining strict page and document metadata.
"""

from typing import List, Dict, Any
import re
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """Normalize whitespace and newlines."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_section_title(text: str) -> str:
    """Extract section or clause headings if present (e.g. 'Section 73', 'Clause 4.1', etc.)."""
    patterns = [
        r"(Section\s+\d+[A-Za-z]?(?:\s*[-–:]\s*[^\n]+)?)",
        r"(Clause\s+\d+(?:\.\d+)*(?:\s*[-–:]\s*[^\n]+)?)",
        r"(Article\s+\d+[A-Za-z]?(?:\s*[-–:]\s*[^\n]+)?)",
        r"(###\s+[^\n]+)",
        r"(ORDER\s+[^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace("###", "").strip()
    return "General"


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split pages into structured chunks with document, page, and section metadata.
    """
    chunks: List[Dict[str, Any]] = []

    for page_info in pages:
        doc_name = page_info.get("document", "document")
        page_num = page_info.get("page", 1)
        raw_text = clean_text(page_info.get("text", ""))

        if not raw_text or raw_text == "[Empty text document]" or raw_text == "[No readable text found on this page]":
            continue

        # Split into logical paragraphs
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        current_chunk_words: List[str] = []
        current_section = "General"

        for para in paragraphs:
            para_section = extract_section_title(para)
            if para_section != "General":
                current_section = para_section

            words = para.split()
            if not words:
                continue

            # If adding this paragraph exceeds chunk_size, flush the current chunk
            if len(current_chunk_words) + len(words) > chunk_size and current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                chunks.append({
                    "chunk_id": f"{doc_name}_p{page_num}_{len(chunks) + 1}",
                    "text": chunk_text,
                    "document": doc_name,
                    "page": page_num,
                    "section": current_section,
                })

                # Carry over overlap words
                current_chunk_words = current_chunk_words[-chunk_overlap:] if chunk_overlap < len(current_chunk_words) else []

            current_chunk_words.extend(words)

        # Flush any remaining words
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunks.append({
                "chunk_id": f"{doc_name}_p{page_num}_{len(chunks) + 1}",
                "text": chunk_text,
                "document": doc_name,
                "page": page_num,
                "section": current_section,
            })

    return chunks
