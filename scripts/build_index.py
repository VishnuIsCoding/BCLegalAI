"""
Build Knowledge Base FAISS Index
Reads authoritative Indian legal files from knowledge_base/raw/,
chunks them, associates rich metadata, and creates the persistent FAISS index in knowledge_base/index/.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import KB_RAW_DIR, KB_METADATA_DIR, KB_INDEX_DIR
from document_processing.txt import process_txt
from rag.chunking import chunk_document_pages
from rag.index import VectorIndex


def build_knowledge_base_index():
    print("==================================================")
    print("INDIAN LEGAL AI - KNOWLEDGE BASE INDEX BUILDER")
    print("==================================================")

    # 1. Load metadata if available
    metadata_file = KB_METADATA_DIR / "sources_metadata.json"
    doc_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            doc_metadata = json.load(f)

    # 2. Collect all raw files
    raw_files = list(KB_RAW_DIR.glob("*.txt"))
    if not raw_files:
        print("[!] No raw text files found in knowledge_base/raw/")
        return

    all_chunks = []
    print(f"[*] Processing {len(raw_files)} authoritative legal documents...")

    for file_path in raw_files:
        filename = file_path.name
        meta = doc_metadata.get(filename, {})
        title = meta.get("title", filename)

        pages = process_txt(file_path, filename=filename)
        chunks = chunk_document_pages(pages, chunk_size=350, chunk_overlap=50)

        # Attach enhanced metadata
        for chunk in chunks:
            chunk["title"] = title
            chunk["jurisdiction"] = meta.get("jurisdiction", "India")
            chunk["date"] = meta.get("date", "Current")
            chunk["version"] = meta.get("version", "In Force")

        print(f"  - {filename}: {len(chunks)} chunks created (Title: {title[:40]}...)")
        all_chunks.extend(chunks)

    # 3. Create FAISS index
    print(f"[*] Generating CPU embeddings and building FAISS index for {len(all_chunks)} total chunks...")
    index = VectorIndex()
    index.add_chunks(all_chunks)

    # 4. Save to disk
    index.save(KB_INDEX_DIR)
    print(f"[✓] Knowledge base index built and saved to: {KB_INDEX_DIR}")
    print(f"    - Index file: {KB_INDEX_DIR / 'index.faiss'}")
    print(f"    - Chunks file: {KB_INDEX_DIR / 'chunks.json'}")
    print(f"    - Total indexed chunks: {index.total_chunks}")


if __name__ == "__main__":
    build_knowledge_base_index()
