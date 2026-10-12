"""
Legal Retrieval Engine
Coordinates retrieval across Knowledge Base index and uploaded Document index.
"""

from typing import List, Dict, Any, Optional
from .index import VectorIndex


def format_context_and_sources(retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format retrieved chunks into prompt context and structured sources.
    """
    if not retrieved_chunks:
        return {
            "context_str": "No matching legal source material found in the database.",
            "sources": [],
            "has_sources": False,
        }

    context_parts = []
    sources = []

    for i, chunk in enumerate(retrieved_chunks):
        doc = chunk.get("document", "Unknown Document")
        page = chunk.get("page", 1)
        section = chunk.get("section", "General")
        text = chunk.get("text", "").strip()

        context_parts.append(
            f"--- [Source {i+1} | {doc} | Page {page} | Section: {section}] ---\n{text}"
        )

        sources.append({
            "index": i + 1,
            "document": doc,
            "page": page,
            "section": section,
            "score": chunk.get("score", 0.0),
            "snippet": text[:180] + ("..." if len(text) > 180 else ""),
        })

    return {
        "context_str": "\n\n".join(context_parts),
        "sources": sources,
        "has_sources": True,
    }


def retrieve_relevant_legal_context(
    query: str,
    doc_index: Optional[VectorIndex] = None,
    kb_index: Optional[VectorIndex] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve top chunks prioritizing uploaded documents if present, supplemented by Knowledge Base.
    """
    all_chunks: List[Dict[str, Any]] = []

    # 1. Search uploaded document index first if available
    if doc_index and doc_index.total_chunks > 0:
        doc_chunks = doc_index.search(query, top_k=top_k, score_threshold=0.15)
        all_chunks.extend(doc_chunks)

    # 2. Search knowledge base index
    if kb_index and kb_index.total_chunks > 0:
        remaining_k = max(2, top_k - len(all_chunks))
        kb_chunks = kb_index.search(query, top_k=remaining_k, score_threshold=0.20)
        all_chunks.extend(kb_chunks)

    # Sort combined by score descending
    all_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_chunks = all_chunks[:top_k]

    return format_context_and_sources(top_chunks)
