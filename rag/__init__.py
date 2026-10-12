"""
RAG Package for Indian Legal AI
"""

from .embeddings import get_embedding_model, generate_embeddings
from .chunking import chunk_document_pages
from .index import VectorIndex
from .retrieval import retrieve_relevant_legal_context, format_context_and_sources

__all__ = [
    "get_embedding_model",
    "generate_embeddings",
    "chunk_document_pages",
    "VectorIndex",
    "retrieve_relevant_legal_context",
    "format_context_and_sources",
]
