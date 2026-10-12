"""
Embedding Generator using sentence-transformers/all-MiniLM-L6-v2
Strictly executed on CPU to preserve GPU VRAM for the Indian Legal Qwen model.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE

_EMBEDDER_INSTANCE = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton getter for the CPU embedding model."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        # Explicitly enforce CPU execution
        _EMBEDDER_INSTANCE = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    return _EMBEDDER_INSTANCE


def generate_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate normalized L2 embeddings for cosine similarity with FAISS.
    Returns np.ndarray of shape (len(texts), 384) with float32 type.
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,  # Normalizing enables dot-product as cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
