"""
FAISS Vector Index Management
Supports in-memory session indices for uploaded documents and persistent knowledge base indices.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from app.config import EMBEDDING_DIM
from .embeddings import generate_embeddings


class VectorIndex:
    """FAISS-backed Vector Index for Legal RAG."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product on normalized vectors = Cosine similarity
        self.chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed chunks and add them to the index."""
        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 3, score_threshold: float = 0.20) -> List[Dict[str, Any]]:
        """
        Search for most relevant chunks.
        Returns sorted list of chunks with their similarity score.
        """
        if self.index.ntotal == 0 or not self.chunks:
            return []

        query_emb = generate_embeddings([query])
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < score_threshold:
                continue

            chunk_data = dict(self.chunks[idx])
            chunk_data["score"] = float(score)
            results.append(chunk_data)

        return results

    def save(self, index_dir: Path):
        """Save FAISS index and chunks metadata to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = index_dir / "index.faiss"
        meta_path = index_dir / "chunks.json"

        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> Optional["VectorIndex"]:
        """Load index from disk if files exist."""
        index_dir = Path(index_dir)
        faiss_path = index_dir / "index.faiss"
        meta_path = index_dir / "chunks.json"

        if not faiss_path.exists() or not meta_path.exists():
            return None

        instance = cls()
        instance.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            instance.chunks = json.load(f)

        return instance

    def clear(self):
        """Reset index and stored chunks."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
