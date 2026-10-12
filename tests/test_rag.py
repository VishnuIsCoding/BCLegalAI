"""
Unit Tests for RAG (Chunking, CPU Embeddings, FAISS Index, Retrieval)
"""

import pytest
import numpy as np
from pathlib import Path
from rag.chunking import chunk_document_pages
from rag.embeddings import generate_embeddings
from rag.index import VectorIndex
from rag.retrieval import retrieve_relevant_legal_context


def test_chunking():
    pages = [
        {
            "document": "contract.pdf",
            "page": 1,
            "text": "Clause 1: Scope of Work.\n\nClause 2: Consideration of Rs. 10,000.\n\nClause 3: Termination.\nEither party may terminate.",
        }
    ]
    chunks = chunk_document_pages(pages, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 1
    assert chunks[0]["document"] == "contract.pdf"
    assert chunks[0]["page"] == 1


def test_cpu_embeddings():
    texts = ["What is breach of contract?", "Bail provisions under BNSS."]
    embeddings = generate_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    # Check normalization: L2 norm should be approximately 1.0
    norm = np.linalg.norm(embeddings[0])
    assert abs(norm - 1.0) < 1e-3


def test_vector_index_and_retrieval(tmp_path):
    index = VectorIndex(dimension=384)
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Section 2(h) provides that an agreement enforceable by law is a contract.",
            "document": "contract_act.txt",
            "page": 1,
            "section": "Section 2(h)",
        },
        {
            "chunk_id": "c2",
            "text": "Under Section 173 BNSS, every cognizable offence information must be recorded as an FIR.",
            "document": "bnss.txt",
            "page": 1,
            "section": "Section 173",
        },
    ]

    added = index.add_chunks(chunks)
    assert added == 2
    assert index.total_chunks == 2

    # Query contract definition
    results = index.search("What is the legal definition of a contract?", top_k=1)
    assert len(results) == 1
    assert "Section 2(h)" in results[0]["text"]

    # Test saving and loading
    index.save(tmp_path)
    loaded_index = VectorIndex.load(tmp_path)
    assert loaded_index is not None
    assert loaded_index.total_chunks == 2

    res_context = retrieve_relevant_legal_context("contract definition", kb_index=loaded_index, top_k=1)
    assert res_context["has_sources"] is True
    assert len(res_context["sources"]) == 1
