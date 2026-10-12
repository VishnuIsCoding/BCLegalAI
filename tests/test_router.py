"""
Unit Tests for Query Router
"""

import pytest
from app.router import classify_query, QueryCategory


def test_legal_questions():
    legal_queries = [
        "What is a contract?",
        "What is breach of contract?",
        "What is bail?",
        "What is an FIR?",
        "What is consideration in contract law?",
        "What is consumer protection?",
        "Can a minor sign a contract?",
        "What is Section 73 of Contract Act?",
        "How do I file a writ petition under Article 32?",
    ]
    for q in legal_queries:
        cat, _ = classify_query(q)
        assert cat == QueryCategory.LEGAL_QUESTION, f"Failed for query: {q}"


def test_non_legal_questions():
    non_legal_queries = [
        "Write Python code to sort a list.",
        "What's the weather today?",
        "Who won yesterday's cricket match?",
        "Tell me a joke.",
        "Can you write a Java program for me?",
        "Who is the best movie actor?",
    ]
    for q in non_legal_queries:
        cat, _ = classify_query(q)
        assert cat == QueryCategory.GENERAL_QUESTION, f"Failed for query: {q}"


def test_document_queries():
    cat, _ = classify_query("Explain clause 8 of this agreement.", has_uploaded_doc=True)
    assert cat == QueryCategory.DOCUMENT_ANALYSIS

    cat2, _ = classify_query("What does this legal notice mean?", has_uploaded_doc=True)
    assert cat2 == QueryCategory.DOCUMENT_ANALYSIS


def test_image_queries():
    cat, _ = classify_query("What does this image say?", has_uploaded_image=True)
    assert cat == QueryCategory.IMAGE_ANALYSIS


def test_greetings():
    greetings = [
        "hello",
        "Hi",
        "hey there",
        "Namaste",
        "good morning",
        "who are you",
        "what can you do",
        "help",
        "how can you help me",
    ]
    for g in greetings:
        cat, _ = classify_query(g)
        assert cat == QueryCategory.GREETING, f"Failed for greeting: {g}"
