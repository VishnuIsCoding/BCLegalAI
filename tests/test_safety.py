"""
Unit Tests for Safety, Sanitization, and Hallucination Control
"""

import pytest
from app.safety import (
    sanitize_overconfident_language,
    extract_cited_sections,
    validate_and_refine_response,
)


def test_sanitize_overconfident_language():
    bad_text = "Do not worry, you will win the case and this is definitely legal."
    cleaned = sanitize_overconfident_language(bad_text)
    assert "you will win" not in cleaned.lower()
    assert "definitely legal" not in cleaned.lower()


def test_extract_cited_sections():
    text = "Under Section 73 and Article 21, the court noted Clause 4.1."
    sections = extract_cited_sections(text)
    assert any("Section 73" in s for s in sections)
    assert any("Article 21" in s for s in sections)
    assert any("Clause 4.1" in s for s in sections)


def test_validate_and_refine_response_with_unverified():
    generated = "## Answer\nBreach occurred.\n## Legal Basis\nSection 999 applies."
    context = "Section 73 provides compensation for breach."
    sources = [{"document": "act.txt", "page": 1, "section": "Section 73"}]

    res = validate_and_refine_response(generated, context, sources)
    assert res["is_safe"] is True
    # Section 999 is not in context, so it should be flagged
    assert any("999" in s for s in res["unverified_citations"])
    assert "not a substitute for" in res["text"].lower()
