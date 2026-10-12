"""
Unit Tests for Cloud Fallback Engine and Difficulty Router
"""

import pytest
from unittest.mock import patch, MagicMock
from app.fallback import (
    CloudFallbackEngine,
    CloudProvider,
    detect_provider,
    mask_api_key,
)
from app.router import is_difficult_legal_query, classify_query, QueryCategory
from app.rag import LegalAssistantCoordinator


def test_detect_provider():
    # Gemini keys
    assert detect_provider("AQ.mock_dummy_test_key_1234567890abcdef") == CloudProvider.GEMINI
    assert detect_provider("AIzaSyD-1234567890abcdef") == CloudProvider.GEMINI
    # Groq keys
    assert detect_provider("gsk_1234567890abcdef") == CloudProvider.GROQ
    # Legacy xAI keys
    assert detect_provider("xai-1234567890abcdef") == CloudProvider.XAI
    # Configured overrides
    assert detect_provider("custom_key", configured_provider=CloudProvider.GEMINI) == CloudProvider.GEMINI
    assert detect_provider("custom_key", configured_provider=CloudProvider.GROQ) == CloudProvider.GROQ
    assert detect_provider("custom_key", configured_provider=CloudProvider.XAI) == CloudProvider.XAI


def test_mask_api_key():
    assert mask_api_key("AQ.mock_dummy_test_key_1234567890abcdef") == "AQ.mock_...cdef"
    assert mask_api_key("") == "None"
    assert mask_api_key("shortkey") == "sho...ey"


def test_is_difficult_legal_query():
    # 1. Basic question with simulated high RAG confidence -> not difficult
    sim_rag_high = {
        "has_sources": True,
        "sources": [{"score": 0.55, "document": "indian_contract_act_1872.txt"}],
    }
    is_diff, reason = is_difficult_legal_query("What is a contract?", retrieval_result=sim_rag_high)
    assert is_diff is False

    # 2. Specialized domain: Companies Act / Director liability -> difficult
    is_diff, reason = is_difficult_legal_query("Can a company director be held personally liable for corporate debt?")
    assert is_diff is True
    assert "Corporate Law" in reason

    # 3. Specialized domain: GST / Taxation -> difficult
    is_diff, reason = is_difficult_legal_query("What are the conditions to claim input tax credit under GST?")
    assert is_diff is True
    assert "Taxation" in reason

    # 4. Specialized domain: Insolvency / IBC -> difficult
    is_diff, reason = is_difficult_legal_query("What happens during moratorium under Section 14 of IBC?")
    assert is_diff is True
    assert "Insolvency" in reason

    # 5. Specialized domain: Hindu Succession -> difficult
    is_diff, reason = is_difficult_legal_query("How is ancestral property partitioned under Hindu Succession Act?")
    assert is_diff is True
    assert "Succession" in reason

    # 6. Comparative query -> difficult
    is_diff, reason = is_difficult_legal_query("What is the difference between theft and extortion?")
    assert is_diff is True
    assert "Comparative" in reason

    # 7. Low RAG confidence -> difficult
    sim_rag_low = {
        "has_sources": True,
        "sources": [{"score": 0.22, "document": "general.txt"}],
    }
    is_diff, reason = is_difficult_legal_query("What is specific performance?", retrieval_result=sim_rag_low, min_confidence=0.35)
    assert is_diff is True
    assert "Low local knowledge base" in reason


def test_fallback_generate_gemini_success():
    engine = CloudFallbackEngine(api_key="AQ.test-gemini-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "## Answer\nA director is generally not personally liable unless fraud is established.\n\n## Legal Basis\nCompanies Act 2013, Section 166.\n\n## Important Note\nGeneral legal info."
                }
            }
        ],
        "usage": {"completion_tokens": 42},
    }

    with patch("requests.post", return_value=mock_resp):
        res = engine.generate("Can a director be liable?", context="Some context")
        assert res["success"] is True
        assert "director is generally not personally liable" in res["text"]
        assert res["provider"] == "gemini"
        assert res["model"] == "gemini-3.5-flash"
        assert res["num_tokens"] == 42


def test_fallback_quota_handling():
    engine = CloudFallbackEngine(api_key="AQ.test-gemini-key")

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {
        "code": "permission-denied",
        "error": "The account has exceeded quota or has credit restrictions.",
    }

    with patch("requests.post", return_value=mock_resp):
        res = engine.generate("What is bail?")
        assert res["success"] is False
        assert "quota" in res["error"] or "credit" in res["error"]


def test_coordinator_smart_routing(monkeypatch):
    coordinator = LegalAssistantCoordinator()

    # Mock the fallback engine to succeed
    mock_fb_res = {
        "success": True,
        "text": "## Answer\nUnder Section 166 of Companies Act 2013, directors owe fiduciary duties.\n\n## Legal Basis\nCompanies Act 2013.\n\n## Explanation\nDetailed analysis.\n\n## Important Note\nNot legal advice.",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "latency_sec": 0.85,
        "tokens_per_sec": 40.0,
        "num_tokens": 50,
    }

    with patch.object(coordinator.fallback_engine, "generate", return_value=mock_fb_res):
        # Query that triggers difficult detection (Corporate law)
        res = coordinator.query(
            "Can a company director be held personally liable for debt under Companies Act?",
            fallback_mode="auto",
        )
        assert res["is_fallback"] is True
        assert res["provider"] == "gemini"
        assert "directors owe fiduciary duties" in res["answer"]
        assert res["fallback_attempted"] is True


def test_coordinator_graceful_local_fallback_when_cloud_fails():
    coordinator = LegalAssistantCoordinator()

    # Mock fallback engine failure (e.g. quota/credit restriction)
    mock_fb_err = {
        "success": False,
        "error": "GEMINI account has 0 credits or quota restrictions.",
        "text": "",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
    }

    with patch.object(coordinator.fallback_engine, "generate", return_value=mock_fb_err):
        # Mock local model generation so we don't load 1.5B weights during unit test
        mock_local_res = {
            "text": "## Answer\nLocal answer.\n\n## Legal Basis\nGeneral.\n\n## Important Note\nNot legal advice.",
            "total_time": 0.5,
            "first_token_time": 0.1,
            "num_tokens": 20,
            "tokens_per_second": 30.0,
            "vram_mb": 0.0,
            "device": "cpu",
        }
        with patch.object(coordinator.model_engine, "is_loaded", True), \
             patch.object(coordinator.model_engine, "generate_full", return_value=mock_local_res):

            res = coordinator.query(
                "Can a company director be held personally liable?",
                fallback_mode="auto",
            )
            # Local model should execute as graceful fallback
            assert res["is_fallback"] is False
            assert res["fallback_attempted"] is True
            assert "quota restrictions" in res["fallback_error"]
            assert "Local answer" in res["answer"]
