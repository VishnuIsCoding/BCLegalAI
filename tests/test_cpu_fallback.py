"""
Test CPU Fallback Mode
Verifies that the model can be loaded and executed on CPU safely if CUDA fails or is disabled.
"""

import pytest
from app.model import LegalModelEngine


def test_cpu_fallback_loading():
    cpu_engine = LegalModelEngine()
    success, msg = cpu_engine.load(force_cpu=True)
    assert success is True
    assert cpu_engine.device == "cpu"
    assert "CPU" in msg

    # Test short generation on CPU
    res = cpu_engine.generate_full("What is an agreement?", mode="fast")
    assert len(res["text"]) > 10
    assert res["device"] == "cpu"
