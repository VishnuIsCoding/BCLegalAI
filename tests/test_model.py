"""
Unit Tests for Model Engine & Lazy Loading
"""

import pytest
from app.model import get_model_engine


def test_model_engine_singleton():
    e1 = get_model_engine()
    e2 = get_model_engine()
    assert e1 is e2


def test_lazy_loading_state():
    engine = get_model_engine()
    # Before explicit load call, engine is either not loaded or initialized
    assert hasattr(engine, "is_loaded")
    assert hasattr(engine, "device")
