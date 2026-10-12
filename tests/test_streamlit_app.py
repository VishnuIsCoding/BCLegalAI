"""
Headless Streamlit UI Test using st.testing.v1.AppTest
Verifies that app/main.py runs without error, local model is loaded by default on startup,
and all UI elements (title, sidebar, file uploader, buttons) are rendered properly.
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_startup_and_ui():
    at = AppTest.from_file("../app/main.py", default_timeout=30)
    at.run()

    # Verify no unhandled exceptions during run
    assert not at.exception, f"Streamlit app threw an exception: {at.exception}"

    # Verify main title and subtitle
    titles = [t.value for t in at.title]
    assert any("INDIAN LEGAL AI" in t for t in titles), f"Titles found: {titles}"

    # Verify sidebar elements
    sidebar_markdown = [m.value for m in at.sidebar.markdown]
    assert any("Indian Legal Qwen 1.5B" in m for m in sidebar_markdown)
    assert any("4-bit (NF4)" in m for m in sidebar_markdown)
    assert any("RAG:" in m for m in sidebar_markdown)

    # Verify model is ready by default
    success_boxes = [s.value for s in at.success]
    assert any("Legal AI Model is Ready" in s for s in success_boxes), f"Success boxes found: {success_boxes}"

    # Verify buttons
    buttons = [b.label for b in at.button]
    assert any("Clear Chat" in b for b in buttons)
    assert any("Clear Docs" in b for b in buttons)


def test_streamlit_chat_greeting():
    at = AppTest.from_file("../app/main.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Type 'hello' in chat input and run
    if at.chat_input:
        at.chat_input[0].set_value("hello").run()
        assert not at.exception
        # Verify assistant responded with greeting
        markdowns = [m.value for m in at.markdown]
        assert any("Namaste! Welcome to Indian Legal AI" in m for m in markdowns)
