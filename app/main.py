"""
Indian Legal AI - Streamlit Web Application
Fast local Indian legal information and document analysis assistant.
Equipped with Smart Cloud Fallback (Google Gemini & Groq Cloud) for complex/difficult legal questions.
Optimized for NVIDIA RTX 3050 (4 GB VRAM) + AMD Ryzen 7 (16 GB RAM).
"""

import sys
import os
import time
import logging
from pathlib import Path
import streamlit as st

# Ensure project root is prioritized over app directory in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger(__name__)

from app.config import (
    CUDA_AVAILABLE,
    GPU_NAME,
    GPU_VRAM_MB,
    NON_LEGAL_REFUSAL,
    LEGAL_DISCLAIMER,
    DEFAULT_FALLBACK_API_KEY,
    FALLBACK_ENABLED,
)
from app.model import get_model_engine, LegalModelEngine
from app.rag import LegalAssistantCoordinator
from app.router import classify_query, QueryCategory
from app.safety import validate_and_refine_response
try:
    from app.prompts import GREETING_RESPONSE
except (ImportError, AttributeError):
    GREETING_RESPONSE = (
        "🙏 **Namaste! Welcome to Indian Legal AI by Team Blank Paper.**\n\n"
        "I am your dedicated legal assistant, specialized in Indian jurisprudence, statutory frameworks, and legal document analysis. I can help you with:\n\n"
        "⚖️ **Criminal & Procedural Law:** Bharatiya Nyaya Sanhita (BNS 2023), Bharatiya Nagarik Suraksha Sanhita (BNSS 2023), Bharatiya Sakshya Adhiniyam (BSA 2023), IPC, CrPC, FIR filing, and Anticipatory/Regular Bail.\n"
        "📜 **Contracts & Commercial Disputes:** Indian Contract Act 1872, Section 138 NI Act (cheque dishonour/bounce), arbitration, and consumer protection rights.\n"
        "🏠 **Property & Tenancy Law:** Rental agreements, landlord-tenant disputes, eviction procedures, security deposit refund, and RERA rights.\n"
        "💼 **Employment & Labor Rights:** Workplace harassment (POSH), wrongful termination, unpaid salary/dues recovery, gratuity, and EPF.\n"
        "👨‍👩‍👧 **Family & Succession Law:** Maintenance (Section 144 BNSS / Section 125 CrPC), domestic violence remedies, divorce, and ancestral property partition.\n"
        "📑 **Document & Image Analysis:** Upload agreements, legal notices, FIR copies, or deeds for instant clause analysis, risk extraction, and key legal terms.\n\n"
        "Feel free to ask any Indian legal question or upload a document/image to begin!"
    )
from app.fallback import (
    mask_api_key,
    detect_provider,
    CloudProvider,
    get_fallback_engine,
)
from document_processing.ocr import pytesseract

# Set page configuration
st.set_page_config(
    page_title="Indian Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_cached_model_engine() -> LegalModelEngine:
    """Cached loader for the Qwen legal model. Loaded by default on startup."""
    engine = get_model_engine()
    if not engine.is_loaded:
        engine.load()
    return engine


@st.cache_resource
def get_coordinator() -> LegalAssistantCoordinator:
    """Returns singleton assistant coordinator."""
    return LegalAssistantCoordinator()


# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "ocr_preview" not in st.session_state:
    st.session_state.ocr_preview = None

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None

if "fallback_strategy" not in st.session_state:
    st.session_state.fallback_strategy = "Auto (Smart Fallback)"

if "fallback_api_key" not in st.session_state:
    st.session_state.fallback_api_key = DEFAULT_FALLBACK_API_KEY

if "fallback_provider" not in st.session_state:
    st.session_state.fallback_provider = "auto"

if "fallback_test_result" not in st.session_state:
    st.session_state.fallback_test_result = None

# Automatically load the local model by default on startup
engine = get_model_engine()
if not engine.is_loaded and not st.session_state.model_loaded:
    try:
        load_cached_model_engine()
        st.session_state.model_loaded = True
    except Exception as e:
        logger.error(f"Auto-loading model on startup encountered error: {e}")

coordinator = get_coordinator()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚖️ System Status")

    # Local Model specifications
    st.markdown("**Model:** Indian Legal Qwen 1.5B")
    st.markdown("**Quantization:** 4-bit (NF4)")

    # Device Status
    if engine.is_loaded:
        device_badge = f":green[CUDA (Active)]" if engine.device == "cuda" else f":orange[CPU Mode]"
        st.markdown(f"**Device:** {device_badge}")
        vram_val = engine.get_vram_usage_mb()
        st.caption(f"VRAM in use: {vram_val} MB / {GPU_VRAM_MB} MB")
    else:
        st.markdown("**Device:** :red[Not Loaded (Standby)]")

    # RAG Status
    kb_chunks = coordinator.kb_index.total_chunks if coordinator.kb_index else 0
    if kb_chunks > 0:
        st.markdown(f"**RAG:** :green[Ready ({kb_chunks} legal chunks)]")
    else:
        st.markdown("**RAG:** :orange[Not Ready (Index missing)]")

    # OCR Status
    try:
        tess_ver = pytesseract.get_tesseract_version()
        st.markdown(f"**OCR:** :green[Ready (Tesseract {tess_ver})]")
    except Exception:
        st.markdown("**OCR:** :red[Not Available]")

    # Cloud Fallback Status
    curr_key = st.session_state.fallback_api_key
    active_prov = detect_provider(curr_key, st.session_state.fallback_provider)
    if active_prov == CloudProvider.GEMINI:
        prov_label = "Google Gemini"
    elif active_prov == CloudProvider.GROQ:
        prov_label = "Groq Cloud"
    else:
        prov_label = "xAI (Grok)"
    key_preview = mask_api_key(curr_key)
    st.markdown(f"**Cloud Fallback:** :blue[{prov_label} ({key_preview})]")

    st.divider()

    # Inference & Fallback Settings
    st.subheader("⚙️ Settings")
    mode_selection = st.radio(
        "Response Mode:",
        ["Fast Mode", "Detailed Mode"],
        index=0,
        help="Fast Mode prioritizes sub-second responses with top 3 sources. Detailed Mode provides deeper legal analysis.",
    )
    mode = "fast" if mode_selection == "Fast Mode" else "detailed"

    st.markdown("**Fallback Routing:**")
    strategy_selection = st.radio(
        "Routing Strategy:",
        [
            "Auto (Smart Fallback)",
            "Always Cloud (Gemini / Groq)",
            "Local Model Only",
        ],
        index=0,
        label_visibility="collapsed",
        help="Auto routes difficult or out-of-index questions to Cloud Fallback. Always Cloud uses Gemini/Groq for all queries. Local uses only Qwen 1.5B.",
    )
    st.session_state.fallback_strategy = strategy_selection

    # Advanced Cloud Settings Expander
    with st.expander("🔑 Cloud Fallback API Key & Provider", expanded=False):
        new_key = st.text_input(
            "Fallback API Key (Gemini / Groq):",
            value=st.session_state.fallback_api_key,
            type="password",
            help="Supports Gemini keys (starts with AQ. or AIza) or Groq keys (starts with gsk_).",
        )
        if new_key != st.session_state.fallback_api_key:
            st.session_state.fallback_api_key = new_key
            st.session_state.fallback_test_result = None
            st.toast("Fallback API Key updated!", icon="🔑")

        provider_opt = st.selectbox(
            "Cloud Provider:",
            ["Auto-Detect", "Google Gemini", "Groq Cloud"],
            index=0 if st.session_state.fallback_provider == "auto" else (1 if st.session_state.fallback_provider == "gemini" else 2),
        )
        st.session_state.fallback_provider = (
            "auto" if provider_opt == "Auto-Detect"
            else ("gemini" if provider_opt == "Google Gemini" else "groq")
        )

        if st.button("🔍 Test Cloud Connection", width="stretch"):
            with st.spinner("Probing cloud endpoint..."):
                t_res = coordinator.fallback_engine.check_connection(
                    key_override=st.session_state.fallback_api_key,
                    provider_override=None if st.session_state.fallback_provider == "auto" else st.session_state.fallback_provider,
                )
                st.session_state.fallback_test_result = t_res

        if st.session_state.fallback_test_result:
            tres = st.session_state.fallback_test_result
            if tres["status"] == "ready":
                st.success(f"✅ {tres['message']}")
            elif tres["status"] == "no_credits":
                st.warning(f"⚠️ {tres['message']}")
            elif tres["status"] == "invalid_key":
                st.error(f"❌ {tres['message']}")
            else:
                st.info(f"ℹ️ {tres['message']}")

    st.divider()

    # Model Controls (Local model is loaded by default)
    st.subheader("⚡ Model Controls")
    if engine.is_loaded:
        st.success("Legal AI Model is Ready (Loaded by Default)", icon="✅")
        if st.button("🔄 Reload Model", width="stretch"):
            with st.spinner("Reloading Indian Legal Qwen 1.5B into 4-bit memory..."):
                load_cached_model_engine()
                st.session_state.model_loaded = True
                st.toast("Model reloaded successfully!", icon="✅")
                st.rerun()
    else:
        if st.button("🚀 Load Legal AI", type="primary", width="stretch"):
            with st.spinner("Loading Indian Legal Qwen 1.5B into 4-bit memory..."):
                load_cached_model_engine()
                st.session_state.model_loaded = True
                st.toast("Model loaded successfully!", icon="✅")
                st.rerun()

    # Clear Buttons
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
    with col_c2:
        if st.button("Clear Docs", width="stretch"):
            coordinator.clear_session_documents()
            st.session_state.uploaded_file_name = None
            st.session_state.ocr_preview = None
            st.session_state.doc_summary = None
            st.rerun()


# ==========================================
# MAIN PAGE
# ==========================================
st.title("INDIAN LEGAL AI")
st.caption("Fast Indian legal intelligence with local 4-bit Qwen 1.5B & Smart Cloud Fallback (Google Gemini / Groq)")

# Document / Image Uploader
with st.expander("📁 Upload Legal Document or Image (PDF, DOCX, TXT, PNG, JPG, JPEG)", expanded=st.session_state.uploaded_file_name is None):
    uploaded_file = st.file_uploader(
        "Select file to analyze",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        with st.spinner(f"Processing and indexing {uploaded_file.name}..."):
            file_bytes = uploaded_file.read()
            proc_res = coordinator.process_and_index_file(file_bytes, uploaded_file.name)
            st.session_state.uploaded_file_name = uploaded_file.name

            if proc_res.get("type") == "image":
                ocr_txt = proc_res.get("extracted_text", "")
                st.session_state.ocr_preview = ocr_txt
                st.session_state.doc_summary = f"Image OCR completed. {len(ocr_txt)} characters extracted."
                st.success(f"Extracted image text: {len(ocr_txt)} characters.")
            else:
                st.session_state.ocr_preview = None
                p_cnt = proc_res.get("pages_count", 1)
                c_cnt = proc_res.get("num_chunks", 0)
                st.session_state.doc_summary = f"Indexed {p_cnt} page(s), {c_cnt} clauses/chunks."
                st.success(f"Indexed {uploaded_file.name}: {p_cnt} page(s), {c_cnt} chunks ready for RAG.")

    if st.session_state.uploaded_file_name:
        st.markdown(f"**Active Document:** `{st.session_state.uploaded_file_name}` — *{st.session_state.doc_summary}*")
        if st.session_state.ocr_preview:
            with st.expander("View Extracted OCR Text"):
                st.text_area("Extracted Text", st.session_state.ocr_preview, height=180, disabled=True)

st.divider()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Display model badge
            if msg.get("is_fallback"):
                prov_raw = msg.get("provider", "Cloud")
                prov_tag = "Google Gemini" if str(prov_raw).lower() == "gemini" else ("Groq Cloud" if str(prov_raw).lower() == "groq" else str(prov_raw).upper())
                model_tag = msg.get("model", "Cloud LLM")
                st.markdown(f":blue[⚡ **Powered by {prov_tag} ({model_tag})**]")
                if msg.get("fallback_reason"):
                    st.caption(f"🧠 *Smart Fallback Trigger: {msg['fallback_reason']}*")
            else:
                st.markdown(":green[🖥️ **Powered by Local Indian Legal Qwen 1.5B**]")

            if msg.get("fallback_error"):
                st.warning(f"⚠️ *Cloud Fallback Notice: {msg['fallback_error']}*")

        st.markdown(msg["content"])

        if "metrics" in msg and msg["metrics"]:
            st.caption(msg["metrics"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Verified Legal Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**Source {src['index']}:** `{src['document']}` | "
                        f"**Page:** {src['page']} | **Section:** `{src['section']}`"
                    )
                    st.caption(f"> \"{src['snippet']}\"")

# Chat Input
user_input = st.chat_input("Ask an Indian legal question or inquire about the uploaded document...")

if user_input:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant Response
    with st.chat_message("assistant"):
        has_doc = coordinator.doc_index.total_chunks > 0
        has_img = coordinator.latest_ocr_text is not None

        category, reason = classify_query(
            user_input,
            has_uploaded_doc=has_doc,
            has_uploaded_image=has_img,
        )

        # 1. Greeting / Introduction (Instant, zero compute, informative welcome)
        if category == QueryCategory.GREETING:
            st.markdown(GREETING_RESPONSE)
            st.session_state.messages.append({
                "role": "assistant",
                "content": GREETING_RESPONSE,
                "metrics": "Routing: Assistant Introduction & Capabilities | Latency: 0.00s",
                "sources": [],
                "is_fallback": False,
            })

        # 2. Non-legal Refusal (Instant, zero compute)
        elif category == QueryCategory.GENERAL_QUESTION:
            refusal_text = NON_LEGAL_REFUSAL
            st.markdown(refusal_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": refusal_text,
                "metrics": "Routing: Legal Filter Refusal | Latency: 0.00s",
                "sources": [],
                "is_fallback": False,
            })

        # 3. Legal / Document Query
        else:
            # Map UI strategy to coordinator fallback mode
            strat = st.session_state.fallback_strategy
            if "Always" in strat:
                fb_mode = "always"
            elif "Local" in strat:
                fb_mode = "never"
            else:
                fb_mode = "auto"

            with st.spinner("Analyzing legal provisions and formulating grounded answer..."):
                start_exec = time.time()
                result = coordinator.query(
                    user_input,
                    mode=mode,
                    fallback_mode=fb_mode,
                    api_key=st.session_state.fallback_api_key,
                    provider=None if st.session_state.fallback_provider == "auto" else st.session_state.fallback_provider,
                )
                latency = round(time.time() - start_exec, 2)

                answer_text = result["answer"]
                sources = result.get("sources", [])
                tok_per_sec = result.get("tokens_per_sec", 0.0)
                vram_used = result.get("vram_mb", 0.0)
                device_used = result.get("device", "cuda")
                is_fb = result.get("is_fallback", False)
                fb_err = result.get("fallback_error")
                fb_reason = result.get("fallback_reason")

                # Display Cloud or Local Badge
                if is_fb:
                    prov_raw = result.get("provider", "Cloud")
                    prov_tag = "Google Gemini" if str(prov_raw).lower() == "gemini" else ("Groq Cloud" if str(prov_raw).lower() == "groq" else str(prov_raw).upper())
                    model_tag = result.get("model", "Cloud LLM")
                    st.markdown(f":blue[⚡ **Powered by {prov_tag} ({model_tag})**]")
                    if fb_reason:
                        st.caption(f"🧠 *Smart Fallback Trigger: {fb_reason}*")
                else:
                    st.markdown(":green[🖥️ **Powered by Local Indian Legal Qwen 1.5B**]")

                if fb_err:
                    st.warning(f"⚠️ *Cloud Fallback Notice: {fb_err}*")

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Verified Legal Sources ({len(sources)})", expanded=False):
                        for src in sources:
                            st.markdown(
                                f"**Source {src['index']}:** `{src['document']}` | "
                                f"**Page:** {src['page']} | **Section:** `{src['section']}`"
                            )
                            st.caption(f"> \"{src['snippet']}\"")

                metrics_str = f"⚡ {latency}s | {tok_per_sec} tok/s | Device: {device_used.upper()} | VRAM: {vram_used} MB"
                st.caption(metrics_str)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "metrics": metrics_str,
                    "sources": sources,
                    "is_fallback": is_fb,
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "fallback_reason": fb_reason,
                    "fallback_error": fb_err,
                })
