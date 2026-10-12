"""
Legal Assistant Coordinator
Orchestrates Router, Document Parsing, RAG Retrieval, Qwen Model Generation, and Safety Verification.
"""

import sys
from typing import Dict, Any, Optional, List, Generator
from pathlib import Path
import logging

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
if str(ROOT_DIR) in sys.path:
    sys.path.remove(str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from app.config import (
    KB_INDEX_DIR,
    NON_LEGAL_REFUSAL,
    FAST_MODE,
    DETAILED_MODE,
    FALLBACK_ENABLED,
    FALLBACK_CONFIDENCE_THRESHOLD,
)
from app.router import classify_query, QueryCategory, is_difficult_legal_query
from app.prompts import (
    build_legal_qa_prompt,
    build_document_analysis_prompt,
    build_image_ocr_analysis_prompt,
)
try:
    from app.prompts import GREETING_RESPONSE
except (ImportError, AttributeError):
    GREETING_RESPONSE = (
        "🙏 **Namaste! Welcome to Indian Legal AI by Team Blank Paper.**\n\n"
        "I am your dedicated legal assistant, specialized in Indian jurisprudence, statutory frameworks, and legal document analysis."
    )
from app.safety import validate_and_refine_response
from app.model import get_model_engine
from app.fallback import get_fallback_engine, CloudFallbackEngine, CloudProvider
from rag.index import VectorIndex
from rag.retrieval import retrieve_relevant_legal_context
from document_processing import process_document
from rag.chunking import chunk_document_pages

logger = logging.getLogger(__name__)


class LegalAssistantCoordinator:
    """Coordinates RAG, knowledge base, document session index, and model inference."""

    def __init__(self):
        self.kb_index = VectorIndex.load(KB_INDEX_DIR)
        self.doc_index = VectorIndex()
        self.model_engine = get_model_engine()
        self.fallback_engine = get_fallback_engine()
        self.uploaded_doc_name = None
        self.latest_ocr_text = None

    def reload_kb_index(self):
        """Reload KB index from disk."""
        self.kb_index = VectorIndex.load(KB_INDEX_DIR)

    def process_and_index_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded document/image and build the session vector index.
        """
        self.doc_index.clear()
        self.uploaded_doc_name = filename
        self.latest_ocr_text = None

        pages = process_document(file_bytes, filename)
        if not pages:
            return {"status": "error", "message": "Failed to extract content from file."}

        # Check if it's an image with OCR text
        if pages[0].get("type") == "image":
            ocr_text = pages[0].get("text", "")
            self.latest_ocr_text = ocr_text
            status = pages[0].get("status", "success")
            msg = pages[0].get("message", "")
            if status != "error" and ocr_text:
                chunks = chunk_document_pages(pages, chunk_size=300, chunk_overlap=40)
                self.doc_index.add_chunks(chunks)
            return {
                "status": status,
                "message": msg,
                "type": "image",
                "extracted_text": ocr_text,
                "num_chunks": self.doc_index.total_chunks,
            }

        # Standard document (PDF/DOCX/TXT)
        chunks = chunk_document_pages(pages, chunk_size=350, chunk_overlap=50)
        added_count = self.doc_index.add_chunks(chunks)

        return {
            "status": "success",
            "message": f"Successfully processed {len(pages)} page(s) and indexed {added_count} chunk(s).",
            "type": "document",
            "pages_count": len(pages),
            "num_chunks": added_count,
        }

    def clear_session_documents(self):
        """Reset session documents and index."""
        self.doc_index.clear()
        self.uploaded_doc_name = None
        self.latest_ocr_text = None

    def query(
        self,
        user_query: str,
        mode: str = "fast",
        fallback_mode: str = "auto",
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes user query end-to-end:
        1. Classifies query (LEGAL vs NON-LEGAL).
        2. Retrieves relevant context.
        3. Assesses query difficulty for smart cloud fallback.
        4. Runs Cloud Fallback (Google Gemini / Groq) or Local Qwen 1.5B inference.
        5. Performs safety verification and fallback error recovery.
        """
        has_doc = self.doc_index.total_chunks > 0
        has_img = self.latest_ocr_text is not None

        # 1. Router Classification
        category, reason = classify_query(
            user_query,
            has_uploaded_doc=has_doc,
            has_uploaded_image=has_img,
        )

        # Conversational greetings & introductions: return warm introduction immediately
        if category == QueryCategory.GREETING:
            return {
                "answer": GREETING_RESPONSE,
                "category": category,
                "sources": [],
                "verified": True,
                "unverified_citations": [],
                "tokens_per_sec": 0.0,
                "latency_sec": 0.00,
                "first_token_sec": 0.0,
                "num_tokens": len(GREETING_RESPONSE.split()),
                "vram_mb": self.model_engine.get_vram_usage_mb(),
                "device": self.model_engine.device,
                "is_fallback": False,
            }

        # Non-legal queries: return refusal immediately without wasting compute
        if category == QueryCategory.GENERAL_QUESTION:
            return {
                "answer": NON_LEGAL_REFUSAL,
                "category": category,
                "sources": [],
                "verified": True,
                "unverified_citations": [],
                "tokens_per_sec": 0.0,
                "latency_sec": 0.01,
                "first_token_sec": 0.0,
                "num_tokens": len(NON_LEGAL_REFUSAL.split()),
                "vram_mb": self.model_engine.get_vram_usage_mb(),
                "device": self.model_engine.device,
                "is_fallback": False,
            }

        top_k = 3 if mode == "fast" else 5

        # 2. RAG Retrieval
        retrieval_res = retrieve_relevant_legal_context(
            query=user_query,
            doc_index=self.doc_index if has_doc else None,
            kb_index=self.kb_index,
            top_k=top_k,
        )
        context_str = retrieval_res["context_str"]
        sources = retrieval_res["sources"]

        # 3. Check for Difficulty / Cloud Fallback Trigger
        is_difficult, diff_reason = is_difficult_legal_query(
            user_query,
            retrieval_result=retrieval_res,
            min_confidence=FALLBACK_CONFIDENCE_THRESHOLD,
        )

        should_use_fallback = (
            (fallback_mode == "always")
            or (fallback_mode == "auto" and is_difficult and FALLBACK_ENABLED)
        )

        fallback_attempted = False
        fallback_error_msg = None

        # 4. Attempt Cloud Fallback if indicated
        if should_use_fallback:
            fallback_attempted = True
            logger.info(f"Triggering cloud fallback ({diff_reason})...")

            fb_res = self.fallback_engine.generate(
                prompt=user_query,
                context=context_str if retrieval_res.get("has_sources") else "",
                key_override=api_key,
                provider_override=provider,
            )

            if fb_res.get("success"):
                safety_res = validate_and_refine_response(
                    generated_text=fb_res["text"],
                    retrieved_context=context_str,
                    sources=sources,
                )
                return {
                    "answer": safety_res["text"],
                    "category": category,
                    "sources": sources,
                    "verified": safety_res["is_safe"],
                    "unverified_citations": safety_res.get("unverified_citations", []),
                    "tokens_per_sec": fb_res.get("tokens_per_sec", 0.0),
                    "latency_sec": fb_res.get("latency_sec", 0.0),
                    "first_token_sec": 0.0,
                    "num_tokens": fb_res.get("num_tokens", 0),
                    "vram_mb": 0.0,
                    "device": f"Cloud ({fb_res['provider'].upper()})",
                    "is_fallback": True,
                    "provider": fb_res["provider"],
                    "model": fb_res["model"],
                    "fallback_reason": diff_reason if fallback_mode == "auto" else "User configured Always Cloud mode",
                    "fallback_attempted": True,
                    "fallback_error": None,
                }
            else:
                fallback_error_msg = fb_res.get("error", "Unknown cloud error")
                logger.warning(
                    f"Cloud fallback generation failed: {fallback_error_msg}. "
                    "Falling back to local model."
                )

        # 5. Local Model Generation (Executed when fallback is not needed OR when cloud fails)
        if not self.model_engine.is_loaded:
            success, msg = self.model_engine.load()
            if not success:
                err_text = f"Error: Model could not be loaded ({msg})."
                if fallback_error_msg:
                    err_text += f"\n\n*Cloud Fallback was also attempted but failed: {fallback_error_msg}*"
                return {
                    "answer": err_text,
                    "category": category,
                    "sources": [],
                    "verified": False,
                    "latency_sec": 0.0,
                    "is_fallback": False,
                    "fallback_attempted": fallback_attempted,
                    "fallback_error": fallback_error_msg,
                }

        # Build prompt for local model
        if category == QueryCategory.IMAGE_ANALYSIS and self.latest_ocr_text:
            prompt = build_image_ocr_analysis_prompt(
                query=user_query,
                ocr_text=self.latest_ocr_text,
                doc_name=self.uploaded_doc_name or "Uploaded Image",
            )
        elif category == QueryCategory.DOCUMENT_ANALYSIS and has_doc:
            prompt = build_document_analysis_prompt(
                query=user_query,
                context=context_str,
                doc_name=self.uploaded_doc_name or "Uploaded Document",
            )
        else:
            prompt = build_legal_qa_prompt(
                query=user_query,
                context=context_str,
            )

        gen_result = self.model_engine.generate_full(prompt, mode=mode)
        raw_text = gen_result["text"]

        # If local model gave an unverified refusal or stub and fallback hasn't been attempted yet, try cloud fallback
        if (
            not fallback_attempted
            and fallback_mode == "auto"
            and FALLBACK_ENABLED
            and (
                "could not verify the specific legal provision" in raw_text.lower()
                or "unverified" in raw_text.lower()
                or len(raw_text.strip().split()) < 25
            )
        ):
            logger.info("Local model could not verify provisions. Retrying with cloud fallback...")
            fb_res = self.fallback_engine.generate(
                prompt=user_query,
                context=context_str if retrieval_res.get("has_sources") else "",
                key_override=api_key,
                provider_override=provider,
            )
            if fb_res.get("success"):
                safety_res = validate_and_refine_response(
                    generated_text=fb_res["text"],
                    retrieved_context=context_str,
                    sources=sources,
                )
                return {
                    "answer": safety_res["text"],
                    "category": category,
                    "sources": sources,
                    "verified": safety_res["is_safe"],
                    "unverified_citations": safety_res.get("unverified_citations", []),
                    "tokens_per_sec": fb_res.get("tokens_per_sec", 0.0),
                    "latency_sec": fb_res.get("latency_sec", 0.0),
                    "first_token_sec": 0.0,
                    "num_tokens": fb_res.get("num_tokens", 0),
                    "vram_mb": 0.0,
                    "device": f"Cloud ({fb_res['provider'].upper()})",
                    "is_fallback": True,
                    "provider": fb_res["provider"],
                    "model": fb_res["model"],
                    "fallback_reason": "Local model lacked coverage; auto-escalated to cloud intelligence",
                    "fallback_attempted": True,
                    "fallback_error": None,
                }
            else:
                fallback_error_msg = fb_res.get("error")

        safety_result = validate_and_refine_response(
            generated_text=raw_text,
            retrieved_context=context_str,
            sources=sources,
        )

        return {
            "answer": safety_result["text"],
            "category": category,
            "sources": sources,
            "verified": safety_result["is_safe"],
            "unverified_citations": safety_result["unverified_citations"],
            "tokens_per_sec": gen_result["tokens_per_second"],
            "latency_sec": gen_result["total_time"],
            "first_token_sec": gen_result["first_token_time"],
            "num_tokens": gen_result["num_tokens"],
            "vram_mb": gen_result["vram_mb"],
            "device": gen_result["device"],
            "is_fallback": False,
            "fallback_attempted": fallback_attempted,
            "fallback_error": fallback_error_msg,
            "fallback_reason": diff_reason if is_difficult else None,
        }
