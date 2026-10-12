"""
Configuration for Ultra-Lightweight Indian Legal AI
Optimized specifically for RTX 3050 (4 GB VRAM) + AMD Ryzen 7 (16 GB RAM).
"""

import os
from pathlib import Path
import torch
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODELS_DIR = BASE_DIR / "models"
LOCAL_MODEL_PATH = MODELS_DIR / "indian-legal-qwen-1.5b"
MODEL_REPO_ID = "GSMS-B/Indian-Legal-Qwen2.5-1.5B"

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
KB_RAW_DIR = KNOWLEDGE_BASE_DIR / "raw"
KB_PROCESSED_DIR = KNOWLEDGE_BASE_DIR / "processed"
KB_METADATA_DIR = KNOWLEDGE_BASE_DIR / "metadata"
KB_INDEX_DIR = KNOWLEDGE_BASE_DIR / "index"

# Cloud Fallback Configuration (Google Gemini / Groq Cloud)
FALLBACK_ENABLED = os.getenv("FALLBACK_ENABLED", "True").lower() in ("true", "1", "yes")
DEFAULT_FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", DEFAULT_FALLBACK_API_KEY)
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai"
)
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
]
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
XAI_API_BASE = os.getenv("XAI_API_BASE", "https://api.x.ai/v1")
XAI_DEFAULT_MODEL = os.getenv("XAI_DEFAULT_MODEL", "grok-2-latest")
FALLBACK_CONFIDENCE_THRESHOLD = float(os.getenv("FALLBACK_CONFIDENCE_THRESHOLD", "0.35"))

# Embedding Model (CPU only to conserve GPU VRAM for Qwen)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_DIM = 384

# Hardware Detection
CUDA_AVAILABLE = torch.cuda.is_available()
DEFAULT_DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"
GPU_VRAM_MB = (
    torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    if CUDA_AVAILABLE
    else 0
)

# Text Chunking Settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# Generation Modes (Expanded for rich, comprehensive legal explanations)
FAST_MODE = {
    "top_k": 3,
    "max_new_tokens": 768,
    "temperature": 0.25,
    "top_p": 0.90,
    "repetition_penalty": 1.1,
}

DETAILED_MODE = {
    "top_k": 5,
    "max_new_tokens": 1536,
    "temperature": 0.35,
    "top_p": 0.92,
    "repetition_penalty": 1.1,
}

# Legal Refusal Standard Response
NON_LEGAL_REFUSAL = (
    "I am an Indian legal-focused AI assistant. "
    "I can help with Indian legal questions, statutory analysis, and legal documents."
)

# Mandatory Legal Disclaimer
LEGAL_DISCLAIMER = (
    "This is general legal information and not a substitute for advice from a qualified legal professional."
)
