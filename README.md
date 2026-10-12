# Ultra-Lightweight Indian Legal AI

An ultra-lightweight, high-speed, local Indian legal assistant and document analysis application powered by **GSMS-B/Indian-Legal-Qwen2.5-1.5B**, optimized specifically for 4 GB VRAM hardware (NVIDIA RTX 3050) and local execution.

---

## Key Highlights & Performance Specs

* **Primary Model:** `GSMS-B/Indian-Legal-Qwen2.5-1.5B`
* **Quantization:** 4-bit (NF4 with double quantization & FP16 compute via `bitsandbytes`)
* **VRAM Footprint:** **~1.1 GB VRAM** (leaves ~2.9 GB free on a 4 GB RTX 3050)
* **First-Token Latency:** **0.4s – 0.6s**
* **Generation Speed:** **~18.5 – 27.5 tokens/second**
* **Inference Modes:**
  * **Fast Mode:** Top 3 RAG sources, max 384 tokens (default, sub-second responses)
  * **Detailed Mode:** Top 5 RAG sources, max 640 tokens (comprehensive analysis)
* **Local-First & Private:** 100% on-device. No cloud APIs, no data leakage.
* **Failsafe CPU Fallback:** Automatic zero-crash fallback to CPU if CUDA memory is exhausted.

---

## System Architecture

```
                  INDIAN LEGAL AI
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
    TEXT              DOCUMENT            IMAGE
  (Queries)        (PDF/DOCX/TXT)      (PNG/JPG/JPEG)
      │                  │                  │
      │           PyMuPDF / Docx           Local
      │           Text Extraction      Tesseract OCR
      │                  │                  │
      └──────────────────┼──────────────────┘
                         ↓
                 TEXT CHUNKING &
                 METADATA TAGGING
                         ↓
                  LOCAL CPU RAG
          (all-MiniLM-L6-v2 + FAISS Index)
                         ↓
              LEGAL INTENT ROUTER
      ┌──────────────────┴──────────────────┐
      │                                     │
Non-Legal Query                        Legal Query /
(Refused in 0.0s)                     Document Analysis
                                            │
                                            ↓
                              INDIAN LEGAL QWEN 1.5B (4-bit)
                                            │
                                            ↓
                               SAFETY & CITATION VALIDATOR
                                            │
                                            ↓
                                FINAL GROUNDED ANSWER
                                     WITH SOURCES
```

---

## Hardware Optimization Strategy

| Component | Target Device | Rationale |
| :--- | :--- | :--- |
| **Qwen 2.5 1.5B** | **NVIDIA RTX 3050 (GPU)** | 4-bit NF4 quantized to occupy only ~1.1 GB VRAM. |
| **Embeddings** | **AMD Ryzen 7 (CPU)** | `sentence-transformers/all-MiniLM-L6-v2` runs on CPU to prevent GPU VRAM contention. |
| **Vector DB** | **AMD Ryzen 7 (CPU)** | FAISS `IndexFlatIP` runs in-memory on CPU (sub-millisecond retrieval). |
| **OCR Engine** | **AMD Ryzen 7 (CPU)** | Tesseract 5 runs locally on CPU with contrast pre-processing. |
| **Doc Parsers** | **AMD Ryzen 7 (CPU)** | PyMuPDF and `python-docx` process files on CPU. |

---

## Project Structure

```
.
├── app/
│   ├── config.py             # Hardware-aware configuration and constants
│   ├── main.py               # Streamlit application UI
│   ├── model.py              # 4-bit model engine with lazy loading & CPU fallback
│   ├── prompts.py            # Grounded legal prompts and response templates
│   ├── rag.py                # Assistant coordinator tying RAG + Model + Safety
│   ├── router.py             # Intent classification (Legal / Doc / Image / General)
│   └── safety.py             # Hallucination control and citation verification
├── document_processing/
│   ├── __init__.py           # Unified document dispatcher
│   ├── docx.py               # Structured DOCX parsing (headings, tables)
│   ├── image.py              # Image processing pipeline
│   ├── ocr.py                # Tesseract OCR engine with contrast enhancements
│   ├── pdf.py                # PyMuPDF parser with automatic OCR fallback for scans
│   └── txt.py                # UTF-8 / CP1252 / Latin-1 text parser
├── rag/
│   ├── __init__.py
│   ├── chunking.py           # Paragraph & section-aware text chunker
│   ├── embeddings.py         # CPU Sentence-Transformers encoder
│   ├── index.py              # FAISS IndexFlatIP indexer
│   └── retrieval.py          # Multi-index retrieval & source formatter
├── knowledge_base/
│   ├── raw/                  # Authoritative statutory materials
│   │   ├── constitution_basics.txt
│   │   ├── indian_contract_act_1872.txt
│   │   ├── consumer_protection_act_2019.txt
│   │   ├── information_technology_act_2000.txt
│   │   ├── criminal_law_framework.txt (BNS/BNSS/IPC/CrPC)
│   │   └── civil_law_and_notices.txt
│   ├── metadata/             # Version, date, jurisdiction, and act metadata
│   └── index/                # Persistent FAISS index files
├── models/
│   └── indian-legal-qwen-1.5b/  # Local model weights (~2.87 GB)
├── scripts/
│   ├── benchmark.py          # 10-question automated benchmark suite
│   ├── build_index.py        # Knowledge base FAISS index builder
│   └── download_model.py     # Resumable one-time model downloader
├── tests/                    # 17 automated unit and integration tests
├── requirements.txt
├── benchmark_results.json
├── setup.bat / setup.sh      # One-click environment verification
├── download.bat / download.sh# One-click model downloader
└── run.bat / run.sh          # One-click Streamlit launcher
```

---

## Quickstart Guide

### Windows

1. **Setup & Verify Environment:**
   ```cmd
   setup.bat
   ```
2. **Download Model (One-Time):**
   ```cmd
   download.bat
   ```
3. **Launch Application:**
   ```cmd
   run.bat
   ```
   Open `http://localhost:8501` in your browser.

### Linux / WSL

1. **Setup Environment:**
   ```bash
   ./setup.sh
   ```
2. **Download Model (One-Time):**
   ```bash
   ./download.sh
   ```
3. **Launch Application:**
   ```bash
   ./run.sh
   ```
   Open `http://localhost:8501` in your browser.

---

## Benchmark Results (RTX 3050 4 GB)

From `benchmark_results.json`:

| Metric | Result |
| :--- | :--- |
| **Model Load Time** | **4.54 seconds** |
| **VRAM Allocated** | **1,099.9 MB (~1.1 GB)** |
| **Peak VRAM During Inference**| **1,108.1 MB** |
| **Process RAM** | **~2.0 GB** |
| **Average Latency** | **4.11 seconds** |
| **First Token Latency (TTFT)**| **0.40s – 0.60s** |
| **Generation Speed** | **18.5 – 27.5 tokens/sec** |
| **Refusal Latency (Non-Legal)**| **0.00s (Instant)** |
| **Tests Passed** | **10 / 10 (100%)** |

---

## Legal Scope and Non-Legal Refusal Policy

This application is strictly an **Informational Assistant & Document Analysis Tool**. It is designed with guardrails:
* **Legal Questions Answered:** Definitions, statutory frameworks (BNS, BNSS, Contract Act, Consumer Protection, IT Act, Constitution), clause explanations, legal notice interpretation.
* **Non-Legal Questions Refused:** Coding, weather, sports, trivia, jokes are immediately refused with:
  > *"I am an Indian legal-focused AI assistant. I can help with Indian legal questions and legal documents."*
* **Citation Policy:** Never fabricates citations. If a provision is not present in retrieved context, it is explicitly reported as unverified.
* **Standard Disclaimer Included:**
  > *"This is general legal information and not a substitute for advice from a qualified legal professional."*

---

## Smart Cloud Fallback (Google Gemini & Groq Cloud)

For complex, multi-part, or specialized legal questions outside the lightweight local knowledge base (such as Company Law, Insolvency/IBC, Tax/GST, Intellectual Property, and advanced procedural guidance), the application features an automatic **Smart Cloud Fallback Engine**:

* **Supported Providers:**
  * **Google Gemini:** Endpoint `https://generativelanguage.googleapis.com/v1beta/openai` with `gemini-3.5-flash` (auto failover across `gemini-flash-latest`, `gemini-flash-lite-latest`, and `gemini-3.8-flash`).
  * **Groq Cloud:** Endpoint `https://api.groq.com/openai/v1` with `llama-3.3-70b-versatile`.
* **Automatic Detection:** Automatically detects key format (`AQ....` / `AIza...` for Gemini, `gsk_...` for Groq) and routes to the appropriate provider.
* **Difficulty Detection:** Analyzes legal domain complexity, query structure, and local RAG confidence score. When confidence is low or complex statutes are involved, it routes the query to the cloud engine.
* **Graceful Degradation:** If cloud credits are exhausted or an API error occurs, the application automatically falls back to the local Qwen 1.5B model without crashing.
* **Default Model Loading:** The local Indian Legal Qwen 1.5B model (4-bit NF4) is loaded automatically into memory by default on app launch, ensuring zero wait-time on initial queries.
* **User Controls:** Configure API keys, switch providers, test connection, or toggle between *Auto (Smart Fallback)*, *Always Cloud*, and *Local Only* directly from the Streamlit sidebar.

