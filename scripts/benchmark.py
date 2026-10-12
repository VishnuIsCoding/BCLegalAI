"""
Benchmark Suite for Indian Legal AI
Measures:
- Model loading time
- First-token latency (TTFT)
- Total response time
- Tokens per second
- GPU VRAM consumption
- System RAM usage
- RAG retrieval latency
Tests all 10 benchmark questions and outputs benchmark_results.json.
"""

import sys
import os
import time
import json
from pathlib import Path
import psutil
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import (
    CUDA_AVAILABLE,
    NON_LEGAL_REFUSAL,
)
from app.model import get_model_engine
from app.rag import LegalAssistantCoordinator
from app.router import classify_query, QueryCategory

TEST_QUESTIONS = [
    {
        "id": 1,
        "query": "What is a contract?",
        "type": "legal_question",
        "expected_action": "Explain definition under Section 2(h) of Indian Contract Act",
    },
    {
        "id": 2,
        "query": "What is breach of contract?",
        "type": "legal_question",
        "expected_action": "Explain breach and Section 73 remedies",
    },
    {
        "id": 3,
        "query": "What is bail?",
        "type": "legal_question",
        "expected_action": "Explain bail provisions under BNSS / CrPC",
    },
    {
        "id": 4,
        "query": "What is an FIR?",
        "type": "legal_question",
        "expected_action": "Explain FIR registration under Section 173 BNSS / 154 CrPC",
    },
    {
        "id": 5,
        "query": "What is consideration in contract law?",
        "type": "legal_question",
        "expected_action": "Explain consideration under Section 2(d) of Indian Contract Act",
    },
    {
        "id": 6,
        "query": "What is consumer protection?",
        "type": "legal_question",
        "expected_action": "Explain Consumer Protection Act, 2019 and consumer rights",
    },
    {
        "id": 7,
        "query": "Explain this uploaded contract.",
        "type": "document_analysis",
        "expected_action": "Analyze uploaded contract clauses and parties",
    },
    {
        "id": 8,
        "query": "What does this legal notice mean?",
        "type": "document_analysis",
        "expected_action": "Analyze uploaded legal notice grievance and demand",
    },
    {
        "id": 9,
        "query": "Who won yesterday's cricket match?",
        "type": "general_refusal",
        "expected_action": "Refuse non-legal query instantly",
    },
    {
        "id": 10,
        "query": "Write a Python program.",
        "type": "general_refusal",
        "expected_action": "Refuse non-legal query instantly",
    },
]

MOCK_CONTRACT_TEXT = """
SERVICE AGREEMENT
This Agreement is entered into on 1st October 2024 between Alpha Tech Private Limited (Service Provider) and Beta Retail Limited (Client).
Clause 1: Scope of Services - Service Provider shall develop and maintain cloud inventory management software for Client.
Clause 2: Consideration - Client shall pay Rs. 5,00,000 monthly upon delivery of milestones.
Clause 3: Confidentiality - Both parties agree not to disclose proprietary trade secrets or customer data.
Clause 4: Termination - Either party may terminate this agreement by providing thirty (30) days prior written notice. In case of material breach, termination is immediate.
Clause 5: Dispute Resolution - Disputes shall be referred to arbitration in New Delhi under the Arbitration and Conciliation Act, 1996.
"""

MOCK_NOTICE_TEXT = """
LEGAL NOTICE
Date: 15th September 2024
To: ABC Construction Limited, Sector 62, Noida, UP
From: Adv. Rajesh Sharma on behalf of Mr. Suresh Kumar
Subject: Legal Notice for recovery of unpaid outstanding dues of Rs. 4,50,000/-
Under instructions from my client, this notice is served upon you regarding non-payment of invoices for supply of building materials delivered between May and July 2024.
You are hereby called upon to pay the outstanding sum of Rs. 4,50,000 along with 18% interest within fifteen (15) days of receipt of this notice, failing which my client will institute appropriate civil and criminal legal proceedings under Section 138 of Negotiable Instruments Act and Order 37 CPC at your cost and risk.
"""


def get_current_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 1)


def run_benchmarks():
    print("==================================================")
    print("INDIAN LEGAL AI - PERFORMANCE BENCHMARK SUITE")
    print("==================================================")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if CUDA_AVAILABLE else 0,
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        },
        "model_loading": {},
        "questions": [],
        "summary": {},
    }

    coordinator = LegalAssistantCoordinator()
    engine = get_model_engine()

    # Step 1: Model Loading Benchmark
    print("\n[*] Step 1: Measuring Model Loading Time...")
    t_load_start = time.time()
    success, load_msg = engine.load()
    load_time = round(time.time() - t_load_start, 2)
    vram_after_load = engine.get_vram_usage_mb()
    ram_after_load = get_current_ram_mb()

    results["model_loading"] = {
        "success": success,
        "message": load_msg,
        "load_time_sec": load_time,
        "device": engine.device,
        "vram_allocated_mb": vram_after_load,
        "ram_used_mb": ram_after_load,
    }
    print(f"[✓] {load_msg}")
    print(f"    - Loading Time: {load_time}s")
    print(f"    - Device: {engine.device.upper()}")
    print(f"    - VRAM: {vram_after_load} MB")
    print(f"    - Process RAM: {ram_after_load} MB")

    total_legal_tokens = 0
    total_legal_gen_time = 0.0

    print("\n[*] Step 2: Running 10 Test Questions...\n")

    for q_data in TEST_QUESTIONS:
        qid = q_data["id"]
        q_text = q_data["query"]
        q_type = q_data["type"]

        print(f"--------------------------------------------------")
        print(f"Test Question {qid}/10: \"{q_text}\"")

        # Mock document attachment for questions 7 & 8
        if qid == 7:
            print("  -> Attaching mock Service Agreement to session RAG...")
            coordinator.process_and_index_file(MOCK_CONTRACT_TEXT.encode("utf-8"), "service_agreement.txt")
        elif qid == 8:
            print("  -> Attaching mock Legal Notice to session RAG...")
            coordinator.process_and_index_file(MOCK_NOTICE_TEXT.encode("utf-8"), "legal_notice.txt")
        elif qid in [9, 10]:
            coordinator.clear_session_documents()

        t_start = time.time()
        res = coordinator.query(q_text, mode="fast")
        t_total = round(time.time() - t_start, 3)

        category = res.get("category", "")
        answer = res.get("answer", "")
        tokens_sec = res.get("tokens_per_sec", 0.0)
        num_tokens = res.get("num_tokens", 0)
        first_token_sec = res.get("first_token_sec", 0.0)
        vram_mb = res.get("vram_mb", engine.get_vram_usage_mb())
        ram_mb = get_current_ram_mb()

        # Check refusal expectations
        is_refusal_expected = q_type == "general_refusal"
        refusal_passed = False
        if is_refusal_expected:
            refusal_passed = (category == QueryCategory.GENERAL_QUESTION and NON_LEGAL_REFUSAL in answer)
            status_mark = "PASSED (Refused)" if refusal_passed else "FAILED"
        else:
            refusal_passed = (category != QueryCategory.GENERAL_QUESTION and len(answer) > 20)
            status_mark = "PASSED (Answered)" if refusal_passed else "FAILED"
            total_legal_tokens += num_tokens
            total_legal_gen_time += t_total

        print(f"  Category:        {category}")
        print(f"  Status:          {status_mark}")
        print(f"  Latency:         {t_total}s (First token: {first_token_sec}s)")
        print(f"  Tokens:          {num_tokens} tokens ({tokens_sec} tok/s)")
        print(f"  VRAM / RAM:      {vram_mb} MB / {ram_mb} MB")
        print(f"  Answer Snippet:  {answer[:120].strip()}...")

        q_result = {
            "id": qid,
            "query": q_text,
            "category": category,
            "passed": refusal_passed,
            "latency_sec": t_total,
            "first_token_sec": first_token_sec,
            "num_tokens": num_tokens,
            "tokens_per_second": tokens_sec,
            "vram_mb": vram_mb,
            "ram_mb": ram_mb,
            "sources_count": len(res.get("sources", [])),
            "answer_preview": answer[:200],
        }
        results["questions"].append(q_result)

    # Summary Metrics
    avg_speed = round(total_legal_tokens / max(0.001, total_legal_gen_time), 2)
    avg_latency = round(sum(q["latency_sec"] for q in results["questions"]) / len(results["questions"]), 2)
    all_passed = all(q["passed"] for q in results["questions"])

    results["summary"] = {
        "total_questions": len(TEST_QUESTIONS),
        "passed_count": sum(1 for q in results["questions"] if q["passed"]),
        "all_passed": all_passed,
        "avg_latency_sec": avg_latency,
        "avg_tokens_per_sec": avg_speed,
        "peak_vram_mb": max(q["vram_mb"] for q in results["questions"]),
        "peak_ram_mb": max(q["ram_mb"] for q in results["questions"]),
    }

    # Save to benchmark_results.json
    out_file = ROOT_DIR / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print("BENCHMARK SUMMARY")
    print("==================================================")
    print(f"Total Tests:        {len(TEST_QUESTIONS)}")
    print(f"Tests Passed:       {results['summary']['passed_count']}/{len(TEST_QUESTIONS)}")
    print(f"Average Latency:    {avg_latency}s")
    print(f"Legal Gen Speed:    {avg_speed} tokens/second")
    print(f"Peak VRAM:          {results['summary']['peak_vram_mb']} MB (RTX 3050 4GB)")
    print(f"Results Saved:      {out_file}")
    print("==================================================")


if __name__ == "__main__":
    run_benchmarks()
