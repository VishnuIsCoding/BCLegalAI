"""
Model Loader and Inference Engine for GSMS-B/Indian-Legal-Qwen2.5-1.5B
Optimized for NVIDIA RTX 3050 (4 GB VRAM) with 4-bit quantization and CPU fallback.
Features lazy loading, memory safety, streaming generation, and low VRAM footprint.
"""

import os
import sys
import time
import logging
from threading import Thread
from typing import Generator, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from app.config import (
    LOCAL_MODEL_PATH,
    MODEL_REPO_ID,
    FAST_MODE,
    DETAILED_MODE,
    CUDA_AVAILABLE,
)
from app.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LegalModelEngine:
    """Manages model lifecycle, 4-bit quantization, and streaming inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "none"
        self.is_loaded = False
        self.load_error = None
        self.load_time_sec = 0.0

    def load(self, force_cpu: bool = False) -> Tuple[bool, str]:
        """
        Loads the model and tokenizer into memory.
        Attempts 4-bit CUDA first; automatically falls back to CPU if OOM or failure occurs.
        """
        if self.is_loaded and self.model is not None:
            return True, f"Model already loaded on {self.device.upper()}."

        # Determine path: prefer local directory, fall back to repo ID
        model_source = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else MODEL_REPO_ID
        logger.info(f"Loading model from: {model_source}")
        start_time = time.time()

        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. Attempt 4-bit CUDA loading if available and not forced to CPU
            if CUDA_AVAILABLE and not force_cpu:
                try:
                    logger.info("Attempting 4-bit quantization loading on CUDA...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    self.device = "cuda"
                    self.is_loaded = True
                    self.load_time_sec = round(time.time() - start_time, 2)
                    return True, f"Loaded in 4-bit on CUDA ({self.load_time_sec}s)."
                except (torch.cuda.OutOfMemoryError, Exception) as cuda_err:
                    logger.warning(f"CUDA loading failed ({cuda_err}). Falling back to CPU...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 3. CPU Fallback (bfloat16/float32)
            logger.info("Loading model on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.device = "cpu"
            self.is_loaded = True
            self.load_time_sec = round(time.time() - start_time, 2)
            return True, f"GPU memory is insufficient or unavailable. Running in CPU mode ({self.load_time_sec}s)."

        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            logger.error(f"Failed to load model: {e}")
            return False, f"Failed to load model: {str(e)}"

    def get_vram_usage_mb(self) -> float:
        """Returns currently allocated VRAM in MB."""
        if torch.cuda.is_available() and self.device == "cuda":
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        return 0.0

    def generate_stream(
        self,
        prompt: str,
        mode: str = "fast",
    ) -> Generator[str, None, None]:
        """
        Generate streaming tokens for real-time UI display.
        """
        if not self.is_loaded or self.model is None:
            yield "Model is not loaded. Please click [ Load Legal AI ] first."
            return

        settings = FAST_MODE if mode.lower() == "fast" else DETAILED_MODE

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            # Apply Qwen chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": settings["max_new_tokens"],
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "repetition_penalty": settings.get("repetition_penalty", 1.1),
            "do_sample": settings["temperature"] > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Run generation in a separate thread so streamer yields tokens
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()

    def generate_full(
        self,
        prompt: str,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """
        Synchronous full response generation with latency, token count, and tokens/sec metrics.
        """
        start_time = time.time()
        full_tokens = []

        first_token_time = None
        for chunk in self.generate_stream(prompt, mode=mode):
            if first_token_time is None:
                first_token_time = time.time()
            full_tokens.append(chunk)

        end_time = time.time()
        output_text = "".join(full_tokens).strip()

        total_time = max(0.001, end_time - start_time)
        ttft = (first_token_time - start_time) if first_token_time else total_time

        # Calculate tokens using tokenizer
        num_tokens = len(self.tokenizer.encode(output_text)) if self.tokenizer else len(output_text.split())
        tokens_per_sec = round(num_tokens / total_time, 2)

        return {
            "text": output_text,
            "total_time": round(total_time, 3),
            "first_token_time": round(ttft, 3),
            "num_tokens": num_tokens,
            "tokens_per_second": tokens_per_sec,
            "vram_mb": self.get_vram_usage_mb(),
            "device": self.device,
        }


# Singleton instance
_MODEL_ENGINE_INSTANCE: Optional[LegalModelEngine] = None


def get_model_engine() -> LegalModelEngine:
    global _MODEL_ENGINE_INSTANCE
    if _MODEL_ENGINE_INSTANCE is None:
        _MODEL_ENGINE_INSTANCE = LegalModelEngine()
    return _MODEL_ENGINE_INSTANCE
