"""
Cloud Fallback Engine for Indian Legal AI
Integrates Google Gemini and Groq Cloud as high-intelligence fallbacks for complex/difficult legal questions.
Supports automatic key-type detection, streaming, resilient model failover, and error diagnostics.
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, Generator, Tuple, List
import requests

from app.config import (
    DEFAULT_FALLBACK_API_KEY,
    GEMINI_API_BASE,
    GEMINI_DEFAULT_MODEL,
    GEMINI_FALLBACK_MODELS,
    GROQ_API_BASE,
    GROQ_DEFAULT_MODEL,
    XAI_API_BASE,
    XAI_DEFAULT_MODEL,
)

logger = logging.getLogger(__name__)


class CloudProvider:
    GEMINI = "gemini"
    GROQ = "groq"
    XAI = "xai"
    AUTO = "auto"


FALLBACK_SYSTEM_PROMPT = """You are an Expert Indian Legal Information and Advisory Assistant (Cloud Intelligence Tier).
Your role is to analyze complex, difficult, or specialized Indian legal questions with deep statutory rigor, precision, and practical procedural guidance.

CRITICAL RULES:
1. JURISDICTION: Focus strictly on Indian law, encompassing Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), Bharatiya Sakshya Adhiniyam (BSA), Indian Penal Code (IPC), CrPC, Indian Contract Act 1872, Constitution of India, Companies Act 2013, Consumer Protection Act 2019, Negotiable Instruments Act, Specific Relief Act, Transfer of Property Act, Arbitration and Conciliation Act, IT Act 2000, IBC 2016, and relevant landmark Supreme Court and High Court precedents.
2. CAUTIOUS & OBJECTIVE TONE: Provide objective legal analysis. Never guarantee judicial outcomes. Use balanced phrases such as "Under the statutory framework...", "The Supreme Court in [Case] observed that...", "Depending on evidentiary proof...".
3. GROUNDING: If context from uploaded documents or verified local sources is provided, incorporate it directly. If the question extends beyond the provided context, synthesize the answer using authoritative Indian statutory provisions and settled jurisprudence.
4. STRUCTURE: You MUST format your response using EXACTLY these headings:

## Answer
[Direct, clear, authoritative answer in 2-4 sentences explaining the legal position]

## Legal Basis
[Cite the exact Acts, Sections, Rules, Articles, or landmark case laws governing the matter]

## Detailed Legal Analysis
[In-depth statutory interpretation, conditions precedent, exceptions, remedies, or rights of the parties]

## Procedural Next Steps & Safeguards
[Practical procedural steps, filing forums (e.g. Police/Magistrate/Civil Court/NCLT/Consumer Commission), limitation periods, or required documentation]

## Important Note
This is general legal information for informational purposes and does not constitute formal advocate-client legal advice. Consult an advocate practicing in the relevant jurisdiction for case-specific representation.
"""


def detect_provider(api_key: str, configured_provider: str = CloudProvider.AUTO) -> str:
    """Detect whether key belongs to Google Gemini, Groq Cloud, or legacy xAI."""
    if configured_provider in (CloudProvider.GEMINI, CloudProvider.GROQ, CloudProvider.XAI):
        return configured_provider

    cleaned_key = (api_key or "").strip()
    if cleaned_key.startswith("AQ.") or cleaned_key.startswith("AIza"):
        return CloudProvider.GEMINI
    if cleaned_key.startswith("gsk_"):
        return CloudProvider.GROQ
    if cleaned_key.startswith("xai-"):
        return CloudProvider.XAI
    if "gemini" in cleaned_key.lower():
        return CloudProvider.GEMINI
    if "xai" in cleaned_key.lower():
        return CloudProvider.XAI
    return CloudProvider.GEMINI  # Default to Google Gemini


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe UI display (e.g., 'AIzaSyD...cdef')."""
    if not api_key:
        return "None"
    cleaned = api_key.strip()
    if len(cleaned) <= 12:
        return cleaned[:3] + "..." + cleaned[-2:]
    return cleaned[:8] + "..." + cleaned[-4:]


class CloudFallbackEngine:
    """
    Manages API calls to Google Gemini and Groq Cloud endpoints.
    Provides synchronous generation, token streaming, and credit/quota diagnosis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = CloudProvider.AUTO,
        model: Optional[str] = None,
        timeout: int = 35,
    ):
        self.api_key = api_key or DEFAULT_FALLBACK_API_KEY
        self.provider = provider
        self.model = model
        self.timeout = timeout

    def get_effective_provider(self, key_override: Optional[str] = None) -> str:
        key = (key_override or self.api_key).strip()
        return detect_provider(key, self.provider)

    def get_effective_model(self, provider: str) -> str:
        if self.model:
            return self.model
        if provider == CloudProvider.GROQ:
            return GROQ_DEFAULT_MODEL
        if provider == CloudProvider.XAI:
            return XAI_DEFAULT_MODEL
        return GEMINI_DEFAULT_MODEL

    def get_base_url(self, provider: str) -> str:
        if provider == CloudProvider.GROQ:
            return GROQ_API_BASE
        if provider == CloudProvider.XAI:
            return XAI_API_BASE
        return GEMINI_API_BASE

    def check_connection(
        self,
        key_override: Optional[str] = None,
        provider_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validates API key credentials and reports connection status.
        """
        key = (key_override or self.api_key).strip()
        if not key:
            return {
                "status": "missing_key",
                "provider": "unknown",
                "message": "No API key configured for cloud fallback.",
            }

        prov = provider_override or detect_provider(key, self.provider)
        base_url = self.get_base_url(prov)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.get(f"{base_url}/models", headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                cnt = len(data) if isinstance(data, list) else 1
                return {
                    "status": "ready",
                    "provider": prov,
                    "message": f"Connected to {prov.upper()} successfully ({cnt} models available).",
                    "models_count": cnt,
                }

            error_data = {}
            try:
                error_data = resp.json()
            except Exception:
                pass

            err_code = error_data.get("code") or error_data.get("error", {}).get("code")
            err_msg = error_data.get("error") if isinstance(error_data.get("error"), str) else error_data.get("error", {}).get("message", resp.text)

            if resp.status_code == 403 and (err_code == "permission-denied" or "credit" in str(err_msg).lower()):
                return {
                    "status": "no_credits",
                    "provider": prov,
                    "message": f"{prov.upper()} authenticated successfully, but account has quota/credit restrictions.",
                    "raw_error": err_msg,
                }
            elif resp.status_code == 401:
                return {
                    "status": "invalid_key",
                    "provider": prov,
                    "message": f"Invalid API Key for {prov.upper()}.",
                    "raw_error": err_msg,
                }
            else:
                return {
                    "status": "error",
                    "provider": prov,
                    "status_code": resp.status_code,
                    "message": f"{prov.upper()} returned HTTP {resp.status_code}: {err_msg}",
                }

        except requests.exceptions.Timeout:
            return {
                "status": "timeout",
                "provider": prov,
                "message": f"Connection to {prov.upper()} timed out.",
            }
        except Exception as e:
            return {
                "status": "network_error",
                "provider": prov,
                "message": f"Could not reach {prov.upper()} API: {str(e)}",
            }

    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        key_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Synchronous full response generation via cloud LLM.
        Automatically retries with resilient fallback models if Gemini encounters temporary demand spikes (503).
        """
        key = (key_override or self.api_key).strip()
        if not key:
            return {
                "success": False,
                "error": "No API key configured for cloud fallback.",
                "error_code": "NO_API_KEY",
                "text": "",
            }

        prov = provider_override or detect_provider(key, self.provider)
        active_model = model_override or self.get_effective_model(prov)
        base_url = self.get_base_url(prov)

        sys_prompt = system_prompt or FALLBACK_SYSTEM_PROMPT

        user_content = prompt
        if context and context.strip() and "No matching legal source material" not in context:
            user_content = f"Retrieved Context from Indian Legal Database / Documents:\n{context}\n\nUser Question:\n{prompt}"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # Candidate models list for resilience
        candidate_models = [active_model]
        if prov == CloudProvider.GEMINI:
            for fb_m in GEMINI_FALLBACK_MODELS:
                if fb_m not in candidate_models:
                    candidate_models.append(fb_m)

        last_resp = None
        last_err_msg = ""
        last_status_code = 500
        start_time = time.time()

        for curr_model in candidate_models:
            payload = {
                "model": curr_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                last_resp = resp
                last_status_code = resp.status_code
                latency = round(time.time() - start_time, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [{}])
                    choice = choices[0] if choices else {}
                    content = (choice.get("message", {}).get("content") or "").strip()
                    usage = data.get("usage", {})
                    num_tokens = usage.get("completion_tokens", len(content.split()))
                    tok_per_sec = round(num_tokens / max(0.01, latency), 1)

                    return {
                        "success": True,
                        "text": content,
                        "provider": prov,
                        "model": curr_model,
                        "latency_sec": latency,
                        "num_tokens": num_tokens,
                        "tokens_per_sec": tok_per_sec,
                        "status_code": 200,
                    }

                # If model is unavailable (503/404) and we have alternate models to try, continue
                if resp.status_code in (503, 404) and len(candidate_models) > 1 and curr_model != candidate_models[-1]:
                    logger.warning(
                        f"Model {curr_model} returned {resp.status_code}. Retrying with backup model..."
                    )
                    continue

                # Parse non-200 responses
                error_data = {}
                try:
                    error_data = resp.json()
                    if isinstance(error_data, list) and error_data:
                        error_data = error_data[0]
                except Exception:
                    pass

                err_code = error_data.get("code") or (
                    error_data.get("error", {}).get("code")
                    if isinstance(error_data.get("error"), dict)
                    else None
                )
                raw_err = error_data.get("error")
                last_err_msg = (
                    raw_err
                    if isinstance(raw_err, str)
                    else (raw_err.get("message") if isinstance(raw_err, dict) else resp.text)
                )

                if resp.status_code == 403 and (err_code == "permission-denied" or "credit" in str(last_err_msg).lower()):
                    diagnostic = (
                        f"{prov.upper()} account has 0 credits or quota restrictions. "
                        "Check billing or provide an alternate key."
                    )
                elif resp.status_code == 401:
                    diagnostic = f"Invalid {prov.upper()} API Key. Please verify the key."
                elif resp.status_code == 429:
                    diagnostic = f"{prov.upper()} rate limit reached. Please wait a moment."
                elif resp.status_code == 503:
                    diagnostic = f"{prov.upper()} is currently experiencing high demand. Please try again shortly."
                else:
                    diagnostic = f"{prov.upper()} error ({resp.status_code}): {last_err_msg}"

                return {
                    "success": False,
                    "error": diagnostic,
                    "raw_error": last_err_msg,
                    "error_code": err_code or f"HTTP_{resp.status_code}",
                    "status_code": resp.status_code,
                    "provider": prov,
                    "model": curr_model,
                    "latency_sec": latency,
                    "text": "",
                }

            except requests.exceptions.Timeout:
                if curr_model != candidate_models[-1]:
                    continue
                return {
                    "success": False,
                    "error": f"Request to {prov.upper()} timed out after {self.timeout}s.",
                    "error_code": "TIMEOUT",
                    "provider": prov,
                    "model": curr_model,
                    "latency_sec": round(time.time() - start_time, 2),
                    "text": "",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to connect to {prov.upper()}: {str(e)}",
                    "error_code": "NETWORK_ERROR",
                    "provider": prov,
                    "model": curr_model,
                    "latency_sec": round(time.time() - start_time, 2),
                    "text": "",
                }

        return {
            "success": False,
            "error": f"All candidate models for {prov.upper()} were unavailable ({last_status_code}): {last_err_msg}",
            "error_code": f"HTTP_{last_status_code}",
            "status_code": last_status_code,
            "provider": prov,
            "model": active_model,
            "latency_sec": round(time.time() - start_time, 2),
            "text": "",
        }

    def generate_stream(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None,
        key_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Streaming generation via Server-Sent Events (SSE).
        Yields text chunks as they arrive.
        """
        key = (key_override or self.api_key).strip()
        if not key:
            yield "Error: No API key configured for cloud fallback."
            return

        prov = provider_override or detect_provider(key, self.provider)
        active_model = model_override or self.get_effective_model(prov)
        base_url = self.get_base_url(prov)
        sys_prompt = system_prompt or FALLBACK_SYSTEM_PROMPT

        user_content = prompt
        if context and context.strip() and "No matching legal source material" not in context:
            user_content = f"Retrieved Context from Indian Legal Database / Documents:\n{context}\n\nUser Question:\n{prompt}"

        payload = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        try:
            with requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                if resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        if isinstance(err_json, list) and err_json:
                            err_json = err_json[0]
                        err_msg = err_json.get("error", resp.text)
                    except Exception:
                        err_msg = resp.text
                    yield f"Error ({resp.status_code}): {err_msg}"
                    return

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            delta = data_json.get("choices", [{}])[0].get("delta", {})
                            content_piece = delta.get("content", "")
                            if content_piece:
                                yield content_piece
                        except Exception:
                            continue

        except Exception as e:
            yield f"\n[Stream connection error: {str(e)}]"


# Singleton instance
_FALLBACK_ENGINE_INSTANCE: Optional[CloudFallbackEngine] = None


def get_fallback_engine() -> CloudFallbackEngine:
    global _FALLBACK_ENGINE_INSTANCE
    if _FALLBACK_ENGINE_INSTANCE is None:
        _FALLBACK_ENGINE_INSTANCE = CloudFallbackEngine()
    return _FALLBACK_ENGINE_INSTANCE
