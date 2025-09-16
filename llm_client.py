# llm_client.py
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/app/prompts")

# Canonical: GOOGLE_API_KEY / GOOGLE_MODEL (mantém compatibilidade com GEMINI_*)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_MODEL   = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"

# OpenAI (opcional para futuro)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_provider = None   # "google" | "openai" | None
_client   = None   # módulo google.generativeai ou cliente OpenAI

def load_prompt(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _lazy_client():
    """Inicializa provedor uma vez; prioriza Google."""
    global _provider, _client
    if _provider is not None:
        return _provider, _client

    # 1) Google (Gemini)
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=GOOGLE_API_KEY)
            _provider, _client = "google", genai
            print(f"[LLM] Using Google Gemini: {GOOGLE_MODEL}", flush=True)
            return _provider, _client
        except Exception as e:
            print(f"[LLM] Google Gemini indisponível: {e}", flush=True)

    # 2) OpenAI
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI  # type: ignore
            _client = OpenAI(api_key=OPENAI_API_KEY)
            _provider = "openai"
            print(f"[LLM] Using OpenAI: {OPENAI_MODEL}", flush=True)
            return _provider, _client
        except Exception as e:
            print(f"[LLM] OpenAI indisponível: {e}", flush=True)

    # 3) Sem LLM
    _provider, _client = None, None
    print("[LLM] Nenhum provedor configurado. Usando fallback extrativo.", flush=True)
    return _provider, _client

def _extract_text_from_google_response(resp) -> str:
    """Compatível com várias versões do SDK: usa .text ou extrai de candidates/parts."""
    try:
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text
    except Exception:
        pass
    # fallback: candidates -> content.parts[].text
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand0 = resp.candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                out = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        out.append(t)
                return "\n".join(out).strip()
    except Exception:
        pass
    return ""

def _try_build_google_model(genai, system_prompt: str, generation_config: dict):
    """
    Tenta diferentes assinaturas do SDK:
    - GenerativeModel(model=..., system_instruction=..., generation_config=...)
    - GenerativeModel(model_name=..., system_instruction=...)
    - GenerativeModel("model-string")
    """
    sysi = (system_prompt or "").strip()
    # 1) assinatura nova (model=)
    try:
        return genai.GenerativeModel(
            model=GOOGLE_MODEL,
            system_instruction=sysi,
            generation_config=generation_config
        )
    except TypeError:
        pass
    # 2) assinatura antiga (model_name=)
    try:
        return genai.GenerativeModel(
            model_name=GOOGLE_MODEL,
            system_instruction=sysi
        )
    except TypeError:
        pass
    # 3) mínima
    try:
        return genai.GenerativeModel(GOOGLE_MODEL)
    except Exception:
        return None

def _call_google(system_prompt: str, user_prompt: str, expect_json: bool, max_tokens: int) -> Tuple[str, Optional[dict]]:
    provider, genai = _lazy_client()
    if provider != "google":
        return "", None

    # Nem todas as versões aceitam response_mime_type em generation_config
    gen_cfg = {"max_output_tokens": max_tokens}
    # Preferimos guiar via prompt para JSON, mas se a versão suportar mime, melhor.
    if expect_json:
        gen_cfg["response_mime_type"] = "application/json"

    try:
        model = _try_build_google_model(genai, system_prompt, gen_cfg)
        if model is None:
            return "", None
        # Nem todas as versões aceitam generation_config no generate_content
        try:
            resp = model.generate_content((user_prompt or "").strip(), generation_config=gen_cfg)
        except TypeError:
            resp = model.generate_content((user_prompt or "").strip())
        text = _extract_text_from_google_response(resp)
    except Exception:
        # falha silenciosa -> caller usa fallback
        return "", None

    if expect_json:
        # Tenta JSON “puro”
        try:
            data = json.loads(text)
            return text, data
        except Exception:
            # Tenta extrair o primeiro bloco {...}
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    data = json.loads(m.group(0))
                    return text, data
                except Exception:
                    return text, None
            return text, None
    return text, None

def _call_openai(system_prompt: str, user_prompt: str, expect_json: bool, max_tokens: int) -> Tuple[str, Optional[dict]]:
    provider, client = _lazy_client()
    if provider != "openai":
        return "", None
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": (system_prompt or "").strip()},
                {"role": "user", "content": (user_prompt or "").strip()},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return "", None

    if expect_json:
        try:
            data = json.loads(text)
            return text, data
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    data = json.loads(m.group(0))
                    return text, data
                except Exception:
                    return text, None
            return text, None
    return text, None

def call_llm(system_prompt: str, user_prompt: str, *, expect_json: bool = False, max_tokens: int = 500) -> Tuple[str, Optional[dict]]:
    """
    Prioriza Google; se indisponível, tenta OpenAI; senão, devolve vazio.
    NUNCA devolve mensagem de erro no texto — falha é silenciosa (caller faz fallback).
    """
    provider, _ = _lazy_client()
    if provider == "google":
        return _call_google(system_prompt, user_prompt, expect_json, max_tokens)
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, expect_json, max_tokens)
    return "", None
