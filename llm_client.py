# llm_client.py
# Este módulo gerencia a interação com diferentes provedores de Large Language Models (LLMs),
# como Google Gemini e OpenAI, fornecendo uma interface unificada para chamadas de LLM.

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Any
from tenacity import retry, stop_after_attempt, wait_exponential

# Diretório onde os arquivos de prompt são armazenados.
PROMPTS_DIR_ENV = os.getenv("PROMPTS_DIR")
PROMPTS_DIR = Path(PROMPTS_DIR_ENV).resolve() if PROMPTS_DIR_ENV else (Path(__file__).resolve().parent / "prompts")

# --- CONFIGURAÇÕES DO PROVEDOR ---
# Variável para forçar um provedor específico ("google" ou "openai"). Se "auto", detecta o disponível.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()

# Configurações para o Google Gemini (prioritário).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_MODEL   = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash-latest"

# Configurações para OpenAI (opcional, para uso futuro ou como fallback).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_provider = None   # Armazena o provedor de LLM ativo ("google" | "openai" | None).
_client   = None   # Armazena o objeto cliente do LLM (módulo google.generativeai ou cliente OpenAI).

def load_prompt(filename: str) -> str:
    """Carrega o conteúdo de um arquivo de prompt do diretório PROMPTS_DIR."""
    path = PROMPTS_DIR / filename
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[LLM] WARN: Não foi possível carregar o prompt '{path}': {e}", flush=True)
        return ""

def _init_google():
    """Tenta inicializar e configurar o cliente Google Gemini."""
    global _provider, _client
    if not GOOGLE_API_KEY:
        return False
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GOOGLE_API_KEY)
        _provider, _client = "google", genai
        print(f"[LLM] Using Google Gemini: {GOOGLE_MODEL}", flush=True)
        return True
    except Exception as e:
        print(f"[LLM] Google Gemini indisponível: {e}", flush=True)
        return False

def _init_openai():
    """Tenta inicializar e configurar o cliente OpenAI."""
    global _provider, _client
    if not OPENAI_API_KEY:
        return False
    try:
        from openai import OpenAI  # type: ignore
        _client = OpenAI(api_key=OPENAI_API_KEY)
        _provider = "openai"
        print(f"[LLM] Using OpenAI: {OPENAI_MODEL}", flush=True)
        return True
    except Exception as e:
        print(f"[LLM] OpenAI indisponível: {e}", flush=True)
        return False

def _lazy_client() -> Tuple[Optional[str], Any]:
    """Inicializa o cliente LLM (Google ou OpenAI) apenas uma vez, com base na configuração."""
    global _provider, _client
    if _provider is not None:
        return _provider, _client

    # Tenta inicializar com base na configuração de LLM_PROVIDER
    if LLM_PROVIDER == "google":
        if _init_google():
            return _provider, _client
    elif LLM_PROVIDER == "openai":
        if _init_openai():
            return _provider, _client
    elif LLM_PROVIDER == "auto":
        # Lógica de detecção automática: prioriza Google
        if _init_google() or _init_openai():
            return _provider, _client

    # Se nenhuma configuração funcionar, usa fallback extrativo (sem LLM).
    _provider, _client = None, None
    print("[LLM] Nenhum provedor configurado ou disponível. Usando fallback extrativo.", flush=True)
    return _provider, _client

def _extract_text_from_google_response(resp) -> str:
    """Extrai texto de uma resposta do modelo Google Gemini, compatível com diferentes versões do SDK."""
    try:
        # Acesso direto ao atributo .text (mais comum em versões recentes)
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text
    except Exception:
        pass
    # Fallback para extrair de candidates/parts se .text não estiver disponível.
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand0 = resp.candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None)
            if parts:
                return "\n".join(p.text for p in parts if hasattr(p, "text")).strip()
    except Exception:
        pass
    return ""

def _try_build_google_model(genai, system_prompt: str, generation_config: dict):
    """Tenta construir um modelo Google GenerativeModel com retrocompatibilidade."""
    sysi = (system_prompt or "").strip()
    try:
        # Assinatura moderna com system_instruction
        return genai.GenerativeModel(
            model_name=GOOGLE_MODEL,
            system_instruction=sysi,
            generation_config=generation_config
        )
    except (TypeError, AttributeError):
        # Fallback para assinatura mais antiga sem system_instruction
        try:
            return genai.GenerativeModel(
                model_name=GOOGLE_MODEL,
                generation_config=generation_config
            )
        except Exception as e:
            print(f"[LLM] ERROR: Falha ao construir o modelo Gemini: {e}", flush=True)
            return None

def _parse_json_from_text(text: str) -> Optional[dict]:
    """Tenta extrair um objeto JSON de uma string, mesmo que haja texto ao redor."""
    if not text:
        return None
    try:
        # Tenta carregar o texto diretamente
        return json.loads(text)
    except json.JSONDecodeError:
        # Se falhar, procura por um bloco JSON aninhado (ex: ```json\n{...}\n```)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def _call_google(system_prompt: str, user_prompt: str, expect_json: bool, max_tokens: int) -> Tuple[str, Optional[dict]]:
    """Realiza uma chamada ao modelo Google Gemini com retentativas."""
    provider, genai = _lazy_client()
    if provider != "google":
        return "", None

    gen_cfg = {"max_output_tokens": max_tokens, "temperature": 0.2}
    if expect_json:
        gen_cfg["response_mime_type"] = "application/json"

    try:
        model = _try_build_google_model(genai, system_prompt, gen_cfg)
        if model is None:
            return "", None
        
        full_prompt = (user_prompt or "").strip()
        # Para modelos mais antigos que não suportam system_instruction, o prompt do sistema é pré-anexado.
        if "system_instruction" not in model.__dir__():
             full_prompt = f"{system_prompt}\n\n{user_prompt}"

        resp = model.generate_content(full_prompt)
        text = _extract_text_from_google_response(resp)
    except Exception as e:
        print(f"[LLM] ERROR: Falha na chamada ao Google Gemini: {e}", flush=True)
        raise  # Re-lança a exceção para acionar a retentativa do Tenacity

    data = _parse_json_from_text(text) if expect_json else None
    return text, data

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def _call_openai(system_prompt: str, user_prompt: str, expect_json: bool, max_tokens: int) -> Tuple[str, Optional[dict]]:
    """Realiza uma chamada ao modelo OpenAI com retentativas."""
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
            response_format={"type": "json_object"} if expect_json else None,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[LLM] ERROR: Falha na chamada ao OpenAI: {e}", flush=True)
        raise  # Re-lança a exceção para acionar a retentativa do Tenacity

    data = _parse_json_from_text(text) if expect_json else None
    return text, data

def call_llm(system_prompt: str, user_prompt: str, *, expect_json: bool = False, max_tokens: int = 500) -> Tuple[str, Optional[dict]]:
    """Função principal para chamar um LLM com lógica de retentativa e fallback."""
    provider, _ = _lazy_client()
    try:
        if provider == "google":
            return _call_google(system_prompt, user_prompt, expect_json, max_tokens)
        if provider == "openai":
            return _call_openai(system_prompt, user_prompt, expect_json, max_tokens)
    except Exception as e:
        # Captura a exceção final após todas as retentativas falharem
        print(f"[LLM] FATAL: Chamada para o provedor '{provider}' falhou após todas as retentativas: {e}", flush=True)

    # Retorno de fallback se nenhum provedor estiver configurado ou a chamada falhar definitivamente
    return "", None
