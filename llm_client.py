# llm_client.py
# Este módulo gerencia a interação com diferentes provedores de Large Language Models (LLMs),
# como Google Gemini e OpenAI, fornecendo uma interface unificada para chamadas de LLM.

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

# Diretório onde os arquivos de prompt são armazenados.
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/app/prompts")

# Configurações para o Google Gemini (prioritário).
# GOOGLE_API_KEY: Chave da API do Google, pode ser definida como GOOGLE_API_KEY ou GEMINI_API_KEY.
# GOOGLE_MODEL: Nome do modelo Gemini a ser usado, pode ser GOOGLE_MODEL ou GEMINI_MODEL, com fallback para "gemini-2.5-flash-lite".
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GOOGLE_MODEL   = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite"

# Configurações para OpenAI (opcional, para uso futuro ou como fallback).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_provider = None   # Armazena o provedor de LLM ativo ("google" | "openai" | None).
_client   = None   # Armazena o objeto cliente do LLM (módulo google.generativeai ou cliente OpenAI).

def load_prompt(filename: str) -> str:
    """Carrega o conteúdo de um arquivo de prompt do diretório PROMPTS_DIR.
    
    Args:
        filename (str): O nome do arquivo de prompt a ser carregado.
        
    Returns:
        str: O conteúdo do arquivo de prompt como uma string, ou uma string vazia se o arquivo não for encontrado ou houver erro.
    """
    path = os.path.join(PROMPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _lazy_client():
    """Inicializa o cliente LLM (Google ou OpenAI) apenas uma vez, priorizando o Google.
    
    Returns:
        Tuple[Optional[str], Any]: Uma tupla contendo o nome do provedor ("google" ou "openai") e o objeto cliente, ou (None, None) se nenhum for configurado.
    """
    global _provider, _client
    if _provider is not None:
        return _provider, _client

    # 1) Tenta configurar o Google Gemini
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=GOOGLE_API_KEY)
            _provider, _client = "google", genai
            print(f"[LLM] Using Google Gemini: {GOOGLE_MODEL}", flush=True)
            return _provider, _client
        except Exception as e:
            print(f"[LLM] Google Gemini indisponível: {e}", flush=True)

    # 2) Tenta configurar o OpenAI
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI  # type: ignore
            _client = OpenAI(api_key=OPENAI_API_KEY)
            _provider = "openai"
            print(f"[LLM] Using OpenAI: {OPENAI_MODEL}", flush=True)
            return _provider, _client
        except Exception as e:
            print(f"[LLM] OpenAI indisponível: {e}", flush=True)

    # 3) Nenhum LLM configurado, usa fallback extrativo (sem LLM).
    _provider, _client = None, None
    print("[LLM] Nenhum provedor configurado. Usando fallback extrativo.", flush=True)
    return _provider, _client

def _extract_text_from_google_response(resp) -> str:
    """Extrai texto de uma resposta do modelo Google Gemini, compatível com diferentes versões do SDK.
    
    Args:
        resp: O objeto de resposta retornado pelo modelo Gemini.
        
    Returns:
        str: O texto extraído da resposta.
    """
    try:
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
    """Tenta construir um modelo Google GenerativeModel usando diferentes assinaturas do SDK.
    
    Args:
        genai: O módulo `google.generativeai`.
        system_prompt (str): O prompt do sistema para o modelo.
        generation_config (dict): Configurações de geração para o modelo.
        
    Returns:
        Optional[genai.GenerativeModel]: O objeto do modelo GenerativeModel se construído com sucesso, caso contrário None.
    """
    sysi = (system_prompt or "").strip()
    # 1) Tenta assinatura nova (model=...)
    try:
        return genai.GenerativeModel(
            model=GOOGLE_MODEL,
            system_instruction=sysi,
            generation_config=generation_config
        )
    except TypeError:
        pass
    # 2) Tenta assinatura antiga (model_name=...)
    try:
        return genai.GenerativeModel(
            model_name=GOOGLE_MODEL,
            system_instruction=sysi
        )
    except TypeError:
        pass
    # 3) Tenta assinatura mínima
    try:
        return genai.GenerativeModel(GOOGLE_MODEL)
    except Exception:
        return None

def _call_google(system_prompt: str, user_prompt: str, expect_json: bool, max_tokens: int) -> Tuple[str, Optional[dict]]:
    """Realiza uma chamada ao modelo Google Gemini.
    
    Args:
        system_prompt (str): O prompt do sistema.
        user_prompt (str): O prompt do usuário.
        expect_json (bool): Se a resposta esperada é um JSON.
        max_tokens (int): O número máximo de tokens na resposta.
        
    Returns:
        Tuple[str, Optional[dict]]: Uma tupla contendo o texto da resposta e o JSON parseado (se `expect_json` for True e o parsing for bem-sucedido).
    """
    provider, genai = _lazy_client()
    if provider != "google":
        return "", None

    gen_cfg = {"max_output_tokens": max_tokens}
    # Se espera JSON, tenta configurar o tipo MIME da resposta (se a versão do SDK suportar).
    if expect_json:
        gen_cfg["response_mime_type"] = "application/json"

    try:
        model = _try_build_google_model(genai, system_prompt, gen_cfg)
        if model is None:
            return "", None
        # Tenta gerar conteúdo, passando generation_config se suportado.
        try:
            resp = model.generate_content((user_prompt or "").strip(), generation_config=gen_cfg)
        except TypeError:
            resp = model.generate_content((user_prompt or "").strip())
        text = _extract_text_from_google_response(resp)
    except Exception:
        # Falha silenciosa, o chamador deve lidar com o fallback.
        return "", None

    if expect_json:
        # Tenta parsear o JSON "puro".
        try:
            data = json.loads(text)
            return text, data
        except Exception:
            # Se falhar, tenta extrair o primeiro bloco JSON {...}.
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
    """Realiza uma chamada ao modelo OpenAI.
    
    Args:
        system_prompt (str): O prompt do sistema.
        user_prompt (str): O prompt do usuário.
        expect_json (bool): Se a resposta esperada é um JSON.
        max_tokens (int): O número máximo de tokens na resposta.
        
    Returns:
        Tuple[str, Optional[dict]]: Uma tupla contendo o texto da resposta e o JSON parseado (se `expect_json` for True e o parsing for bem-sucedido).
    """
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
    """Função principal para chamar um LLM, priorizando Google Gemini e usando OpenAI como fallback.
    
    Args:
        system_prompt (str): O prompt do sistema a ser enviado ao LLM.
        user_prompt (str): O prompt do usuário a ser enviado ao LLM.
        expect_json (bool): Se a resposta esperada do LLM deve ser em formato JSON. Padrão é False.
        max_tokens (int): O número máximo de tokens que o LLM deve gerar na resposta. Padrão é 500.
        
    Returns:
        Tuple[str, Optional[dict]]: Uma tupla contendo o texto bruto da resposta do LLM e um dicionário (se `expect_json` for True e o JSON for válido).
        Retorna strings vazias e None em caso de falha, permitindo que o chamador implemente lógicas de fallback.
    """
    provider, _ = _lazy_client()
    if provider == "google":
        return _call_google(system_prompt, user_prompt, expect_json, max_tokens)
    if provider == "openai":
        return _call_openai(system_prompt, user_prompt, expect_json, max_tokens)
    return "", None


