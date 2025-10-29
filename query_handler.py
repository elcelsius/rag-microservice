# query_handler.py
# Este módulo é responsável por processar as consultas dos usuários, realizar a busca no vetorstore FAISS,
# extrair informações relevantes e formatar a resposta final usando um LLM.

# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
import os
import math
import time  # Importado para telemetria
import yaml  # requer PyYAML

from typing import Any, Dict, List, Tuple, Optional, Iterable, Set
from rapidfuzz import fuzz, distance
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from llm_client import load_prompt, call_llm
from telemetry import log_event
from text_normalizer import normalize_text, normalize_documents

try:
    from sentence_transformers import CrossEncoder  # reranker
except Exception:  # se não tiver sentence_transformers, segue sem rerank
    CrossEncoder = None  # type: ignore
    # Fallback: usar Transformers direto se CrossEncoder pedir prompt interativo
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        torch = None
        AutoTokenizer = None
        AutoModelForSequenceClassification = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:  # pragma: no cover - dependência opcional
    BM25Okapi = None  # type: ignore


# ==== Configurações do Reranker ====
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        v = int(str(raw).strip())
    except Exception:
        v = default
    if min_val is not None and v < min_val:
        v = min_val
    if max_val is not None and v > max_val:
        v = max_val
    return v


def _env_float(name: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    raw = os.getenv(name)
    try:
        v = float(str(raw).strip()) if raw is not None else default
    except Exception:
        v = default
    if min_val is not None and v < min_val:
        v = min_val
    if max_val is not None and v > max_val:
        v = max_val
    return v


def _env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return val.strip() if val is not None else default


_RERANKER_PRESETS = {
    "off": {
        "name": "",
        "candidates": 0,
        "top_k": 0,
        "max_len": 256,
        "device": "cpu",
        "trust_remote_code": False,
    },
    "fast": {
        "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "candidates": 24,
        "top_k": 6,
        "max_len": 384,
        "device": "cpu",
        "trust_remote_code": False,
    },
    "balanced": {
        "name": "BAAI/bge-reranker-v2-m3",
        "candidates": 40,
        "top_k": 8,
        "max_len": 512,
        "device": "cpu",
        "trust_remote_code": False,
    },
    "full": {
        "name": "jinaai/jina-reranker-v2-base-multilingual",
        "candidates": 42,
        "top_k": 6,
        "max_len": 512,
        "device": "cpu",
        "trust_remote_code": True,
    },
}

_raw_preset = _env_str("RERANKER_PRESET", "").lower()
_env_flag_raw = os.getenv("RERANKER_ENABLED")
_env_flag = _env_bool("RERANKER_ENABLED", True) if _env_flag_raw is not None else None

if _raw_preset not in _RERANKER_PRESETS:
    _raw_preset = "balanced"

if _env_flag is False:
    preset_key = "off"
else:
    preset_key = _raw_preset

RERANKER_PRESET = preset_key
_preset_conf = _RERANKER_PRESETS[RERANKER_PRESET]

RERANKER_ENABLED = RERANKER_PRESET != "off" and _env_flag is not False
RERANKER_NAME = _env_str("RERANKER_NAME", _preset_conf["name"])
RERANKER_CANDIDATES = _env_int("RERANKER_CANDIDATES", _preset_conf["candidates"], 1, 100) if RERANKER_ENABLED else 0
RERANKER_TOP_K = _env_int("RERANKER_TOP_K", _preset_conf["top_k"], 1, 20) if RERANKER_ENABLED else 0
RERANKER_MAX_LEN = _env_int("RERANKER_MAX_LEN", _preset_conf["max_len"], 128, 1024)
RERANKER_DEVICE = _env_str("RERANKER_DEVICE", _preset_conf.get("device", "cpu")).lower()
RERANKER_TRUST_REMOTE_CODE = _env_bool("RERANKER_TRUST_REMOTE_CODE", _preset_conf["trust_remote_code"])
# (novo) Forçar rota via ENV para diagnóstico: "", "lexical" ou "vector"
ROUTE_FORCE = _env_str("ROUTE_FORCE", "").lower()

# === Novas configs (híbrido, limiares e cap por fonte) ===
HYBRID_ENABLED   = (_env_bool("HYBRID_ENABLED", True))    # ativa merge lexical+vetorial antes do reranker
LEXICAL_THRESHOLD= _env_int("LEXICAL_THRESHOLD", 90, 60, 100)  # antes estava fixo (=86) em _sentence_hits_by_name
DEPT_BONUS       = _env_int("DEPT_BONUS", 8, 0, 100)      # antes estava fixo (=8) no cálculo de score por depto
MAX_PER_SOURCE   = _env_int("MAX_PER_SOURCE", 2, 1, 10)   # limita diversidade por 'source' no merge híbrido

RETRIEVAL_FETCH_K = _env_int("RETRIEVAL_FETCH_K", 30, 1, 500)
RETRIEVAL_K = _env_int("RETRIEVAL_K", 8, 1, 50)
RETRIEVAL_MMR_LAMBDA = _env_float("RETRIEVAL_MMR_LAMBDA", 0.3, 0.0, 1.0)
RETRIEVAL_MMR_LAMBDA_SHORT = _env_float("RETRIEVAL_MMR_LAMBDA_SHORT", 0.25, 0.0, 1.0)
RETRIEVAL_MIN_SCORE = _env_float("RETRIEVAL_MIN_SCORE", 0.25, 0.0, 1.0)

CACHE_DEFAULT_VERSION = os.getenv("INDEX_VERSION", "0")

RRF_ENABLED = _env_bool("RRF_ENABLED", True)
RRF_K = _env_int("RRF_K", 60, 1, 1000)


def pipeline_cache_fingerprint() -> Dict[str, Any]:
    """Snapshot das configs que impactam a forma como o RAG responde."""
    return {
        "cache_version": CACHE_DEFAULT_VERSION,
        "confidence_min_default": CONFIDENCE_MIN,
        "confidence_min_query": CONFIDENCE_MIN_QUERY,
        "confidence_min_agent": CONFIDENCE_MIN_AGENT,
        "require_context": REQUIRE_CONTEXT,
        "structured_answer": STRUCTURED_ANSWER,
        "route_force": ROUTE_FORCE,
        "mq_enabled": MQ_ENABLED,
        "mq_variants": MQ_VARIANTS,
        "mq_use_llm": MQ_USE_LLM,
        "mq_llm_max_variants": MQ_LLM_MAX_VARIANTS,
        "hybrid_enabled": HYBRID_ENABLED,
        "reranker_preset": RERANKER_PRESET,
        "reranker_enabled": RERANKER_ENABLED,
        "reranker_candidates": RERANKER_CANDIDATES,
        "reranker_top_k": RERANKER_TOP_K,
        "reranker_max_len": RERANKER_MAX_LEN,
        "max_per_source": MAX_PER_SOURCE,
        "retrieval_fetch_k": RETRIEVAL_FETCH_K,
        "retrieval_k": RETRIEVAL_K,
        "retrieval_mmr_lambda": RETRIEVAL_MMR_LAMBDA,
        "retrieval_mmr_lambda_short": RETRIEVAL_MMR_LAMBDA_SHORT,
        "retrieval_min_score": RETRIEVAL_MIN_SCORE,
        "rrf_enabled": RRF_ENABLED,
        "rrf_k": RRF_K,
    }



# ==== Configurações de Multi-Query e Confiança ====
# Ativa ou desativa a geração de múltiplas variações da pergunta do usuário para busca.
MQ_ENABLED = os.getenv("MQ_ENABLED", "true").lower() == "true"
# Número de variações da pergunta a serem geradas.
MQ_VARIANTS = int(os.getenv("MQ_VARIANTS", "3"))
MQ_USE_LLM = _env_bool("MQ_USE_LLM", False)
MQ_LLM_MAX_VARIANTS = _env_int("MQ_LLM_MAX_VARIANTS", 2, 0, 10)
# Limiar mínimo de confiança do reranker para considerar uma resposta válida.
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.32"))
CONFIDENCE_MIN_QUERY = float(os.getenv("CONFIDENCE_MIN_QUERY", str(CONFIDENCE_MIN)))
CONFIDENCE_MIN_AGENT = float(os.getenv("CONFIDENCE_MIN_AGENT", str(CONFIDENCE_MIN)))
# Exige que um contexto seja encontrado nos documentos para gerar uma resposta.
REQUIRE_CONTEXT = os.getenv("REQUIRE_CONTEXT", "true").lower() == "true"

# === Structured answer / citations ===
STRUCTURED_ANSWER = os.getenv("STRUCTURED_ANSWER", "true").lower() == "true"
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "5"))


def _dedupe_preserve_order(items, key=lambda x: x):
    seen = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def _doc_identity(doc: Document) -> tuple:
    """Retorna uma chave estável para identificar documentos em fusões."""
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("source") or "",
        meta.get("chunk"),
        (getattr(doc, "page_content", "") or "")[:64],
    )


def _collect_citations_from_docs(docs, max_sources=MAX_SOURCES):
    """
    Gera citações compactas a partir dos docs (page_content + metadata).
    Deduplica por (source, chunk). Limita em max_sources.
    """
    cites = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("url") or meta.get("source") or meta.get("file") or "desconhecido"
        chunk = meta.get("chunk") or 1
        preview = (getattr(d, "page_content", "") or "").strip().replace("\\n", " ")
        if len(preview) > 200:
            preview = preview[:200].rstrip() + "..."
        cites.append({"source": source, "chunk": chunk, "preview": preview})

    cites = _dedupe_preserve_order(cites, key=lambda c: (c["source"], c["chunk"]))
    return cites[:max_sources]


def _format_answer_markdown(question: str, answer_text: str, citations: list, confidence: float | None):
    """
    Monta um Markdown simples com seções: Resumo / Fontes.
    """
    conf_str = f"\n_(confiança: {confidence:.2f})_" if (confidence is not None) else ""
    fontes = []
    for i, c in enumerate(citations, 1):
        fontes.append(f"{i}. **{c['source']}** (chunk {c['chunk']}): {c['preview']}")
    fontes_md = "\n".join(fontes) if fontes else "_Nenhuma fonte encontrada_"
    md = (
        f"### Resumo{conf_str}\n"
        f"{(answer_text or '').strip()}\n\n"
        f"### Fontes\n"
        f"{fontes_md}\n"
    )
    return md


# INÍCIO DO CÓDIGO INSERIDO (DEBUG HELPERS)
# === Debug & Telemetry ===
DEBUG_LOG = os.getenv("DEBUG_LOG", "true").lower() == "true"
DEBUG_PAYLOAD = os.getenv("DEBUG_PAYLOAD", "true").lower() == "true"


def _now():
    return time.perf_counter()


def _elapsed_ms(t0):
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _pack_doc(doc, score=None, limit_preview=160):
    meta = getattr(doc, "metadata", {}) or {}
    txt = (getattr(doc, "page_content", "") or "").replace("\\n", " ").strip()
    if len(txt) > limit_preview:
        txt = txt[:limit_preview].rstrip() + "..."
    return {
        "source": meta.get("source") or meta.get("file") or "desconhecido",
        "chunk": int(meta.get("chunk") or 1),
        "department": meta.get("department") or "",
        # GARANTIA: sempre float, nunca None
        "score": 0.0 if score is None else float(score),
        "vector_score": float(meta.get("vector_score", 0.0)),
        "preview": txt,
    }


def _log_debug(msg: str):
    if DEBUG_LOG:
        print(f"[DEBUG] {msg}", flush=True)


# FIM DO CÓDIGO INSERIDO (DEBUG HELPERS)

def _strip_accents_lower_safe(text: str) -> str:
    """Remove acentos, lower-case e lida com None de forma segura."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()

# === Dicionário externo (departamentos, aliases, sinônimos, boosts) ===
TERMS_PATH = os.getenv("TERMS_YAML", "/app/config/ontology/terms.yml")


def load_terms(path: str = TERMS_PATH):
    """
    Carrega um dicionário de termos (departamentos, aliases, sinônimos) de um arquivo YAML.
    Isso centraliza a configuração e permite ajustes sem alterar o código.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Garante que as chaves principais existam para evitar erros.
        data.setdefault("departments", {})
        data.setdefault("aliases", {})
        data.setdefault("synonyms", {})
        data.setdefault("boosts", {})
        data.setdefault("mq_expansions", {})
        return data
    except FileNotFoundError:
        print(f"[DICT] Arquivo não encontrado: {path} - usando defaults vazios.")
        return {
            "departments": {},
            "aliases": {},
            "synonyms": {},
            "boosts": {},
            "mq_expansions": {},
        }


# Carrega os termos na inicialização do módulo.
TERMS = load_terms()
DEPARTMENTS = TERMS["departments"]  # dict slug -> label canônica
ALIASES = TERMS["aliases"]  # dict termo -> [variações]
SYNONYMS = TERMS["synonyms"]  # dict sigla -> [expansões]
CUSTOM_MQ_EXPANSIONS = {
    key: [str(v).strip() for v in (values or []) if str(v).strip()]
    for key, values in (TERMS.get("mq_expansions") or {}).items()
}
BOOSTS = TERMS["boosts"]  # dict de boosts opcionais
DEPARTMENT_TOKEN_TO_SLUG: Dict[str, str] = {}
for slug, label in DEPARTMENTS.items():
    slug_norm = _strip_accents_lower_safe(slug)
    label_norm = _strip_accents_lower_safe(label)
    if slug_norm:
        DEPARTMENT_TOKEN_TO_SLUG[slug_norm] = slug_norm
    if label_norm:
        DEPARTMENT_TOKEN_TO_SLUG[label_norm] = slug_norm
    for expansion in SYNONYMS.get(slug, []) or []:
        token_norm = _strip_accents_lower_safe(expansion)
        if token_norm:
            DEPARTMENT_TOKEN_TO_SLUG[token_norm] = slug_norm
CONTACT_INDEX_READY = False
PERSON_CONTACTS: Dict[str, Dict[str, Any]] = {}
CONTACT_ALIAS_MAP: Dict[str, str] = {}

CONTACT_PHONE_KEYWORDS = (
    "telefone",
    "tel",
    "ramal",
    "numero",
    "número",
    "celular",
    "whatsapp",
)
CONTACT_EMAIL_KEYWORDS = (
    "email",
    "e-mail",
    "mail",
)

def _strip_accents_lower_safe(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()

def _build_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for base, alias_list in ALIASES.items():
        canonical = _strip_accents_lower_safe(base)
        lookup[canonical] = canonical
        for alias in alias_list or []:
            lookup[_strip_accents_lower_safe(alias)] = canonical
    return lookup

ALIAS_LOOKUP = _build_alias_lookup()
DEPT_DISPLAY_TO_SLUG = {
    _strip_accents_lower_safe(name): slug
    for slug, name in DEPARTMENTS.items()
}
print(
    f"[DICT] departamentos={len(DEPARTMENTS)} aliases={len(ALIASES)} "
    f"synonyms={len(SYNONYMS)} mq_expansions={len(CUSTOM_MQ_EXPANSIONS)}"
)

_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n+")
_TITLE_PREFIX_RE = re.compile(r"^(prof(?:essor)?a?|prof\.?|profa\.?|profa|professora|dr\.?|dra\.?|doutor(?:a)?)\s+", re.IGNORECASE)
_GENERIC_NAME_TOKENS = {
    "bacharelado",
    "departamento",
    "secretaria",
    "curso",
    "corpo",
    "docente",
    "telefones",
    "contato",
    "impressao",
    "reserva",
    "servico",
    "faculdade",
}

def _split_blocks(text: str) -> List[str]:
    return [blk for blk in _BLOCK_SPLIT_RE.split(text) if blk and blk.strip()]

def _clean_person_name(raw: str) -> str:
    if not raw:
        return ""
    name = raw.strip(":- \u2013\u2014")
    name = _TITLE_PREFIX_RE.sub("", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()

def _unique_citations(citations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for cit in citations:
        if not isinstance(cit, dict):
            continue
        key = (cit.get("source"), cit.get("chunk"), cit.get("preview"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(cit)
    return unique

def _dept_hint_from_question(question: str) -> Optional[str]:
    q_norm = _strip_accents_lower(question)
    if any(term in q_norm for term in ("computacao", "computação", "comp.", "bcc", "bsi", "dco")):
        return "computacao"
    if "psicologia" in q_norm or "dpsi" in q_norm:
        return "psicologia"
    if "biologia" in q_norm or "dcb" in q_norm:
        return "biologia"
    return None

def _ensure_contact_index(vectorstore: FAISS) -> None:
    global CONTACT_INDEX_READY, PERSON_CONTACTS, ALIAS_LOOKUP, CONTACT_ALIAS_MAP
    if CONTACT_INDEX_READY or vectorstore is None:
        return
    try:
        raw_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
    except Exception:
        raw_docs = []
    docs = normalize_documents(raw_docs)
    contacts: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        if not text.strip():
            continue
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or ""
        dept = _guess_dept_from_source(source)
        for block in _split_blocks(text):
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if not lines:
                continue
            name_candidate = _clean_person_name(lines[0])
            if len(name_candidate.split()) < 2:
                continue
            name_tokens = {_strip_accents_lower(tok) for tok in _tokenize_letters(name_candidate)}
            if name_tokens & _GENERIC_NAME_TOKENS:
                continue
            contacts_found = _extract_contacts([block])
            phones = contacts_found.get("phones", [])
            emails = contacts_found.get("emails", [])
            if not phones and not emails:
                continue
            canonical = _strip_accents_lower(name_candidate)
            CONTACT_ALIAS_MAP[canonical] = canonical
            entry = contacts.setdefault(canonical, {
                "name": name_candidate,
                "phones": set(),
                "emails": set(),
                "departments": set(),
                "dept_slugs": set(),
                "citations": [],
            })
            entry["phones"].update(phones)
            entry["emails"].update(emails)
            if dept:
                dept_slug = DEPT_DISPLAY_TO_SLUG.get(_strip_accents_lower_safe(dept), _strip_accents_lower_safe(dept))
                entry["departments"].add(dept)
                if dept_slug:
                    entry["dept_slugs"].add(dept_slug)
            entry["citations"].append(_as_citation(doc))

            # Enrich alias lookup with individual tokens
            for token in _tokenize_letters(name_candidate):
                token_norm = _strip_accents_lower(token)
                if len(token_norm) >= 4 and token_norm not in _GENERIC_NAME_TOKENS:
                    CONTACT_ALIAS_MAP[token_norm] = canonical
                    ALIAS_LOOKUP.setdefault(token_norm, canonical)

            # Map configured aliases (from terms.yml) that match this name
            for alias_key, alias_variants in list(ALIASES.items()):
                alias_key_norm = _strip_accents_lower(alias_key)
                alias_variants_norm = {_strip_accents_lower(a) for a in alias_variants or []}
                if (alias_key_norm in name_tokens) or (alias_variants_norm & name_tokens):
                    CONTACT_ALIAS_MAP[alias_key_norm] = canonical
                    ALIAS_LOOKUP[alias_key_norm] = canonical
                    for alias_norm in alias_variants_norm:
                        if len(alias_norm) >= 3:
                            CONTACT_ALIAS_MAP[alias_norm] = canonical
                            ALIAS_LOOKUP[alias_norm] = canonical

    # Materialize citations and propagate aliases
    for canonical, entry in list(contacts.items()):
        entry["phones"] = sorted(entry["phones"])
        entry["emails"] = sorted(entry["emails"])
        entry["departments"] = {d for d in entry["departments"] if d}
        entry["dept_slugs"] = {d for d in entry["dept_slugs"] if d}
        entry["citations"] = _unique_citations(entry["citations"])
        for alias, canonical_target in list(ALIAS_LOOKUP.items()):
            canonical_target_norm = _strip_accents_lower_safe(canonical_target)
            if canonical_target_norm == canonical:
                CONTACT_ALIAS_MAP[alias] = canonical
        for token in _tokenize_letters(entry["name"]):
            token_norm = _strip_accents_lower(token)
            if len(token_norm) >= 4 and token_norm not in _GENERIC_NAME_TOKENS:
                CONTACT_ALIAS_MAP[token_norm] = canonical
                ALIAS_LOOKUP.setdefault(token_norm, canonical)

    PERSON_CONTACTS = contacts
    CONTACT_INDEX_READY = True
    sample_keys = list(PERSON_CONTACTS.keys())[:5]
    sample_aliases = list(CONTACT_ALIAS_MAP.items())[:8]
    print(f"[CONTACT] index built with {len(PERSON_CONTACTS)} people; sample={sample_keys}; aliases={sample_aliases}", flush=True)

def _contact_lookup(question: str) -> Optional[Dict[str, Any]]:
    if not PERSON_CONTACTS:
        return None
    q_norm = _strip_accents_lower(question)
    print(f"[CONTACT] question_norm={q_norm}", flush=True)
    keywords = CONTACT_PHONE_KEYWORDS + CONTACT_EMAIL_KEYWORDS + ("contato",)
    if not any(term in q_norm for term in keywords):
        print("[CONTACT] no contact keyword found", flush=True)
        return None

    tokens = {_strip_accents_lower(t) for t in _candidate_terms(question)}
    dept_hint = _dept_hint_from_question(question)
    print(f"[CONTACT] lookup tokens={tokens} dept_hint={dept_hint}", flush=True)
    for token in tokens:
        canonical = CONTACT_ALIAS_MAP.get(token)
        if canonical is None:
            alias_target = ALIAS_LOOKUP.get(token)
            if alias_target:
                canonical = CONTACT_ALIAS_MAP.get(_strip_accents_lower(alias_target), _strip_accents_lower(alias_target))
            else:
                canonical = token
        print(f"[CONTACT] token={token} resolves_to={canonical if canonical else 'None'}", flush=True)
        entry = PERSON_CONTACTS.get(canonical)
        if not entry:
            print(f"[CONTACT] no entry for {canonical}", flush=True)
            continue
        dept_slugs = entry.get("dept_slugs") or set()
        if dept_hint and dept_slugs and dept_hint not in dept_slugs:
            print(f"[CONTACT] entry {entry['name']} skipped due to dept mismatch {dept_slugs}", flush=True)
            continue
        print(f"[CONTACT] hit {entry['name']} with phones={entry.get('phones')}", flush=True)
        return {
            "canonical": canonical,
            "matched_token": token,
            "entry": entry,
            "dept_hint": dept_hint,
        }
    return None

def _format_contact_answer(entry: Dict[str, Any], question: str) -> str:
    name = entry.get("name") or "Contato"
    phones_all: List[str] = list(entry.get("phones") or [])
    emails_all: List[str] = list(entry.get("emails") or [])
    tokens = _question_tokens(question)

    phones = _select_best_contacts(set(phones_all), tokens, top_n=2) if phones_all else []
    emails = _select_best_contacts(set(emails_all), tokens, top_n=3) if emails_all else []

    if not phones and phones_all:
        phones = phones_all[:2]
    if not emails and emails_all:
        emails = emails_all[:3]

    depts = entry.get("departments") or set()
    lines = [f"### Contato", f"**Nome:** {name}"]
    if depts:
        lines.append(f"**Departamento(s):** {', '.join(sorted(depts))}")
    if phones:
        lines.append(f"**Telefone(s):** {', '.join(phones)}")
    if emails:
        lines.append(f"**E-mail(s):** {', '.join(emails)}")
    return "\n".join(lines)

#### RERANK -----------------------------------------------------------------------
# Variável global para armazenar o modelo do reranker e evitar recarregá-lo.
_reranker_model = None  # cache em memória


def _safe_score(x):
    try:
        return float(x)
    except Exception:
        return float("-inf")


def _get_reranker():
    """Carrega um reranker sob demanda.
    1) Tenta CrossEncoder (sentence-transformers).
    2) Se falhar (prompt interativo / trust_remote_code), usa fallback via transformers (HF) com trust_remote_code=True.
    Nunca levanta exceção para cima.
    """
    global _reranker_model
    if not RERANKER_ENABLED:
        return None

    if _reranker_model is not None:
        return _reranker_model

    device = os.getenv("RERANKER_DEVICE", "cpu")
    print(f"[INFO] RERANK: enabled=1 name={RERANKER_NAME} max_len={RERANKER_MAX_LEN} device={device}", flush=True)

    # 1) Caminho direto: CrossEncoder (quando funciona, é o mais simples)
    if CrossEncoder is not None:
        try:
            _reranker_model = CrossEncoder(
                RERANKER_NAME,
                max_length=RERANKER_MAX_LEN,
                device=device,
                automodel_args={"trust_remote_code": True},
                tokenizer_args={"trust_remote_code": True},
            )
            print("[INFO] RERANK: CrossEncoder carregado com sucesso.", flush=True)
            return _reranker_model
        except Exception as e:
            print(f"[WARN] RERANK: CrossEncoder falhou: {e} — tentando fallback HF.", flush=True)

    # 2) Fallback HF: importar dentro da função para evitar NameError de escopo
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    except Exception as e_imp:
        print(f"[WARN] RERANK: Transformers/torch indisponíveis ({e_imp}) — desabilitando rerank.", flush=True)
        _reranker_model = None
        return None

    try:
        tok = AutoTokenizer.from_pretrained(RERANKER_NAME, trust_remote_code=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME, trust_remote_code=True)
        mdl.to(device)
        mdl.eval()

        class _HFCE:
            def __init__(self, tokenizer, model, max_length, device):
                self.tok = tokenizer
                self.model = model
                self.max_length = int(max_length)
                self.device = device

            def predict(self, pairs):
                # pairs: List[Tuple[str,str]]
                if not pairs:
                    return []
                a = [p[0] or "" for p in pairs]
                b = [p[1] or "" for p in pairs]
                enc = self.tok(
                    a, b,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                # mover tensores para device
                for k in enc:
                    enc[k] = enc[k].to(self.device)
                with torch.no_grad():
                    out = self.model(**enc).logits
                    # 1 logit => sigmoid; 2 logits => prob classe 1
                    if out.shape[-1] == 1:
                        scores = torch.sigmoid(out).squeeze(-1)
                    else:
                        probs = torch.softmax(out, dim=-1)
                        scores = probs[..., 1]
                    return [float(x) for x in scores.detach().cpu().tolist()]

        _reranker_model = _HFCE(tok, mdl, RERANKER_MAX_LEN, device)
        print("[INFO] RERANK: fallback HF carregado com sucesso.", flush=True)
        return _reranker_model

    except Exception as e_hf:
        print(f"[WARN] RERANK: Fallback HF falhou: {repr(e_hf)} — desabilitando rerank.", flush=True)
        _reranker_model = None
        return None


def _apply_rerank(cands, top_k=5, user_query: str | None = None, dbg: dict | None = None):
    """
    cands: lista de dicts com {"text","preview","source","chunk", "idx" opcional}
    Retorna (docs_ordenados, scores_float) ORDENADOS por score (desc) para TODOS os candidatos.
    - Garante que scores SEMPRE sejam floats.
    - Preenche debug.rerank.* e debug.timing_ms.reranker (ms).
    """
    if not cands:
        if dbg is not None:
            dbg.setdefault("rerank", {})
            dbg["rerank"]["enabled"] = False
            dbg["rerank"]["name"] = None
            dbg["rerank"]["top_k"] = top_k
            dbg.setdefault("timing_ms", {})
            dbg["timing_ms"]["reranker"] = 0.0
        return [], []

    model = _get_reranker()
    # Caso não haja modelo, degrade graceful mantendo floats 0.0
    if model is None:
        docs = list(cands)
        scores = [0.0] * len(docs)
        if dbg is not None:
            dbg.setdefault("rerank", {})
            dbg["rerank"]["enabled"] = False
            dbg["rerank"]["name"] = None
            dbg["rerank"]["top_k"] = top_k
            dbg["rerank"]["scored"] = [
                {
                    "source": d.get("source"),
                    "chunk": d.get("chunk"),
                    "preview": (d.get("preview") or d.get("text",""))[:120],
                    "score": 0.0,
                }
                for d in docs[: min(top_k, len(docs))]
            ]
            dbg.setdefault("timing_ms", {})
            dbg["timing_ms"]["reranker"] = 0.0
        return docs, scores

    if user_query is None:
        user_query = ""

    pairs = [(user_query, d.get("text","") or "") for d in cands]

    t_rerank0 = time.perf_counter()
    try:
        raw_scores = model.predict(pairs)  # sequência de floats
        # Sanitiza possíveis None/NaN do modelo (por segurança)
        scores_clean = []
        for s in raw_scores:
            try:
                v = float(s)
            except Exception:
                v = 0.0
            if math.isnan(v):  # requer 'import math' no topo do arquivo
                v = 0.0
            scores_clean.append(v)
        raw_scores = scores_clean
    except Exception as e:
        print(f"[WARN] Rerank falhou em runtime: {e}. Prosseguindo sem rerank.")
        docs = list(cands)
        scores = [0.0] * len(docs)
        if dbg is not None:
            dbg.setdefault("rerank", {})
            dbg["rerank"]["enabled"] = False
            dbg["rerank"]["name"] = None
            dbg["rerank"]["top_k"] = top_k
            dbg.setdefault("timing_ms", {})
            dbg["timing_ms"]["reranker"] = round((time.perf_counter() - t_rerank0) * 1000.0, 1)
        return docs, scores

    # Ordena TODOS os candidatos por score desc; truncagem apenas para debug.exibição
    order_all = sorted(range(len(raw_scores)), key=lambda i: float(raw_scores[i]), reverse=True)
    docs_sorted = [cands[i] for i in order_all]
    scores_sorted = [float(raw_scores[i]) for i in order_all]

    if dbg is not None:
        dbg.setdefault("rerank", {})
        dbg["rerank"]["enabled"] = True
        dbg["rerank"]["name"] = RERANKER_NAME
        dbg["rerank"]["top_k"] = top_k
        dbg["rerank"]["scored"] = [
            {
                "source": docs_sorted[i].get("source"),
                "chunk": docs_sorted[i].get("chunk"),
                "preview": (docs_sorted[i].get("preview") or docs_sorted[i].get("text",""))[:120],
                "score": scores_sorted[i],
            }
            for i in range(min(top_k, len(docs_sorted)))
        ]
        dbg.setdefault("timing_ms", {})
        dbg["timing_ms"]["reranker"] = round((time.perf_counter() - t_rerank0) * 1000.0, 1)

    return docs_sorted, scores_sorted


def _build_candidates_from_docs(docs):
    """
    Normaliza resultados do FAISS para candidatos com texto completo.
    Aceita listas de `Document` ou pares (Document, score).
    """
    cands = []
    for i, d in enumerate(docs):
        doc = d[0] if isinstance(d, (tuple, list)) and len(d) >= 1 else d
        meta = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or ""
        cands.append({
            "idx": i,  # preserva o índice original para reordenar Documentos
            "text": text,
            "preview": text[:200],
            "source": meta.get("source"),
            "chunk": meta.get("chunk"),
            "vector_score": meta.get("vector_score"),
        })
    return cands

# --- FIM helpers RERANK ---

# ==== Utilitários de Geração de Multi-Query e Processamento de Texto ====


def _gen_variants_local(q: str) -> List[str]:
    """
    Gera variações simples e determinísticas de uma pergunta em português.
    Serve como um fallback caso a geração via LLM falhe.
    """
    qn = q.strip()
    base = [qn]
    # Variações comuns: minúsculas, sem interrogação, expandindo "qual" para "qual é", etc.
    extra = [
        qn.lower(),
        qn.replace("?", "").strip(),
        qn.replace("quais", "quais são").replace("qual", "qual é"),
    ]
    return _dedupe_preserve_order(base + extra)




def _question_tokens(question: str) -> Set[str]:
    tokens = {_strip_accents_lower(tok) for tok in _candidate_terms(question)}
    question_norm = _strip_accents_lower(question)
    tokens.update(tok for tok in question_norm.split() if tok)
    return {tok for tok in tokens if tok}

def _question_wants_email(question: str) -> bool:
    q_norm = _strip_accents_lower(question)
    return any(keyword in q_norm for keyword in CONTACT_EMAIL_KEYWORDS)

def _question_wants_phone(question: str) -> bool:
    q_norm = _strip_accents_lower(question)
    return any(keyword in q_norm for keyword in CONTACT_PHONE_KEYWORDS)

def _collect_metadata_contacts(docs: List[Document]) -> Dict[str, Set[str]]:
    emails: Set[str] = set()
    phones: Set[str] = set()
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        for email in meta.get("emails") or []:
            email_norm = (email or "").strip()
            if email_norm:
                emails.add(email_norm)
        for phone in meta.get("phones") or []:
            phone_norm = (phone or "").strip()
            if phone_norm:
                phones.add(phone_norm)
    return {"emails": emails, "phones": phones}

def _select_best_contacts(items: Set[str], question_tokens: Set[str], top_n: int = 3) -> List[str]:
    scored: List[Tuple[int, int, str]] = []
    for item in items:
        item_norm = _strip_accents_lower(item)
        score = 0
        for token in question_tokens:
            if token and token in item_norm:
                score += 1
        scored.append((score, -len(item), item))
    scored.sort(reverse=True)
    best = [item for score, _neg_len, item in scored if score > 0][:top_n]
    if not best:
        best = [item for _score, _neg_len, item in scored[:top_n]]
    return best

def _build_metadata_answer(question: str, docs: List[Document], dbg: dict) -> Optional[str]:
    wants_email = _question_wants_email(question)
    wants_phone = _question_wants_phone(question)
    if not wants_email and not wants_phone:
        return None
    contacts = _collect_metadata_contacts(docs)
    tokens = _question_tokens(question)
    selected_emails = _select_best_contacts(contacts["emails"], tokens) if wants_email else []
    selected_phones = _select_best_contacts(contacts["phones"], tokens) if wants_phone else []
    if wants_email and not selected_emails:
        return None
    if wants_phone and not selected_phones:
        return None
    lines = ["### Contato"]
    if selected_emails:
        lines.append(f"**E-mail(s):** {', '.join(selected_emails)}")
    if selected_phones:
        lines.append(f"**Telefone(s):** {', '.join(selected_phones)}")
    if dbg is not None:
        dbg.setdefault("metadata", {})["direct_answer"] = {
            "emails": selected_emails,
            "phones": selected_phones,
        }
    return "\n".join(lines)

def _normalize_variant_candidate(text: str) -> str:
    """Limpa bullets/números e normaliza espaços para uma variação de consulta."""
    variant = (text or "").strip()
    variant = re.sub(r'^[-*\d\.\)\(]+\s*', '', variant)
    variant = re.sub(r'\s+', ' ', variant).strip()
    return variant

def _gen_variants_llm(user_query: str, limit: int) -> List[str]:
    """Gera variações extras via LLM usando o prompt dedicado."""
    if not MQ_USE_LLM or limit <= 0:
        return []
    if not _MQ_VARIANTS_PROMPT:
        return []
    system_prompt = _MQ_VARIANTS_PROMPT.replace("{MAX_VARIANTS}", str(limit))
    try:
        text, _ = call_llm(system_prompt, user_query, expect_json=False, max_tokens=200)
    except Exception as exc:
        print(f"[MQ] WARN: Falha ao gerar variações com LLM: {exc}", flush=True)
        return []

    variants: List[str] = []
    for raw in text.splitlines():
        variant = _normalize_variant_candidate(raw)
        if not variant:
            continue
        if variant.lower() == user_query.strip().lower():
            continue
        if variant in variants:
            continue
        variants.append(variant)
        if len(variants) >= limit:
            break
    if variants:
        _log_debug(f"MQ LLM variants: {variants}")
    return variants

def _gen_multi_queries(user_query: str, n: int, llm=None) -> List[str]:
    """
    Gera variações para a busca combinando heurísticas locais, dicionário e LLM.
    """
    requested_n = max(1, n)

    base_local = _gen_variants_local(user_query)

    s_norm = _strip_accents_lower(user_query)
    syn_vars: List[str] = []
    custom_vars: List[str] = []
    for key, exps in SYNONYMS.items():
        if key in s_norm:
            syn_vars.extend(exps)
    for key, exps in CUSTOM_MQ_EXPANSIONS.items():
        if key in s_norm:
            custom_vars.extend(exps)

    llm_vars: List[str] = []
    if llm is not None:
        try:
            prompt = (
                "Gere variações curtas e diferentes (1 por linha) desta pergunta, "
                "mantendo o sentido, para busca em base de conhecimento:\n"
                f"{user_query}\n"
                f"(gere no máximo {n})"
            )
            txt = llm.generate_variants(prompt, n)  # adapte ao seu wrapper, se existir
            llm_vars = [
                _normalize_variant_candidate(ln)
                for ln in txt.splitlines()
                if _normalize_variant_candidate(ln)
            ]
        except Exception:
            llm_vars = []

    llm_vars_env: List[str] = []
    if MQ_USE_LLM:
        budget = max(0, requested_n - len(custom_vars) - len(base_local) - len(syn_vars) - len(llm_vars))
        budget = min(budget, MQ_LLM_MAX_VARIANTS)
        if budget > 0:
            llm_vars_env = _gen_variants_llm(user_query, budget)

    merged = _dedupe_preserve_order(
        [user_query] + custom_vars + base_local + syn_vars + llm_vars + llm_vars_env
    )
    effective_n = max(requested_n, 1 + len(custom_vars) + len(llm_vars) + len(llm_vars_env))
    return merged[:effective_n]
    # 1) variações locais
    base_local = _gen_variants_local(user_query)

    # 2) variações por sinônimos do dicionário
    s_norm = _strip_accents_lower(user_query)
    syn_vars = []
    custom_vars = []
    for key, exps in SYNONYMS.items():
        if key in s_norm:
            syn_vars.extend(exps)
    for key, exps in CUSTOM_MQ_EXPANSIONS.items():
        if key in s_norm:
            custom_vars.extend(exps)

    # 3) (opcional) pedir variantes ao LLM (se houver wrapper compatível)
    llm_vars = []
    if llm is not None:
        try:
            prompt = (
                "Gere variações curtas e diferentes (1 por linha) desta pergunta, "
                "mantendo o sentido, para busca em base de conhecimento:\\n"
                f"{user_query}\\n"
                f"(gere no máximo {n})"
            )
            txt = llm.generate_variants(prompt, n)  # adapte ao seu wrapper, se existir
            llm_vars = [ln.strip(" -•\\t") for ln in txt.splitlines() if ln.strip()]
        except Exception:
            llm_vars = []

    # 4) mescla e corta em n
    merged = _dedupe_preserve_order([user_query] + custom_vars + base_local + syn_vars + llm_vars)
    effective_n = max(requested_n, 1 + len(custom_vars))
    return merged[:effective_n]


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\;\:])\s+|\n+")  # Regex para dividir texto em sentenças
_WS = re.compile(r"\s+")  # Regex para normalizar espaços em branco
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)  # Regex para encontrar e-mails
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d{4,5}-\d{4}")  # Regex para encontrar números de telefone
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")  # Regex para encontrar palavras (letras acentuadas incluídas)
PERSON_BLOCK_SPLIT = re.compile(r"\n\s*\n+")

# Palavras de parada (stopwords) comuns em português, usadas para filtrar termos de busca.
STOP = {
    "que", "qual", "quais", "sobre", "para", "como", "onde", "quem",
    "contato", "email", "e-mail", "fone", "telefone", "ramal",
    "da", "de", "do", "um", "uma", "o", "a", "os", "as"
}


def _norm_ws(s: str) -> str:
    """Normaliza espaços em branco em uma string, substituindo múltiplos espaços por um único e removendo espaços nas extremidades."""
    return _WS.sub(" ", (s or "").strip())


def _strip_accents_lower(s: str) -> str:
    """Remove acentos e converte uma string para minúsculas para facilitar comparações."""
    nfkd = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def _tokenize_letters(text: str) -> List[str]:
    """Tokeniza um texto, extraindo apenas sequências de letras como palavras."""
    return WORD_RE.findall(text or "")


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokeniza texto para BM25 (minúsculas e sem acentos)."""
    return [
        _strip_accents_lower(token)
        for token in _tokenize_letters(text or "")
        if len(token) >= 2
    ]


def _person_block(text: str, name_term: Optional[str]) -> Optional[str]:
    """Extrai o bloco textual referente a uma pessoa (linhas separadas por espaços)."""
    if not text:
        return None
    variants = []
    if name_term:
        for candidate in [name_term, *_term_variants(name_term)]:
            norm = _strip_accents_lower(candidate)
            if norm and norm not in variants:
                variants.append(norm)
    if not variants:
        return None
    for segment in PERSON_BLOCK_SPLIT.split(text):
        seg_norm = _strip_accents_lower(segment)
        if any(v in seg_norm for v in variants):
            return _norm_ws(segment)
    return None


def _guess_dept_from_source(src: str) -> Optional[str]:
    """Tenta adivinhar o departamento a partir do texto da fonte do documento."""
    s = _strip_accents_lower(src or "")
    # percorre os slugs carregados do YAML
    for slug, label in DEPARTMENTS.items():
        if slug in s:
            return label
    return None


def _dept_hints_in_question(q: str) -> List[str]:
    """Identifica possíveis departamentos mencionados na pergunta do usuário."""
    s = _strip_accents_lower(q)
    hints = []
    for slug, label in DEPARTMENTS.items():
        if slug in s and label not in hints:
            hints.append(label)
    return hints


def _dept_slugs_in_question(q: str) -> List[str]:
    """
    Identifica slugs de departamento (ex: 'computacao', 'biologia') na pergunta do usuário.
    Usa o dicionário DEPARTMENTS carregado do arquivo YAML para encontrar correspondências.
    """
    s = _strip_accents_lower(q or "")
    hits: List[str] = []
    for slug in DEPARTMENTS.keys():
        if slug in s:
            hits.append(slug)
    return hits


# ==== Funções de Extração de Termos ====
def _candidate_terms(question: str) -> List[str]:
    """Extrai termos candidatos da pergunta, incluindo e-mails e palavras com 3+ caracteres, excluindo stopwords."""
    q = (question or "").strip()
    if not q:
        return []
    emails = EMAIL_RE.findall(q)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", q)  # Encontra palavras com 3 ou mais caracteres
    terms, seen = [], set()
    for w in emails + words:
        k = _strip_accents_lower(w)
        if k not in STOP and k not in seen:
            terms.append(w)
            seen.add(k)
    return terms


def _term_variants(term: str) -> List[str]:
    """Gera variantes de um termo (ex: com/sem acento, aliases) para busca flexível."""
    t = _strip_accents_lower(term)
    vs = {t}
    # Adiciona aliases de nomes comuns (agora carregados do YAML)
    for base, al in ALIASES.items():
        if t == base or t in al:
            vs.update([base] + al)
    # Lida com variações de "eia" vs "ea"
    if t.endswith("eia"):
        vs.add(t[:-3] + "ea")
    if t.endswith("ea"):
        vs.add(t[:-2] + "eia")
    # Normalizações simples de caracteres
    vs.add(
        t.replace("ç", "c").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u"))
    return list(vs)


def _is_name_like(token: str) -> bool:
    """Verifica se um token parece ser um nome (apenas letras, 5+ caracteres)."""
    return token.isalpha() and len(token) >= 5


def _name_token_match(token_norm: str, term_norm: str) -> bool:
    """Compara um token com um termo de busca usando a distância de Levenshtein normalizada."""
    if abs(len(token_norm) - len(term_norm)) > 1:
        return False
    sim = distance.Levenshtein.normalized_similarity(token_norm, term_norm)
    return sim >= 0.86


def _sentence_hits_by_name(text: str, terms: List[str]) -> List[Tuple[str, int]]:
    """Encontra sentenças em um texto que correspondem a uma lista de termos de busca (nomes)."""
    out: List[Tuple[str, int]] = []
    if not text or not terms:
        return out
    sentences = [s for s in _SENT_SPLIT.split(text) if _norm_ws(s)]
    all_norms = list({tn for t in terms for tn in _term_variants(t)})

    for sent in sentences:
        s_clean = _norm_ws(sent)
        s_norm = _strip_accents_lower(s_clean)
        best = 0
        tokens = _tokenize_letters(s_clean)
        tokens_norm = [_strip_accents_lower(tk) for tk in tokens]
        hit = False
        for tn in all_norms:
            for tk in tokens_norm:
                if _is_name_like(tk) and _name_token_match(tk, tn):
                    sc = int(100 * distance.Levenshtein.normalized_similarity(tk, tn))
                    best = max(best, sc)
                    hit = True
        if not hit:
            for tn in all_norms:
                sc = fuzz.partial_ratio(tn, s_norm)
                best = max(best, sc)
        if best >= LEXICAL_THRESHOLD:
            out.append((s_clean, best))

    out.sort(key=lambda x: x[1], reverse=True)
    seen, uniq = set(), []
    for s, sc in out:
        k = s.lower()
        if k not in seen:
            uniq.append((s, sc))
            seen.add(k)
    return uniq[:3]


def _coerce_sentence_hits(snippets: Iterable[Tuple[str, Any]]) -> List[Tuple[str, int]]:
    """Normaliza escores de sentenças para inteiros, mantendo até 3 entradas."""
    normalized: List[Tuple[str, int]] = []
    for snippet, score in snippets or []:  # type: ignore
        if not snippet:
            continue
        try:
            score_val = int(round(float(score)))
        except Exception:
            score_val = 0
        normalized.append((snippet, score_val))
        if len(normalized) >= 3:
            break
    return normalized


NAME_SEQ_RE = re.compile(r"([A-ZÁ-Ü][a-zá-ü]+(?:\s+[A-ZÁ-Ü][a-zá-ü]+){1,5})")  # Regex para sequências de nomes


def _normalize_phone(p: str) -> str:
    """Normaliza um número de telefone para um formato padrão."""
    digits = re.sub(r"\\D", "", p or "")
    if len(digits) == 11:
        return f"({digits[0:2]}) {digits[2:7]}-{digits[7:11]}"
    if len(digits) == 10:
        return f"({digits[0:2]}) {digits[2:6]}-{digits[6:10]}"
    return p


def _extract_contacts(texts: List[str]) -> Dict[str, List[str]]:
    """Extrai e-mails e telefones de uma lista de textos."""
    emails, phones = set(), set()
    for t in texts:
        for e in EMAIL_RE.findall(t or ""):
            emails.add(e.lower())
        for p in PHONE_RE.findall(t or ""):
            phones.add(_normalize_phone(p))
    return {"emails": sorted(emails), "phones": sorted(phones)}


def _extract_name(snippets: List[str], term: str) -> Optional[str]:
    """Extrai um nome completo de uma lista de snippets de texto, com base em um termo de busca."""
    variants: List[str] = []
    if term:
        for candidate in [term, *_term_variants(term)]:
            norm = _strip_accents_lower(candidate)
            if norm and norm not in variants:
                variants.append(norm)
    for s in snippets:
        cands = NAME_SEQ_RE.findall(s or "")
        if not cands:
            continue
        best, best_sim = None, 0.0
        for cand in cands:
            for tk in _tokenize_letters(cand):
                token_norm = _strip_accents_lower(tk)
                targets = variants or [token_norm]
                for tgt in targets:
                    sim = distance.Levenshtein.normalized_similarity(token_norm, tgt)
                    if sim > best_sim:
                        best_sim = sim
                        best = cand
        if best and (best_sim >= 0.85 if variants else best_sim >= 0.80):
            return best
    if not variants:
        for s in snippets:
            cands = NAME_SEQ_RE.findall(s or "")
            if cands:
                return cands[0]
    return None


def _as_citation(doc: Document) -> Dict[str, Any]:
    """Formata um documento LangChain como uma citação para a resposta final."""
    src = doc.metadata.get("source") or "desconhecido"
    chunk = doc.metadata.get("chunk")
    preview = _norm_ws(doc.page_content)[:180]
    return {"source": src, "chunk": int(chunk) if str(chunk).isdigit() else None, "preview": preview}


def _top_sentences(question: str, texts: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
    """Seleciona as sentenças mais relevantes de uma lista de textos, com base na similaridade com a pergunta."""
    sentences: List[str] = []
    for t in texts:
        for s in _SENT_SPLIT.split(t or ""):
            s = _norm_ws(s)
            if s:
                sentences.append(s)
    seen, uniq = set(), []
    for s in sentences:
        k = s.lower()
        if k not in seen:
            uniq.append(s)
            seen.add(k)
    q = _norm_ws(question)
    scored = [(s, fuzz.token_set_ratio(q, s) / 100.0) for s in uniq]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def _context_window_around_name(
    doc_text: str,
    person_name: str,
    chars_before: int = 220,
    chars_after: int = 300,
) -> Optional[str]:
    """Extrai uma janela de contexto de texto ao redor de um nome de pessoa."""
    if not (doc_text and person_name):
        return None
    txt = doc_text
    name = person_name.strip()
    idx = txt.find(name)
    if idx < 0:
        compact_name = _norm_ws(name)
        idx = _norm_ws(txt).find(compact_name)
        if idx < 0:
            return None
    start = max(0, idx - chars_before)
    end = min(len(txt), idx + len(name) + chars_after)
    return _norm_ws(txt[start:end])


# ==== Funções de Interação com LLM ====
# Carrega os prompts para triagem, pedido de informação e resposta final.
_TRIAGE_PROMPT = load_prompt("triagem_prompt.txt")
_PEDIR_INFO_PROMPT = load_prompt("pedir_info_prompt.txt")
_RESPOSTA_PROMPT = load_prompt("resposta_final_prompt.txt")
_MQ_VARIANTS_PROMPT = load_prompt("mq_variants_prompt.txt")


def _llm_triage(question: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    if not _TRIAGE_PROMPT:
        return {"action": "AUTO_RESOLVER", "ask": ""}
    qn = _norm_ws(question)
    user_payload = f"Pergunta: {qn}\\n\\nSinais:\\n{signals}"
    _, data = call_llm(_TRIAGE_PROMPT, user_payload, expect_json=True, max_tokens=150)
    if isinstance(data, dict) and "action" in data:
        return {"action": data.get("action"), "ask": data.get("ask", "")}
    return {"action": "AUTO_RESOLVER", "ask": ""}


def _llm_pedir_info(question: str, options: List[str], prefer_attr: str = "departamento") -> str:
    """Usa o LLM para gerar uma pergunta de esclarecimento ao usuário."""
    if not _PEDIR_INFO_PROMPT:
        if options:
            opts = ", ".join(options[:3])
            return f"Encontrei múltiplas possibilidades: {opts}. Pode dizer qual delas?"
        return "Pode informar o departamento/unidade para eu localizar a pessoa correta?"
    payload = f"Pergunta original: {question}\\nOpções (máx 3): {options[:3]}\\nAtributo preferencial: {prefer_attr}"
    text, _ = call_llm(_PEDIR_INFO_PROMPT, payload, expect_json=False, max_tokens=120)
    return (text or "").strip() or "Pode informar o departamento/unidade?"


def _format_direct(name: Optional[str], dept: Optional[str], contacts: Dict[str, List[str]],
                   context_snippets: List[str]) -> str:
    """Formata uma resposta direta com as informações encontradas, sem usar o LLM."""
    parts = []
    if name:
        parts.append(name)
    if dept:
        parts.append(dept)
    if contacts.get("phones"):
        parts.append("📞 " + ", ".join(contacts["phones"]))
    if contacts.get("emails"):
        parts.append("✉️ " + ", ".join(contacts["emails"]))
    if parts:
        return " — ".join(parts)
    return (context_snippets[0] if context_snippets else "Não encontrei.")[:400]


def _llm_resposta_final(question: str, context_snippets: List[str], name: Optional[str], dept: Optional[str],
                        contacts: Dict[str, List[str]]) -> str:
    """Usa o LLM para gerar uma resposta final em linguagem natural, com base no contexto e metadados extraídos."""
    if not _RESPOSTA_PROMPT:
        return _format_direct(name, dept, contacts, context_snippets)
    ctx = "\\n\\n".join(f"- {s}" for s in context_snippets[:6])
    meta = {"nome": name or "", "departamento": dept or "", "telefones": contacts.get("phones", []),
            "emails": contacts.get("emails", [])}
    user = f"Pergunta: {question}\\n\\nContexto (trechos):\\n{ctx}\\n\\nMetadados extraídos:\\n{meta}\\n\\nFormato: Nome — Departamento — 📞 — ✉️ (curto)."
    text, _ = call_llm(_RESPOSTA_PROMPT, user, expect_json=False, max_tokens=220)
    if not (text or "").strip():
        return _format_direct(name, dept, contacts, context_snippets)
    return text.strip()


def _initialize_debug_payload(question: str) -> dict:
    """Initializes the debug dictionary for a request."""
    return {
        "question": question,
        "timing_ms": {},
        "route": None,
        "mq_variants": [],
        "faiss": {"k": None, "candidates": []},
        "rerank": {
            "enabled": bool(RERANKER_ENABLED),
            "name": RERANKER_NAME if RERANKER_ENABLED else None,
            "preset": RERANKER_PRESET,
            "top_k": RERANKER_TOP_K if RERANKER_ENABLED else None,
            "scored": []
        },
        "chosen": {"confidence": None},
    }

def _perform_lexical_search(question: str, all_docs: List[Document]) -> List[Tuple[Document, List[Tuple[str, int]], float]]:
    """Performs a lexical search leveraging BM25 (fallbacks to legacy heuristics when unavailable)."""
    if not all_docs:
        return []

    terms = _candidate_terms(question)
    if not terms:
        return []

    if BM25Okapi is None:
        return _perform_lexical_search_legacy(question, all_docs, terms)

    query_tokens = _tokenize_for_bm25(question)
    if not query_tokens:
        return []

    dept_hints = {_strip_accents_lower(h) for h in _dept_hints_in_question(question)}
    corpus_tokens: List[List[str]] = []
    corpus_docs: List[Document] = []
    for doc in all_docs:
        tokens = _tokenize_for_bm25(getattr(doc, "page_content", "") or "")
        if not tokens:
            continue
        corpus_tokens.append(tokens)
        corpus_docs.append(doc)

    if not corpus_tokens:
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)

    lex_hits: List[Tuple[Document, List[Tuple[str, int]], float]] = []
    limit = min(RETRIEVAL_FETCH_K, len(ranked_indices))

    for idx in ranked_indices[:limit]:
        score = float(scores[idx])
        if score <= 0:
            continue
        doc = corpus_docs[idx]
        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata["lexical_score"] = round(score, 6)
        doc.metadata = metadata

        raw_hits = _sentence_hits_by_name(doc.page_content or "", terms)
        if not raw_hits:
            top_sentences = _top_sentences(question, [doc.page_content], top_n=2)
            raw_hits = [(sent, int(round(s * 100))) for sent, s in top_sentences]
        hits = _coerce_sentence_hits(raw_hits)

        src_norm = _strip_accents_lower(metadata.get("source", "") or "")
        score_bonus = DEPT_BONUS if dept_hints and any(h in src_norm for h in dept_hints) else 0.0
        lex_hits.append((doc, hits, score + score_bonus))

    lex_hits.sort(key=lambda item: item[2], reverse=True)
    return lex_hits[:RETRIEVAL_FETCH_K]


def _perform_lexical_search_legacy(
    question: str,
    all_docs: List[Document],
    terms: List[str],
) -> List[Tuple[Document, List[Tuple[str, int]], float]]:
    """Legacy lexical search based on fuzzy matching (used when BM25 is unavailable)."""
    dept_hints = _dept_hints_in_question(question)
    hits_out: List[Tuple[Document, List[Tuple[str, int]], float]] = []
    for doc in all_docs:
        src = (doc.metadata.get("source") or "").lower()
        score_bonus = DEPT_BONUS if any(_strip_accents_lower(h) in src for h in dept_hints) else 0.0
        hits = _sentence_hits_by_name(doc.page_content or "", terms)
        if not hits:
            continue
        best_score = float(max(sc for _, sc in hits)) + score_bonus
        hits_out.append((doc, hits, best_score))

    hits_out.sort(key=lambda item: item[2], reverse=True)
    return hits_out[:min(6, RETRIEVAL_FETCH_K)]

def _handle_lexical_route(question: str, lex_hits: list, dbg: dict) -> Optional[Dict[str, Any]]:
    """
    Handles the entire logic for the lexical-only route.
    If this route is taken, it returns a complete final answer.
    Otherwise, it returns None, and the main pipeline continues.
    """
    if (ROUTE_FORCE == "vector") or not lex_hits or HYBRID_ENABLED:
        return None

    dbg["route"] = "lexical"
    _log_debug("Route: lexical")

    # 1. Extract context and metadata from lexical hits
    sentence_snippets = []
    similar_docs = [d for d, _, _ in lex_hits]
    for doc, hits, _ in lex_hits:
        sentence_snippets.extend(s for s, _ in hits[:2])
        if len(sentence_snippets) >= 6:
            break
    
    contacts = _extract_contacts(sentence_snippets)
    name_terms = [t for t in _candidate_terms(question) if t.isalpha()]
    name = _extract_name(sentence_snippets, name_terms[0]) if name_terms else None
    
    temp_citations = [_as_citation(doc) for doc in similar_docs]
    dept = _guess_dept_from_source(temp_citations[0]["source"]) if temp_citations else None

    context_snippets = list(sentence_snippets)
    if name and not contacts["phones"]:
        ref_doc = lex_hits[0][0]
        win = _context_window_around_name(ref_doc.page_content or "", name, 220, 300)
        if win:
            context_snippets.insert(0, win)
            more_contacts = _extract_contacts([win])
            contacts["phones"] = list(dict.fromkeys(contacts["phones"] + more_contacts["phones"]))
            contacts["emails"] = list(dict.fromkeys(contacts["emails"] + more_contacts["emails"]))

    # 2. Triage with LLM to decide whether to ask for clarification
    triage_signals = {"dept_hints_in_question": _dept_hints_in_question(question), "candidates_found": len(lex_hits), "have_name": bool(name)}
    tri = _llm_triage(question, triage_signals)
    if tri.get("action") == "PEDIR_INFO":
        options = [nm for d, hits, _b in lex_hits[:3] if (nm := _extract_name([hits[0][0]], name_terms[0] if name_terms else ""))]
        ask_text = tri.get("ask") or _llm_pedir_info(question, options, prefer_attr="departamento")
        return {"answer": ask_text, "citations": [], "context_found": False, "needs_clarification": True}

    # 3. Generate final answer
    t_llm = _now()
    final_text = _llm_resposta_final(question, context_snippets or sentence_snippets, name, dept, contacts)
    dbg["timing_ms"]["llm"] = _elapsed_ms(t_llm)

    citations_clean = _collect_citations_from_docs(similar_docs, max_sources=MAX_SOURCES)
    conf = None # Confidence is not calculated in the lexical route
    dbg["chosen"]["confidence"] = conf

    answer_out = final_text
    if STRUCTURED_ANSWER:
        answer_out = _format_answer_markdown(question, final_text, citations_clean, conf)

    # 4. Assemble and return final result
    result = {
        "answer": answer_out,
        "citations": citations_clean,
        "context_found": bool(similar_docs),
        "confidence": conf,
    }
    # Populate debug info for this route
    dbg["faiss"]["k"] = len(similar_docs)
    dbg["faiss"]["candidates"] = [_pack_doc(d, (getattr(d, "metadata", {}) or {}).get("vector_score")) for d in similar_docs[:10]]
    dbg["rerank"]["enabled"] = False
    dbg["rerank"]["scored"] = [_pack_doc(d, 0.0) for d in similar_docs[:5]]
    
    return result


def _apply_question_boosts(question: str, docs: List[Document]) -> None:
    """Aplica boosts condicionais de acordo com o setor do chunk."""
    boosts = (BOOSTS.get("department") or {}) if BOOSTS else {}
    if not boosts or not docs:
        return
    question_norm = _strip_accents_lower(question)
    preferred_slugs = {
        slug for token, slug in DEPARTMENT_TOKEN_TO_SLUG.items()
        if token and token in question_norm
    }
    if not preferred_slugs:
        return
    for doc in docs:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        slug = _strip_accents_lower(metadata.get("sector_slug") or metadata.get("department_slug") or "")
        if not slug or slug not in boosts or slug not in preferred_slugs:
            continue
        factor = float(boosts.get(slug, 1.0))
        if factor <= 0:
            continue
        vector_score = float(metadata.get("vector_score", 0.0))
        lexical_score = float(metadata.get("lexical_score", 0.0))
        if vector_score > 0:
            metadata["vector_score"] = vector_score * factor
        if lexical_score > 0:
            metadata["lexical_score"] = lexical_score * factor
        metadata.setdefault("boost_trace", []).append({"slug": slug, "factor": factor})
        doc.metadata = metadata


def _doc_similarity(doc_a: Document, doc_b: Document) -> float:
    """Calcula similaridade textual aproximada entre dois documentos (0-1)."""
    text_a = _strip_accents_lower(_norm_ws((doc_a.page_content or "")[:512]))
    text_b = _strip_accents_lower(_norm_ws((doc_b.page_content or "")[:512]))
    if not text_a or not text_b:
        return 0.0
    return fuzz.token_set_ratio(text_a, text_b) / 100.0


def _apply_mmr_selection(
    candidates: List[Document],
    top_k: int,
    lambda_param: float,
    dbg: dict,
) -> List[Document]:
    """Applies Maximal Marginal Relevance to encourage diversity before reranking."""
    if not candidates or top_k <= 0 or len(candidates) <= top_k or lambda_param <= 0:
        return candidates

    selected: List[Document] = []
    remaining = list(candidates)
    lambda_clamped = max(0.0, min(lambda_param, 1.0))

    def _relevance(doc: Document) -> float:
        meta = getattr(doc, "metadata", {}) or {}
        vector_score = float(meta.get("vector_score", 0.0))
        lexical_score = float(meta.get("lexical_score", 0.0))
        base = vector_score if vector_score > 0 else lexical_score
        return math.log1p(max(base, 0.0))

    while remaining and len(selected) < top_k:
        best_doc: Optional[Document] = None
        best_score = -float("inf")
        for doc in remaining:
            rel = _relevance(doc)
            div = 0.0
            if selected:
                div = max(_doc_similarity(doc, picked) for picked in selected)
            mmr_score = lambda_clamped * rel - (1.0 - lambda_clamped) * div
            if mmr_score > best_score:
                best_score = mmr_score
                best_doc = doc
        if best_doc is None:
            break
        selected.append(best_doc)
        remaining.remove(best_doc)

    if dbg is not None:
        dbg.setdefault("retrieval", {})["mmr"] = {
            "lambda": lambda_clamped,
            "pre": len(candidates),
            "post": len(selected),
        }
    return selected


def _perform_vector_search(question: str, vectorstore: FAISS, dbg: dict) -> List[Document]:
    """Performs vector search, including multi-query generation."""
    t_retrieval0 = _now()
    
    # 1. Generate query variants
    queries = [question]
    if MQ_ENABLED:
        queries = _gen_multi_queries(question, MQ_VARIANTS)
    dbg["mq_variants"] = list(queries)
    _log_debug(f"MQ variants: {queries}")

    # 2. Determine search budget per query
    reranker = _get_reranker()
    rerank_budget = RERANKER_CANDIDATES if reranker is not None else RETRIEVAL_FETCH_K
    total_budget = max(RETRIEVAL_FETCH_K, rerank_budget, RETRIEVAL_K)
    per_q = max(3, math.ceil(total_budget / max(1, len(queries))))

    # 3. Perform search for each query and collect candidates
    cands = []
    for query_variant in queries:
        try:
            hits_raw = vectorstore.similarity_search_with_score(query_variant, k=per_q)
            hits = [(doc, score) for doc, score in hits_raw]
            docs_only = [doc for doc, _ in hits]
            normalize_documents(docs_only)
            for doc, score in hits:
                try:
                    dist = float(score)
                except Exception:
                    dist = 0.0
                similarity = 1.0 / (1.0 + max(dist, 0.0))
                metadata = dict(getattr(doc, "metadata", {}) or {})
                metadata["vector_score"] = similarity
                metadata.setdefault("query_variants", []).append(query_variant)
                doc.metadata = metadata
                cands.append(doc)
        except Exception as e:
            print(f"[ERROR] Falha na busca vetorial para a query '{query_variant}': {e}")

    dbg["timing_ms"]["retrieval"] = _elapsed_ms(t_retrieval0)
    
    # 4. Deduplicate and return
    def _doc_key(d: Document) -> tuple:
        return (d.metadata.get("source", ""), d.metadata.get("chunk"), (d.page_content or "")[:64])

    unique_cands = _dedupe_preserve_order(cands, key=_doc_key)
    _apply_question_boosts(question, unique_cands)
    unique_cands.sort(
        key=lambda d: float((getattr(d, "metadata", {}) or {}).get("vector_score", 0.0)),
        reverse=True,
    )
    top_cands = unique_cands[:RETRIEVAL_FETCH_K]
    retrieval_dbg = dbg.setdefault("retrieval", {})
    retrieval_dbg["vector_top"] = [
        _pack_doc(doc, (getattr(doc, "metadata", {}) or {}).get("vector_score"))
        for doc in top_cands[:30]
    ]
    _log_debug(f"FAISS candidates from vector search: {len(top_cands)} (raw={len(unique_cands)})")
    return top_cands

def _merge_hybrid_results(vector_cands: List[Document], lex_hits: list, dbg: dict) -> List[Document]:
    """Merges lexical and vector search results for the hybrid pipeline."""
    if not HYBRID_ENABLED or not lex_hits or (ROUTE_FORCE == "vector"):
        dbg["route"] = "vector"
        return vector_cands

    lex_docs = [d for d, _, _ in lex_hits]

    if not RRF_ENABLED:
        dbg["route"] = "hybrid"
        _log_debug("Route: hybrid (concat)")
        all_merged = _dedupe_preserve_order(vector_cands + lex_docs, key=_doc_identity)
    else:
        dbg["route"] = "hybrid_rrf"
        _log_debug("Route: hybrid (RRF)")
        scores: Dict[tuple, Dict[str, Any]] = {}

        for rank, doc in enumerate(vector_cands, start=1):
            key = _doc_identity(doc)
            entry = scores.setdefault(key, {"doc": doc, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["ranks"]["vector"] = rank

        for rank, (doc, _hits, _score) in enumerate(lex_hits, start=1):
            key = _doc_identity(doc)
            entry = scores.setdefault(key, {"doc": doc, "score": 0.0, "ranks": {}})
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["ranks"]["lexical"] = rank

        fused = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
        dbg.setdefault("hybrid", {})["rrf_scores"] = [
            {
                "source": (getattr(item["doc"], "metadata", {}) or {}).get("source"),
                "chunk": (getattr(item["doc"], "metadata", {}) or {}).get("chunk"),
                "score": round(item["score"], 6),
                "ranks": item["ranks"],
            }
            for item in fused[:10]
        ]

        all_merged = [item["doc"] for item in fused]

    by_src: Dict[str, int] = {}
    cands_merged: List[Document] = []
    for d in all_merged:
        src = (getattr(d, "metadata", {}) or {}).get("source", "")
        by_src.setdefault(src, 0)
        by_src[src] += 1
        if by_src[src] <= MAX_PER_SOURCE:
            cands_merged.append(d)

    _log_debug(f"Hybrid merged candidates: {len(cands_merged)}")
    return cands_merged

def _rerank_and_get_confidence(candidates: List[Document], question: str, dbg: dict) -> Tuple[List[Document], float]:
    """Applies reranker to candidates, returns top docs and final confidence score."""
    reranker = _get_reranker()
    dbg["rerank"]["enabled"] = bool(reranker)
    dbg["rerank"]["name"] = RERANKER_NAME if reranker else None
    dbg["rerank"]["top_k"] = RERANKER_TOP_K if reranker else None

    if not reranker or not candidates:
        dbg["rerank"]["enabled"] = False
        dbg["timing_ms"]["reranker"] = 0.0
        sorted_docs = sorted(
            candidates,
            key=lambda d: (getattr(d, "metadata", {}) or {}).get("vector_score", 0.0),
            reverse=True,
        )
        top_limit = RERANKER_TOP_K if RERANKER_TOP_K else min(6, len(sorted_docs))
        top_limit = max(top_limit, 1)
        fallback_docs = sorted_docs[:top_limit]
        conf = float((fallback_docs[0].metadata.get("vector_score", 0.0)) if fallback_docs else 0.0)
        dbg["chosen"]["confidence"] = conf
        return fallback_docs, conf

    # Rerank logic
    cand_dicts = _build_candidates_from_docs(candidates)
    docs_norm_sorted, scores_sorted = _apply_rerank(
        cand_dicts, top_k=RERANKER_TOP_K, user_query=question, dbg=dbg
    )

    # Reorder original Document objects based on reranked order
    order = [d.get("idx") for d in docs_norm_sorted if d.get("idx") is not None]
    order = [int(i) for i in order if isinstance(i, int)][:RERANKER_TOP_K]
    
    final_docs = [candidates[i] for i in order]
    final_scores = [scores_sorted[i] for i in range(len(order))]
    
    _log_debug(f"Rerank top: {len(final_docs)}")

    # Apply heuristics to boost contact-related documents when question asks for phone/contato
    final_docs, final_scores = _apply_contact_boost(question, final_docs, final_scores, dbg)

    # Calculate final confidence
    conf = 0.0
    if final_scores:
        max_s = float(max(final_scores))
        # Jina v2 scores are already in [0, 1]
        conf = max_s if (0.0 <= max_s <= 1.0) else (1.0 / (1.0 + math.exp(-max_s)))
    
    dbg["chosen"]["confidence"] = float(conf)
    print(f"[RETRIEVE] mq={MQ_ENABLED} variants={len(dbg['mq_variants'])} merged={len(candidates)} reranked={len(final_docs)} conf={conf:.3f}")
    
    return final_docs, conf

def _apply_contact_boost(question: str, docs: List[Document], scores: List[float], dbg: dict) -> Tuple[List[Document], List[float]]:
    """Boost rerank scores for documents that match contact intents (telephone, ramal, etc.)."""
    if not docs:
        return docs, scores

    question_norm = _strip_accents_lower(question)
    wants_contact = any(term in question_norm for term in ("telefone", "telefone?", "tel", "ramal", "contato", "numero", "número", "celular", "whatsapp"))
    if not wants_contact:
        return docs, scores

    raw_name_terms = {
        _strip_accents_lower(t)
        for t in _candidate_terms(question)
        if t.isalpha() and len(t) >= 4
    }
    generic_terms = {
        "computacao", "computação", "comp", "bcc", "bsi",
        "departamento", "departamentos", "curso", "cursos",
        "faculdade", "secao", "seção", "stgp", "stf", "dco",
        "telefone", "telefones", "contato", "contatos", "ramal",
        "qual", "onde", "email", "e-mail", "numero", "número"
    }

    name_terms: set[str] = set()
    for term in raw_name_terms:
        if term in generic_terms:
            continue
        matched = False
        canonical = CONTACT_ALIAS_MAP.get(term)
        if canonical:
            name_terms.add(canonical)
            matched = True
        else:
            for base_norm, alias_list in ALIASES.items():
                base_norm_s = _strip_accents_lower(base_norm)
                alias_norms = {_strip_accents_lower(a) for a in alias_list}
                variants = alias_norms | {base_norm_s}
                CONTACT_ALIAS_MAP.setdefault(base_norm_s, base_norm_s)
                for alias in alias_norms:
                    CONTACT_ALIAS_MAP.setdefault(alias, base_norm_s)
            if term in variants:
                name_terms.add(base_norm_s)
                matched = True
                break
        if not matched:
            name_terms.add(term)

    boosted: List[Tuple[float, float, float, Document]] = []
    aliases_for_names = {
        alias for alias, canon in CONTACT_ALIAS_MAP.items()
        if canon in name_terms
    }

    for doc, base_score in zip(docs, scores):
        text = getattr(doc, "page_content", "") or ""
        text_norm = _strip_accents_lower(text)
        contacts = _extract_contacts([text])
        doc_phones = contacts.get("phones", [])

        bonus = 0.0
        has_name = bool(name_terms) and (
            any(canon in text_norm for canon in name_terms) or
            any(alias in text_norm for alias in aliases_for_names)
        )
        has_phone = bool(doc_phones)
        source = (getattr(doc, "metadata", {}) or {}).get("source", "") or ""
        if has_name:
            bonus += 10.0
        else:
            bonus -= 10.0
        if has_phone:
            bonus += 6.0
        else:
            bonus -= 6.0
        if has_name and has_phone:
            bonus += 12.0
        if "docentes" in source and "dep_" in source:
            bonus += 4.0

        boosted.append((float(base_score) + bonus, float(base_score), bonus, doc))

    boosted.sort(key=lambda item: item[0], reverse=True)

    new_docs = [doc for _, _, _, doc in boosted]
    new_scores = [score for score, _, _, _ in boosted]

    if dbg is not None:
        rerank_dbg = dbg.setdefault("rerank", {})
        top_k = int(rerank_dbg.get("top_k") or len(new_docs))
        previews = []
        for score_new, score_base, bonus, doc in boosted[:top_k]:
            meta = getattr(doc, "metadata", {}) or {}
            preview = _norm_ws((doc.page_content or "")[:120])
            previews.append({
                "source": meta.get("source"),
                "chunk": meta.get("chunk"),
                "preview": preview,
                "score": score_new,
                "base_score": score_base,
                "bonus": bonus,
            })
        rerank_dbg["scored"] = previews

    return new_docs, new_scores

def _finalize_result(question: str, docs: List[Document], conf: float, dbg: dict) -> Dict[str, Any]:
    """Generates the final text answer with LLM and packages the full response dictionary."""
    metadata_answer = _build_metadata_answer(question, docs, dbg)
    citations_clean = _collect_citations_from_docs(docs, max_sources=MAX_SOURCES)
    if metadata_answer:
        answer_out = metadata_answer
        if STRUCTURED_ANSWER:
            answer_out = _format_answer_markdown(question, metadata_answer, citations_clean, conf)
        return {
            "answer": answer_out,
            "citations": citations_clean,
            "context_found": bool(docs),
            "confidence": conf,
        }

    # 1. Prepare context snippets para o LLM
    top_texts = [d.page_content for d in docs]
    sent_scores = _top_sentences(question, top_texts, top_n=3)

    if sent_scores and sent_scores[0][1] >= 0.45:
        context_snippets = [s for s, _ in sent_scores]
    else:
        body = _norm_ws(docs[0].page_content)
        if len(body) > 500:
            body = body[:500].rstrip() + "..."
        context_snippets = [body]

    # 2. Generate final answer text using LLM
    t_llm = _now()
    final_text = _llm_resposta_final(question, context_snippets, None, None, {"emails": [], "phones": []})
    dbg["timing_ms"]["llm"] = _elapsed_ms(t_llm)

    if not final_text:
        final_text = " ".join(context_snippets)

    # 3. Format answer and citations
    answer_out = final_text
    if STRUCTURED_ANSWER:
        answer_out = _format_answer_markdown(question, final_text, citations_clean, conf)

    # 4. Assemble final payload
    result = {
        "answer": answer_out,
        "citations": citations_clean,
        "context_found": bool(docs),
        "confidence": conf,
    }
    return result


def _people_candidates_from_hits(question: str, lex_hits: list) -> List[Dict[str, Any]]:
    """Aggregate possible person matches from lexical hits for disambiguation."""
    if not lex_hits:
        return []

    name_terms = [t for t in _candidate_terms(question) if t.isalpha()]
    primary_term = name_terms[0] if name_terms else ""

    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc, hits, score in lex_hits:
        doc_text = getattr(doc, "page_content", "") or ""
        snippets = [s for s, _ in hits[:3]]
        if doc_text and not snippets:
            snippets = [_norm_ws(doc_text[:320])]

        name = _extract_name(snippets, primary_term) if snippets else None
        if not name and doc_text:
            name = _extract_name([doc_text], primary_term) if primary_term else None

        block = _person_block(doc_text, name or primary_term)
        if block:
            snippets = [block]
        elif name:
            window = _context_window_around_name(doc_text, name, 100, 200)
            if window:
                snippets = [window]

        meta = getattr(doc, "metadata", {}) or {}
        dept = _guess_dept_from_source(meta.get("source"))

        contacts = _extract_contacts(snippets)
        if not name:
            if primary_term:
                continue
            if not contacts["phones"] and not contacts["emails"]:
                continue
        elif primary_term:
            tokens_norm = [_strip_accents_lower(tk) for tk in _tokenize_letters(name)]
            target = _strip_accents_lower(primary_term)
            if all(distance.Levenshtein.normalized_similarity(tn, target) < 0.9 for tn in tokens_norm):
                continue

        key_name = _strip_accents_lower(name or meta.get("source") or "")
        key = (key_name, dept or "")

        source_val = meta.get("url") or meta.get("source") or ""

        cand = buckets.get(key)
        if cand is None:
            buckets[key] = {
                "name": name,
                "dept": dept,
                "contacts": {
                    "phones": sorted(set(contacts.get("phones", []))),
                    "emails": sorted(set(contacts.get("emails", []))),
                },
                "docs": [doc],
                "snippets": [s for s in snippets if s],
                "score": score,
                "source": source_val,
            }
            continue

        # Merge with existing candidate
        cand["name"] = cand["name"] or name
        cand["dept"] = cand["dept"] or dept
        cand["score"] = max(cand["score"], score)
        cand["contacts"]["phones"] = sorted(set(cand["contacts"]["phones"] + contacts.get("phones", [])))
        cand["contacts"]["emails"] = sorted(set(cand["contacts"]["emails"] + contacts.get("emails", [])))
        cand["docs"].append(doc)
        for snip in snippets:
            if snip and snip not in cand["snippets"]:
                cand["snippets"].append(snip)

    candidates = list(buckets.values())
    candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    return candidates


def _build_retry_query(question: str, lex_hits: list) -> Optional[str]:
    """Gera uma consulta alternativa expandindo sinônimos e setores observados."""
    question_norm = _strip_accents_lower(question)
    expansions: List[str] = []
    seen: Set[str] = set()

    for term in _candidate_terms(question):
        key = _strip_accents_lower(term)
        for exp in SYNONYMS.get(key, []) or []:
            exp_norm = _strip_accents_lower(exp)
            if exp_norm and exp_norm not in seen and exp_norm not in question_norm:
                expansions.append(exp.strip())
                seen.add(exp_norm)

    for doc, hits, _score in lex_hits[:3]:
        meta = getattr(doc, "metadata", {}) or {}
        sector = meta.get("sector")
        if sector:
            sector_norm = _strip_accents_lower(sector)
            if sector_norm and sector_norm not in seen and sector_norm not in question_norm:
                expansions.append(str(sector).strip())
                seen.add(sector_norm)
        for sigla in meta.get("siglas") or []:
            sig_norm = _strip_accents_lower(sigla)
            if sig_norm and sig_norm not in seen and sig_norm not in question_norm:
                expansions.append(sigla.strip())
                seen.add(sig_norm)
        if hits:
            snippet = hits[0][0]
            for token in _tokenize_letters(snippet):
                token_norm = _strip_accents_lower(token)
                if token_norm in DEPARTMENT_TOKEN_TO_SLUG and token_norm not in seen and token_norm not in question_norm:
                    expansions.append(token.strip())
                    seen.add(token_norm)

    expansions = _dedupe_preserve_order([exp for exp in expansions if exp])
    if not expansions:
        return None
    addition = " ".join(expansions[:3])
    retry_query = f"{question} {addition}".strip()
    return retry_query if retry_query and retry_query != question else None


def _build_low_confidence_response(question: str, conf: float, lex_hits: list) -> Dict[str, Any]:
    """Creates a clarification response when confidence is insufficient."""
    candidates = _people_candidates_from_hits(question, lex_hits)
    low_notice = conf is None or (conf is not None and conf < RETRIEVAL_MIN_SCORE)

    if candidates:
        if len(candidates) == 1:
            cand = candidates[0]
            answer_text = _format_direct(cand["name"], cand["dept"], cand["contacts"], cand["snippets"])
            citations = _collect_citations_from_docs(cand["docs"])
            return {
                "answer": answer_text,
                "citations": citations,
                "context_found": True,
                "confidence": conf,
            }

        lines = []
        for idx, cand in enumerate(candidates[:3], start=1):
            segments = []
            if cand["name"]:
                segments.append(cand["name"])
            if cand["dept"]:
                segments.append(cand["dept"])
            phones = cand["contacts"].get("phones") or []
            emails = cand["contacts"].get("emails") or []
            if phones:
                segments.append("📞 " + ", ".join(phones))
            if emails:
                segments.append("✉️ " + ", ".join(emails))
            if not segments:
                segments.append(cand["source"] or "sem detalhes")
            lines.append(f"{idx}. {' — '.join(segments)}")
        clarification = (
            "Encontrei mais de uma pessoa compatível. Informe o departamento, curso ou nome completo, "
            "ou escolha uma das opções:\n" + "\n".join(lines)
        )
        if low_notice:
            clarification = "Não localizei trechos com confiança suficiente. " + clarification
        citations = _collect_citations_from_docs([doc for cand in candidates[:3] for doc in cand["docs"]])
        return {
            "answer": clarification,
            "citations": citations,
            "context_found": False,
            "confidence": conf,
            "needs_clarification": True,
        }

    options = []
    for doc, _hits, _score in lex_hits[:3]:
        src = (getattr(doc, "metadata", {}) or {}).get("source") or ""
        if src:
            label = os.path.splitext(os.path.basename(src))[0].replace("_", " ")
            if label and label not in options:
                options.append(label)

    clarification = _llm_pedir_info(question, options, prefer_attr="departamento") if options else "Poderia indicar o departamento, curso ou unidade relacionado?"
    if low_notice:
        clarification = "Não localizei contexto suficiente na base para responder. " + clarification

    return {
        "answer": clarification,
        "citations": [],
        "context_found": False,
        "confidence": conf,
        "needs_clarification": True,
    }

def _log_telemetry_event(question: str, dbg: dict, result: dict):
    """Logs the telemetry event for the request."""
    try:
        log_event(
            os.getenv("LOG_DIR", "./logs"),
            {
                "question": question,
                "route": dbg.get("route"),
                "confidence": dbg.get("chosen", {}).get("confidence"),
                "timing_ms": dbg.get("timing_ms", {}),
                "mq_variants": dbg.get("mq_variants", []),
                "faiss_top": (dbg.get("faiss", {}) or {}).get("candidates", [])[:10],
                "retrieval_vector": (dbg.get("retrieval", {}) or {}).get("vector_top", [])[:10],
                "retrieval_lexical": (dbg.get("retrieval", {}) or {}).get("lexical_top", [])[:10],
                "retrieval_pre": (dbg.get("retrieval", {}) or {}).get("pre_rerank", [])[:8],
                "retrieval_post": (dbg.get("retrieval", {}) or {}).get("post_rerank", [])[:8],
                "used_query": (dbg.get("retrieval", {}) or {}).get("used_query"),
                "ctx_docs": len(result.get("citations", [])),
                "reranker_preset": RERANKER_PRESET,
                "answer_len": len((result.get("answer") or "")),
            },
        )
    except Exception as e:
        print(f"[WARN] Failed to log telemetry event: {e}")


# ==== Pipeline Principal de Resposta (Refatorado) ====
def _run_retrieval_round(
    question: str,
    all_docs: List[Document],
    vectorstore: FAISS,
    dbg_ctx: dict,
    *,
    fetch_limit: int,
    top_k: int,
) -> Tuple[List[Document], float, List[Tuple[Document, List[Tuple[str, int]], float]]]:
    """Executes one retrieval + rerank cycle and fills the provided debug dictionary."""
    lex_hits = _perform_lexical_search(question, all_docs)
    retrieval_dbg = dbg_ctx.setdefault("retrieval", {})
    retrieval_dbg["lexical_top"] = [
        {
            "source": (getattr(doc, "metadata", {}) or {}).get("source"),
            "chunk": (getattr(doc, "metadata", {}) or {}).get("chunk"),
            "score": round(float(score), 4),
            "preview": hits[0][0][:120] if hits else "",
        }
        for doc, hits, score in lex_hits[:10]
    ]

    vector_cands = _perform_vector_search(question, vectorstore, dbg_ctx)
    merged_cands = _merge_hybrid_results(vector_cands, lex_hits, dbg_ctx)
    retrieval_dbg["hybrid_top"] = [
        _pack_doc(doc, (getattr(doc, "metadata", {}) or {}).get("vector_score"))
        for doc in merged_cands[:30]
    ]

    mmr_target = min(fetch_limit, len(merged_cands)) if fetch_limit > 0 else len(merged_cands)
    mmr_target = max(top_k, mmr_target)
    question_tokens = _tokenize_letters(question)
    lambda_dynamic = RETRIEVAL_MMR_LAMBDA
    if RETRIEVAL_MMR_LAMBDA_SHORT > 0 and len(question_tokens) <= 6:
        lambda_dynamic = min(RETRIEVAL_MMR_LAMBDA, RETRIEVAL_MMR_LAMBDA_SHORT)
    retrieval_dbg["mmr_lambda"] = lambda_dynamic
    mmr_cands = _apply_mmr_selection(merged_cands, mmr_target, lambda_dynamic, dbg_ctx)
    retrieval_dbg["pre_rerank"] = [
        _pack_doc(doc, (getattr(doc, "metadata", {}) or {}).get("vector_score"))
        for doc in mmr_cands[:30]
    ]

    dbg_ctx["faiss"]["k"] = len(mmr_cands)
    dbg_ctx["faiss"]["candidates"] = retrieval_dbg["pre_rerank"]

    final_docs, conf = _rerank_and_get_confidence(mmr_cands, question, dbg_ctx)
    retrieval_dbg["post_rerank"] = [
        _pack_doc(doc, (getattr(doc, "metadata", {}) or {}).get("vector_score"))
        for doc in final_docs[:top_k]
    ]

    return final_docs, conf, lex_hits


def answer_question(
    question: str,
    embeddings_model: HuggingFaceEmbeddings,
    vectorstore: FAISS,
    *,
    k: int | None = None,
    fetch_k: int | None = None,
    debug: bool = False,
    confidence_min: float | None = None,
) -> Dict[str, Any]:
    """
    Pipeline principal para responder a uma pergunta usando uma abordagem híbrida (busca lexical e vetorial).
    Esta função foi refatorada para orquestrar o fluxo de RAG através de funções auxiliares.
    """
    t0 = _now()
    q = _norm_ws(question)
    dbg = _initialize_debug_payload(q)
    threshold = confidence_min if confidence_min is not None else CONFIDENCE_MIN
    dbg["confidence_threshold"] = threshold
    top_k_effective = k if k is not None else RETRIEVAL_K
    fetch_limit = fetch_k if fetch_k is not None else RETRIEVAL_FETCH_K
    if RERANKER_TOP_K:
        top_k_effective = max(top_k_effective, RERANKER_TOP_K)
    fetch_limit = max(fetch_limit, top_k_effective)
    dbg.setdefault("config", {})["retrieval"] = {
        "k": top_k_effective,
        "fetch_k": fetch_limit,
        "mmr_lambda": RETRIEVAL_MMR_LAMBDA,
        "mmr_lambda_short": RETRIEVAL_MMR_LAMBDA_SHORT,
        "min_score_retry": RETRIEVAL_MIN_SCORE,
    }

    if not q:
        return {"answer": "Não entendi a pergunta. Pode reformular?", "citations": [], "context_found": False}

    _ensure_contact_index(vectorstore)
    contact_match = _contact_lookup(q)
    if contact_match is not None:
        entry = contact_match["entry"]
        answer_text = _format_contact_answer(entry, q)
        citations = _unique_citations(entry.get("citations", []))[:MAX_SOURCES]
        result = {
            "answer": answer_text,
            "citations": citations,
            "context_found": True,
            "confidence": 0.99,
        }
        dbg["route"] = "contact_fallback"
        dbg["chosen"]["confidence"] = 0.99
        dbg["contact_lookup"] = {
            "matched_token": contact_match["matched_token"],
            "canonical": contact_match["canonical"],
            "dept_hint": contact_match["dept_hint"],
            "phones": entry.get("phones", []),
            "emails": entry.get("emails", []),
        }
        dbg["timing_ms"]["total"] = _elapsed_ms(t0)
        if debug and DEBUG_PAYLOAD:
            result["debug"] = dbg
        _log_telemetry_event(q, dbg, result)
        return result

    # 1. Carregar todos os documentos da base (para busca lexical)
    try:
        all_docs: List[Document] = list(getattr(vectorstore.docstore, "_dict", {}).values())
    except Exception:
        all_docs = []
    all_docs = normalize_documents(all_docs)

    # 2. Realizar busca lexical inicial
    lex_hits = _perform_lexical_search(q, all_docs)

    # 3. Tentar a rota "somente lexical" (se aplicável)
    # Esta rota é um atalho para perguntas muito específicas (ex: nomes) e retorna a resposta diretamente.
    lexical_result = _handle_lexical_route(q, lex_hits, dbg)
    if lexical_result is not None:
        # Finaliza e retorna o resultado da rota lexical
        dbg["timing_ms"]["total"] = _elapsed_ms(t0)
        if debug and DEBUG_PAYLOAD:
            lexical_result["debug"] = dbg
        _log_telemetry_event(q, dbg, lexical_result)
        return lexical_result

    # --- Início da Rota Vetorial / Híbrida ---

    final_docs, conf, lex_hits = _run_retrieval_round(
        q,
        all_docs,
        vectorstore,
        dbg,
        fetch_limit=fetch_limit,
        top_k=top_k_effective,
    )
    used_query = q

    if conf < RETRIEVAL_MIN_SCORE:
        retry_meta = dbg.setdefault("retry", {})
        retry_meta["previous_confidence"] = conf
        retry_query = _build_retry_query(q, lex_hits)
        if retry_query and retry_query != q:
            retry_dbg = _initialize_debug_payload(retry_query)
            retry_docs, retry_conf, retry_lex_hits = _run_retrieval_round(
                retry_query,
                all_docs,
                vectorstore,
                retry_dbg,
                fetch_limit=fetch_limit,
                top_k=top_k_effective,
            )
            retry_meta["variant"] = retry_query
            retry_meta["confidence"] = retry_conf
            retry_meta["details"] = retry_dbg
            if retry_conf > conf:
                retry_meta["adopted"] = True
                retry_meta["first_confidence"] = conf
                retry_meta["first_retrieval"] = dbg.get("retrieval")
                retry_meta["first_faiss"] = dbg.get("faiss")
                retry_meta["first_rerank"] = dbg.get("rerank")
                final_docs = retry_docs
                conf = retry_conf
                lex_hits = retry_lex_hits
                used_query = retry_query
                dbg["retrieval"] = retry_dbg.get("retrieval", {})
                dbg["faiss"] = retry_dbg.get("faiss", dbg.get("faiss", {}))
                dbg["rerank"] = retry_dbg.get("rerank", dbg.get("rerank", {}))
                dbg["chosen"] = retry_dbg.get("chosen", dbg.get("chosen", {}))
                dbg["mq_variants"] = retry_dbg.get("mq_variants", dbg.get("mq_variants", []))
                dbg["timing_ms"].update(retry_dbg.get("timing_ms", {}))
            else:
                retry_meta["adopted"] = False
        else:
            retry_meta["skipped"] = True

    dbg.setdefault("retrieval", {})["used_query"] = used_query

    # 7. Verificar se a confiança é suficiente para responder
    if (not final_docs) or (conf < threshold):
        if REQUIRE_CONTEXT:
            result = _build_low_confidence_response(q, conf, lex_hits)
            dbg["timing_ms"]["total"] = _elapsed_ms(t0)
            if debug and DEBUG_PAYLOAD:
                result["debug"] = dbg
            _log_telemetry_event(q, dbg, result)
            return result
    
    # 8. Gerar a resposta final com LLM e formatar o resultado
    result = _finalize_result(q, final_docs, conf, dbg)

    # 9. Finalizar, adicionar debug info e logar telemetria
    dbg["timing_ms"]["total"] = _elapsed_ms(t0)
    if debug and DEBUG_PAYLOAD:
        result["debug"] = dbg
    
    _log_telemetry_event(q, dbg, result)
    
    _log_debug(f"Timing(ms): {dbg['timing_ms']}")
    return result
