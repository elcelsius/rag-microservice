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
        "name": "jinaai/jina-reranker-v1-base-multilingual",
        "candidates": 36,
        "top_k": 6,
        "max_len": 512,
        "device": "cpu",
        "trust_remote_code": True,
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

# ==== Configurações de Multi-Query e Confiança ====
# Ativa ou desativa a geração de múltiplas variações da pergunta do usuário para busca.
MQ_ENABLED = os.getenv("MQ_ENABLED", "true").lower() == "true"
# Número de variações da pergunta a serem geradas.
MQ_VARIANTS = int(os.getenv("MQ_VARIANTS", "3"))
# Limiar mínimo de confiança do reranker para considerar uma resposta válida.
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.32"))
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
        return data
    except FileNotFoundError:
        print(f"[DICT] Arquivo não encontrado: {path} — usando defaults vazios.")
        return {"departments": {}, "aliases": {}, "synonyms": {}, "boosts": {}}


# Carrega os termos na inicialização do módulo.
TERMS = load_terms()
DEPARTMENTS = TERMS["departments"]  # dict slug -> label canônica
ALIASES = TERMS["aliases"]  # dict termo -> [variações]
SYNONYMS = TERMS["synonyms"]  # dict sigla -> [expansões]
CUSTOM_MQ_EXPANSIONS = {
    'reserva': [
        'reserva anfiteatro stpg',
        'email stpg reserva anfiteatro',
        'solicitar sala pos graduacao stpg'
    ],
    'anfiteatro': [
        'anfiteatro pos graduacao stpg',
        'reserva sala anfiteatro faculdade de ciencias',
        'agendar anfiteatro fc bauru'
    ],
    'impressao': [
        'impressao ldc passo a passo',
        'comprar cotas impressao ldc',
        'envio arquivo impressao laboratorio didatico computacional'
    ],
    'cota': [
        'cotas impressao pagamento pix',
        'valor impressao frente verso ldc',
        'tabela preco impressao faculdade ciencias'
    ],
    'ldc': [
        'laboratorio didatico computacional agenda',
        'agendar uso ldc lepec',
        'horario funcionamento lepec'
    ],
    'contato': [
        'telefone secretaria faculdade ciencias',
        'email secretaria apoio administrativo',
        'ramal suporte ldc'
    ]
}
BOOSTS = TERMS["boosts"]  # dict de boosts opcionais
print(f"[DICT] departamentos={len(DEPARTMENTS)} aliases={len(ALIASES)} synonyms={len(SYNONYMS)}")

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


def _gen_multi_queries(user_query: str, n: int, llm=None) -> List[str]:
    """
    Gera variações para a busca:
    1) variações locais simples
    2) expansões vindas de SYNONYMS (do YAML)
    3) (opcional) variações via LLM, se um wrapper for passado em `llm`
    """
    n = max(1, n)

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
    merged = _dedupe_preserve_order([user_query] + base_local + syn_vars + custom_vars + llm_vars)
    return merged[:n]


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

def _perform_lexical_search(question: str, all_docs: List[Document]) -> List[Tuple[Document, List[Tuple[str, int]], int]]:
    """Performs a lexical search over all documents based on terms from the question."""
    dept_hints = _dept_hints_in_question(question)
    terms = _candidate_terms(question)
    if not terms or not all_docs:
        return []

    lex_hits = []
    for d in all_docs:
        src = (d.metadata.get("source") or "").lower()
        score_bonus = DEPT_BONUS if any(_strip_accents_lower(h) in src for h in dept_hints) else 0
        hits = _sentence_hits_by_name(d.page_content or "", terms)
        if hits:
            best_score = max(sc for _, sc in hits) + score_bonus
            lex_hits.append((d, hits, best_score))
    
    lex_hits.sort(key=lambda x: x[2], reverse=True)
    return lex_hits[:6]

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
    reranker = _get_reranker() # Check if reranker is available to set budget
    k_candidates_total = RERANKER_CANDIDATES if reranker is not None else 10
    per_q = max(3, math.ceil(k_candidates_total / max(1, len(queries))))

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
                doc.metadata = metadata
                cands.append(doc)
        except Exception as e:
            print(f"[ERROR] Falha na busca vetorial para a query '{query_variant}': {e}")

    dbg["timing_ms"]["retrieval"] = _elapsed_ms(t_retrieval0)
    
    # 4. Deduplicate and return
    def _doc_key(d: Document) -> tuple:
        return (d.metadata.get("source", ""), d.metadata.get("chunk"), d.page_content[:64])
    
    unique_cands = _dedupe_preserve_order(cands, key=_doc_key)
    _log_debug(f"FAISS candidates from vector search: {len(unique_cands)}")
    return unique_cands

def _merge_hybrid_results(vector_cands: List[Document], lex_hits: list, dbg: dict) -> List[Document]:
    """Merges lexical and vector search results for the hybrid pipeline."""
    if not HYBRID_ENABLED or not lex_hits or (ROUTE_FORCE == "vector"):
        dbg["route"] = "vector"
        return vector_cands

    dbg["route"] = "hybrid"
    _log_debug("Route: hybrid")
    
    lex_docs = [d for d, _, _ in lex_hits]
    
    def _doc_key(d: Document) -> tuple:
        return (d.metadata.get("source", ""), d.metadata.get("chunk"), d.page_content[:64])

    # Combine and deduplicate, preserving order (vector results first)
    all_merged = _dedupe_preserve_order(vector_cands + lex_docs, key=_doc_key)

    # Enforce diversity by limiting docs per source
    by_src = {}
    cands_merged = []
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

    # Calculate final confidence
    conf = 0.0
    if final_scores:
        max_s = float(max(final_scores))
        # Jina v2 scores are already in [0, 1]
        conf = max_s if (0.0 <= max_s <= 1.0) else (1.0 / (1.0 + math.exp(-max_s)))
    
    dbg["chosen"]["confidence"] = float(conf)
    print(f"[RETRIEVE] mq={MQ_ENABLED} variants={len(dbg['mq_variants'])} merged={len(candidates)} reranked={len(final_docs)} conf={conf:.3f}")
    
    return final_docs, conf

def _finalize_result(question: str, docs: List[Document], conf: float, dbg: dict) -> Dict[str, Any]:
    """Generates the final text answer with LLM and packages the full response dictionary."""
    # 1. Prepare context snippets for the LLM
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
    citations_clean = _collect_citations_from_docs(docs, max_sources=MAX_SOURCES)
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


def _build_low_confidence_response(question: str, conf: float, lex_hits: list) -> Dict[str, Any]:
    """Creates a clarification response when confidence is insufficient."""
    candidates = _people_candidates_from_hits(question, lex_hits)

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
                "faiss_top": (dbg.get("faiss", {}) or {}).get("candidates", [])[:5],
                "ctx_docs": len(result.get("citations", [])),
                "reranker_preset": RERANKER_PRESET,
                "answer_len": len((result.get("answer") or "")),
            },
        )
    except Exception as e:
        print(f"[WARN] Failed to log telemetry event: {e}")


# ==== Pipeline Principal de Resposta (Refatorado) ====
def answer_question(question: str, embeddings_model: HuggingFaceEmbeddings, vectorstore: FAISS, *, k: int = 5,
                    fetch_k: int = 20, debug: bool = False) -> Dict[str, Any]:
    """
    Pipeline principal para responder a uma pergunta usando uma abordagem híbrida (busca lexical e vetorial).
    Esta função foi refatorada para orquestrar o fluxo de RAG através de funções auxiliares.
    """
    t0 = _now()
    q = _norm_ws(question)
    dbg = _initialize_debug_payload(q)

    if not q:
        return {"answer": "Não entendi a pergunta. Pode reformular?", "citations": [], "context_found": False}

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

    # 4. Realizar busca vetorial (com multi-query)
    vector_cands = _perform_vector_search(q, vectorstore, dbg)

    # 5. Mesclar resultados lexicais e vetoriais (se rota híbrida estiver ativa)
    merged_cands = _merge_hybrid_results(vector_cands, lex_hits, dbg)
    
    # Preencher snapshot de candidatos para debug
    dbg["faiss"]["k"] = len(merged_cands)
    dbg["faiss"]["candidates"] = [_pack_doc(d, (getattr(d, "metadata", {}) or {}).get("vector_score")) for d in merged_cands]

    # 6. Reordenar candidatos com Reranker e calcular confiança
    final_docs, conf = _rerank_and_get_confidence(merged_cands, q, dbg)

    # 7. Verificar se a confiança é suficiente para responder
    if (not final_docs) or (conf < CONFIDENCE_MIN):
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
