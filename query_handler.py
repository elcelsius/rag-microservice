# query_handler.py
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

from rapidfuzz import fuzz, distance
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# =======================
# Regras e utilidades
# =======================
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\;\:])\s+|\n+")
_WS = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d{4,5}-\d{4}")

STOP = {
    "que","qual","quais","sobre","para","como","onde","quem",
    "contato","email","e-mail","fone","telefone","ramal",
    "da","de","do","um","uma","o","a","os","as"
}

DEPT_LABELS = [
    ("computacao", "Computação"),
    ("biologia", "Biologia"),
    ("stpg", "STPG"),
    ("staepe", "STAEPE"),
    ("dti", "DTI"),
    ("administracao", "Administração"),
    ("congregacao", "Congregação"),
    ("diretoria", "Diretoria Técnica Acadêmica"),
]

def _norm_ws(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def _strip_accents_lower(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s or "")
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()

def _guess_dept_from_source(src: str) -> str | None:
    s = _strip_accents_lower(src or "")
    for key, label in DEPT_LABELS:
        if key in s:
            return label
    return None

# =======================
# Termos e variantes
# =======================
def _candidate_terms(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return []
    emails = EMAIL_RE.findall(q)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", q)
    terms, seen = [], set()
    for w in emails + words:
        k = _strip_accents_lower(w)
        if k not in STOP and k not in seen:
            terms.append(w)
            seen.add(k)
    return terms

def _term_variants(term: str) -> List[str]:
    """Gera variantes simples p/ nomes: Andreia ~ Andrea; Joao ~ João etc."""
    t = _strip_accents_lower(term)
    vs = {t}
    # Andreia -> Andrea (remove 'i' pós 'e')
    if t.endswith("eia"):
        vs.add(t[:-3] + "ea")
    # normalizações mínimas
    vs.add(t.replace("ç", "c").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u"))
    return list(vs)

# =======================
# Matching de nomes (token)
# =======================
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

def _tokenize_letters(text: str) -> List[str]:
    return WORD_RE.findall(text or "")

def _is_name_like(token: str) -> bool:
    # heurística: só letras e >= 5 chars
    return token.isalpha() and len(token) >= 5

def _name_token_match(token_norm: str, term_norm: str) -> bool:
    """
    Aproximação para NOMES:
      - comprimentos parecidos (diferença <= 1)
      - alta similaridade (>= 0.86)
      -> evita 'andre' quando pedem 'andreia' (diferença de comprimento = 2)
    """
    if abs(len(token_norm) - len(term_norm)) > 1:
        return False
    sim = distance.Levenshtein.normalized_similarity(token_norm, term_norm)  # 0..1
    return sim >= 0.86

def _sentence_hits(text: str, terms: List[str]) -> List[Tuple[str, int]]:
    """
    Marca sentenças/linhas que contenham um TOKEN parecido com o termo (regra de nome),
    ou, se não for nome, usa fuzzy por frase. Retorna [(sentenca,score)].
    """
    out: List[Tuple[str, int]] = []
    if not text or not terms:
        return out

    sentences = [s for s in _SENT_SPLIT.split(text) if _norm_ws(s)]
    terms_norm_lists = [_term_variants(t) for t in terms]
    # flattens + dedup
    all_norms = list({tn for lst in terms_norm_lists for tn in lst})

    for sent in sentences:
        s_clean = _norm_ws(sent)
        s_norm = _strip_accents_lower(s_clean)
        best = 0
        # 1) tente por tokens (nomes)
        tokens = _tokenize_letters(s_clean)
        tokens_norm = [_strip_accents_lower(tk) for tk in tokens]
        hit = False
        for tn in all_norms:
            for tk in tokens_norm:
                if _is_name_like(tk) and _name_token_match(tk, tn):
                    # score baseado na similaridade de nome
                    sc = int(100 * distance.Levenshtein.normalized_similarity(tk, tn))
                    best = max(best, sc)
                    hit = True
        # 2) fallback por frase (se ainda não bateu como nome)
        if not hit:
            for tn in all_norms:
                sc = fuzz.partial_ratio(tn, s_norm)
                best = max(best, sc)

        if best >= 86:  # limiar mais alto p/ evitar André vs Andreia
            out.append((s_clean, best))

    # ordena e dedup
    out.sort(key=lambda x: x[1], reverse=True)
    seen, uniq = set(), []
    for s, sc in out:
        k = s.lower()
        if k not in seen:
            uniq.append((s, sc))
            seen.add(k)
    return uniq[:3]

# =======================
# Extração de contatos e nome
# =======================
def _extract_contacts(texts: List[str]) -> Dict[str, List[str]]:
    emails, phones = set(), set()
    for t in texts:
        for e in EMAIL_RE.findall(t or ""):
            emails.add(e)
        for p in PHONE_RE.findall(t or ""):
            phones.add(p)
    return {"emails": sorted(emails), "phones": sorted(phones)}

NAME_SEQ_RE = re.compile(r"([A-ZÁ-Ü][a-zá-ü]+(?:\s+[A-ZÁ-Ü][a-zá-ü]+){1,5})")

def _extract_name(snippets: List[str], term: str) -> str | None:
    """Tenta extrair um nome próprio que contenha o termo (sem acento) nos snippets."""
    tnorm = _strip_accents_lower(term or "")
    for s in snippets:
        cands = NAME_SEQ_RE.findall(s or "")
        if not cands:
            continue
        # tente o candidato que contém o termo aproximado
        best = None
        best_sim = 0.0
        for cand in cands:
            # escolha o token do cand mais parecido com o termo
            for tk in _tokenize_letters(cand):
                sim = distance.Levenshtein.normalized_similarity(_strip_accents_lower(tk), tnorm)
                if sim > best_sim:
                    best_sim = sim
                    best = cand
        if best and best_sim >= 0.82:
            return best
    # fallback: primeiro candidato do primeiro snippet
    for s in snippets:
        cands = NAME_SEQ_RE.findall(s or "")
        if cands:
            return cands[0]
    return None

def _as_citation(doc: Document) -> Dict[str, Any]:
    src = doc.metadata.get("source") or "desconhecido"
    chunk = doc.metadata.get("chunk")
    preview = _norm_ws(doc.page_content)[:180]
    return {
        "source": src,
        "chunk": int(chunk) if (isinstance(chunk, int) or (isinstance(chunk, str) and str(chunk).isdigit())) else None,
        "preview": preview,
    }

# =======================
# Fallback por embeddings
# =======================
def _top_sentences(question: str, texts: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
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
    scored: List[Tuple[str, float]] = [(s, fuzz.token_set_ratio(q, s) / 100.0) for s in uniq]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

# =======================
# Pipeline principal
# =======================
def answer_question(
    question: str,
    embeddings_model: HuggingFaceEmbeddings,
    vectorstore: FAISS,
    *,
    k: int = 5,
    fetch_k: int = 20,
) -> Dict[str, Any]:
    q = _norm_ws(question)
    if not q:
        return {"answer": "Não entendi a pergunta. Pode reformular?", "citations": [], "context_found": False}

    # ---- 1) Busca lexical por sentenças (ótimo p/ nomes) ----
    terms = _candidate_terms(q)
    try:
        all_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
    except Exception:
        all_docs = []

    lex_results: List[Tuple[Document, List[Tuple[str,int]], int]] = []
    if terms and all_docs:
        for d in all_docs:
            hits = _sentence_hits(d.page_content or "", terms)
            if hits:
                best = max(sc for _, sc in hits)
                lex_results.append((d, hits, best))
        lex_results.sort(key=lambda x: x[2], reverse=True)
        lex_results = lex_results[:5]

    if lex_results:
        snippets, citations = [], []
        for doc, hits, _best in lex_results:
            for s, _ in hits[:2]:
                snippets.append(s)
            citations.append(_as_citation(doc))
            if len(snippets) >= 6:
                break

        info = _extract_contacts(snippets)

        # tenta achar um nome alinhado ao termo principal (se houver só um)
        name = None
        name_terms = [t for t in terms if t.isalpha()]
        if name_terms:
            name = _extract_name(snippets, name_terms[0])

        # departamento via nome do arquivo
        dept = None
        if citations:
            dept = _guess_dept_from_source(citations[0]["source"])

        # monta resposta curta e direta
        parts = []
        if name:
            parts.append(name)
        if dept:
            parts.append(dept)
        if info["phones"]:
            parts.append(f"📞 {', '.join(info['phones'])}")
        if info["emails"]:
            parts.append(f"✉️ {', '.join(info['emails'])}")

        answer = " — ".join(parts) if parts else " ".join(snippets[:2]).strip()
        if not answer:
            answer = "Encontrei referências relacionadas, mas não localizei um trecho textual claro."

        return {"answer": answer, "citations": citations[:3], "context_found": True}

    # ---- 2) Fallback: embeddings (MMR) ----
    try:
        docs: List[Document] = vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=fetch_k)
    except Exception:
        docs = vectorstore.similarity_search(q, k=k)

    if not docs:
        return {"answer": "Não encontrei nada relacionado nos documentos indexados.", "citations": [], "context_found": False}

    top_texts = [d.page_content for d in docs]
    sent_scores = _top_sentences(q, top_texts, top_n=3)

    if sent_scores and sent_scores[0][1] >= 0.45:
        answer = " ".join([s for s, _ in sent_scores][:3])
    else:
        answer = _norm_ws(docs[0].page_content)
        if len(answer) > 500:
            answer = answer[:500].rstrip() + "..."

    citations = [_as_citation(d) for d in docs[:3]]
    return {"answer": answer, "citations": citations, "context_found": True}
