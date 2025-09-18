# query_handler.py
# Este módulo é responsável por processar as consultas dos usuários, realizar a busca no vetorstore FAISS,
# extrair informações relevantes e formatar a resposta final usando um LLM.

from __future__ import annotations
import re, unicodedata
from typing import Any, Dict, List, Tuple, Optional

from rapidfuzz import fuzz, distance
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from llm_client import load_prompt, call_llm

# ==== Utilitários de Processamento de Texto ====
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\;\:])\s+|\n+") # Regex para dividir texto em sentenças
_WS = re.compile(r"\s+") # Regex para normalizar espaços em branco
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE) # Regex para encontrar e-mails
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d{4,5}-\d{4}") # Regex para encontrar números de telefone
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+") # Regex para encontrar palavras (letras acentuadas incluídas)

# Palavras de parada (stopwords) comuns em português, usadas para filtrar termos de busca.
STOP = {
    "que","qual","quais","sobre","para","como","onde","quem",
    "contato","email","e-mail","fone","telefone","ramal",
    "da","de","do","um","uma","o","a","os","as"
}

# Rótulos de departamentos/unidades e suas chaves normalizadas para busca.
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

# Aliases para nomes comuns, usados para normalização e busca flexível.
ALIASES = {
    "andreia": ["andrea", "andria", "andréia", "andréa", "andreá"],
    "andréa": ["andreia", "andrea", "andréia"],
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

def _guess_dept_from_source(src: str) -> Optional[str]:
    """Tenta adivinhar o departamento a partir do texto da fonte do documento."""
    s = _strip_accents_lower(src or "")
    for key, label in DEPT_LABELS:
        if key in s:
            return label
    return None

def _dept_hints_in_question(q: str) -> List[str]:
    """Identifica possíveis departamentos mencionados na pergunta do usuário."""
    s = _strip_accents_lower(q)
    hints = []
    for key, label in DEPT_LABELS:
        if key in s and label not in hints:
            hints.append(label)
    return hints

# ==== Funções de Extração de Termos ====
def _candidate_terms(question: str) -> List[str]:
    """Extrai termos candidatos da pergunta, incluindo e-mails e palavras com 3+ caracteres, excluindo stopwords."""
    q = (question or "").strip()
    if not q:
        return []
    emails = EMAIL_RE.findall(q)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", q) # Encontra palavras com 3 ou mais caracteres
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
    # Adiciona aliases de nomes comuns
    for base, al in ALIASES.items():
        if t == base or t in al:
            vs.update([base] + al)
    # Lida com variações de "eia" vs "ea"
    if t.endswith("eia"):
        vs.add(t[:-3] + "ea")
    if t.endswith("ea"):
        vs.add(t[:-2] + "eia")
    # Normalizações simples de caracteres
    vs.add(t.replace("ç","c").replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u"))
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
                    best = max(best, sc); hit = True
        if not hit:
            for tn in all_norms:
                sc = fuzz.partial_ratio(tn, s_norm)
                best = max(best, sc)
        if best >= 86:
            out.append((s_clean, best))

    out.sort(key=lambda x: x[1], reverse=True)
    seen, uniq = set(), []
    for s, sc in out:
        k = s.lower()
        if k not in seen:
            uniq.append((s, sc)); seen.add(k)
    return uniq[:3]

NAME_SEQ_RE = re.compile(r"([A-ZÁ-Ü][a-zá-ü]+(?:\s+[A-ZÁ-Ü][a-zá-ü]+){1,5})") # Regex para sequências de nomes

def _normalize_phone(p: str) -> str:
    """Normaliza um número de telefone para um formato padrão."""
    digits = re.sub(r"\D", "", p or "")
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
    tnorm = _strip_accents_lower(term or "")
    for s in snippets:
        cands = NAME_SEQ_RE.findall(s or "")
        if not cands: continue
        best, best_sim = None, 0.0
        for cand in cands:
            for tk in _tokenize_letters(cand):
                sim = distance.Levenshtein.normalized_similarity(_strip_accents_lower(tk), tnorm)
                if sim > best_sim:
                    best_sim = sim; best = cand
        if best and best_sim >= 0.82:
            return best
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
            if s: sentences.append(s)
    seen, uniq = set(), []
    for s in sentences:
        k = s.lower()
        if k not in seen:
            uniq.append(s); seen.add(k)
    q = _norm_ws(question)
    scored = [(s, fuzz.token_set_ratio(q, s) / 100.0) for s in uniq]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def _context_window_around_name(doc_text: str, person_name: str, chars_before: int = 220, chars_after: int = 300) -> Optional[str]:
    """Extrai uma janela de contexto de texto ao redor de um nome de pessoa."""
    if not (doc_text and person_name):
        return None
    txt = doc_text; name = person_name.strip()
    idx = txt.find(name)
    if idx < 0:
        compact_name = _norm_ws(name)
        idx = _norm_ws(txt).find(compact_name)
        if idx < 0: return None
    start = max(0, idx - chars_before)
    end = min(len(txt), idx + len(name) + chars_after)
    return _norm_ws(txt[start:end])

# ==== Funções de Interação com LLM ====
# Carrega os prompts para triagem, pedido de informação e resposta final.
_TRIAGE_PROMPT   = load_prompt("prompts/triagem_prompt.txt")
_PEDIR_INFO_PROMPT = load_prompt("prompts/pedir_info_prompt.txt")
_RESPOSTA_PROMPT = load_prompt("prompts/resposta_final_prompt.txt")

def _llm_triage(question: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    """Usa o LLM para fazer a triagem da pergunta, decidindo se deve responder ou pedir mais informações.
    
    Args:
        question (str): A pergunta original do usuário.
        signals (Dict[str, Any]): Sinais extraídos da pergunta para auxiliar na triagem.
        
    Returns:
        Dict[str, Any]: Um dicionário contendo a ação decidida ("AUTO_RESOLVER" ou "PEDIR_INFO") e uma possível pergunta de esclarecimento.
    """
    if not _TRIAGE_PROMPT:
        return {"action": "AUTO_RESOLVER", "ask": ""}
    user_payload = f"Pergunta: {question}\n\nSinais:\n{signals}"
    _, data = call_llm(_TRIAGE_PROMPT, user_payload, expect_json=True, max_tokens=150)
    if isinstance(data, dict) and "action" in data:
        return {"action": data.get("action"), "ask": data.get("ask", "")}
    return {"action": "AUTO_RESOLVER", "ask": ""}

def _llm_pedir_info(question: str, options: List[str], prefer_attr: str = "departamento") -> str:
    """Usa o LLM para gerar uma pergunta de esclarecimento ao usuário.
    
    Args:
        question (str): A pergunta original do usuário.
        options (List[str]): Opções de termos encontrados que podem precisar de esclarecimento.
        prefer_attr (str): Atributo preferencial para o qual o esclarecimento é solicitado (ex: "departamento").
        
    Returns:
        str: A pergunta de esclarecimento gerada pelo LLM.
    """
    if not _PEDIR_INFO_PROMPT:
        if options:
            opts = ", ".join(options[:3])
            return f"Encontrei múltiplas possibilidades: {opts}. Pode dizer qual delas?"
        return "Pode informar o departamento/unidade para eu localizar a pessoa correta?"
    payload = f"Pergunta original: {question}\nOpções (máx 3): {options[:3]}\nAtributo preferencial: {prefer_attr}"
    text, _ = call_llm(_PEDIR_INFO_PROMPT, payload, expect_json=False, max_tokens=120)
    return (text or "").strip() or "Pode informar o departamento/unidade?"

def _format_direct(name: Optional[str], dept: Optional[str], contacts: Dict[str, List[str]], context_snippets: List[str]) -> str:
    """Formata uma resposta direta com as informações encontradas, sem usar o LLM.
    
    Args:
        name (Optional[str]): Nome da pessoa encontrada.
        dept (Optional[str]): Departamento encontrado.
        contacts (Dict[str, List[str]]): Contatos (e-mails, telefones) encontrados.
        context_snippets (List[str]): Trechos de contexto relevantes.
        
    Returns:
        str: A resposta formatada.
    """
    parts = []
    if name: parts.append(name)
    if dept: parts.append(dept)
    if contacts.get("phones"): parts.append("📞 " + ", ".join(contacts["phones"]))
    if contacts.get("emails"): parts.append("✉️ " + ", ".join(contacts["emails"]))
    if parts: return " — ".join(parts)
    return (context_snippets[0] if context_snippets else "Não encontrei.")[:400]

def _llm_resposta_final(question: str, context_snippets: List[str], name: Optional[str], dept: Optional[str], contacts: Dict[str, List[str]]) -> str:
    """Usa o LLM para gerar uma resposta final em linguagem natural, com base no contexto e metadados extraídos.
    
    Args:
        question (str): A pergunta original do usuário.
        context_snippets (List[str]): Trechos de contexto relevantes.
        name (Optional[str]): Nome da pessoa encontrada.
        dept (Optional[str]): Departamento encontrado.
        contacts (Dict[str, List[str]]): Contatos (e-mails, telefones) encontrados.
        
    Returns:
        str: A resposta final gerada pelo LLM.
    """
    if not _RESPOSTA_PROMPT:
        return _format_direct(name, dept, contacts, context_snippets)
    ctx = "\n\n".join(f"- {s}" for s in context_snippets[:6])
    meta = {"nome": name or "", "departamento": dept or "", "telefones": contacts.get("phones", []), "emails": contacts.get("emails", [])}
    user = f"Pergunta: {question}\n\nContexto (trechos):\n{ctx}\n\nMetadados extraídos:\n{meta}\n\nFormato: Nome — Departamento — 📞 — ✉️ (curto)."
    text, _ = call_llm(_RESPOSTA_PROMPT, user, expect_json=False, max_tokens=220)
    if not (text or "").strip():
        return _format_direct(name, dept, contacts, context_snippets)
    return text.strip()

# ==== Pipeline Principal de Resposta ====
def answer_question(question: str, embeddings_model: HuggingFaceEmbeddings, vectorstore: FAISS, *, k: int = 5, fetch_k: int = 20) -> Dict[str, Any]:
    """Pipeline principal para responder a uma pergunta usando uma abordagem híbrida (busca lexical e vetorial).
    
    Args:
        question (str): A pergunta do usuário.
        embeddings_model (HuggingFaceEmbeddings): O modelo de embeddings para gerar vetores.
        vectorstore (FAISS): O vetorstore FAISS para busca de documentos.
        k (int): Número de documentos a serem retornados pela busca vetorial.
        fetch_k (int): Número de documentos a serem buscados inicialmente para MMR.
        
    Returns:
        Dict[str, Any]: Um dicionário contendo a resposta, citações e se o contexto foi encontrado.
    """
    q = _norm_ws(question)
    if not q:
        return {"answer": "Não entendi a pergunta. Pode reformular?", "citations": [], "context_found": False}

    try:
        all_docs: List[Document] = list(getattr(vectorstore.docstore, "_dict", {}).values())
    except Exception:
        all_docs = []
    dept_hints = _dept_hints_in_question(q)

    terms = _candidate_terms(q)
    lex_hits: List[Tuple[Document, List[Tuple[str,int]], int]] = []
    if terms and all_docs:
        for d in all_docs:
            src = (d.metadata.get("source") or "").lower()
            score_bonus = 8 if any(_strip_accents_lower(h) in src for h in dept_hints) else 0
            hits = _sentence_hits_by_name(d.page_content or "", terms)
            if hits:
                best = max(sc for _, sc in hits) + score_bonus
                lex_hits.append((d, hits, best))
        lex_hits.sort(key=lambda x: x[2], reverse=True)
        lex_hits = lex_hits[:6]

    if lex_hits:
        sentence_snippets: List[str] = []
        citations = []
        for doc, hits, _best in lex_hits:
            for s, _ in hits[:2]:
                sentence_snippets.append(s)
            citations.append(_as_citation(doc))
            if len(sentence_snippets) >= 6: break

        contacts = _extract_contacts(sentence_snippets)
        name_terms = [t for t in terms if t.isalpha()]
        name = _extract_name(sentence_snippets, name_terms[0]) if name_terms else None
        dept = _guess_dept_from_source(citations[0]["source"]) if citations else None

        # Janela de contexto extra ao redor do nome para encontrar mais informações
        context_snippets = list(sentence_snippets)
        if name and (not contacts["phones"]):
            ref_doc = None
            for d, hits, _b in lex_hits:
                if any(name in s for s,_ in hits):
                    ref_doc = d; break
            ref_doc = ref_doc or lex_hits[0][0]
            win = _context_window_around_name(ref_doc.page_content or "", name, 220, 300)
            if win:
                context_snippets.insert(0, win)
                more = _extract_contacts([win])
                phones = list(dict.fromkeys(contacts["phones"] + more["phones"]))
                emails = list(dict.fromkeys(contacts["emails"] + more["emails"]))
                contacts = {"phones": phones, "emails": emails}

        triage_signals = {
            "dept_hints_in_question": dept_hints,
            "candidates_found": len(lex_hits),
            "have_name": bool(name),
            "phones_found": contacts.get("phones", []),
            "emails_found": contacts.get("emails", []),
        }
        tri = _llm_triage(q, triage_signals)
        if tri.get("action") == "PEDIR_INFO":
            options: List[str] = []
            if len(lex_hits) >= 2:
                for d, hits, _b in lex_hits[:3]:
                    nm = _extract_name([hits[0][0]], name_terms[0] if name_terms else "")
                    if nm and nm not in options:
                        options.append(nm)
            ask = tri.get("ask") or _llm_pedir_info(q, options, prefer_attr="departamento")
            return {"answer": ask, "citations": [], "context_found": False, "needs_clarification": True}

        final = _llm_resposta_final(q, context_snippets or sentence_snippets, name, dept, contacts)
        return {"answer": final, "citations": citations[:3], "context_found": True}

    # Fallback para busca por embeddings se a busca lexical falhar
    try:
        docs: List[Document] = vectorstore.max_marginal_relevance_search(q, k=k, fetch_k=fetch_k)
    except Exception:
        docs = vectorstore.similarity_search(q, k=k)
    if not docs:
        return {"answer": "Não encontrei nada relacionado nos documentos indexados.", "citations": [], "context_found": False}

    top_texts = [d.page_content for d in docs]
    sent_scores = _top_sentences(q, top_texts, top_n=3)
    if sent_scores and sent_scores[0][1] >= 0.45:
        body = " ".join([s for s,_ in sent_scores][:3])
    else:
        body = _norm_ws(docs[0].page_content)
        if len(body) > 500: body = body[:500].rstrip() + "..."
    citations = [_as_citation(d) for d in docs[:3]]
    final = _llm_resposta_final(q, [s for s,_ in sent_scores] if sent_scores else [body], None, None, {"emails": [], "phones": []})
    if not final: final = body
    return {"answer": final, "citations": citations, "context_found": True}


