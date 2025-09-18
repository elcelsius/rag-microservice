# query_handler.py
# Este módulo é responsável por processar as consultas dos usuários, realizar a busca no vetorstore FAISS,
# extrair informações relevantes e formatar a resposta final usando um LLM.

from __future__ import annotations
import re, unicodedata
import os
import math
from typing import Any, Dict, List, Tuple, Optional, Iterable, Set

from rapidfuzz import fuzz, distance
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from llm_client import load_prompt, call_llm
from sentence_transformers import CrossEncoder

# ==== Configurações do Reranker ====
# Ativa ou desativa o uso do reranker para reordenar os resultados da busca.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
# Nome do modelo CrossEncoder a ser usado para o rerank.
RERANKER_NAME = os.getenv("RERANKER_NAME", "jinaai/jina-reranker-v2-base-multilingual")
# Número de documentos candidatos a serem enviados para o reranker.
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "30"))
# Número de documentos a serem retornados pelo reranker (top K).
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
# Comprimento máximo da sequência para o modelo reranker.
RERANKER_MAX_LEN = int(os.getenv("RERANKER_MAX_LEN", "4096"))

# ==== Configurações de Multi-Query e Confiança ====
# Ativa ou desativa a geração de múltiplas variações da pergunta do usuário para busca.
MQ_ENABLED = os.getenv("MQ_ENABLED", "true").lower() == "true"
# Número de variações da pergunta a serem geradas.
MQ_VARIANTS = int(os.getenv("MQ_VARIANTS", "3"))
# Limiar mínimo de confiança do reranker para considerar uma resposta válida.
CONFIDENCE_MIN = float(os.getenv("CONFIDENCE_MIN", "0.32"))
# Exige que um contexto seja encontrado nos documentos para gerar uma resposta.
REQUIRE_CONTEXT = os.getenv("REQUIRE_CONTEXT", "true").lower() == "true"

# === Dicionário externo (departamentos, aliases, sinônimos, boosts) ===
import yaml  # requer PyYAML

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
BOOSTS = TERMS["boosts"]  # dict de boosts opcionais
print(f"[DICT] departamentos={len(DEPARTMENTS)} aliases={len(ALIASES)} synonyms={len(SYNONYMS)}")

# Variável global para armazenar o modelo do reranker e evitar recarregá-lo.
_reranker_model = None


def _get_reranker():
    """Carrega o modelo CrossEncoder do reranker sob demanda e o mantém em memória."""
    global _reranker_model
    if _reranker_model is None and RERANKER_ENABLED:
        # carrega uma vez, fica em memória
        print(f"[INFO] Carregando o modelo reranker: {RERANKER_NAME}")
        _reranker_model = CrossEncoder(RERANKER_NAME, max_length=RERANKER_MAX_LEN)
    return _reranker_model


def _apply_rerank(query: str, docs: List[Document]) -> Tuple[List[Document], List[float]]:
    """
    Reclassifica uma lista de documentos com base na sua relevância para a consulta,
    aplicando um bônus (boost) para documentos que mencionam departamentos citados na pergunta.

    Args:
        query (str): A consulta do usuário.
        docs (List[Document]): Lista de documentos do LangChain a serem reranqueados.

    Returns:
        Tuple[List[Document], List[float]]: Uma tupla contendo a lista de documentos reordenada
                                            e a lista de seus respectivos scores de confiança.
    """
    # Retorna listas vazias se não houver documentos para processar.
    if not docs:
        return [], []
    # Se o reranker estiver desabilitado, retorna os top K documentos com score 0.0.
    if not RERANKER_ENABLED:
        top = docs[:min(len(docs), RERANKER_TOP_K)]
        return top, [0.0] * len(top)

    rr = _get_reranker()
    # Se o modelo não puder ser carregado, retorna os top K documentos com score 0.0.
    if rr is None:
        top = docs[:min(len(docs), RERANKER_TOP_K)]
        return top, [0.0] * len(top)

    # Cria pares de (consulta, conteúdo do documento) para o modelo.
    pairs = [(query, getattr(d, "page_content", str(d))) for d in docs]
    # O modelo prediz os scores. Modelos como Jina v2 retornam scores de 0 a 1 (maior = melhor).
    scores = rr.predict(pairs)

    # --- boost por departamento (leve) ---
    # Identifica slugs de departamento na pergunta para dar um bônus aos documentos correspondentes.
    wanted = set(_dept_slugs_in_question(query))
    boosted: List[Tuple[object, float]] = []
    for d, s in zip(docs, scores):
        b = float(s)
        # Se algum slug de departamento foi encontrado na pergunta...
        if wanted:
            meta = getattr(d, "metadata", {}) or {}
            # Concatena os campos de metadados onde o departamento pode ser encontrado.
            src = " ".join([
                str(meta.get("source") or ""),
                str(meta.get("file") or ""),
                str(meta.get("department") or ""),
            ])
            src_norm = _strip_accents_lower(src)
            # Verifica se algum dos slugs desejados está nos metadados do documento.
            if any(slug in src_norm for slug in wanted):
                # Aplica um boost padrão para o documento correspondente.
                b += 0.05
                # Aplica um boost adicional opcional, configurado no arquivo YAML.
                # Ex: boosts.department.<slug>
                dep_boosts = (BOOSTS.get("department") or {})
                for slug in wanted:
                    w = dep_boosts.get(slug)
                    if isinstance(w, (int, float)):
                        # O valor no YAML é tratado como um percentual (ex: 1.5 -> +0.015)
                        b += float(w) * 0.01
        boosted.append((d, b))

    # Ordena os documentos com base nos scores (possivelmente com boost).
    ranked = sorted(boosted, key=lambda x: float(x[1]), reverse=True)

    # Seleciona os top K pares.
    top_pairs = ranked[:RERANKER_TOP_K]
    if not top_pairs:
        print(f"[RERANK/BOOST] wanted={list(wanted)} top=0")
        return [], []

    # Separa os documentos e os scores em duas listas distintas.
    top_docs = [d for d, s in top_pairs]
    top_scores = [float(s) for d, s in top_pairs]

    # Log para depuração do processo de rerank e boost.
    print(f"[RERANK/BOOST] wanted={list(wanted)} top={len(top_docs)}")

    return top_docs, top_scores


# ==== Utilitários de Geração de Multi-Query e Processamento de Texto ====

def _dedupe_preserve_order(items: Iterable, key=lambda x: x) -> list:
    """
    Remove duplicatas de uma lista/iterável, preservando a ordem original dos elementos.

    Args:
        items (Iterable): A lista de itens a serem desduplicados.
        key (function): Uma função para extrair a chave de comparação de cada item.

    Returns:
        list: Uma nova lista sem duplicatas.
    """
    seen: Set = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


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
    for key, exps in SYNONYMS.items():
        if key in s_norm:
            syn_vars.extend(exps)

    # 3) (opcional) pedir variantes ao LLM (se houver wrapper compatível)
    llm_vars = []
    if llm is not None:
        try:
            prompt = (
                "Gere variações curtas e diferentes (1 por linha) desta pergunta, "
                "mantendo o sentido, para busca em base de conhecimento:\n"
                f"{user_query}\n"
                f"(gere no máximo {n})"
            )
            txt = llm.generate_variants(prompt, n)  # adapte ao seu wrapper, se existir
            llm_vars = [ln.strip(" -•\t") for ln in txt.splitlines() if ln.strip()]
        except Exception:
            llm_vars = []

    # 4) mescla e corta em n
    merged = _dedupe_preserve_order([user_query] + base_local + syn_vars + llm_vars)
    return merged[:n]


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?\;\:])\s+|\n+")  # Regex para dividir texto em sentenças
_WS = re.compile(r"\s+")  # Regex para normalizar espaços em branco
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)  # Regex para encontrar e-mails
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d{4,5}-\d{4}")  # Regex para encontrar números de telefone
WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")  # Regex para encontrar palavras (letras acentuadas incluídas)

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
                    best = max(best, sc);
                    hit = True
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
            uniq.append((s, sc));
            seen.add(k)
    return uniq[:3]


NAME_SEQ_RE = re.compile(r"([A-ZÁ-Ü][a-zá-ü]+(?:\s+[A-ZÁ-Ü][a-zá-ü]+){1,5})")  # Regex para sequências de nomes


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
                    best_sim = sim;
                    best = cand
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
            uniq.append(s);
            seen.add(k)
    q = _norm_ws(question)
    scored = [(s, fuzz.token_set_ratio(q, s) / 100.0) for s in uniq]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def _context_window_around_name(doc_text: str, person_name: str, chars_before: int = 220, chars_after: int = 300) -> \
        Optional[str]:
    """Extrai uma janela de contexto de texto ao redor de um nome de pessoa."""
    if not (doc_text and person_name):
        return None
    txt = doc_text;
    name = person_name.strip()
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
_TRIAGE_PROMPT = load_prompt("prompts/triagem_prompt.txt")
_PEDIR_INFO_PROMPT = load_prompt("prompts/pedir_info_prompt.txt")
_RESPOSTA_PROMPT = load_prompt("prompts/resposta_final_prompt.txt")


def _llm_triage(question: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    """Usa o LLM para fazer a triagem da pergunta, decidindo se deve responder ou pedir mais informações."""
    if not _TRIAGE_PROMPT:
        return {"action": "AUTO_RESOLVER", "ask": ""}
    user_payload = f"Pergunta: {question}\n\nSinais:\n{signals}"
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
    payload = f"Pergunta original: {question}\nOpções (máx 3): {options[:3]}\nAtributo preferencial: {prefer_attr}"
    text, _ = call_llm(_PEDIR_INFO_PROMPT, payload, expect_json=False, max_tokens=120)
    return (text or "").strip() or "Pode informar o departamento/unidade?"


def _format_direct(name: Optional[str], dept: Optional[str], contacts: Dict[str, List[str]],
                   context_snippets: List[str]) -> str:
    """Formata uma resposta direta com as informações encontradas, sem usar o LLM."""
    parts = []
    if name: parts.append(name)
    if dept: parts.append(dept)
    if contacts.get("phones"): parts.append("📞 " + ", ".join(contacts["phones"]))
    if contacts.get("emails"): parts.append("✉️ " + ", ".join(contacts["emails"]))
    if parts: return " — ".join(parts)
    return (context_snippets[0] if context_snippets else "Não encontrei.")[:400]


def _llm_resposta_final(question: str, context_snippets: List[str], name: Optional[str], dept: Optional[str],
                        contacts: Dict[str, List[str]]) -> str:
    """Usa o LLM para gerar uma resposta final em linguagem natural, com base no contexto e metadados extraídos."""
    if not _RESPOSTA_PROMPT:
        return _format_direct(name, dept, contacts, context_snippets)
    ctx = "\n\n".join(f"- {s}" for s in context_snippets[:6])
    meta = {"nome": name or "", "departamento": dept or "", "telefones": contacts.get("phones", []),
            "emails": contacts.get("emails", [])}
    user = f"Pergunta: {question}\n\nContexto (trechos):\n{ctx}\n\nMetadados extraídos:\n{meta}\n\nFormato: Nome — Departamento — 📞 — ✉️ (curto)."
    text, _ = call_llm(_RESPOSTA_PROMPT, user, expect_json=False, max_tokens=220)
    if not (text or "").strip():
        return _format_direct(name, dept, contacts, context_snippets)
    return text.strip()


# ==== Pipeline Principal de Resposta ====
def answer_question(question: str, embeddings_model: HuggingFaceEmbeddings, vectorstore: FAISS, *, k: int = 5,
                    fetch_k: int = 20) -> Dict[str, Any]:
    """
    Pipeline principal para responder a uma pergunta usando uma abordagem híbrida (busca lexical e vetorial).

    Args:
        question (str): A pergunta do usuário.
        embeddings_model (HuggingFaceEmbeddings): O modelo de embeddings para gerar vetores (não usado diretamente aqui, mas parte da arquitetura).
        vectorstore (FAISS): O vetorstore FAISS para busca de documentos.
        k (int): Número de documentos a serem retornados pela busca vetorial (usado como fallback se o reranker estiver desabilitado).
        fetch_k (int): Número de documentos a serem buscados inicialmente para MMR (não mais usado diretamente, mas mantido por compatibilidade).

    Returns:
        Dict[str, Any]: Um dicionário contendo a resposta, citações, confiança e se o contexto foi encontrado.
    """
    q = _norm_ws(question)
    if not q:
        return {"answer": "Não entendi a pergunta. Pode reformular?", "citations": [], "context_found": False}

    # Estratégia 1: Busca Lexical por Nomes (rápida e precisa para contatos)
    try:
        all_docs: List[Document] = list(getattr(vectorstore.docstore, "_dict", {}).values())
    except Exception:
        all_docs = []

    dept_hints = _dept_hints_in_question(q)
    terms = _candidate_terms(q)
    lex_hits: List[Tuple[Document, List[Tuple[str, int]], int]] = []

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
        # Se a busca lexical encontrou candidatos fortes, processa-os.
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

        context_snippets = list(sentence_snippets)
        if name and (not contacts["phones"]):
            ref_doc = lex_hits[0][0]  # Usa o documento de maior score como referência
            win = _context_window_around_name(ref_doc.page_content or "", name, 220, 300)
            if win:
                context_snippets.insert(0, win)
                more = _extract_contacts([win])
                phones = list(dict.fromkeys(contacts["phones"] + more["phones"]))
                emails = list(dict.fromkeys(contacts["emails"] + more["emails"]))
                contacts = {"phones": phones, "emails": emails}

        # Triagem com LLM para decidir se pede esclarecimento.
        triage_signals = {"dept_hints_in_question": dept_hints, "candidates_found": len(lex_hits),
                          "have_name": bool(name)}
        tri = _llm_triage(q, triage_signals)
        if tri.get("action") == "PEDIR_INFO":
            options = [nm for d, hits, _b in lex_hits[:3] if
                       (nm := _extract_name([hits[0][0]], name_terms[0] if name_terms else "")) and nm]
            ask = tri.get("ask") or _llm_pedir_info(q, options, prefer_attr="departamento")
            return {"answer": ask, "citations": [], "context_found": False, "needs_clarification": True}

        # Gera a resposta final com base nos dados extraídos.
        final = _llm_resposta_final(q, context_snippets or sentence_snippets, name, dept, contacts)
        return {"answer": final, "citations": citations[:3], "context_found": True}

    # Estratégia 2: Fallback para Busca Vetorial com Multi-Query e Rerank
    # Este bloco é executado se a busca lexical não encontrar resultados.

    k_candidates_total = RERANKER_CANDIDATES if RERANKER_ENABLED else 10

    # 1. Gera variações da pergunta para uma busca mais abrangente.
    queries = [q]
    if MQ_ENABLED:
        queries = _gen_multi_queries(q, MQ_VARIANTS)

    # 2. Distribui o "orçamento" de busca entre as queries geradas.
    per_q = max(3, math.ceil(k_candidates_total / max(1, len(queries))))

    # 3. Busca para cada query e mescla os resultados.
    cands = []
    for query_variant in queries:
        try:
            hits = vectorstore.similarity_search(query_variant, k=per_q)
            cands.extend(hits)
        except Exception as e:
            print(f"[ERROR] Falha na busca vetorial para a query '{query_variant}': {e}")
            continue

    def _doc_key(d: Document) -> tuple:
        # Cria uma chave única para cada documento para desduplicação.
        src = d.metadata.get("source", "")
        chk = d.metadata.get("chunk")
        return (src, chk, d.page_content[:64])

    cands = _dedupe_preserve_order(cands, key=_doc_key)

    # 4. Reordena os candidatos com o reranker e obtém os scores de confiança.
    docs, scores = _apply_rerank(q, cands)

    # Função Sigmoid para normalizar scores, se necessário.
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except (OverflowError, ValueError):
            return 0.0

    # Calcula a confiança final. Usa o score máximo do reranker.
    if scores:
        max_s = float(max(scores))
        # O Jina v2 já retorna scores em [0, 1]. Se não, o sigmoid normalizaria.
        conf = max_s if (0.0 <= max_s <= 1.0) else _sigmoid(max_s)
    else:
        conf = 0.0

    # Log de diagnóstico para a busca vetorial.
    print(
        f"[RETRIEVE] mq={MQ_ENABLED} variants={len(queries)} merged={len(cands)} reranked={len(docs)} conf={conf:.3f}")

    # 5. Verifica se a confiança é suficiente para responder.
    if (not docs) or (conf < CONFIDENCE_MIN):
        if REQUIRE_CONTEXT:
            # Retorna uma resposta segura se a confiança for muito baixa.
            return {
                "answer": "Não encontrei informação suficiente nos documentos para responder com segurança.",
                "citations": [],
                "context_found": False,
                "confidence": conf,
            }

    # 6. Se a confiança for aceitável, processa os documentos para a resposta final.
    top_texts = [d.page_content for d in docs]
    sent_scores = _top_sentences(q, top_texts, top_n=3)

    if sent_scores and sent_scores[0][1] >= 0.45:
        # Usa as sentenças mais relevantes se a similaridade for boa.
        context_snippets = [s for s, _ in sent_scores]
    else:
        # Caso contrário, usa o início do documento principal como contexto.
        body = _norm_ws(docs[0].page_content)
        if len(body) > 500: body = body[:500].rstrip() + "..."
        context_snippets = [body]

    citations = [_as_citation(d) for d in docs[:3]]
    # Gera a resposta final com o LLM, mas sem metadados de nome/depto, pois esta é uma busca genérica.
    final = _llm_resposta_final(q, context_snippets, None, None, {"emails": [], "phones": []})

    if not final:
        final = " ".join(context_snippets)

    return {"answer": final, "citations": citations, "context_found": True, "confidence": conf}