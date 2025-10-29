# api.py
from __future__ import annotations
import os, time, uuid, json
from collections import Counter
from typing import Any, Dict, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Importações dos Módulos da Aplicação ---
# Tenta importar o cliente LLM de forma lazy. Se falhar, define como None.
try:
    from llm_client import _lazy_client as _llm_lazy_client
except ImportError:
    _llm_lazy_client = None

# Importa a função de RAG direto e a função de execução do agente.
from query_handler import (
    answer_question,
    pipeline_cache_fingerprint,
    CONFIDENCE_MIN_QUERY,
    CONFIDENCE_MIN_AGENT,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_K,
)
from agent_workflow import run_agent, AGENT_REFINE_MAX_ATTEMPTS
from text_normalizer import normalize_documents, normalize_text
from cache_backend import (
    AGENT_NAMESPACE,
    QUERY_NAMESPACE,
    cache_fetch,
    cache_store,
)

# --- Variáveis de Estado de Prontidão ---
APP_READY = False
FAISS_OK = False
LLM_OK = False
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in ("1", "true", "yes")

# --- Configurações do Ambiente ---
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

# --- Métricas e Inicialização do Flask ---
METRICS = Counter()
START_TS = time.time()
METRICS["cache_hits_total"] = 0
METRICS["cache_misses_total"] = 0
METRICS["queries_low_confidence_total"] = 0
METRICS["agent_refine_attempts_total"] = 0
METRICS["agent_refine_success_total"] = 0
METRICS["agent_refine_exhausted_total"] = 0
METRICS["agent_low_confidence_total"] = 0

app = Flask(__name__)
# Habilita o CORS para permitir que a UI (em outro domínio/porta) chame a API.
CORS(app)


# --- Inicialização dos Modelos e Vectorstore ---
# Esta parte do código roda apenas uma vez, quando a aplicação é iniciada.
embeddings_model = None
vectorstore = None

def _normalize_for_cache(text: str) -> str:
    normalized = normalize_text(text or "")
    return " ".join(normalized.split()).lower()


def _base_cache_context() -> Dict[str, Any]:
    return {
        "index_version": os.getenv("INDEX_VERSION", "0"),
        "google_model": os.getenv("GOOGLE_MODEL", ""),
        "embeddings_model": EMBEDDINGS_MODEL,
    }


def _query_cache_payload(question: str, *, k: int, fetch_k: int, confidence_min: float) -> Dict[str, Any]:
    payload = {
        "question": _normalize_for_cache(question),
        "k": k,
        "fetch_k": fetch_k,
        "confidence_min_override": confidence_min,
    }
    payload.update(_base_cache_context())
    payload.update(pipeline_cache_fingerprint())
    return payload


def _normalize_messages_for_cache(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).strip().lower() or "user"
        content = _normalize_for_cache(str(msg.get("content", "")))
        normalized.append({"role": role, "content": content})
    return normalized


def _agent_cache_payload(question: str, normalized_messages: List[Dict[str, str]], *, confidence_min: float, max_refine_allowed: int) -> Dict[str, Any]:
    payload = {
        "question": _normalize_for_cache(question),
        "messages": normalized_messages,
        "agent_version": os.getenv("AGENT_CACHE_VERSION", "1"),
        "confidence_min_override": confidence_min,
        "max_refine_allowed": max_refine_allowed,
    }
    payload.update(_base_cache_context())
    payload.update(pipeline_cache_fingerprint())
    return payload


def _initialize_models():
    """Carrega o modelo de embeddings e o índice FAISS na memória."""
    global embeddings_model, vectorstore, FAISS_OK, APP_READY
    try:

        print(f"[API] Carregando modelo de embeddings: {EMBEDDINGS_MODEL}", flush=True)
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

        print(f"[API] Carregando FAISS de: {FAISS_STORE_DIR}", flush=True)
        if os.path.exists(FAISS_STORE_DIR):
            vectorstore = FAISS.load_local(
                FAISS_STORE_DIR,
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            print("[API] FAISS carregado com sucesso.", flush=True)
            try:
                normalize_documents(getattr(vectorstore.docstore, "_dict", {}).values())
                print("[API] Documentos normalizados para remoção de mojibake.", flush=True)
            except Exception as exc:
                print(f"[API] WARN: falha ao normalizar documentos: {exc}", flush=True)
            FAISS_OK = True
        else:
            print("[API] WARN: Diretório do FAISS não encontrado. A busca não funcionará.", flush=True)
            FAISS_OK = False

    except Exception as e:
        print(f"[API] CRÍTICO: Falha ao carregar FAISS ou modelo de embeddings: {e}", flush=True)
        FAISS_OK = False
    
    _update_readiness()

# --- Funções de Verificação de Prontidão (Probes) ---
def _probe_llm() -> bool:
    """Verifica se o cliente LLM está disponível e configurado."""
    if _llm_lazy_client is None: return False
    try:
        provider, _ = _llm_lazy_client()
        return provider in ("google", "openai")
    except Exception:
        return False

def _update_readiness():
    """Atualiza o estado de prontidão global da aplicação."""
    global LLM_OK, APP_READY
    LLM_OK = _probe_llm()
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)


# --- Rotas da API Flask ---

@app.get("/")
def root():
    return jsonify({"status": "ok"})

@app.get("/healthz")
def healthz():
    """Endpoint de health check, usado por orquestradores como Kubernetes."""
    _update_readiness() # Reavalia a prontidão a cada chamada
    status = {
        "ready": bool(APP_READY),
        "faiss_ok": bool(FAISS_OK),
        "llm_ok": bool(LLM_OK),
        "require_llm_ready": REQUIRE_LLM_READY,
    }
    code = 200 if status["ready"] else 503
    return jsonify(status), code

@app.get("/debug/dict")
def debug_dict():
    try:
        from query_handler import DEPARTMENTS, ALIASES, SYNONYMS, BOOSTS
        return {
            "departments": DEPARTMENTS,
            "aliases_keys": list(ALIASES.keys()),
            "synonyms_keys": list(SYNONYMS.keys()),
            "boosts": BOOSTS,
        }, 200
    except Exception as exc:
        return {"error": str(exc)}, 500

@app.get("/debug/env")
def debug_env():
    keys = [
        "TERMS_YAML",
        "FAISS_STORE_DIR",
        "EMBEDDINGS_MODEL",
        "RERANKER_ENABLED", "RERANKER_NAME",
        "MQ_ENABLED", "MQ_VARIANTS",
        "CONFIDENCE_MIN", "REQUIRE_CONTEXT",
    ]
    return {k: os.getenv(k) for k in keys}, 200

@app.get("/metrics")
def metrics():
    uptime = time.time() - START_TS
    payload = {"uptime_sec": int(uptime), "counters": dict(METRICS)}
    return jsonify(payload), 200

@app.post("/query")
def query():
    """Endpoint legado para processar as consultas do usuário via RAG direto."""
    rid = str(uuid.uuid4())
    ts0 = time.time()

    if not FAISS_OK or vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1
        return jsonify({"error": "O índice vetorial não está carregado."}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "O campo 'question' é obrigatório."}), 400

    debug_flag = bool(data.get("debug")) or (request.args.get("debug", "").lower() == "true")

    cache_payload = None
    cache_key = None
    cache_available = False
    cached_response = None
    res = None
    status = "ok"
    confidence_threshold = CONFIDENCE_MIN_QUERY
    low_confidence = False

    if not debug_flag:
        cache_payload = _query_cache_payload(
            question, k=RETRIEVAL_K, fetch_k=RETRIEVAL_FETCH_K, confidence_min=confidence_threshold
        )
        cached_response, cache_key, cache_available = cache_fetch(QUERY_NAMESPACE, cache_payload)
        if cache_available and cached_response is not None:
            METRICS["cache_hits_total"] += 1
            res = cached_response
            status = "cache_hit"
        else:
            if cache_available:
                METRICS["cache_misses_total"] += 1

    if res is None:
        try:
            res = answer_question(
                question,
                embeddings_model,
                vectorstore,
                debug=debug_flag,
                confidence_min=confidence_threshold,
            )
            status = "ok"
        except Exception as e:
            status = f"error:{type(e).__name__}"
            METRICS["errors_internal"] += 1
            res = {"error": f"Ocorreu uma falha interna: {e}"}

    success = not status.startswith("error")

    if success:
        METRICS["queries_total"] += 1
        if res.get("needs_clarification"):
            METRICS["queries_ambiguous"] += 1
            low_confidence = True
        elif not res.get("context_found"):
            METRICS["queries_not_found"] += 1
            low_confidence = True
        else:
            METRICS["queries_answered"] += 1

        confidence_value = res.get("confidence")
        if confidence_value is not None and confidence_value < confidence_threshold:
            low_confidence = True

        if (
            not debug_flag
            and cache_available
            and cached_response is None
            and cache_payload is not None
            and "error" not in res
        ):
            cache_store(QUERY_NAMESPACE, cache_payload, res, key=cache_key)

        if low_confidence:
            METRICS["queries_low_confidence_total"] += 1

    log = {
        "rid": rid,
        "status": status,
        "took_ms": int((time.time() - ts0) * 1000),
        "question": question[:400],
        "confidence_threshold": confidence_threshold,
        "low_confidence": low_confidence,
    }
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)

# --- NOVO ENDPOINT PARA O AGENTE ---
# --- NOVO ENDPOINT PARA O AGENTE ---

@app.post("/api/ask")
def api_ask():
    """Alias compatível que reutiliza a lógica do /query."""
    return query()

@app.post("/agent/ask")
def agent_ask():
    """Endpoint principal que utiliza o fluxo do agente com LangGraph."""
    rid = str(uuid.uuid4())
    ts0 = time.time()

    if not FAISS_OK or not LLM_OK or vectorstore is None or embeddings_model is None:
        METRICS["error_agent_not_ready"] += 1
        return jsonify({"error": "O agente não está pronto. Verifique o status do FAISS e do LLM."}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "O campo 'question' é obrigatório."}), 400

    debug_flag = bool(data.get("debug")) or (request.args.get("debug", "").lower() == "true")
    raw_messages = data.get("messages", [])
    if not isinstance(raw_messages, list):
        raw_messages = []

    normalized_messages = _normalize_messages_for_cache(raw_messages)

    message_history = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = msg.get("content", "")
        if role == "user":
            message_history.append(HumanMessage(content=content))
        elif role in ("assistant", "ia", "ai"):
            message_history.append(AIMessage(content=content))

    confidence_threshold = CONFIDENCE_MIN_AGENT
    max_refine_override = data.get("max_refine_attempts")
    try:
        max_refine_override_int = int(max_refine_override) if max_refine_override is not None else AGENT_REFINE_MAX_ATTEMPTS
    except (TypeError, ValueError):
        max_refine_override_int = AGENT_REFINE_MAX_ATTEMPTS
    max_refine_override_int = max(0, max_refine_override_int)
    max_refine_allowed = min(max_refine_override_int, AGENT_REFINE_MAX_ATTEMPTS)

    cache_payload = None
    cache_key = None
    cache_available = False
    cached_response = None

    if not debug_flag:
        cache_payload = _agent_cache_payload(
            question,
            normalized_messages,
            confidence_min=confidence_threshold,
            max_refine_allowed=max_refine_allowed,
        )
        cached_response, cache_key, cache_available = cache_fetch(AGENT_NAMESPACE, cache_payload)

    served_from_cache = False
    low_confidence = False
    if cache_available and cached_response is not None:
        METRICS["cache_hits_total"] += 1
        res = cached_response
        status = "cache_hit"
        served_from_cache = True
    else:
        if cache_available:
            METRICS["cache_misses_total"] += 1
        try:
            res = run_agent(
                question,
                message_history,
                embeddings_model,
                vectorstore,
                confidence_min=confidence_threshold,
                max_refine_attempts=max_refine_override_int,
                debug=debug_flag,
            )
            status = "ok"
        except Exception as e:
            status = f"error:{type(e).__name__}"
            METRICS["errors_agent_internal"] += 1
            res = {"error": f"Ocorreu uma falha interna no agente: {e}"}
            import traceback
            print(traceback.format_exc(), flush=True)

    success = not status.startswith("error")
    meta: Dict[str, Any] = res.get("meta") or {}
    meta_confidence = meta.get("confidence")

    if success:
        METRICS["agent_queries_total"] += 1
        if res.get("action") == "PEDIR_INFO":
            METRICS["agent_clarification_needed"] += 1
            low_confidence = True
        else:
            METRICS["agent_answered"] += 1

        if not served_from_cache:
            refine_attempts = int(meta.get("refine_attempts") or 0)
            if refine_attempts:
                METRICS["agent_refine_attempts_total"] += refine_attempts
                if meta.get("refine_success"):
                    METRICS["agent_refine_success_total"] += 1
                else:
                    METRICS["agent_refine_exhausted_total"] += 1

        if meta_confidence is not None and meta_confidence < confidence_threshold:
            low_confidence = True

        if (
            not debug_flag
            and cache_available
            and cached_response is None
            and "error" not in res
        ):
            cache_store(AGENT_NAMESPACE, cache_payload, res, key=cache_key)

        if low_confidence:
            METRICS["agent_low_confidence_total"] += 1

    log = {
        "rid": rid,
        "status": status,
        "took_ms": int((time.time() - ts0) * 1000),
        "question": question[:400],
        "confidence_threshold": confidence_threshold,
        "low_confidence": low_confidence,
    }
    if meta:
        log["meta"] = {
            "refine_attempts": int(meta.get("refine_attempts") or 0),
            "confidence": meta.get("confidence"),
            "refine_success": bool(meta.get("refine_success")),
            "query_hash": meta.get("query_hash"),
            "refine_prompt_hashes": meta.get("refine_prompt_hashes"),
            "max_refine_allowed": meta.get("max_refine_allowed"),
            "confidence_threshold": confidence_threshold,
        }
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)





if __name__ == "__main__":
    # A inicialização dos modelos é chamada antes de iniciar o servidor.
    _initialize_models()
    # Inicia o servidor de desenvolvimento do Flask.
    app.run(host="0.0.0.0", port=5000)
