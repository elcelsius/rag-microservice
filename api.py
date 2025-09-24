# api.py
from __future__ import annotations
import os, time, uuid, json, hashlib
from collections import Counter
from flask import Flask, request, jsonify
import redis

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from query_handler import answer_question

# Tenta importar o cliente LLM de forma lazy (preguiçosa).
try:
    from llm_client import _lazy_client as _llm_lazy_client
except Exception:
    _llm_lazy_client = None

# --- Variáveis de Estado de Prontidão ---
APP_READY = False
FAISS_OK = False
LLM_OK = False
REDIS_OK = False # Novo estado para a prontidão do Redis
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in ("1", "true", "yes")

# --- Configurações do Ambiente ---
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or os.getenv("EMBEDDINGS_MODEL_NAME") or "intfloat/multilingual-e5-large"

# --- Configurações do Cache Redis ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 86400)) # Padrão: 24 horas

# --- Métricas Simples para Monitoramento ---
METRICS = Counter()
START_TS = time.time()

app = Flask(__name__)

# --- Inicialização do Cliente Redis ---
redis_client = None
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping() # Verifica a conexão
    print(f"[API] Conectado ao Redis em {REDIS_HOST}:{REDIS_PORT}", flush=True)
    REDIS_OK = True
except Exception as e:
    print(f"[API] AVISO: Não foi possível conectar ao Redis: {e}", flush=True)
    REDIS_OK = False


# --- Rotas de Debug ---
@app.get("/debug/dict")
def debug_dict():
    try:
        from query_handler import DEPARTMENTS, ALIASES, SYNONYMS, BOOSTS
        return {"departments": DEPARTMENTS, "aliases_keys": list(ALIASES.keys()), "synonyms_keys": list(SYNONYMS.keys()), "boosts": BOOSTS}, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/debug/env")
def debug_env():
    keys = ["TERMS_YAML", "FAISS_STORE_DIR", "EMBEDDINGS_MODEL", "RERANKER_ENABLED", "RERANKER_NAME", "MQ_ENABLED", "MQ_VARIANTS", "CONFIDENCE_MIN", "REQUIRE_CONTEXT", "REDIS_HOST", "REDIS_PORT"]
    return {k: os.getenv(k) for k in keys}, 200


# --- Funções de Verificação de Prontidão (Probes) ---
def _probe_faiss(vs) -> bool:
    try: return len(list(getattr(vs.docstore, "_dict", {}).values())) > 0
    except Exception: return False

def _probe_llm() -> bool:
    if _llm_lazy_client is None: return False
    try:
        provider, _ = _llm_lazy_client()
        return provider in ("google", "openai")
    except Exception: return False

def _probe_redis() -> bool:
    if redis_client is None: return False
    try: return redis_client.ping()
    except Exception: return False

def _update_readiness(vs=None):
    global APP_READY, FAISS_OK, LLM_OK, REDIS_OK
    if vs is not None: FAISS_OK = _probe_faiss(vs)
    LLM_OK = _probe_llm()
    REDIS_OK = _probe_redis()
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)


# --- Inicialização dos Modelos e Vectorstore ---
embeddings_model = None
vectorstore = None
try:
    print(f"[API] Carregando modelo de embeddings: {EMBEDDINGS_MODEL}", flush=True)
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    print(f"[API] Carregando FAISS de: {FAISS_STORE_DIR}", flush=True)
    vectorstore = FAISS.load_local(FAISS_STORE_DIR, embeddings_model, allow_dangerous_deserialization=True)
    print("[API] FAISS carregado com sucesso.", flush=True)
except Exception as e:
    print(f"[API] CRÍTICO: Falha ao carregar FAISS ou modelo de embeddings: {e}", flush=True)

_update_readiness(vectorstore)


# --- Rotas da API Flask ---
@app.get("/")
def root():
    return jsonify({"status": "ok"})

@app.get("/healthz")
def healthz():
    _update_readiness(vectorstore) # Reavalia a prontidão a cada chamada
    status = {
        "ready": bool(APP_READY),
        "faiss": bool(FAISS_OK),
        "llm": bool(LLM_OK),
        "redis": bool(REDIS_OK), # Adiciona status do Redis
        "require_llm_ready": REQUIRE_LLM_READY,
        "faiss_store_dir": FAISS_STORE_DIR,
        "embeddings_model": EMBEDDINGS_MODEL,
    }
    code = 200 if status["ready"] else 503
    return jsonify(status), code

@app.get("/metrics")
def metrics():
    uptime = time.time() - START_TS
    payload = {"uptime_sec": int(uptime), "counters": dict(METRICS)}
    return jsonify(payload), 200

@app.post("/query")
def query():
    rid = str(uuid.uuid4())
    ts0 = time.time()

    if vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1
        return jsonify({"answer": "O índice vetorial não está carregado.", "citations": [], "context_found": False}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "O campo 'question' é obrigatório."}), 400

    debug_flag = bool(data.get("debug")) or (request.args.get("debug", "").lower() == "true")

    # --- Lógica de Cache (Leitura) ---
    cache_key = None
    if redis_client and REDIS_OK:
        normalized_question = question.lower().strip()
        cache_key = f"rag_query:{hashlib.sha256(normalized_question.encode()).hexdigest()}"
        try:
            cached_response = redis_client.get(cache_key)
            if cached_response:
                METRICS["queries_cached"] += 1
                print(f"[API] Cache HIT para a chave: {cache_key}", flush=True)
                cached_data = json.loads(cached_response)
                cached_data["X-Cache-Status"] = "hit"
                return jsonify(cached_data)
        except Exception as e:
            print(f"[API] AVISO: Falha ao ler do cache Redis: {e}", flush=True)
    
    print(f"[API] Cache MISS para a chave: {cache_key or 'N/A'}", flush=True)
    # --- Fim da Lógica de Cache (Leitura) ---

    res = {}
    status = "ok"
    try:
        res = answer_question(question, embeddings_model, vectorstore, debug=debug_flag)
        METRICS["queries_total"] += 1
        if res.get("needs_clarification"): METRICS["queries_ambiguous"] += 1
        elif not res.get("context_found"): METRICS["queries_not_found"] += 1
        else: METRICS["queries_answered"] += 1

        # --- Lógica de Cache (Escrita) ---
        if redis_client and REDIS_OK and cache_key and res.get("context_found"):
            try:
                redis_client.set(cache_key, json.dumps(res), ex=CACHE_TTL_SECONDS)
                print(f"[API] Resposta armazenada no cache com a chave: {cache_key}", flush=True)
            except Exception as e:
                print(f"[API] AVISO: Falha ao escrever no cache Redis: {e}", flush=True)
        # --- Fim da Lógica de Cache (Escrita) ---

    except Exception as e:
        status = f"error:{type(e).__name__}"
        METRICS["errors_internal"] += 1
        res = {"error": f"Ocorreu uma falha interna: {e}"}

    took_ms = int((time.time() - ts0) * 1000)
    log = {"rid": rid, "status": status, "took_ms": took_ms, "question": question[:400], "ready": APP_READY, "faiss": FAISS_OK, "llm": LLM_OK, "redis": REDIS_OK}
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)

@app.post("/api/ask")
def api_ask():
    return query()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
