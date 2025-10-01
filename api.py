# api.py
from __future__ import annotations
import os, time, uuid, json, hashlib
from collections import Counter

from flask import Flask, request, jsonify
import redis

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agent_workflow import compiled_graph
from query_handler import answer_question

try:
    from llm_client import _lazy_client as _llm_lazy_client
except Exception:
    _llm_lazy_client = None

APP_READY = False
FAISS_OK = False
LLM_OK = False
REDIS_OK = False
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in {"1", "true", "yes"}

FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or os.getenv("EMBEDDINGS_MODEL_NAME") or "intfloat/multilingual-e5-large"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 86400))

METRICS = Counter()
START_TS = time.time()

app = Flask(__name__)

redis_client = None
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    print(f"[API] Conectado ao Redis em {REDIS_HOST}:{REDIS_PORT}", flush=True)
    REDIS_OK = True
except Exception as exc:
    print(f"[API] AVISO: NÃ£o foi possÃ­vel conectar ao Redis: {exc}", flush=True)
    REDIS_OK = False

embeddings_model = None
vectorstore = None
try:
    print(f"[API] Carregando modelo de embeddings: {EMBEDDINGS_MODEL}", flush=True)
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    print(f"[API] Carregando FAISS de: {FAISS_STORE_DIR}", flush=True)
    vectorstore = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True,
    )
    print("[API] FAISS carregado com sucesso.", flush=True)
except Exception as exc:
    print(f"[API] CRÃTICO: Falha ao carregar FAISS ou embeddings: {exc}", flush=True)


def _probe_faiss(vs) -> bool:
    try:
        return len(list(getattr(vs.docstore, "_dict", {}).values())) > 0
    except Exception:
        return False


def _probe_llm() -> bool:
    if _llm_lazy_client is None:
        return False
    try:
        provider, _ = _llm_lazy_client()
        return provider in {"google", "openai"}
    except Exception:
        return False


def _probe_redis() -> bool:
    if redis_client is None:
        return False
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


def _update_readiness(vs=None):
    global APP_READY, FAISS_OK, LLM_OK, REDIS_OK
    if vs is not None:
        FAISS_OK = _probe_faiss(vs)
    LLM_OK = _probe_llm()
    REDIS_OK = _probe_redis()
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)


_update_readiness(vectorstore)


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
        "RERANKER_ENABLED",
        "RERANKER_NAME",
        "MQ_ENABLED",
        "MQ_VARIANTS",
        "CONFIDENCE_MIN",
        "REQUIRE_CONTEXT",
        "REDIS_HOST",
        "CACHE_TTL_SECONDS",
    ]
    return {key: os.getenv(key) for key in keys}, 200


@app.get("/")
def root():
    return jsonify({"status": "ok"})


@app.get("/healthz")
def healthz():
    _update_readiness(vectorstore)
    status = {
        "ready": bool(APP_READY),
        "faiss": bool(FAISS_OK),
        "llm": bool(LLM_OK),
        "redis": bool(REDIS_OK),
        "require_llm_ready": REQUIRE_LLM_READY,
        "faiss_store_dir": FAISS_STORE_DIR,
        "embeddings_model": EMBEDDINGS_MODEL,
    }
    code = 200 if status["ready"] else 503
    return jsonify(status), code


@app.get("/metrics")
def metrics():
    uptime = int(time.time() - START_TS)
    return jsonify({"uptime_sec": uptime, "counters": dict(METRICS)}), 200


@app.post("/query")
def query():
    rid = str(uuid.uuid4())
    ts0 = time.time()

    if vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1
        return (
            jsonify(
                {
                    "answer": "O Ã­ndice vetorial nÃ£o estÃ¡ carregado. Execute o ETL e reinicie a aplicaÃ§Ã£o.",
                    "citations": [],
                    "context_found": False,
                }
            ),
            503,
        )

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "O campo 'question' (ou 'query'/'q') Ã© obrigatÃ³rio."}), 400

    debug_flag = bool(data.get("debug")) or request.args.get("debug", "").lower() == "true"

    cache_key = None
    if not debug_flag and redis_client and REDIS_OK:
        normalized = question.lower().strip()
        cache_key = f"rag_query:{hashlib.sha256(normalized.encode()).hexdigest()}"
        try:
            cached_response = redis_client.get(cache_key)
            if cached_response:
                METRICS["queries_cached"] += 1
                cached_payload = json.loads(cached_response)
                cached_payload["X-Cache-Status"] = "hit"
                print(f"[API] Cache HIT -> {cache_key}", flush=True)
                return jsonify(cached_payload)
        except Exception as exc:
            print(f"[API] AVISO: Falha ao ler do cache Redis: {exc}", flush=True)

    res: dict[str, object] = {}
    status = "ok"

    try:
        if debug_flag:
            res = answer_question(question, embeddings_model, vectorstore, debug=True)
        else:
            agent_state = compiled_graph.invoke({"pergunta": question})
            res = {
                "answer": agent_state.get("resposta"),
                "citations": agent_state.get("citacoes", []),
                "context_found": bool(agent_state.get("citacoes")),
                "needs_clarification": agent_state.get("acao_final") == "PEDIR_INFO",
            }

        METRICS["queries_total"] += 1
        if res.get("needs_clarification"):
            METRICS["queries_ambiguous"] += 1
        elif not res.get("context_found"):
            METRICS["queries_not_found"] += 1
        else:
            METRICS["queries_answered"] += 1

        if (
            not debug_flag
            and redis_client
            and REDIS_OK
            and cache_key
            and not res.get("needs_clarification")
            and res.get("context_found")
        ):
            try:
                redis_client.set(cache_key, json.dumps(res), ex=CACHE_TTL_SECONDS)
                print(f"[API] Resposta armazenada no cache -> {cache_key}", flush=True)
            except Exception as exc:
                print(f"[API] AVISO: Falha ao escrever no cache Redis: {exc}", flush=True)
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        METRICS["errors_internal"] += 1
        res = {"error": f"Ocorreu uma falha interna ao processar a pergunta: {exc}"}

    took_ms = int((time.time() - ts0) * 1000)
    log_payload = {
        "rid": rid,
        "status": status,
        "took_ms": took_ms,
        "question": question[:400],
        "ready": APP_READY,
        "faiss": FAISS_OK,
        "llm": LLM_OK,
        "redis": REDIS_OK,
        "debug": debug_flag,
    }
    print(json.dumps(log_payload, ensure_ascii=False), flush=True)

    if not debug_flag and cache_key:
        res.setdefault("X-Cache-Status", "miss")

    return jsonify(res)


@app.post("/api/ask")
def api_ask():
    return query()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
