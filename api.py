# api.py
from __future__ import annotations
import os, time, uuid, json
from collections import Counter
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from query_handler import answer_question

try:
    from llm_client import _lazy_client as _llm_lazy_client
except Exception:
    _llm_lazy_client = None

# Readiness
APP_READY = False
FAISS_OK = False
LLM_OK = False
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in ("1","true","yes")

# Config
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Metrics (simples)
METRICS = Counter()
START_TS = time.time()

app = Flask(__name__)

def _probe_faiss(vs) -> bool:
    try:
        all_docs = list(getattr(vs.docstore, "_dict", {}).values())
        return len(all_docs) > 0
    except Exception:
        return False

def _probe_llm() -> bool:
    if _llm_lazy_client is None:
        return False
    try:
        provider, _ = _llm_lazy_client()
        return provider in ("google", "openai")
    except Exception:
        return False

def _update_readiness(vs=None):
    global APP_READY, FAISS_OK, LLM_OK
    if vs is not None:
        FAISS_OK = _probe_faiss(vs)
    LLM_OK = _probe_llm()
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)

embeddings_model = None
vectorstore = None

try:
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    print(f"[API] FAISS carregado: {FAISS_STORE_DIR}", flush=True)
except Exception as e:
    print(f"[API] Falha ao carregar FAISS em '{FAISS_STORE_DIR}': {e}", flush=True)

_update_readiness(vectorstore)

@app.get("/")
def root():
    return jsonify({"status": "ok"})

@app.get("/healthz")
def healthz():
    status = {
        "ready": bool(APP_READY),
        "faiss": bool(FAISS_OK),
        "llm": bool(LLM_OK),
        "require_llm_ready": REQUIRE_LLM_READY,
        "faiss_store_dir": FAISS_STORE_DIR,
        "embeddings_model": EMBEDDINGS_MODEL,
    }
    code = 200 if status["ready"] else 503
    return jsonify(status), code

@app.get("/metrics")
def metrics():
    uptime = time.time() - START_TS
    payload = {
        "uptime_sec": int(uptime),
        "counters": dict(METRICS),
    }
    return jsonify(payload), 200

@app.post("/query")
def query():
    rid = str(uuid.uuid4())
    ts0 = time.time()

    if vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1
        return jsonify({
            "answer": "Índice não carregado. Rode o ETL para gerar o FAISS e tente novamente.",
            "citations": [],
            "context_found": False
        }), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "Informe 'question' (ou 'query'/'q') no JSON."}), 400

    res = {}
    status = "ok"
    try:
        res = answer_question(question, embeddings_model, vectorstore)
        METRICS["queries_total"] += 1
        if res.get("needs_clarification"):
            METRICS["queries_ambiguous"] += 1
        elif not res.get("context_found"):
            METRICS["queries_not_found"] += 1
        else:
            METRICS["queries_answered"] += 1
    except Exception as e:
        status = f"error:{type(e).__name__}"
        METRICS["errors_internal"] += 1
        res = {"error": f"Falha ao processar pergunta: {e}"}

    took_ms = int((time.time() - ts0) * 1000)
    log = {
        "rid": rid,
        "status": status,
        "took_ms": took_ms,
        "question": question[:400],
        "ready": APP_READY,
        "faiss": FAISS_OK,
        "llm": LLM_OK,
    }
    print(json.dumps(log, ensure_ascii=False), flush=True)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
