# api.py
from __future__ import annotations
import os, time, uuid, json
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage

# --- Importações dos Módulos da Aplicação ---
# Tenta importar o cliente LLM de forma lazy. Se falhar, define como None.
try:
    from llm_client import _lazy_client as _llm_lazy_client
except ImportError:
    _llm_lazy_client = None

# Importa a função de RAG direto e a função de execução do agente.
from query_handler import answer_question
from agent_workflow import run_agent

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

app = Flask(__name__)
# Habilita o CORS para permitir que a UI (em outro domínio/porta) chame a API.
CORS(app)


# --- Inicialização dos Modelos e Vectorstore ---
# Esta parte do código roda apenas uma vez, quando a aplicação é iniciada.
embeddings_model = None
vectorstore = None

def _initialize_models():
    """Carrega o modelo de embeddings e o índice FAISS na memória."""
    global embeddings_model, vectorstore, FAISS_OK, APP_READY
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

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

    res = {}
    status = "ok"
    try:
        res = answer_question(question, embeddings_model, vectorstore, debug=debug_flag)
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
        res = {"error": f"Ocorreu uma falha interna: {e}"}

    log = {"rid": rid, "status": status, "took_ms": int((time.time() - ts0) * 1000), "question": question[:400]}
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)

# --- NOVO ENDPOINT PARA O AGENTE ---
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

    # Reconstrói o histórico de mensagens a partir do payload da requisição.
    message_history = []
    for msg in data.get("messages", []):
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "user":
            message_history.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ia" or role == "ai":
            message_history.append(AIMessage(content=content))

    res = {}
    status = "ok"
    try:
        # Executa o agente, injetando os modelos e o histórico.
        res = run_agent(question, message_history, embeddings_model, vectorstore)
        METRICS["agent_queries_total"] += 1
        if res.get("action") == "PEDIR_INFO":
            METRICS["agent_clarification_needed"] += 1
        else:
            METRICS["agent_answered"] += 1
    except Exception as e:
        status = f"error:{type(e).__name__}"
        METRICS["errors_agent_internal"] += 1
        res = {"error": f"Ocorreu uma falha interna no agente: {e}"}
        # Loga a exceção para debug
        import traceback
        print(traceback.format_exc(), flush=True)

    log = {"rid": rid, "status": status, "took_ms": int((time.time() - ts0) * 1000), "question": question[:400]}
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)


if __name__ == "__main__":
    # A inicialização dos modelos é chamada antes de iniciar o servidor.
    _initialize_models()
    # Inicia o servidor de desenvolvimento do Flask.
    app.run(host="0.0.0.0", port=5000)
