import os
from typing import Any, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from query_handler import answer_question
from triage import run_triage

load_dotenv()

app = Flask(__name__)
CORS(app)

EMBEDDINGS_MODEL_NAME = os.getenv(
    "EMBEDDINGS_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)
VECTOR_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")

# Inicializa embeddings (CPU por padrão)
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDINGS_MODEL_NAME,
    # sem model_kwargs={"device": ...} -> evita dependência de torch no import
)

vectorstore = None
_vector_error = None

# Espera arquivos do índice existirem (compose já espera, mas deixamos redundante e tolerante)
def _wait_index(path: str, tries: int = 60, sleep_s: float = 1.0) -> bool:
    import time
    for _ in range(tries):
        if os.path.isfile(os.path.join(path, "index.faiss")) and os.path.isfile(os.path.join(path, "index.pkl")):
            return True
        time.sleep(sleep_s)
    return False

if _wait_index(VECTOR_DIR, tries=10, sleep_s=1.0):
    try:
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        _vector_error = f"Falha ao carregar índice em '{VECTOR_DIR}': {e}"
else:
    _vector_error = f"Arquivos do índice não encontrados em '{VECTOR_DIR}'."

def _get_question_from_request(req) -> str:
    payload: Dict[str, Any] = {}
    try:
        payload = req.get_json(silent=True) or {}
    except Exception:
        payload = {}
    q = payload.get("question") or payload.get("query") or payload.get("q") or ""
    return (q or "").strip()

def _error(msg: str, code: int = 400):
    return jsonify({"error": msg}), code

def _handle_question(question: str):
    if _vector_error:
        # Resposta degradada, mas o container fica de pé e o healthcheck passa
        return jsonify({
            "answer": f"Serviço está no ar, porém o índice não foi carregado. Detalhe: {_vector_error}",
            "citations": [],
            "context_found": False
        }), 200

    triage = run_triage(question)
    if triage.get("action") == "ask_clarification":
        return jsonify({
            "answer": triage.get("message", "Poderia detalhar melhor a sua pergunta?"),
            "citations": [],
            "context_found": False,
            "needs_clarification": True
        }), 200

    result = answer_question(question, embeddings_model, vectorstore)
    return jsonify(result), 200

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = _get_question_from_request(request)
        return _handle_question(question)
    except Exception as e:
        print(f"[ERROR] /ask -> {e}")
        return _error(f"Falha ao processar a pergunta: {e}", 500)

@app.route("/query", methods=["GET", "POST"])
def query():
    try:
        if request.method == "GET":
            q = request.args.get("q") or request.args.get("query") or request.args.get("question") or ""
        else:
            q = _get_question_from_request(request)
        return _handle_question(q)
    except Exception as e:
        print(f"[ERROR] /query -> {e}")
        return _error(f"Falha ao processar a pergunta: {e}", 500)

@app.route("/", methods=["GET"])
def health():
    status = {"status": "ok"}
    if _vector_error:
        status["warning"] = _vector_error
    return jsonify(status), 200

if __name__ == "__main__":
    # bind 0.0.0.0 para aceitar chamadas de outros containers
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
