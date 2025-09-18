# api.py
from __future__ import annotations
import os, time, uuid, json
from collections import Counter
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from query_handler import answer_question

# Tenta importar o cliente LLM de forma lazy. Se falhar, define como None.
try:
    from llm_client import _lazy_client as _llm_lazy_client
except Exception:
    _llm_lazy_client = None

# Variáveis de estado para verificar a prontidão da aplicação
APP_READY = False  # Indica se a aplicação está pronta para servir requisições
FAISS_OK = False   # Indica se o índice FAISS foi carregado com sucesso
LLM_OK = False     # Indica se o LLM está acessível e pronto
# Configura se a prontidão do LLM é um requisito para a aplicação estar pronta
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in ("1","true","yes")

# Configurações do ambiente, com valores padrão
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or os.getenv("EMBEDDINGS_MODEL_NAME") or "intfloat/multilingual-e5-large"

# Métricas simples para monitoramento
METRICS = Counter()  # Contador para armazenar métricas como erros e consultas
START_TS = time.time() # Timestamp de início da aplicação para cálculo de uptime

app = Flask(__name__) # Inicializa a aplicação Flask

# --- Funções de Probing para Verificação de Prontidão ---
def _probe_faiss(vs) -> bool:
    """Verifica se o vetorstore FAISS está carregado e contém documentos."""
    try:
        # Tenta acessar o dicionário de documentos do FAISS para verificar se há conteúdo
        all_docs = list(getattr(vs.docstore, "_dict", {}).values())
        return len(all_docs) > 0
    except Exception:
        return False

def _probe_llm() -> bool:
    """Verifica se o cliente LLM está disponível e configurado corretamente."""
    if _llm_lazy_client is None:
        return False
    try:
        # Tenta obter o provedor do LLM para confirmar sua inicialização
        provider, _ = _llm_lazy_client()
        return provider in ("google", "openai") # Suporta Google Gemini e OpenAI
    except Exception:
        return False

def _update_readiness(vs=None):
    """Atualiza o estado de prontidão da aplicação com base nos probes."""
    global APP_READY, FAISS_OK, LLM_OK
    if vs is not None:
        FAISS_OK = _probe_faiss(vs)
    LLM_OK = _probe_llm()
    # A aplicação está pronta se o FAISS estiver OK e, opcionalmente, o LLM também
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)

# --- Inicialização do Modelo de Embeddings e Vectorstore FAISS ---
embeddings_model = None
vectorstore = None

try:
    # Carrega o modelo de embeddings do HuggingFace
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    # Carrega o índice FAISS salvo localmente
    vectorstore = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True # Permite a desserialização de objetos pickle
    )
    print(f"[API] FAISS carregado: {FAISS_STORE_DIR}", flush=True)
except Exception as e:
    print(f"[API] Falha ao carregar FAISS em \'{FAISS_STORE_DIR}\': {e}", flush=True)

_update_readiness(vectorstore) # Atualiza o estado de prontidão após tentar carregar o FAISS

# --- Rotas da API Flask ---

@app.get("/")
def root():
    """Rota raiz para verificar se a API está respondendo."""
    return jsonify({"status": "ok"})

@app.get("/healthz")
def healthz():
    """Endpoint de health check para monitorar a prontidão da aplicação."""
    status = {
        "ready": bool(APP_READY), # Estado geral de prontidão
        "faiss": bool(FAISS_OK),   # Estado do FAISS
        "llm": bool(LLM_OK),     # Estado do LLM
        "require_llm_ready": REQUIRE_LLM_READY, # Se o LLM é obrigatório para prontidão
        "faiss_store_dir": FAISS_STORE_DIR,     # Diretório do FAISS
        "embeddings_model": EMBEDDINGS_MODEL,   # Modelo de embeddings usado
    }
    # Retorna 200 se pronto, 503 se não estiver pronto
    code = 200 if status["ready"] else 503
    return jsonify(status), code

@app.get("/metrics")
def metrics():
    """Endpoint para expor métricas simples da aplicação."""
    uptime = time.time() - START_TS # Calcula o tempo de atividade
    payload = {
        "uptime_sec": int(uptime), # Uptime em segundos
        "counters": dict(METRICS), # Contadores de métricas (erros, queries, etc.)
    }
    return jsonify(payload), 200

@app.post("/query")
def query():
    """Endpoint principal para processar consultas RAG."""
    rid = str(uuid.uuid4()) # Gera um ID de requisição único
    ts0 = time.time()       # Timestamp de início da requisição

    # Verifica se o vetorstore e o modelo de embeddings estão carregados
    if vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1 # Incrementa contador de erro
        return jsonify({
            "answer": "Índice não carregado. Rode o ETL para gerar o FAISS e tente novamente.",
            "citations": [],
            "context_found": False
        }), 503 # Retorna erro de serviço indisponível

    # Extrai a pergunta do corpo da requisição JSON
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1 # Incrementa contador de requisição inválida
        return jsonify({"error": "Informe \'question\' (ou \'query\'/\'q\') no JSON."}), 400 # Retorna erro de bad request

    res = {}     # Resultado da resposta
    status = "ok" # Status inicial da requisição
    try:
        # Chama a função principal para responder à pergunta usando RAG
        res = answer_question(question, embeddings_model, vectorstore)
        METRICS["queries_total"] += 1 # Incrementa contador de queries totais
        # Classifica o tipo de resposta para métricas
        if res.get("needs_clarification"):
            METRICS["queries_ambiguous"] += 1
        elif not res.get("context_found"):
            METRICS["queries_not_found"] += 1
        else:
            METRICS["queries_answered"] += 1
    except Exception as e:
        status = f"error:{type(e).__name__}" # Captura o tipo de erro
        METRICS["errors_internal"] += 1       # Incrementa contador de erros internos
        res = {"error": f"Falha ao processar pergunta: {e}"} # Mensagem de erro

    took_ms = int((time.time() - ts0) * 1000) # Tempo de processamento da requisição
    log = {
        "rid": rid,           # ID da requisição
        "status": status,     # Status final da requisição
        "took_ms": took_ms,   # Tempo de processamento
        "question": question[:400], # Pergunta (truncada para log)
        "ready": APP_READY,   # Estado de prontidão da aplicação
        "faiss": FAISS_OK,    # Estado do FAISS
        "llm": LLM_OK,        # Estado do LLM
    }
    print(json.dumps(log, ensure_ascii=False), flush=True) # Loga a requisição
    return jsonify(res) # Retorna a resposta JSON

if __name__ == "__main__":
    # Inicia o servidor Flask, acessível externamente na porta 5000
    app.run(host="0.0.0.0", port=5000)


