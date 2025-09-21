# api.py
from __future__ import annotations
import os, time, uuid, json
from collections import Counter
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from query_handler import answer_question

# Tenta importar o cliente LLM de forma lazy (preguiçosa). Se falhar, define como None.
# Isso evita que a aplicação quebre se a dependência do LLM não estiver configurada.
try:
    from llm_client import _lazy_client as _llm_lazy_client
except Exception:
    _llm_lazy_client = None

# --- Variáveis de Estado de Prontidão ---
# Estas variáveis controlam se a aplicação está pronta para receber tráfego.
APP_READY = False  # Indica se a aplicação como um todo está pronta.
FAISS_OK = False  # Indica se o índice vetorial FAISS foi carregado com sucesso.
LLM_OK = False  # Indica se o LLM está acessível e pronto para uso.
# Configura se a prontidão do LLM é um requisito obrigatório para a aplicação.
# O valor é lido da variável de ambiente REQUIRE_LLM_READY.
REQUIRE_LLM_READY = os.getenv("REQUIRE_LLM_READY", "false").lower() in ("1", "true", "yes")

# --- Configurações do Ambiente ---
# Carrega configurações a partir de variáveis de ambiente, com valores padrão de fallback.
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/app/vector_store/faiss_index")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL") or os.getenv(
    "EMBEDDINGS_MODEL_NAME") or "intfloat/multilingual-e5-large"

# --- Métricas Simples para Monitoramento ---
METRICS = Counter()  # Um contador para armazenar métricas simples, como número de erros e consultas.
START_TS = time.time()  # Timestamp de quando a aplicação iniciou, usado para calcular o uptime.

app = Flask(__name__)  # Inicializa a aplicação Flask.


# --- Rotas de Debug ---
# Endpoint para inspecionar dicionários carregados em memória (departamentos, sinônimos, etc).
# Útil para verificar se os termos foram carregados corretamente.
@app.get("/debug/dict")
def debug_dict():
    try:
        # Importa do módulo central para garantir que estamos vendo o que está em memória.
        from query_handler import DEPARTMENTS, ALIASES, SYNONYMS, BOOSTS
        return {
            "departments": DEPARTMENTS,  # Mapeamento de slug -> nome canônico
            "aliases_keys": list(ALIASES.keys()),
            "synonyms_keys": list(SYNONYMS.keys()),
            "boosts": BOOSTS,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


# Endpoint para inspecionar as variáveis de ambiente relevantes para a aplicação.
# Facilita o debug de configurações sem precisar acessar o servidor.
@app.get("/debug/env")
def debug_env():
    import os
    keys = [
        "TERMS_YAML",
        "FAISS_STORE_DIR",
        "EMBEDDINGS_MODEL",
        "RERANKER_ENABLED", "RERANKER_NAME",
        "MQ_ENABLED", "MQ_VARIANTS",
        "CONFIDENCE_MIN", "REQUIRE_CONTEXT",
    ]
    return {k: os.getenv(k) for k in keys}, 200


# --- Funções de Verificação de Prontidão (Probes) ---

def _probe_faiss(vs) -> bool:
    """Verifica se o vectorstore FAISS está carregado e funcional."""
    try:
        # A forma mais simples de verificar é ver se o docstore interno contém documentos.
        all_docs = list(getattr(vs.docstore, "_dict", {}).values())
        return len(all_docs) > 0
    except Exception:
        return False


def _probe_llm() -> bool:
    """Verifica se o cliente LLM está disponível e configurado."""
    if _llm_lazy_client is None:
        return False
    try:
        # Tenta invocar o cliente para confirmar que ele foi inicializado corretamente.
        provider, _ = _llm_lazy_client()
        return provider in ("google", "openai")  # Checa se é um dos provedores suportados.
    except Exception:
        return False


def _update_readiness(vs=None):
    """Atualiza o estado de prontidão global da aplicação com base nos probes."""
    global APP_READY, FAISS_OK, LLM_OK
    if vs is not None:
        FAISS_OK = _probe_faiss(vs)
    LLM_OK = _probe_llm()
    # A aplicação está pronta se o FAISS estiver OK. O LLM é opcional, a menos que exigido.
    APP_READY = FAISS_OK and (LLM_OK if REQUIRE_LLM_READY else True)


# --- Inicialização dos Modelos e Vectorstore ---
# Esta parte do código roda apenas uma vez, quando a aplicação é iniciada.
embeddings_model = None
vectorstore = None

try:
    # 1. Carrega o modelo de embeddings do HuggingFace que transforma texto em vetores.
    print(f"[API] Carregando modelo de embeddings: {EMBEDDINGS_MODEL}", flush=True)
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    # 2. Carrega o índice FAISS, que é a base de conhecimento vetorial.
    print(f"[API] Carregando FAISS de: {FAISS_STORE_DIR}", flush=True)
    vectorstore = FAISS.load_local(
        FAISS_STORE_DIR,
        embeddings_model,
        allow_dangerous_deserialization=True  # Necessário para carregar índices salvos com pickle.
    )
    print(f"[API] FAISS carregado com sucesso.", flush=True)
except Exception as e:
    print(f"[API] CRÍTICO: Falha ao carregar FAISS ou modelo de embeddings: {e}", flush=True)

_update_readiness(vectorstore)  # Atualiza o estado de prontidão após a tentativa de carregamento.


# --- Rotas da API Flask ---

@app.get("/")
def root():
    """Rota raiz, útil para um teste rápido de que a API está no ar."""
    return jsonify({"status": "ok"})


@app.get("/healthz")
def healthz():
    """Endpoint de health check (verificação de saúde), usado por orquestradores como Kubernetes."""
    status = {
        "ready": bool(APP_READY),  # O estado geral de prontidão.
        "faiss": bool(FAISS_OK),  # O estado específico do FAISS.
        "llm": bool(LLM_OK),  # O estado específico do LLM.
        "require_llm_ready": REQUIRE_LLM_READY,  # Informa se o LLM é obrigatório.
        "faiss_store_dir": FAISS_STORE_DIR,  # Diretório configurado para o FAISS.
        "embeddings_model": EMBEDDINGS_MODEL,  # Modelo de embeddings em uso.
    }
    # Retorna código 200 (OK) se estiver pronto, ou 503 (Service Unavailable) se não estiver.
    code = 200 if status["ready"] else 503
    return jsonify(status), code


@app.get("/metrics")
def metrics():
    """Endpoint para expor métricas simples, compatível com sistemas como o Prometheus."""
    uptime = time.time() - START_TS  # Calcula há quanto tempo a aplicação está rodando.
    payload = {
        "uptime_sec": int(uptime),
        "counters": dict(METRICS),  # Expõe os contadores de erros, queries, etc.
    }
    return jsonify(payload), 200


@app.post("/query")
def query():
    """Endpoint principal para processar as consultas do usuário via RAG."""
    rid = str(uuid.uuid4())  # Gera um ID de requisição único para rastreabilidade.
    ts0 = time.time()  # Marca o tempo de início do processamento.

    # Verifica se os componentes essenciais (vectorstore) estão carregados.
    # Se não estiverem, a API não pode funcionar e retorna um erro 503.
    if vectorstore is None or embeddings_model is None:
        METRICS["error_no_index"] += 1
        return jsonify({
            "answer": "O índice vetorial não está carregado. Execute o ETL para gerar os arquivos FAISS e reinicie a aplicação.",
            "citations": [],
            "context_found": False
        }), 503

    # Extrai a pergunta do corpo da requisição JSON.
    data = request.get_json(silent=True) or {}
    # Aceita 'question', 'query' ou 'q' como chaves para a pergunta.
    question = (data.get("question") or data.get("query") or data.get("q") or "").strip()
    if not question:
        METRICS["bad_request"] += 1
        return jsonify({"error": "O campo 'question' (ou 'query'/'q') é obrigatório no corpo do JSON."}), 400

    # NOVO: Aceita o parâmetro de debug tanto do corpo do JSON quanto da querystring da URL.
    # Ex: POST /query?debug=true ou POST /query com body {"question": "...", "debug": true}
    debug_flag = bool(data.get("debug")) or (request.args.get("debug", "").lower() == "true")

    res = {}  # Dicionário para armazenar o resultado.
    status = "ok"  # Status inicial da requisição.
    try:
        # Chama a função principal que executa a lógica de RAG (busca e geração).
        # NOVO: Repassa o flag de debug para o handler da consulta.
        res = answer_question(question, embeddings_model, vectorstore, debug=debug_flag)

        METRICS["queries_total"] += 1  # Incrementa o contador total de queries.
        # Classifica o resultado para métricas mais detalhadas.
        if res.get("needs_clarification"):
            METRICS["queries_ambiguous"] += 1
        elif not res.get("context_found"):
            METRICS["queries_not_found"] += 1
        else:
            METRICS["queries_answered"] += 1
    except Exception as e:
        status = f"error:{type(e).__name__}"  # Captura o tipo do erro para o log.
        METRICS["errors_internal"] += 1  # Incrementa o contador de erros internos.
        res = {"error": f"Ocorreu uma falha interna ao processar a pergunta: {e}"}

    took_ms = int((time.time() - ts0) * 1000)  # Calcula o tempo total de processamento.
    # Monta um log estruturado em JSON para facilitar a análise posterior.
    log = {
        "rid": rid,  # Request ID
        "status": status,  # Status final ('ok' ou 'error:...')
        "took_ms": took_ms,  # Tempo de processamento em milissegundos
        "question": question[:400],  # Pergunta (truncada para evitar logs muito longos)
        "ready": APP_READY,  # Estado de prontidão da aplicação no momento da requisição
        "faiss": FAISS_OK,
        "llm": LLM_OK,
    }
    # Imprime o log no console (stdout).
    print(json.dumps(log, ensure_ascii=False), flush=True)

    return jsonify(res)  # Retorna a resposta final como JSON.

@app.post("/api/ask")
def api_ask():
    # Alias compatível; reaproveita exatamente a mesma lógica de /query
    return query()


if __name__ == "__main__":
    # Inicia o servidor de desenvolvimento do Flask.
    # host="0.0.0.0" torna a aplicação acessível de fora do container/máquina.
    app.run(host="0.0.0.0", port=5000)