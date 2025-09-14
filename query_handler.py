# query_handler.py
import os
import torch
import psycopg2
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# --- CONFIGURAÇÕES E INICIALIZAÇÃO DE MODELOS ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
VECTOR_STORE_PATH = "vector_store/faiss_index"
# Carrega o threshold do .env, convertendo para float
FAISS_SCORE_THRESHOLD = float(os.getenv("FAISS_SCORE_THRESHOLD", "1.0"))

# Carregamento dos modelos na inicialização do módulo para reuso
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': DEVICE}
)
index = faiss.read_index(f"{VECTOR_STORE_PATH}/index.faiss")


# --- FUNÇÕES AUXILIARES ---
def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados Postgres."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)


def _formatar_citacoes(db_results: list, query: str) -> list[dict]:
    """Formata os resultados do banco de dados em uma lista de citações legível."""
    if not db_results:
        return []

    citations = []
    for row in db_results:
        source_file, chunk_text = row
        citations.append({
            "documento": source_file,
            "trecho": chunk_text.strip()
        })
    return citations


def generate_answer(context: str, query: str) -> str:
    """Usa o Gemini para gerar uma resposta baseada no contexto e na pergunta."""
    if not GOOGLE_API_KEY:
        return "ERRO: A chave de API do Google não foi configurada."

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
    Você é o IA Compilot, um assistente especialista na base de conhecimento da UNESP.
    Responda à pergunta do usuário de forma clara e objetiva, utilizando APENAS as informações do CONTEXTO fornecido.
    Se a resposta não estiver no contexto, diga educadamente que não encontrou a informação.

    CONTEXTO:
    ---
    {context}
    ---
    PERGUNTA DO USUÁRIO: {query}
    RESPOSTA:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"ERRO DETALHADO DO GEMINI: {e}")
        return f"Ocorreu um erro ao chamar a API do Gemini: {e}"


# --- FUNÇÃO PRINCIPAL DO RAG (COM THRESHOLD DE CONFIANÇA) ---
def find_answer_for_query(query_text: str, top_k: int = 5) -> dict:
    """
    Executa o pipeline RAG e RETORNA um dicionário com a resposta, citações e status.
    """
    # --- DEBUG AGENTE: Imprime a pergunta exata que o RAG recebeu ---
    print(f"\n\n\n--- DEBUG RAG: RECEBIDA PERGUNTA PARA BUSCA ---")
    print(f"Query: '{query_text}'")
    # --- FIM DEBUG ---

    query_embedding = embeddings_model.embed_query(query_text)
    query_vector = np.array([query_embedding], dtype=np.float32)

    try:
        distances, ids = index.search(query_vector, top_k)

        # --- DEBUG RAG: Mostra os resultados brutos da busca vetorial ---
        print("\n--- DEBUG RAG: PASSO 1 - BUSCA FAISS BRUTA ---")
        print(f"Distâncias (scores) encontradas: {distances[0]}")
        print(f"Índices (IDs) dos chunks encontrados: {ids[0]}")
        print("(Lembre-se: score mais baixo = mais relevante)")
        # --- FIM DEBUG ---

    except Exception as e:
        print(f"ERRO: Falha na busca do FAISS: {e}")
        return {
            "answer": "Desculpe, estou com um problema técnico para acessar minha base de conhecimento.",
            "citations": [],
            "context_found": False
        }

    faiss_indices = []
    if ids.any() and ids[0][0] != -1:
        filtered_results = [
            (int(i), d) for i, d in zip(ids[0], distances[0])
            if i != -1 and d < FAISS_SCORE_THRESHOLD
        ]

        # --- DEBUG RAG: Mostra o que passou pelo filtro de confiança ---
        print("\n--- DEBUG RAG: PASSO 2 - FILTRO DE THRESHOLD ---")
        print(f"Threshold de score definido no .env: {FAISS_SCORE_THRESHOLD}")
        print(f"Resultados que passaram no filtro (índice, score): {filtered_results}")
        # --- FIM DEBUG ---

        if filtered_results:
            faiss_indices = tuple(res[0] for res in filtered_results)

    if not faiss_indices:
        # --- DEBUG RAG: Informa que a busca não retornou nada após o filtro ---
        print(
            "\n--- DEBUG RAG: Nenhum documento relevante encontrado após filtro. O contexto para o LLM estará VAZIO. ---")
        # --- FIM DEBUG ---
        print(
            f"--- RAG: Nenhum documento relevante encontrado após filtro de score (threshold: {FAISS_SCORE_THRESHOLD}) ---")
        return {
            "answer": "Não encontrei informações suficientemente relevantes sobre este tópico. Você poderia reformular a pergunta?",
            "citations": [],
            "context_found": False
        }

    print(f"\n--- DEBUG RAG: {len(faiss_indices)} documento(s) recuperado(s) do banco de dados. ---")

    conn = get_db_connection()
    cur = conn.cursor()
    query_sql = "SELECT source_file, chunk_text FROM document_chunks WHERE faiss_index IN %s"
    cur.execute(query_sql, (faiss_indices,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    context_text = "\n\n".join([f"Trecho do arquivo {row[0]}:\n{row[1]}" for row in results])

    # --- DEBUG RAG: Mostra o contexto exato que será enviado para o Gemini ---
    print("\n--- DEBUG RAG: PASSO 3 - CONTEXTO FINAL ENVIADO AO LLM ---")
    print("===================== INÍCIO DO CONTEXTO =====================")
    print(context_text)
    print("====================== FIM DO CONTEXTO =======================")
    # --- FIM DEBUG ---

    final_answer = generate_answer(context_text, query_text)
    citations = _formatar_citacoes(results, query_text)

    return {
        "answer": final_answer,
        "citations": citations,
        "context_found": True
    }