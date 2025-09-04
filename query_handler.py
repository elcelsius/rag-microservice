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

# --- CONFIGURAÇÕES ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# --- FUNÇÕES AUXILIARES QUE ESTAVAM FALTANDO ---
def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados Postgres."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

def generate_answer(context: str, query: str):
    """Usa o Gemini para gerar uma resposta baseada no contexto e na pergunta."""
    if not GOOGLE_API_KEY:
        return "ERRO: A chave de API do Google não foi configurada."
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Você é um assistente especialista no projeto. Responda à pergunta do usuário de forma clara, utilizando APENAS as informações do CONTEXTO. Se a resposta não estiver no contexto, diga que não encontrou a informação.

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
        # Adicionamos um print para vermos o erro detalhado no log da API
        print(f"ERRO DETALHADO DO GEMINI: {e}")
        return f"Ocorreu um erro ao chamar a API do Gemini: {e}"

# --- FUNÇÃO PRINCIPAL REUTILIZÁVEL ---
def find_answer_for_query(query_text: str, embeddings_model, index, top_k: int = 5):
    """Executa o pipeline RAG e RETORNA a resposta final."""
    query_embedding = embeddings_model.embed_query(query_text)
    query_vector = np.array([query_embedding], dtype=np.float32)

    distances, ids = index.search(query_vector, top_k)
    
    if not ids.any():
        return "Desculpe, não encontrei nenhum documento relevante para sua pergunta."
        
    faiss_indices = tuple(int(i) for i in ids[0])
    
    conn = get_db_connection() # Agora esta função existe e o NameError será resolvido
    cur = conn.cursor()
    query_sql = "SELECT source_file, chunk_text FROM document_chunks WHERE faiss_index IN %s"
    cur.execute(query_sql, (faiss_indices,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    context_text = "\n\n".join([f"Trecho do arquivo {row[0]}:\n{row[1]}" for row in results])
    final_answer = generate_answer(context_text, query_text)
    return final_answer