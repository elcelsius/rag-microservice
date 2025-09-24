# etl_orchestrator.py
# Este script orquestra o processo de ETL para construir e atualizar a base de conhecimento vetorial.

import os
import torch
import psycopg2
import sys
import hashlib
import redis
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loaders import pdf_loader, docx_loader, txt_loader, md_loader, code_loader

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÕES GLOBAIS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
VECTOR_STORE_PATH = "vector_store/faiss_index"

# Configurações do banco de dados PostgreSQL
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# --- Configurações do Cache Redis ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

LOADER_MAPPING = {
    ".pdf": pdf_loader, ".docx": docx_loader, ".md": md_loader, ".txt": txt_loader,
    ".php": code_loader, ".sql": code_loader, ".json": code_loader, ".xml": code_loader,
    ".ini": code_loader, ".config": code_loader, ".example": code_loader,
    ".yml": code_loader, ".yaml": code_loader,
}


# --- FUNÇÕES AUXILIARES ---
def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def setup_database():
    print("INFO: Conectando ao banco de dados para configurar as tabelas...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(1024) NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_index BIGINT UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(1024) UNIQUE NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Tabelas verificadas/criadas com sucesso.")
    except Exception as e:
        print(f"ERROR: Não foi possível configurar o banco de dados: {e}")
        raise

def clear_rag_cache():
    """Conecta ao Redis e limpa todas as chaves de cache do RAG."""
    print("INFO: Tentando limpar o cache do RAG no Redis...")
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        keys_to_delete = [key.decode('utf-8') for key in redis_client.scan_iter("rag_query:*")]
        if keys_to_delete:
            deleted_count = redis_client.delete(*keys_to_delete)
            print(f"SUCCESS: {deleted_count} chaves de cache removidas do Redis.")
        else:
            print("INFO: Nenhuma chave de cache para limpar foi encontrada no Redis.")
    except Exception as e:
        print(f"WARNING: Não foi possível limpar o cache do Redis: {e}")

def process_and_embed_documents(docs_to_process, embeddings_model):
    if not docs_to_process: return None, None
    print(f"INFO: {len(docs_to_process)} documentos serão processados.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
    split_chunks = text_splitter.split_documents(docs_to_process)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")
    print("INFO: Gerando embeddings...")
    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    print("SUCCESS: Embeddings gerados com sucesso.")
    return split_chunks, vector_store


# --- LÓGICAS DE EXECUÇÃO PRINCIPAIS ---
def run_full_rebuild():
    print("\n--- INICIANDO REBUILD COMPLETO ---")
    conn = get_db_connection()
    cur = conn.cursor()
    print("INFO: Limpando dados antigos (tabelas document_chunks e processed_files)...")
    cur.execute("TRUNCATE TABLE document_chunks, processed_files RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()

    all_docs, file_hashes = [], {}
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in LOADER_MAPPING:
                try:
                    all_docs.extend(LOADER_MAPPING[file_ext].load(file_path))
                    file_hashes[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"ERROR: Falha ao carregar {file_path}: {e}")

    if not all_docs: return print("WARNING: Nenhum documento encontrado para processar.")

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    split_chunks, vector_store = process_and_embed_documents(all_docs, embeddings_model)

    if vector_store:
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"SUCCESS: Novo índice FAISS salvo em '{VECTOR_STORE_PATH}'")
        conn = get_db_connection()
        cur = conn.cursor()
        print("INFO: Salvando chunks e hashes de arquivos no PostgreSQL...")
        for i, doc in enumerate(split_chunks):
            cur.execute("INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)", (doc.metadata.get('source', 'desconhecido'), doc.page_content, i))
        for file_path, file_hash in file_hashes.items():
            cur.execute("INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s)", (file_path, file_hash))
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Metadados salvos no PostgreSQL.")
        clear_rag_cache() # Limpa o cache após o rebuild

def run_incremental_update():
    print("\n--- INICIANDO ATUALIZAÇÃO INCREMENTAL ---")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT source_file, file_hash FROM processed_files")
    processed_files_db = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()

    docs_to_add, files_to_update_in_db = [], {}
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in LOADER_MAPPING:
                current_hash = get_file_hash(file_path)
                if file_path not in processed_files_db or processed_files_db[file_path] != current_hash:
                    print(f"INFO: Detectado arquivo novo/modificado: {file_path}")
                    try:
                        docs_to_add.extend(LOADER_MAPPING[file_ext].load(file_path))
                        files_to_update_in_db[file_path] = current_hash
                    except Exception as e:
                        print(f"ERROR: Falha ao carregar {file_path}: {e}")

    if not docs_to_add: return print("SUCCESS: Nenhuma alteração detectada. A base de conhecimento já está atualizada.")

    modified_files_exist = any(path in processed_files_db for path in files_to_update_in_db)
    if modified_files_exist:
        print("WARNING: Modificações em arquivos existentes foram detectadas.")
        print("Recomendação: Execute um rebuild completo para garantir a consistência.")
        return

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    split_chunks, new_vector_store = process_and_embed_documents(docs_to_add, embeddings_model)

    if new_vector_store:
        if os.path.exists(VECTOR_STORE_PATH):
            existing_vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
            existing_vector_store.add_documents(split_chunks)
            existing_vector_store.save_local(VECTOR_STORE_PATH)
            print(f"SUCCESS: Índice FAISS atualizado e salvo. Total de vetores: {existing_vector_store.index.ntotal}")

            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT max(faiss_index) FROM document_chunks")
            last_index = cur.fetchone()[0] or -1
            print("INFO: Salvando novos chunks e hashes no PostgreSQL...")
            for i, doc in enumerate(split_chunks):
                cur.execute("INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)", (doc.metadata.get('source', 'desconhecido'), doc.page_content, last_index + 1 + i))
            for file_path, file_hash in files_to_update_in_db.items():
                cur.execute("INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s) ON CONFLICT (source_file) DO UPDATE SET file_hash = EXCLUDED.file_hash, processed_at = CURRENT_TIMESTAMP", (file_path, file_hash))
            conn.commit()
            cur.close()
            conn.close()
            print("SUCCESS: Banco de dados atualizado.")
            clear_rag_cache() # Limpa o cache após a atualização
        else:
            print("WARNING: Nenhum índice FAISS existente encontrado. Executando um rebuild completo como fallback.")
            run_full_rebuild()


if __name__ == "__main__":
    setup_database()
    execution_mode = "rebuild"
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        execution_mode = "update"

    if execution_mode == "update":
        run_incremental_update()
    else:
        run_full_rebuild()

    print("\n--- PROCESSO DE ETL CONCLUÍDO ---")
