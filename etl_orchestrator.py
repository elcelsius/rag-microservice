# etl_orchestrator.py
# Orquestra o processo de ETL para construir e atualizar a base de conhecimento vetorial.

import os
import sys
import hashlib

import psycopg2
import redis
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from loaders import code_loader, docx_loader, md_loader, pdf_loader, txt_loader

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
VECTOR_STORE_PATH = "vector_store/faiss_index"

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

LOADER_MAPPING = {
    ".pdf": pdf_loader,
    ".docx": docx_loader,
    ".md": md_loader,
    ".txt": txt_loader,
    ".php": code_loader,
    ".sql": code_loader,
    ".json": code_loader,
    ".xml": code_loader,
    ".ini": code_loader,
    ".config": code_loader,
    ".example": code_loader,
    ".yml": code_loader,
    ".yaml": code_loader,
}


def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def get_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handler:
        hasher.update(handler.read())
    return hasher.hexdigest()


def setup_database() -> None:
    print("INFO: Conectando ao banco de dados para configurar as tabelas...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(1024) NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_index BIGINT UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_files (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(1024) UNIQUE NOT NULL,
                file_hash VARCHAR(64) NOT NULL,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Tabelas verificadas/criadas com sucesso.")
    except Exception as exc:
        print(f"ERROR: NÃ£o foi possÃ­vel configurar o banco de dados: {exc}")
        raise


def clear_rag_cache() -> None:
    print("INFO: Tentando limpar o cache do RAG no Redis...")
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        keys = [key.decode("utf-8") for key in client.scan_iter("rag_query:*")]
        if keys:
            deleted = client.delete(*keys)
            print(f"SUCCESS: {deleted} chaves de cache removidas do Redis.")
        else:
            print("INFO: Nenhuma chave de cache do RAG encontrada para remoÃ§Ã£o.")
    except Exception as exc:
        print(f"WARNING: NÃ£o foi possÃ­vel limpar o cache do Redis: {exc}")


def process_and_embed_documents(docs_to_process, embeddings_model):
    if not docs_to_process:
        return None, None

    print(f"INFO: {len(docs_to_process)} documentos serÃ£o processados.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    split_chunks = splitter.split_documents(docs_to_process)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")

    print("INFO: Gerando embeddings...")
    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    print("SUCCESS: Embeddings gerados com sucesso.")
    return split_chunks, vector_store


def _load_documents_and_hashes():
    documents = []
    hashes = {}
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            extension = os.path.splitext(filename)[1].lower()
            if extension in LOADER_MAPPING:
                try:
                    documents.extend(LOADER_MAPPING[extension].load(file_path))
                    hashes[file_path] = get_file_hash(file_path)
                except Exception as exc:
                    print(f"ERROR: Falha ao carregar {file_path}: {exc}")
    return documents, hashes


def run_full_rebuild() -> None:
    print("\n--- INICIANDO REBUILD COMPLETO ---")

    conn = get_db_connection()
    cur = conn.cursor()
    print("INFO: Limpando dados antigos (document_chunks e processed_files)...")
    cur.execute("TRUNCATE TABLE document_chunks, processed_files RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()

    all_docs, file_hashes = _load_documents_and_hashes()
    if not all_docs:
        print("WARNING: Nenhum documento encontrado para processar.")
        return

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": DEVICE})
    split_chunks, vector_store = process_and_embed_documents(all_docs, embeddings_model)
    if vector_store is None:
        print("ERROR: Falha na geraÃ§Ã£o de embeddings.")
        return

    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"SUCCESS: Novo Ã­ndice FAISS salvo em '{VECTOR_STORE_PATH}'")

    conn = get_db_connection()
    cur = conn.cursor()
    print("INFO: Salvando chunks e hashes de arquivos no PostgreSQL...")
    for idx, doc in enumerate(split_chunks):
        cur.execute(
            "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
            (doc.metadata.get("source", "desconhecido"), doc.page_content, idx),
        )
    for file_path, file_hash in file_hashes.items():
        cur.execute(
            "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s)",
            (file_path, file_hash),
        )
    conn.commit()
    cur.close()
    conn.close()
    print("SUCCESS: Metadados atualizados.")

    clear_rag_cache()


def run_incremental_update() -> None:
    print("\n--- INICIANDO ATUALIZAÃ‡ÃƒO INCREMENTAL ---")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT source_file, file_hash FROM processed_files")
    processed_files_db = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()

    docs_to_add = []
    files_to_update = {}

    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            extension = os.path.splitext(filename)[1].lower()
            if extension in LOADER_MAPPING:
                current_hash = get_file_hash(file_path)
                if file_path not in processed_files_db or processed_files_db[file_path] != current_hash:
                    print(f"INFO: Detectado arquivo novo/modificado: {file_path}")
                    try:
                        docs_to_add.extend(LOADER_MAPPING[extension].load(file_path))
                        files_to_update[file_path] = current_hash
                    except Exception as exc:
                        print(f"ERROR: Falha ao carregar {file_path}: {exc}")

    if not docs_to_add:
        print("SUCCESS: Nenhuma alteraÃ§Ã£o detectada. A base jÃ¡ estÃ¡ atualizada.")
        return

    modified_existing = any(path in processed_files_db for path in files_to_update)
    if modified_existing:
        print("WARNING: ModificaÃ§Ãµes em arquivos jÃ¡ indexados foram detectadas.")
        print("Para garantir consistÃªncia, execute um rebuild completo (ex.: python etl_orchestrator.py --rebuild ou ./treinar_ia.sh).")
        return

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": DEVICE})
    split_chunks, new_vector_store = process_and_embed_documents(docs_to_add, embeddings_model)
    if new_vector_store is None:
        print("ERROR: Falha na geraÃ§Ã£o de embeddings incrementais.")
        return

    if os.path.exists(VECTOR_STORE_PATH):
        existing_vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
        existing_vector_store.add_documents(split_chunks)
        existing_vector_store.save_local(VECTOR_STORE_PATH)
        print(f"SUCCESS: Ãndice FAISS atualizado. Total de vetores: {existing_vector_store.index.ntotal}")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT max(faiss_index) FROM document_chunks")
        last_index = cur.fetchone()[0] or -1
        print("INFO: Gravando novos chunks e hashes no PostgreSQL...")
        for offset, doc in enumerate(split_chunks, start=1):
            cur.execute(
                "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
                (
                    doc.metadata.get("source", "desconhecido"),
                    doc.page_content,
                    last_index + offset,
                ),
            )
        for file_path, file_hash in files_to_update.items():
            cur.execute(
                """
                INSERT INTO processed_files (source_file, file_hash)
                VALUES (%s, %s)
                ON CONFLICT (source_file)
                DO UPDATE SET file_hash = EXCLUDED.file_hash, processed_at = CURRENT_TIMESTAMP
                """,
                (file_path, file_hash),
            )
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Banco de dados atualizado com novos documentos.")
        clear_rag_cache()
    else:
        print("WARNING: Ãndice FAISS inexistente. Executando rebuild completo como fallback.")
        run_full_rebuild()


if __name__ == "__main__":
    setup_database()

    mode = "rebuild"
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        mode = "update"

    if mode == "update":
        run_incremental_update()
    else:
        run_full_rebuild()

    print("\n--- PROCESSO DE ETL CONCLUÃDO ---")
