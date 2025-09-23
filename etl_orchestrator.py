# etl_orchestrator.py
# Este script orquestra o processo de ETL (Extract, Transform, Load) para construir e atualizar a base de conhecimento vetorial.
# Ele suporta tanto a reconstrução completa do índice quanto atualizações incrementais.

import os
import torch
import psycopg2
import psycopg2.extras
import sys
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# Importa o módulo de loaders unificado
import loaders

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÕES GLOBAIS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
VECTOR_STORE_PATH = "vector_store/faiss_index"
BATCH_SIZE = 32

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

# Lista de extensões suportadas pelo loader unificado. Usada para otimizar a busca de arquivos.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".md", ".txt", ".php", ".sql", ".json", ".xml",
    ".ini", ".config", ".example", ".yml", ".yaml", ".csv"
}

# --- FUNÇÕES DE BANCO DE DADOS E ARQUIVOS ---

def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados PostgreSQL."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

def get_file_hash(file_path):
    """Calcula o hash SHA256 de um arquivo para detectar modificações."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def setup_database():
    """Garante que as tabelas necessárias existam no banco de dados."""
    print("INFO: Conectando ao banco de dados para configurar as tabelas...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
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
        print("SUCCESS: Tabelas verificadas/criadas com sucesso.")
    except Exception as e:
        print(f"ERROR: Não foi possível configurar o banco de dados: {e}")
        raise

def get_processed_files_from_db(conn) -> Dict[str, str]:
    """Recupera os arquivos já processados e seus hashes do banco de dados."""
    with conn.cursor() as cur:
        cur.execute("SELECT source_file, file_hash FROM processed_files")
        return {row[0]: row[1] for row in cur.fetchall()}

def get_files_on_disk(path: str) -> Dict[str, str]:
    """Mapeia todos os arquivos suportados no disco e seus hashes."""
    disk_files = {}
    for root, _, files in os.walk(path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
                try:
                    disk_files[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"WARN: Não foi possível calcular o hash de {file_path}: {e}")
    return disk_files

# --- FUNÇÕES DO PIPELINE DE ETL ---

def process_and_embed_documents(docs_to_process: List[Document], embeddings_model) -> Tuple[List[Document], FAISS]:
    """Divide documentos em chunks e gera embeddings para eles."""
    if not docs_to_process:
        return [], None

    print(f"INFO: {len(docs_to_process)} documentos serão divididos e embutidos.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    split_chunks = text_splitter.split_documents(docs_to_process)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")

    if not split_chunks:
        return [], None

    print("INFO: Gerando embeddings...")
    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    print("SUCCESS: Embeddings gerados com sucesso.")
    return split_chunks, vector_store

def remove_docs_from_store(files_to_remove: Set[str], vector_store: FAISS, conn):
    """Remove os dados de arquivos específicos do índice FAISS e do banco de dados."""
    if not files_to_remove:
        return

    print(f"INFO: Removendo {len(files_to_remove)} arquivo(s) obsoleto(s)/modificado(s)...")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT faiss_index FROM document_chunks WHERE source_file = ANY(%s)",
            (list(files_to_remove),)
        )
        faiss_indices_to_remove = [row[0] for row in cur.fetchall() if row[0] is not None]

        if faiss_indices_to_remove:
            # A remoção de vetores do FAISS pode ser complexa. A biblioteca FAISS não suporta remoção direta por ID de forma eficiente em todos os cenários.
            # A abordagem mais segura para consistência total é um rebuild. No entanto, para muitos casos, a remoção funciona.
            try:
                removed_count = vector_store.delete(faiss_indices_to_remove)
                if removed_count:
                     print(f"INFO: Removidos {len(faiss_indices_to_remove)} vetores do índice FAISS.")
            except Exception as e:
                print(f"WARN: Falha ao remover vetores do FAISS: {e}. Recomenda-se um rebuild para consistência total.")

        cur.execute("DELETE FROM document_chunks WHERE source_file = ANY(%s)", (list(files_to_remove),))
        cur.execute("DELETE FROM processed_files WHERE source_file = ANY(%s)", (list(files_to_remove),))
        conn.commit()
    print(f"SUCCESS: Entradas de {len(files_to_remove)} arquivo(s) removidas do banco de dados.")

def add_docs_to_store(files_to_add: Set[str], vector_store: FAISS, conn, embeddings_model):
    """Processa e adiciona novos documentos ao índice FAISS e ao banco de dados."""
    if not files_to_add:
        return

    print(f"INFO: Adicionando {len(files_to_add)} arquivo(s) novo(s)/modificado(s)...")
    
    docs_to_process = []
    file_hashes = {}
    for file_path in files_to_add:
        # A função unificada load_document lida com os diferentes tipos de arquivo e seus erros.
        loaded_docs = loaders.load_document(file_path)
        if loaded_docs:
            docs_to_process.extend(loaded_docs)
            try:
                file_hashes[file_path] = get_file_hash(file_path)
            except Exception as e:
                 print(f"WARN: Não foi possível calcular o hash de {file_path} após o carregamento: {e}")

    if not docs_to_process:
        return

    split_chunks, new_vector_store = process_and_embed_documents(docs_to_process, embeddings_model)

    if not new_vector_store:
        return

    vector_store.merge_from(new_vector_store)
    print(f"INFO: {len(split_chunks)} novos vetores mesclados ao índice FAISS.")

    with conn.cursor() as cur:
        cur.execute("SELECT max(id) FROM document_chunks")
        last_id = (cur.fetchone()[0] or 0)

        chunk_data = [
            (doc.metadata.get('source', 'desconhecido'), doc.page_content, last_id + 1 + i)
            for i, doc in enumerate(split_chunks)
        ]
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
            chunk_data
        )

        file_data = list(file_hashes.items())
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s) ON CONFLICT (source_file) DO UPDATE SET file_hash = EXCLUDED.file_hash, processed_at = CURRENT_TIMESTAMP",
            file_data
        )
        conn.commit()
    print("SUCCESS: Banco de dados atualizado com os novos metadados.")

# --- LÓGICAS DE EXECUÇÃO PRINCIPAIS (REFATORADAS) ---

def run_full_rebuild():
    """Executa um rebuild completo da base de conhecimento de forma eficiente em memória."""
    print("\n--- INICIANDO REBUILD COMPLETO ---")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            print("INFO: Limpando dados antigos (tabelas document_chunks e processed_files)...")
            cur.execute("TRUNCATE TABLE document_chunks, processed_files RESTART IDENTITY;")
            conn.commit()

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    
    vector_store = None
    all_files_on_disk = list(get_files_on_disk(DATA_PATH).keys())
    
    for i in range(0, len(all_files_on_disk), BATCH_SIZE):
        batch_paths = all_files_on_disk[i:i+BATCH_SIZE]
        print(f"\n--- Processando lote {i//BATCH_SIZE + 1}/{(len(all_files_on_disk) + BATCH_SIZE - 1)//BATCH_SIZE} ---")
        
        docs_to_process = []
        file_hashes = {}
        for file_path in batch_paths:
            loaded_docs = loaders.load_document(file_path)
            if loaded_docs:
                docs_to_process.extend(loaded_docs)
                try:
                    file_hashes[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"WARN: Não foi possível calcular o hash de {file_path} após o carregamento: {e}")
        
        if not docs_to_process:
            continue

        split_chunks, batch_vector_store = process_and_embed_documents(docs_to_process, embeddings_model)

        if not batch_vector_store:
            continue
            
        if vector_store is None:
            vector_store = batch_vector_store
        else:
            vector_store.merge_from(batch_vector_store)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                chunk_data = [(doc.metadata.get('source', 'desconhecido'), doc.page_content) for doc in split_chunks]
                psycopg2.extras.execute_batch(cur, "INSERT INTO document_chunks (source_file, chunk_text) VALUES (%s, %s)", chunk_data)
                
                file_data = list(file_hashes.items())
                psycopg2.extras.execute_batch(cur, "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s)", file_data)
                conn.commit()

    if vector_store:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, source_file, chunk_text FROM document_chunks ORDER BY id")
                db_chunks = cur.fetchall()
                update_data = [(i, db_chunks[i][0]) for i in range(len(db_chunks))]
                psycopg2.extras.execute_batch(cur, "UPDATE document_chunks SET faiss_index = %s WHERE id = %s", update_data)
                conn.commit()

        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"\nSUCCESS: Novo índice FAISS salvo em '{VECTOR_STORE_PATH}' com {vector_store.index.ntotal} vetores.")
    else:
        print("WARNING: Nenhum documento foi processado. O índice não foi criado.")

def run_incremental_update():
    """Executa uma atualização incremental real, lidando com arquivos novos, modificados e excluídos."""
    print("\n--- INICIANDO ATUALIZAÇÃO INCREMENTAL ---")

    if not os.path.exists(VECTOR_STORE_PATH):
        print("WARNING: Nenhum índice FAISS existente encontrado. Executando um rebuild completo como fallback.")
        run_full_rebuild()
        return

    with get_db_connection() as conn:
        processed_files_db = get_processed_files_from_db(conn)
        files_on_disk = get_files_on_disk(DATA_PATH)

        disk_path_set = set(files_on_disk.keys())
        db_path_set = set(processed_files_db.keys())

        new_files = disk_path_set - db_path_set
        deleted_files = db_path_set - disk_path_set
        
        potential_modified = disk_path_set.intersection(db_path_set)
        modified_files = {path for path in potential_modified if processed_files_db[path] != files_on_disk[path]}

        if not new_files and not deleted_files and not modified_files:
            print("SUCCESS: Nenhuma alteração detectada. A base de conhecimento já está atualizada.")
            return

        print(f"INFO: Detectado: {len(new_files)} novo(s), {len(modified_files)} modificado(s), {len(deleted_files)} excluído(s).")

        embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)

        files_to_remove = modified_files.union(deleted_files)
        if files_to_remove:
            remove_docs_from_store(files_to_remove, vector_store, conn)

        files_to_add = new_files.union(modified_files)
        if files_to_add:
            add_docs_to_store(files_to_add, vector_store, conn, embeddings_model)

        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"\nSUCCESS: Índice FAISS atualizado e salvo. Total de vetores: {vector_store.index.ntotal}")


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
