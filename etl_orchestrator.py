# etl_orchestrator.py
# Este script orquestra o processo de ETL (Extract, Transform, Load) para construir e atualizar a base de conhecimento vetorial.
# Ele suporta tanto a reconstrução completa do índice quanto atualizações incrementais.

import os
import torch
import psycopg2
import sys
import hashlib
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loaders import pdf_loader, docx_loader, txt_loader, md_loader, code_loader

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÕES GLOBAIS ---
# Determina o dispositivo a ser usado (GPU se disponível, caso contrário CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Caminho para a pasta de dados dentro do container Docker
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
# Caminho onde o índice FAISS será salvo/carregado
VECTOR_STORE_PATH = "vector_store/faiss_index"

# Configurações do banco de dados PostgreSQL, obtidas de variáveis de ambiente
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# --- CORREÇÃO: Nome do modelo de embeddings alinhado com a API ---
# Usa a variável de ambiente EMBEDDINGS_MODEL para consistência,
# com um valor padrão igual ao recomendado na API e no README.
MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

# Mapeamento de extensões de arquivo para seus respectivos loaders
LOADER_MAPPING = {
    ".pdf": pdf_loader, ".docx": docx_loader, ".md": md_loader, ".txt": txt_loader,
    ".php": code_loader, ".sql": code_loader, ".json": code_loader, ".xml": code_loader,
    ".ini": code_loader, ".config": code_loader, ".example": code_loader,
    ".yml": code_loader, ".yaml": code_loader,
}


# --- FUNÇÕES AUXILIARES ---

def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados PostgreSQL."""
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)


def get_file_hash(file_path):
    """Calcula o hash SHA256 de um arquivo para detectar modificações.
    Isso é crucial para a funcionalidade de atualização incremental.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def setup_database():
    """Garante que as tabelas 'document_chunks' e 'processed_files' existam no banco de dados.
    Cria as tabelas se elas ainda não existirem.
    """
    print("INFO: Conectando ao banco de dados para configurar as tabelas...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Tabela para armazenar os pedaços de texto (chunks) e seus metadados
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                source_file VARCHAR(1024) NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_index BIGINT UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Tabela para rastrear arquivos processados e seus hashes, permitindo atualizações incrementais
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
        raise # Re-lança a exceção para indicar falha crítica


def process_and_embed_documents(docs_to_process, embeddings_model):
    """Carrega, divide documentos em chunks e gera embeddings para eles.
    Retorna os chunks divididos e o vetorstore FAISS resultante.
    """
    if not docs_to_process:
        return None, None

    print(f"INFO: {len(docs_to_process)} documentos serão processados.")
    # Inicializa o text splitter para dividir documentos grandes em pedaços menores (chunks)
    # Usa separadores para tentar manter a coesão dos chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Tamanho máximo de cada chunk
        chunk_overlap=200, # Sobreposição entre chunks para manter o contexto
        separators=["\n\n", "\n", " ", ""] # Separadores para divisão de texto
    )
    split_chunks = text_splitter.split_documents(docs_to_process)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")

    print("INFO: Gerando embeddings...")
    # Gera embeddings para os chunks e cria um vetorstore FAISS
    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    print("SUCCESS: Embeddings gerados com sucesso.")

    return split_chunks, vector_store


# --- LÓGICAS DE EXECUÇÃO PRINCIPAIS ---

def run_full_rebuild():
    """Executa um rebuild completo da base de conhecimento.
    Limpa todos os dados existentes (chunks e arquivos processados) e reconstrói o índice do zero.
    """
    print("\n--- INICIANDO REBUILD COMPLETO ---")

    # Limpa as tabelas no banco de dados para garantir um estado limpo
    conn = get_db_connection()
    cur = conn.cursor()
    print("INFO: Limpando dados antigos (tabelas document_chunks e processed_files)...")
    cur.execute("TRUNCATE TABLE document_chunks, processed_files RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()

    # Carrega todos os documentos da pasta DATA_PATH usando os loaders apropriados
    all_docs = []
    file_hashes = {}
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in LOADER_MAPPING:
                try:
                    # Carrega o documento e armazena seu hash
                    all_docs.extend(LOADER_MAPPING[file_ext].load(file_path))
                    file_hashes[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"ERROR: Falha ao carregar {file_path}: {e}")

    if not all_docs:
        print("WARNING: Nenhum documento encontrado para processar.")
        return

    # Inicializa o modelo de embeddings com o dispositivo selecionado (CPU/GPU)
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    # Processa os documentos e gera o vetorstore
    split_chunks, vector_store = process_and_embed_documents(all_docs, embeddings_model)

    if vector_store:
        # Salva o índice FAISS localmente
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"SUCCESS: Novo índice FAISS salvo em \'{VECTOR_STORE_PATH}\'")

        # Salva os metadados dos chunks e dos arquivos processados no PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()
        print("INFO: Salvando chunks e hashes de arquivos no PostgreSQL...")
        # Salva cada chunk individualmente
        for i, doc in enumerate(split_chunks):
            cur.execute(
                "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
                (doc.metadata.get('source', 'desconhecido'), doc.page_content, i)
            )
        # Salva o hash de cada arquivo processado
        for file_path, file_hash in file_hashes.items():
            cur.execute(
                "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s)",
                (file_path, file_hash)
            )
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Metadados salvos no PostgreSQL.")


def run_incremental_update():
    """Executa uma atualização incremental da base de conhecimento.
    Processa apenas arquivos novos ou modificados, mesclando-os ao índice existente.
    """
    print("\n--- INICIANDO ATUALIZAÇÃO INCREMENTAL ---")

    # Recupera os arquivos já processados e seus hashes do banco de dados
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT source_file, file_hash FROM processed_files")
    processed_files_db = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()

    docs_to_add = []
    files_to_update_in_db = {}

    # Identifica arquivos novos ou modificados na pasta DATA_PATH
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in LOADER_MAPPING:
                current_hash = get_file_hash(file_path)
                # Se o arquivo não foi processado antes ou seu hash mudou
                if file_path not in processed_files_db or processed_files_db[file_path] != current_hash:
                    print(f"INFO: Detectado arquivo novo/modificado: {file_path}")
                    try:
                        docs_to_add.extend(LOADER_MAPPING[file_ext].load(file_path))
                        files_to_update_in_db[file_path] = current_hash
                    except Exception as e:
                        print(f"ERROR: Falha ao carregar {file_path}: {e}")

    if not docs_to_add:
        print("SUCCESS: Nenhuma alteração detectada. A base de conhecimento já está atualizada.")
        return

    # Verifica se houve modificações em arquivos existentes. A remoção de vetores específicos do FAISS é complexa.
    # Por simplicidade e segurança, se um arquivo existente foi modificado, recomenda-se um rebuild completo.
    modified_files_exist = any(path in processed_files_db for path in files_to_update_in_db)
    if modified_files_exist:
        print("WARNING: Modificações em arquivos existentes foram detectadas.")
        print("Para garantir a consistência, a remoção de dados antigos é complexa.")
        print("Recomendação: Execute um rebuild completo com ./treinar_ia.sh para refletir as alterações.")
        return

    # Inicializa o modelo de embeddings com o dispositivo selecionado
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    # Processa os novos/modificados documentos e gera um novo vetorstore temporário
    split_chunks, new_vector_store = process_and_embed_documents(docs_to_add, embeddings_model)

    if new_vector_store:
        # Carrega o índice antigo e mescla com o novo
        print("INFO: Carregando índice FAISS existente para mesclagem...")
        if os.path.exists(VECTOR_STORE_PATH):
            existing_vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model,
                                                     allow_dangerous_deserialization=True)

            # Adiciona os novos documentos ao índice existente
            existing_vector_store.add_documents(split_chunks)
            existing_vector_store.save_local(VECTOR_STORE_PATH)

            total_vectors = existing_vector_store.index.ntotal
            print(f"SUCCESS: Índice FAISS atualizado e salvo. Total de vetores: {total_vectors}")

            # Atualiza o banco de dados com os novos chunks e hashes de arquivos
            conn = get_db_connection()
            cur = conn.cursor()
            # Obtém o último índice FAISS para garantir que os novos índices sejam únicos
            cur.execute("SELECT max(faiss_index) FROM document_chunks")
            last_index = cur.fetchone()[0] or -1

            print("INFO: Salvando novos chunks e hashes no PostgreSQL...")
            for i, doc in enumerate(split_chunks):
                cur.execute(
                    "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
                    (doc.metadata.get('source', 'desconhecido'), doc.page_content, last_index + 1 + i)
                )
            for file_path, file_hash in files_to_update_in_db.items():
                # Usa ON CONFLICT para atualizar o hash se o arquivo já existir, ou inserir se for novo
                cur.execute(
                    "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s) ON CONFLICT (source_file) DO UPDATE SET file_hash = EXCLUDED.file_hash, processed_at = CURRENT_TIMESTAMP",
                    (file_path, file_hash)
                )
            conn.commit()
            cur.close()
            conn.close()
            print("SUCCESS: Banco de dados atualizado.")

        else:
            print("WARNING: Nenhum índice FAISS existente encontrado. Executando um rebuild completo como fallback.")
            run_full_rebuild()


if __name__ == "__main__":
    # Configura o banco de dados ao iniciar o script
    setup_database()

    # Determina o modo de execução (rebuild completo ou atualização incremental) com base nos argumentos da linha de comando
    execution_mode = "rebuild"
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        execution_mode = "update"

    if execution_mode == "update":
        run_incremental_update()
    else:
        run_full_rebuild()

    print("\n--- PROCESSO DE ETL CONCLUÍDO ---")