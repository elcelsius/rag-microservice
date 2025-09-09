# app/etl_orchestrator.py (COM LIMPEZA DE DADOS)

import os
import torch
import psycopg2
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from loaders import pdf_loader, docx_loader, txt_loader, md_loader, code_loader

load_dotenv()

# --- LÓGICA DE DETECÇÃO DE GPU ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Usando dispositivo: {DEVICE} ---")

# --- CONFIGURAções GERAIS ---
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
VECTOR_STORE_PATH = "vector_store/faiss_index"
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

LOADER_MAPPING = {
    ".pdf": pdf_loader, ".docx": docx_loader, ".md": md_loader, ".txt": txt_loader,
    ".php": code_loader, ".sql": code_loader, ".json": code_loader, ".xml": code_loader,
    ".ini": code_loader, ".config": code_loader, ".example": code_loader,
    ".yml": code_loader, ".yaml": code_loader,
}

def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

def setup_database():
    print("INFO: Conectando ao banco de dados para configurar a tabela...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY, source_file VARCHAR(512) NOT NULL,
                chunk_text TEXT NOT NULL, faiss_index INTEGER UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("SUCCESS: Tabela 'document_chunks' verificada/criada com sucesso.")
    except Exception as e:
        print(f"ERROR: Não foi possível configurar o banco de dados: {e}")
        raise

# --- NOVA FUNÇÃO DE LIMPEZA ---
def clean_document_content(docs):
    cleaned_docs = []
    # Padrões de texto de ruído para remover (use regex para mais flexibilidade)
    patterns_to_remove = [
        r"Carregando\.\.\.",
        r"Navegar para",
        r"Acesso rápido",
        r"Ir para o item anterior",
        r"Ir par ao próximo item",
        r"Ir ao item número \d+",
        r"PÁGINA ATUAL:.*",
        r"Plugin de acessibilidade da Hand Talk",
        r"Acessível em Libras",
        r"Recursos Assistivos",
        r"FUNDAÇÕES",
        r"GOVERNO",
        r"SISTEMAS",
        r"SAÚDE",
        r"Editora Unesp",
        r"Fundunesp",
        r"Fundação Vunesp",
        r"Governo de São Paulo",
        r"Conselho de Reitores",
        r"Transparência Unesp",
        # Remove linhas com apenas uma palavra curta (geralmente links de menu)
        r"^\s*\b\w{1,10}\b\s*$",
    ]
    
    # Compila os padrões regex para eficiência
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns_to_remove]

    for doc in docs:
        content = doc.page_content
        # Aplica cada padrão de remoção
        for pattern in compiled_patterns:
            content = pattern.sub("", content)
        
        # Remove linhas em branco excessivas
        content = re.sub(r'\n\s*\n', '\n', content)
        
        doc.page_content = content
        if len(content) > 50: # Apenas adiciona documentos que ainda têm conteúdo substancial
            cleaned_docs.append(doc)
            
    return cleaned_docs

def process_documents():
    all_docs_from_loaders = []
    print(f"\nINFO: Iniciando varredura recursiva de documentos em '{DATA_PATH}'...")
    for root, dirs, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in LOADER_MAPPING:
                loader_module = LOADER_MAPPING[file_ext]
                try:
                    loaded_docs = loader_module.load(file_path)
                    all_docs_from_loaders.extend(loaded_docs)
                except Exception as e:
                    print(f"ERROR: Falha ao carregar o arquivo {filename}: {e}")

    if not all_docs_from_loaders:
        print("WARNING: Nenhum documento foi carregado. A base de conhecimento ficará vazia.")
        return

    print(f"\nINFO: {len(all_docs_from_loaders)} páginas/documentos carregados. Iniciando a limpeza do conteúdo...")
    # --- APLICA A LIMPEZA AQUI ---
    cleaned_docs = clean_document_content(all_docs_from_loaders)
    print(f"SUCCESS: Documentos limpos. Restaram {len(cleaned_docs)} documentos com conteúdo relevante.")


    print(f"\nINFO: Iniciando o 'chunking' dos documentos limpos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_chunks = text_splitter.split_documents(cleaned_docs)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")

    # O resto do código permanece o mesmo...
    print("\nINFO: Iniciando a geração de embeddings...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': DEVICE}
    )

    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"SUCCESS: Índice FAISS criado e salvo em: '{VECTOR_STORE_PATH}'")

    print("\nINFO: Salvando metadados dos chunks no PostgreSQL...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        print("INFO: Limpando dados antigos da tabela 'document_chunks'...")
        cur.execute("TRUNCATE TABLE document_chunks RESTART IDENTITY;")
        for i, doc in enumerate(split_chunks):
            source = doc.metadata.get('source', 'desconhecido')
            content = doc.page_content.replace('\x00', '')
            cur.execute(
                "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
                (source, content, i)
            )
        conn.commit()
        cur.close()
        conn.close()
        print(f"SUCCESS: {len(split_chunks)} chunks de metadados salvos no PostgreSQL.")
    except Exception as e:
        print(f"ERROR: Falha ao salvar metadados no PostgreSQL: {e}")
        raise

if __name__ == "__main__":
    print("--- INICIANDO PROCESSO DE ETL PARA O SISTEMA DE RAG ---")
    setup_database()
    process_documents()
    print("\n--- PROCESSO DE ETL CONCLUÍDO COM SUCESSO ---")