#!/usr/bin/env python3
# tools/check_index.py
import os
import sys

# Adiciona o diretório raiz ao path para permitir importações de outros módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configurações ---
INDEX_PATH = "/app/vector_store/faiss_index"
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
SEARCH_TERM = "andreia" # Termo a ser buscado (case-insensitive)

def main():
    """Carrega o índice FAISS e busca por um termo nos documentos."""
    print(f"--- Verificador de Índice ---")
    print(f"Carregando índice de: {INDEX_PATH}")
    print(f"Usando modelo de embeddings: {EMBEDDINGS_MODEL_NAME}")
    print(f"Buscando pelo termo: '{SEARCH_TERM}' (case-insensitive)")
    print("-" * 20)

    if not os.path.exists(INDEX_PATH):
        print(f"[ERRO] O diretório do índice FAISS não foi encontrado em '{INDEX_PATH}'. O ETL rodou corretamente?")
        return

    try:
        # 1. Carrega o modelo de embeddings
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        
        # 2. Carrega o índice FAISS local
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True # Necessário para carregar índices do LangChain
        )
        print("[SUCESSO] Índice FAISS carregado.")
    except Exception as e:
        print(f"[ERRO] Falha ao carregar o índice FAISS: {e}")
        return

    # 3. Acessa todos os documentos no docstore
    try:
        docstore = getattr(vectorstore, "docstore", None)
        if not docstore or not hasattr(docstore, "_dict"):
            print("[ERRO] Docstore não encontrado ou em formato inesperado no índice.")
            return

        all_docs = list(getattr(docstore, "_dict", {}).values())
        print(f"Total de documentos no índice: {len(all_docs)}")

        if not all_docs:
            print("[AVISO] O índice foi carregado, mas não contém nenhum documento.")
            return

    except Exception as e:
        print(f"[ERRO] Falha ao extrair documentos do docstore: {e}")
        return

    # 4. Busca pelo termo nos documentos
    found_count = 0
    search_term_lower = SEARCH_TERM.lower()

    for i, doc in enumerate(all_docs):
        content = getattr(doc, "page_content", "").lower()
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "desconhecido")

        if search_term_lower in content:
            found_count += 1
            print(f"\n--- Documento Encontrado #{found_count} ---")
            print(f"Fonte (Arquivo): {source}")
            print(f"Metadados: {metadata}")
            print("Conteúdo do Chunk:")
            print(getattr(doc, "page_content", ""))
            print("-" * 20)

    if found_count == 0:
        print(f"\n[RESULTADO] O termo '{SEARCH_TERM}' não foi encontrado em nenhum dos {len(all_docs)} documentos do índice.")
    else:
        print(f"\n[RESULTADO] Fim da busca. O termo '{SEARCH_TERM}' foi encontrado em {found_count} documento(s).")

if __name__ == "__main__":
    main()
