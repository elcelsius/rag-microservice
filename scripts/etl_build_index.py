#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de ETL (Extract, Transform, Load) para a base de conhecimento da IA.

Funcionalidades:
1.  **Extract (Extração):** Varre o diretório de dados (`/app/data`) de forma recursiva
    em busca de arquivos com extensões permitidas (.txt, .md, .pdf, .docx).
2.  **Transform (Transformação):**
    - Lê e extrai o conteúdo textual de cada arquivo.
    - Divide os textos longos em pedaços menores ("chunks") para otimizar a busca.
    - Converte cada chunk de texto em um vetor numérico (embedding) usando um
      modelo de linguagem da HuggingFace.
3.  **Load (Carga):**
    - Armazena os vetores e os textos correspondentes em um índice vetorial FAISS.
    - Salva o índice final no disco para ser utilizado pela aplicação principal.
"""
from __future__ import annotations

import io
import os
import sys
from typing import List

# LangChain é usado para dividir textos, criar embeddings e gerenciar o índice vetorial.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Configuração a partir de Variáveis de Ambiente
# ----------------------------
# Define as extensões de arquivo que serão processadas.
ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}

# Define os diretórios de entrada (dados) e saída (índice).
# os.environ.get() permite configurar via Docker Compose, com um valor padrão caso não seja definido.
DATA_ROOT = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("FAISS_OUT_DIR", "/app/vector_store/faiss_index")

# Define o modelo de embedding a ser usado. Essencial que seja o mesmo usado pela API.
EMB_MODEL = (
        os.environ.get("EMBEDDINGS_MODEL")
        or os.environ.get("EMBEDDINGS_MODEL_NAME")
        or "intfloat/multilingual-e5-large"  # Modelo padrão multilíngue e de alta performance.
)


# ----------------------------
# Funções de Leitura de Arquivos (Readers)
# ----------------------------

def _read_txt_like(path: str) -> str:
    """Lê arquivos baseados em texto (.txt, .md) com fallback de encoding."""
    # Tenta primeiro com 'utf-8' (padrão universal) e depois com 'latin-1' (compatibilidade).
    for enc in ("utf-8", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            # Se der erro de decodificação, tenta o próximo encoding.
            continue
        except Exception as e:
            # O 'flush=True' garante que o print apareça imediatamente no log do Docker.
            print(f"[ETL] Falha lendo texto {path}: {e}", flush=True)
            return ""
    # Se tudo falhar, tenta ler como binário e decodificar forçadamente, ignorando erros.
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"[ETL] Falha lendo (bin) {path}: {e}", flush=True)
        return ""


def _read_pdf(path: str) -> str:
    """Extrai texto de um arquivo PDF, página por página."""
    try:
        # Importação local para não exigir a biblioteca se nenhum PDF for usado.
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            # Extrai o texto de cada página e o adiciona à lista.
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)  # Junta o texto de todas as páginas.
    except Exception as e:
        print(f"[ETL] Falha lendo PDF {path}: {e}", flush=True)
        return ""


def _read_docx(path: str) -> str:
    """Extrai texto de um arquivo DOCX."""
    try:
        from docx import Document  # Biblioteca python-docx
    except Exception as e:
        print(f"[ETL] python-docx nao instalado; ignorando {path}. Erro: {e}", flush=True)
        return ""
    try:
        doc = Document(path)
        # Extrai o texto de cada parágrafo e junta tudo.
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[ETL] Falha lendo DOCX {path}: {e}", flush=True)
        return ""


# ----------------------------
# Coleta e Carga de Arquivos
# ----------------------------

def _walk_files(root: str) -> List[str]:
    """Encontra todos os arquivos com extensões permitidas em um diretório, recursivamente."""
    found: List[str] = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXTS:
                found.append(os.path.join(base, fn))
    return sorted(found)


def _load_text(path: str) -> str:
    """Seleciona a função de leitura apropriada com base na extensão do arquivo."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return _read_txt_like(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    print(f"[ETL] Extensao nao suportada no momento: {ext} ({path})", flush=True)
    return ""


# ----------------------------
# ETL Principal
# ----------------------------

def main() -> int:
    """Executa o pipeline completo de ETL."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[ETL] Raiz de dados: {DATA_ROOT}", flush=True)

    files = _walk_files(DATA_ROOT)

    if not files:
        print(f"[ETL] Nenhum arquivo {sorted(ALLOWED_EXTS)} encontrado de forma recursiva em {DATA_ROOT}.", flush=True)
        print("[ETL] Gerando índice de placeholder.", flush=True)
    else:
        print(f"[ETL] Encontrados {len(files)} arquivos para indexar (recursivo).", flush=True)
        # Mostra uma amostra dos primeiros 10 arquivos encontrados.
        for sample in files[:10]:
            print(f"[ETL]   - {sample}", flush=True)

    # `RecursiveCharacterTextSplitter` divide o texto em pedaços (chunks) de tamanho
    # definido, com uma sobreposição (overlap) para não perder contexto entre eles.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    texts: List[str] = []
    metas: List[dict] = []

    for path in files or []:
        txt = _load_text(path)
        if not (txt and txt.strip()):
            print(f"[ETL] Vazio/indecifrável: {path}", flush=True)
            continue
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            # Metadados são importantes para saber a origem da informação.
            metas.append({"source": path, "chunk": i + 1})

    # Se, após processar tudo, não houver texto, cria um item dummy para evitar erros.
    if not texts:
        texts = ["Base sem documentos. Adicione .txt/.md/.pdf em /app/data e recrie o índice."]
        metas = [{"source": "dummy", "chunk": 1}]

    print(f"[ETL] Gerando embeddings com {EMB_MODEL} para {len(texts)} chunks ...", flush=True)
    # Inicializa o modelo de embedding que será baixado e executado pela HuggingFace.
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    # Cria o índice FAISS a partir dos textos e embeddings.
    # FAISS é uma biblioteca do Facebook AI para busca de similaridade eficiente.
    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)

    # Salva o índice no disco para uso futuro.
    vs.save_local(OUT_DIR)
    print(f"[ETL] Índice FAISS salvo em: {OUT_DIR}", flush=True)
    return 0


# Ponto de entrada do script.
if __name__ == "__main__":
    sys.exit(main())