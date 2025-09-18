#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL que:
- Varre /app/data (recursivo) e ingere .txt, .md, .pdf e .docx (opcional)
- Usa o MESMO modelo de embeddings do runtime (via EMBEDDINGS_MODEL / EMBEDDINGS_MODEL_NAME)
- Salva FAISS em /app/vector_store/faiss_index
"""
from __future__ import annotations

import io
import os
import sys
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# Config por ambiente
# ----------------------------
ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}  # .docx opcional (requer python-docx)
DATA_ROOT = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR = os.environ.get("FAISS_OUT_DIR", "/app/vector_store/faiss_index")
EMB_MODEL = (
    os.environ.get("EMBEDDINGS_MODEL")
    or os.environ.get("EMBEDDINGS_MODEL_NAME")
    or "intfloat/multilingual-e5-large"
)

# ----------------------------
# Readers
# ----------------------------
def _read_txt_like(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[ETL] Falha lendo texto {path}: {e}", flush=True)
            return ""
    # fallback binário
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"[ETL] Falha lendo (bin) {path}: {e}", flush=True)
        return ""

def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)
    except Exception as e:
        print(f"[ETL] Falha lendo PDF {path}: {e}", flush=True)
        return ""

def _read_docx(path: str) -> str:
    try:
        from docx import Document  # python-docx
    except Exception as e:
        print(f"[ETL] python-docx nao instalado; ignorando {path}. Erro: {e}", flush=True)
        return ""
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[ETL] Falha lendo DOCX {path}: {e}", flush=True)
        return ""

# ----------------------------
# Coleta & carga de arquivos
# ----------------------------
def _walk_files(root: str) -> List[str]:
    found: List[str] = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXTS:
                found.append(os.path.join(base, fn))
    return sorted(found)

def _load_text(path: str) -> str:
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
# ETL principal
# ----------------------------
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[ETL] Raiz de dados: {DATA_ROOT}", flush=True)

    files = _walk_files(DATA_ROOT)

    if not files:
        print(f"[ETL] Nenhum arquivo {sorted(ALLOWED_EXTS)} encontrado de forma recursiva em {DATA_ROOT}.", flush=True)
        print("[ETL] Gerando índice de placeholder.", flush=True)
    else:
        print(f"[ETL] Encontrados {len(files)} arquivos para indexar (recursivo).", flush=True)
        for sample in files[:10]:
            print(f"[ETL]   - {sample}", flush=True)

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
            metas.append({"source": path, "chunk": i + 1})

    if not texts:
        texts = ["Base sem documentos. Adicione .txt/.md/.pdf em /app/data e recrie o índice."]
        metas = [{"source": "dummy", "chunk": 1}]

    print(f"[ETL] Gerando embeddings com {EMB_MODEL} para {len(texts)} chunks ...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
    vs.save_local(OUT_DIR)
    print(f"[ETL] Índice FAISS salvo em: {OUT_DIR}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
