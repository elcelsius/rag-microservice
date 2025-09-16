#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL que:
- varre /app/data (recursivo) e ingere .txt, .md, .pdf
- usa o MESMO modelo de embeddings do runtime (via EMBEDDINGS_MODEL)
- salva FAISS em /app/vector_store/faiss_index
"""
from __future__ import annotations
import os, io, sys
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_ROOT = os.environ.get("DATA_DIR", "/app/data")
OUT_DIR   = os.environ.get("FAISS_OUT_DIR", "/app/vector_store/faiss_index")
EMB_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _read_txt_like(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "ignore")

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

def _walk_files(root: str) -> List[str]:
    exts = {".txt", ".md", ".pdf"}
    found = []
    for base, _dirs, files in os.walk(root):
        for fn in files:
            _, ext = os.path.splitext(fn.lower())
            if ext in exts:
                found.append(os.path.join(base, fn))
    return sorted(found)

def _load_text(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in (".txt", ".md"):
        return _read_txt_like(path)
    if ext == ".pdf":
        return _read_pdf(path)
    return ""

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[ETL] Raiz de dados: {DATA_ROOT}", flush=True)

    files = _walk_files(DATA_ROOT)
    if not files:
        print("[ETL] Nenhum arquivo (.txt/.md/.pdf) encontrado. Gerando índice de placeholder.", flush=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    texts, metas = [], []

    for path in files or []:
        txt = _load_text(path)
        if not (txt and txt.strip()):
            continue
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": path, "chunk": i + 1})

    if not texts:
        texts = ["Base sem documentos. Adicione .txt/.md/.pdf em /app/data e recrie o índice."]
        metas = [{"source": "dummy", "chunk": 1}]

    print(f"[ETL] Gerando embeddings com {EMB_MODEL} para {len(texts)} chunks ...", flush=True)
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)
    vs.save_local(OUT_DIR)
    print(f"[ETL] Índice FAISS salvo em: {OUT_DIR}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
