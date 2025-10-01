#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETL para índice FAISS do RAG.

- Varre ./data (recursivo) e lê arquivos suportados.
- Suporta DOIS estilos de loader em ./loaders:
  1) read_<ext>(path) -> str  (ex.: read_csv, read_json)
  2) load(file_path) -> list[Document]  (seus loaders atuais de txt/pdf/docx/md/code)
- Faz chunking configurável e cria embeddings com HuggingFace.
- Salva FAISS em /app/vector_store/faiss_index (ou o que for passado).
"""

from __future__ import annotations

import os
import sys
import io
import argparse
import importlib.util
from pathlib import Path
from typing import List, Callable, Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Defaults via ENV (podem ser sobrescritos por flags)
# ----------------------------
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "./data")
DEFAULT_OUT_DIR = os.environ.get("FAISS_OUT_DIR", "/app/vector_store/faiss_index")
DEFAULT_EMB = (
    os.environ.get("EMBEDDINGS_MODEL")
    or os.environ.get("EMBEDDINGS_MODEL_NAME")
    or "intfloat/multilingual-e5-large"
)
DEFAULT_LOADERS_DIR = os.environ.get("LOADERS_DIR", "./loaders")
DEFAULT_EXTS = "txt,md,pdf,docx"  # adicione csv,json,... se tiver read_csv/read_json

# ----------------------------
# Leitores nativos (fallback)
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
        from docx import Document
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
# Loaders customizados (./loaders)
# ----------------------------
def _import_module(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def _load_custom_loaders(loaders_dir: str) -> Tuple[Dict[str, Callable[[str], str]], List[Callable]]:
    """
    Carrega dinamicamente loaders de loaders_dir.
    Retorna:
      - readers: {'.ext': read_<ext>(path)->str}
      - loaders_load: [load(file_path)->list[Document]]
    """
    readers: Dict[str, Callable[[str], str]] = {}
    loaders_load: List[Callable] = []
    ld = Path(loaders_dir)
    if not ld.exists():
        return readers, loaders_load

    for py in ld.glob("*.py"):
        try:
            mod = _import_module(py)
            # prioriza read_<ext>
            for name in dir(mod):
                if name.startswith("read_"):
                    ext = "." + name.split("_", 1)[1].lower()
                    fn = getattr(mod, name)
                    if callable(fn):
                        readers[ext] = fn
            # também registra loaders no estilo load(file_path)
            lf = getattr(mod, "load", None)
            if callable(lf):
                loaders_load.append(lf)
        except Exception as e:
            print(f"[ETL] Ignorando loader {py}: {e}", flush=True)
    return readers, loaders_load

def _read_any(path: str, readers: Dict[str, Callable[[str], str]], loaders_load: List[Callable]) -> str:
    ext = Path(path).suffix.lower()

    # 1) prioridade: read_<ext>(path)->str
    if ext in readers:
        try:
            return readers[ext](path) or ""
        except Exception as e:
            print(f"[ETL] read_{ext[1:]} falhou em {path}: {e}", flush=True)

    # 2) fallback: load(file_path)->list[Document]
    for lf in loaders_load:
        try:
            docs = lf(path) or []
            if isinstance(docs, list) and docs:
                parts = []
                for d in docs:
                    pc = getattr(d, "page_content", "")
                    if pc:
                        parts.append(str(pc))
                if parts:
                    return "\n".join(parts)
        except Exception:
            continue

    # 3) fallback nativo
    if ext in {".txt", ".md"}:
        return _read_txt_like(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)

    return ""

# ----------------------------
# Coleta de arquivos
# ----------------------------
def _walk_files(root: str, exts: set[str]) -> List[str]:
    """Lista arquivos recursivamente, filtrando por 'exts' recebidas (CLI/ENV)."""
    found: List[str] = []
    root_p = Path(root)
    if not root_p.exists():
        return found
    for p in root_p.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            found.append(str(p))
    return sorted(found)

# ----------------------------
# Build FAISS
# ----------------------------
def build_faiss(
    data_dir: str,
    out_dir: str,
    emb_model: str,
    chunk_size: int,
    chunk_overlap: int,
    exts_csv: str,
    loaders_dir: str,
) -> None:
    exts = {"."+e.strip().lower() for e in exts_csv.split(",") if e.strip()}
    readers, loaders_load = _load_custom_loaders(loaders_dir)

    print(f"[ETL] data={data_dir} out={out_dir} emb={emb_model}", flush=True)
    print(f"[ETL] exts={sorted(exts)} loaders_dir={loaders_dir}", flush=True)
    print(f"[ETL] readers={sorted(readers.keys()) or 'nenhum'} loaders_load={len(loaders_load)}", flush=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    files = _walk_files(data_dir, exts)
    if not files:
        print(f"[ETL] Nenhum arquivo {sorted(exts)} encontrado em {data_dir}.", flush=True)

    texts: List[str] = []
    metas: List[dict] = []

    for path in files:
        raw = _read_any(path, readers, loaders_load).strip()
        if not raw:
            print(f"[ETL] Vazio/indecifrável: {path}", flush=True)
            continue
        chunks = splitter.split_text(raw)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": path, "chunk": i + 1})

    if not texts:
        texts = ["Base sem documentos. Adicione arquivos em /app/data e recrie o índice."]
        metas = [{"source": "dummy", "chunk": 1}]
        print("[ETL] WARN: índice dummy criado (sem textos válidos).", flush=True)

    print(f"[ETL] Gerando embeddings com {emb_model} para {len(texts)} chunks ...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)

    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(out_dir)
    print(f"[ETL] Índice FAISS salvo em: {out_dir}", flush=True)

# ----------------------------
# CLI
# ----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="ETL para FAISS (RAG).")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="pasta com documentos (default: ./data)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="pasta de saída do FAISS")
    parser.add_argument("--embeddings", default=DEFAULT_EMB, help="modelo de embeddings HF")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--exts", default=DEFAULT_EXTS, help="extensões suportadas (ex: txt,md,pdf,docx[,csv,json])")
    parser.add_argument("--loaders", default=DEFAULT_LOADERS_DIR, help="pasta com loaders customizados")
    args = parser.parse_args()

    build_faiss(
        data_dir=args.data,
        out_dir=args.out,
        emb_model=args.embeddings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        exts_csv=args.exts,
        loaders_dir=args.loaders,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
