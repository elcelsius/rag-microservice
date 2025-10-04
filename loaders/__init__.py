# -*- coding: utf-8 -*-
"""
Módulo unificado para carregamento de documentos.

Este módulo funciona como uma fábrica (factory) que seleciona o carregador apropriado
com base na extensão do arquivo, centralizando a lógica de extração de texto e
o tratamento de erros.
"""
import os
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

try:
    from text_normalizer import normalize_documents
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    normalize_documents = import_module('text_normalizer').normalize_documents


def _extract_leading_url(text: str) -> Optional[str]:
    """Retorna a primeira URL (http/https) presente no início do texto."""
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("http://", "https://")):
            return line
        # encontrou texto não vazio antes de uma URL => aborta
        break
    return None


def _finalize_docs(docs: List[Document], file_path: str) -> List[Document]:
    docs = normalize_documents(docs)
    if not docs:
        return docs

    url = None
    for doc in docs:
        cand = _extract_leading_url(doc.page_content or "")
        if cand:
            url = cand
            break

    for doc in docs:
        meta = doc.metadata or {}
        if file_path:
            meta.setdefault("source", file_path)
        if url:
            meta.setdefault("url", url)
        doc.metadata = meta

    return docs

# --- Funções de Carregamento Customizadas (para CSV e JSON) ---

def _read_csv_to_markdown(path: str) -> str:
    """Lê um CSV e converte para uma tabela formatada em Markdown."""
    try:
        # Tenta usar pandas para uma conversão mais robusta
        import pandas as pd
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        # Converte o DataFrame para Markdown, que é um formato de texto bem estruturado
        return df.to_markdown(index=False)
    except Exception:
        # Fallback para o módulo csv padrão se o pandas não estiver disponível ou falhar
        import csv
        out = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                # Formata cada linha como uma linha de tabela Markdown
                out.append("| " + " | ".join([str(x).replace("\n", " ").strip() for x in row]) + " |")
        return "\n".join(out)

def _read_json_to_flat_text(path: str) -> str:
    """Lê um JSON e o 'achata' (flattens) para um formato de texto chave:valor."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    def flatten(obj, prefix=""):
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                lines.extend(flatten(v, f"{prefix}{k}."))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                lines.extend(flatten(v, f"{prefix}{i}."))
        else:
            s = str(obj).replace("\n", " ").strip()
            key = prefix[:-1] if prefix.endswith(".") else prefix
            lines.append(f"{key}: {s}")
        return lines

    return "\n".join(flatten(data))


# --- Função Principal da Fábrica de Loaders ---

def load_document(file_path: str) -> List[Document]:
    """
    Carrega um documento de um arquivo, selecionando o loader apropriado.

    Esta função atua como uma fábrica, inspecionando a extensão do arquivo para
    determinar a melhor forma de extrair o texto. Ela encapsula a lógica para
    diferentes tipos de arquivo e inclui um tratamento de erro universal para
    garantir que a falha no carregamento de um único arquivo não interrompa um
    processo de ETL em lote.

    Args:
        file_path (str): O caminho completo para o arquivo a ser carregado.

    Returns:
        List[Document]: Uma lista de objetos Document da LangChain.
                        Retorna uma lista vazia se o carregamento falhar.
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"INFO: Carregando arquivo: {file_path} (tipo: {file_ext})")
        
        # --- Lógica de Seleção do Loader ---

        # Loader para PDF (robusto)
        if file_ext == ".pdf":
            from langchain_community.document_loaders import UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(file_path, mode="single")
            docs = loader.load()
            return _finalize_docs(docs, file_path)

        # Loader para DOCX
        elif file_ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return _finalize_docs(docs, file_path)

        # Loader para Markdown
        elif file_ext == ".md":
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            return _finalize_docs(docs, file_path)

        # Loader para TXT e formatos de código (com fallback de encoding)
        elif file_ext in [".txt", ".php", ".sql", ".xml", ".ini", ".config", ".example", ".yml", ".yaml"]:
            from langchain_community.document_loaders import TextLoader
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                return _finalize_docs(docs, file_path)
            except Exception:
                # Fallback para latin-1 se UTF-8 falhar
                print(f"WARN: Falha ao carregar {file_path} com UTF-8. Tentando com 'latin-1'.")
                loader = TextLoader(file_path, encoding='latin-1')
                docs = loader.load()
                return _finalize_docs(docs, file_path)

        # Loader customizado para CSV (converte para Markdown)
        elif file_ext == ".csv":
            text_content = _read_csv_to_markdown(file_path)
            metadata = {"source": file_path}
            docs = [Document(page_content=text_content, metadata=metadata)]
            return _finalize_docs(docs, file_path)

        # Loader customizado para JSON (converte para texto achatado)
        elif file_ext == ".json":
            text_content = _read_json_to_flat_text(file_path)
            metadata = {"source": file_path}
            docs = [Document(page_content=text_content, metadata=metadata)]
            return _finalize_docs(docs, file_path)

        else:
            # Se a extensão não for reconhecida, imprime um aviso mas não quebra o processo
            print(f"WARN: Nenhum loader específico encontrado para a extensão '{file_ext}' do arquivo '{file_path}'. O arquivo será ignorado.")
            return []

    except Exception as e:
        # Tratamento de erro universal para qualquer falha durante o carregamento
        print(f"ERROR: Falha crítica ao carregar o arquivo '{file_path}'. Erro: {e}. O arquivo será ignorado.")
        return []
