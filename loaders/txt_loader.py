# -*- coding: utf-8 -*-
"""
Este módulo é responsável por carregar arquivos de texto simples (.txt).
Foi implementado um tratamento de erro para lidar com diferentes codificações
de caracteres, tornando-o mais robusto.
"""
from typing import Optional

from langchain_community.document_loaders import TextLoader

from text_normalizer import normalize_documents


def _extract_leading_url(text: str) -> Optional[str]:
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("http://", "https://")):
            return line
        break
    return None


def _finalize_docs(docs, file_path: str):
    docs = normalize_documents(docs)
    if not docs:
        return docs

    url = None
    for doc in docs:
        url = _extract_leading_url(getattr(doc, "page_content", ""))
        if url:
            break

    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        if file_path:
            meta.setdefault("source", file_path)
        if url:
            meta.setdefault("url", url)
        doc.metadata = meta

    return docs


def load(file_path: str):
    """
    Carrega o conteúdo de um arquivo de texto simples (.txt), tratando possíveis erros de encoding.

    A função primeiro tenta abrir o arquivo com a codificação 'utf-8'. Se falhar,
    ela tenta a codificação 'latin-1' como uma alternativa segura.

    Args:
        file_path (str): O caminho completo para o arquivo TXT a ser carregado.

    Returns:
        list: Uma lista de Documentos da LangChain.
              Retorna uma lista vazia se o arquivo não puder ser carregado.
    """
    print(f"INFO: Carregando TXT: {file_path}")
    try:
        # 1ª Tentativa: UTF-8. É a codificação mais comum e recomendada.
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        return _finalize_docs(docs, file_path)
    except Exception as e:
        # Se a primeira tentativa falhar, informa o usuário e tenta a próxima.
        print(f"WARNING: Falha ao carregar {file_path} com UTF-8: {e}. Tentando com 'latin-1'.")
        try:
            # 2ª Tentativa: latin-1. É uma codificação de fallback que não falha na decodificação.
            loader = TextLoader(file_path, encoding='latin-1')
            docs = loader.load()
            return _finalize_docs(docs, file_path)
        except Exception as e2:
            # Se todas as tentativas falharem, loga o erro final e retorna uma lista vazia.
            print(f"ERROR: Falha ao carregar {file_path} com todos os encodings tentados: {e2}")
            return []
