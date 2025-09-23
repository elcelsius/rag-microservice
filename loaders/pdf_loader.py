# -*- coding: utf-8 -*-
"""
Este módulo fornece a funcionalidade para carregar e extrair texto de arquivos PDF.
"""
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from typing import List

def load(file_path: str) -> List[Document]:
    """
    Carrega o conteúdo de um arquivo PDF de forma segura.

    Utiliza o UnstructuredPDFLoader para extrair texto de forma robusta.
    Se ocorrer qualquer erro durante o processamento do PDF (ex: arquivo corrompido,
    protegido por senha ou apenas com imagens), a função registrará um aviso
    e retornará uma lista vazia, evitando a interrupção do pipeline de ETL.

    Args:
        file_path (str): O caminho completo para o arquivo PDF a ser carregado.

    Returns:
        List[Document]: Uma lista de Documentos da LangChain, ou uma lista vazia em caso de falha.
    """
    try:
        print(f"INFO: Carregando PDF: {file_path}")
        # Instancia o carregador de PDF. O modo "single" é uma estratégia para
        # obter o texto de forma mais unificada.
        loader = UnstructuredPDFLoader(file_path, mode="single")

        # O método .load() processa o PDF e retorna o texto extraído.
        return loader.load()
    except Exception as e:
        print(f"WARN: Falha ao carregar o arquivo PDF '{file_path}'. Erro: {e}. O arquivo será ignorado.")
        # Retorna uma lista vazia para garantir que o pipeline de ETL não seja interrompido.
        return []
