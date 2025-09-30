# -*- coding: utf-8 -*-
"""
Este módulo fornece a funcionalidade para carregar e extrair texto de arquivos PDF.
"""
from langchain_community.document_loaders import UnstructuredPDFLoader


def load(file_path: str):
    """
    Carrega o conteúdo de um arquivo PDF.

    Utiliza o UnstructuredPDFLoader, que é mais flexível para extrair texto
    de PDFs com layouts complexos (colunas, tabelas, etc.) em comparação
    com outros carregadores mais simples.

    O argumento `mode="single"` instrui o carregador a tratar todas as páginas
    do PDF como um único bloco de texto, o que pode ajudar a manter a coesão
    do conteúdo. Outros modos, como "paged", criariam um Documento por página.

    Args:
        file_path (str): O caminho completo para o arquivo PDF a ser carregado.

    Returns:
        list: Uma lista de Documentos da LangChain com o conteúdo textual do PDF.
    """
    print(f"INFO: Carregando PDF: {file_path}")

    # Instancia o carregador de PDF. O modo "single" é uma estratégia para
    # obter o texto de forma mais unificada.
    loader = UnstructuredPDFLoader(file_path, mode="single")

    # O método .load() processa o PDF e retorna o texto extraído.
    return loader.load()