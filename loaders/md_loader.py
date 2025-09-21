# -*- coding: utf-8 -*-
"""
Este módulo é dedicado ao carregamento de arquivos no formato Markdown (.md).
Ele utiliza uma biblioteca capaz de interpretar a estrutura do Markdown.
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader


def load(file_path: str):
    """
    Carrega o conteúdo de um arquivo Markdown.

    O UnstructuredMarkdownLoader é capaz de interpretar elementos do Markdown
    (como títulos, listas e tabelas) e extrair o texto de forma estruturada.

    Args:
        file_path (str): O caminho completo para o arquivo Markdown a ser carregado.

    Returns:
        list: Uma lista de Documentos da LangChain, onde o texto do Markdown é extraído.
    """
    print(f"INFO: Carregando Markdown: {file_path}")

    # Instancia o carregador otimizado para arquivos Markdown.
    loader = UnstructuredMarkdownLoader(file_path)

    # Chama o método .load() para realizar a leitura e extração.
    return loader.load()