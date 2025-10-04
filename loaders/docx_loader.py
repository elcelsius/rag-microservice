# -*- coding: utf-8 -*-
"""
Este módulo contém a funcionalidade para carregar e extrair o conteúdo
de texto de arquivos Microsoft Word (.docx).
"""
from langchain_community.document_loaders import Docx2txtLoader


def load(file_path: str):
    """
    Carrega o conteúdo de um arquivo DOCX.

    Utiliza a biblioteca 'docx2txt' por baixo dos panos, que é eficiente
    para extrair apenas o texto de documentos Word, ignorando imagens e formatação complexa.

    Args:
        file_path (str): O caminho completo para o arquivo DOCX a ser carregado.

    Returns:
        list: Uma lista contendo um único Documento da LangChain com o texto do arquivo.
    """
    print(f"INFO: Carregando DOCX: {file_path}")

    # Instancia o carregador específico para arquivos .docx.
    loader = Docx2txtLoader(file_path)

    # O método .load() lê o arquivo e retorna uma lista de objetos Document.
    return loader.load()