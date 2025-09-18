# -*- coding: utf-8 -*-
"""
Este módulo é responsável por carregar arquivos de texto simples (.txt).
Foi implementado um tratamento de erro para lidar com diferentes codificações
de caracteres, tornando-o mais robusto.
"""
from langchain_community.document_loaders import TextLoader


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
        return loader.load()
    except Exception as e:
        # Se a primeira tentativa falhar, informa o usuário e tenta a próxima.
        print(f"WARNING: Falha ao carregar {file_path} com UTF-8: {e}. Tentando com 'latin-1'.")
        try:
            # 2ª Tentativa: latin-1. É uma codificação de fallback que não falha na decodificação.
            loader = TextLoader(file_path, encoding='latin-1')
            return loader.load()
        except Exception as e2:
            # Se todas as tentativas falharem, loga o erro final e retorna uma lista vazia.
            print(f"ERROR: Falha ao carregar {file_path} com todos os encodings tentados: {e2}")
            return []