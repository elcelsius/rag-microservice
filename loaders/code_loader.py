# -*- coding: utf-8 -*-
"""
Este módulo é responsável por carregar arquivos de código-fonte ou texto plano.
Ele foi projetado para ser robusto a diferentes codificações de caracteres,
tentando múltiplos formatos antes de falhar.
"""
from langchain_community.document_loaders import TextLoader


def load(file_path: str):
    """
    Carrega o conteúdo de um arquivo de código ou texto, tratando possíveis erros de encoding.

    A função primeiro tenta abrir o arquivo com a codificação 'utf-8', que é a mais comum
    hoje em dia. Se isso falhar (geralmente por um UnicodeDecodeError), ela tenta
    a codificação 'latin-1', que é mais permissiva e raramente falha, embora possa
    não interpretar os caracteres especiais corretamente.

    Args:
        file_path (str): O caminho completo para o arquivo a ser carregado.

    Returns:
        list: Uma lista de Documentos da LangChain. Cada Documento contém o conteúdo
              do arquivo na propriedade `page_content` e metadados na `metadata`.
              Retorna uma lista vazia se o arquivo não puder ser carregado com os
              encodings tentados.
    """
    print(f"INFO: Carregando arquivo de código/texto: {file_path}")
    try:
        # 1ª Tentativa: UTF-8
        # Tenta carregar o arquivo usando a codificação UTF-8, o padrão moderno
        # para arquivos de texto e código.
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    except Exception as e:
        # O 'Exception as e' captura o erro específico para que possamos logá-lo.
        print(f"WARNING: Falha ao carregar {file_path} com UTF-8: {e}. Tentando com 'latin-1'.")
        try:
            # 2ª Tentativa: latin-1
            # Se UTF-8 falhar, tenta carregar com a codificação 'latin-1'.
            # Esta codificação mapeia cada valor de byte para um caractere,
            # então a leitura não falhará, sendo uma boa alternativa de fallback.
            loader = TextLoader(file_path, encoding='latin-1')
            return loader.load()
        except Exception as e2:
            # Se ambas as tentativas falharem, notifica o erro final e desiste.
            print(f"ERROR: Falha ao carregar {file_path} com todos os encodings tentados: {e2}")
            return []