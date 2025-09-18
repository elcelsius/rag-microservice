from langchain_community.document_loaders import TextLoader

def load(file_path: str):
    """
    Carrega o conteúdo de arquivos de código ou texto plano, tratando possíveis erros de encoding.
    
    Args:
        file_path (str): O caminho completo para o arquivo a ser carregado.
        
    Returns:
        list: Uma lista de documentos carregados pela LangChain.
              Retorna uma lista vazia se o arquivo não puder ser carregado com os encodings tentados.
    """
    print(f"INFO: Carregando arquivo de código/texto: {file_path}")
    try:
        # Tenta carregar o arquivo usando codificação UTF-8
        loader = TextLoader(file_path, encoding=\'utf-8\')
        return loader.load()
    except Exception as e:
        print(f"WARNING: Falha ao carregar {file_path} com UTF-8: {e}. Tentando com \'latin-1\'.")
        try:
            # Se UTF-8 falhar, tenta carregar com codificação latin-1
            loader = TextLoader(file_path, encoding=\'latin-1\')
            return loader.load()
        except Exception as e2:
            # Se ambos falharem, imprime um erro e retorna uma lista vazia
            print(f"ERROR: Falha ao carregar {file_path} com todos os encodings tentados: {e2}")
            return []


