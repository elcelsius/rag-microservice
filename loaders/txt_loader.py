from langchain_community.document_loaders import TextLoader

def load(file_path: str):
    """Carrega o conteúdo de um arquivo de texto simples (.txt).
    
    Args:
        file_path (str): O caminho completo para o arquivo TXT a ser carregado.
        
    Returns:
        list: Uma lista de documentos carregados pela LangChain.
    """
    print(f"INFO: Carregando TXT: {file_path}")
    # O TextLoader é usado para arquivos de texto simples, com codificação UTF-8 por padrão.
    loader = TextLoader(file_path, encoding=\'utf-8\')
    return loader.load()


