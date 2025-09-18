from langchain_community.document_loaders import UnstructuredMarkdownLoader

def load(file_path: str):
    """Carrega o conteúdo de um arquivo Markdown.
    
    Args:
        file_path (str): O caminho completo para o arquivo Markdown a ser carregado.
        
    Returns:
        list: Uma lista de documentos carregados pela LangChain.
    """
    print(f"INFO: Carregando Markdown: {file_path}")
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


