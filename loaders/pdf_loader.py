from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

def load(file_path: str):
    """Carrega o conteúdo de um arquivo PDF.
    
    Args:
        file_path (str): O caminho completo para o arquivo PDF a ser carregado.
        
    Returns:
        list: Uma lista de documentos carregados pela LangChain.
    """
    print(f"INFO: Carregando PDF: {file_path}")
    # Usando UnstructuredPDFLoader para maior flexibilidade na extração de texto de PDFs complexos.
    # O modo "single" tenta extrair o texto de forma mais coesa.
    loader = UnstructuredPDFLoader(file_path, mode="single")
    return loader.load()


