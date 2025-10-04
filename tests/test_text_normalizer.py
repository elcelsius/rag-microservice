from langchain.schema import Document

from text_normalizer import normalize_text, normalize_documents


def test_normalize_text_fixes_mojibake_sequences():
    assert normalize_text("Computa§£o") == "Computação"
    assert normalize_text("Andr©a Carla") == "Andréa Carla"
    assert normalize_text("Hist³rico do curso") == "Histórico do curso"
    assert normalize_text("Inªs e gªnero") == "Inês e gênero"


def test_normalize_text_preserves_ordinals_and_professional_titles():
    assert normalize_text("Profª. Dra. Nome") == "Profª. Dra. Nome"
    assert normalize_text("nº 123") == "nº 123"
    assert normalize_text("ºltimos encontros") == "últimos encontros"


def test_normalize_documents_updates_page_content_in_place():
    docs = [Document(page_content="Jo£o e Ant´nio", metadata={"source": "dummy.txt"})]
    normalized = normalize_documents(docs)
    assert normalized[0].page_content == "João e Antônio"
    # Garantir que o objeto original também foi atualizado
    assert docs[0].page_content == "João e Antônio"
