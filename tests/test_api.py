import pytest

import api as api_module
from api import app as flask_app


@pytest.fixture
def client():
    """Cria um cliente de testes da API Flask."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_state():
    """Garante que os estados globais sejam limpos entre os testes."""
    api_module.FAISS_OK = False
    api_module.LLM_OK = False
    api_module.APP_READY = False
    api_module.embeddings_model = None
    api_module.vectorstore = None
    yield
    api_module.FAISS_OK = False
    api_module.LLM_OK = False
    api_module.APP_READY = False
    api_module.embeddings_model = None
    api_module.vectorstore = None


def test_root_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.get_json() == {'status': 'ok'}


def test_healthz_not_ready(client, mocker):
    mocker.patch('api._probe_llm', return_value=False)
    response = client.get('/healthz')
    assert response.status_code == 503
    json_data = response.get_json()
    assert json_data['ready'] is False
    assert json_data['faiss_ok'] is False


def test_healthz_ready(client, mocker):
    api_module.FAISS_OK = True
    mocker.patch('api._probe_llm', return_value=True)
    response = client.get('/healthz')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['ready'] is True
    assert json_data['faiss_ok'] is True
    assert json_data['llm_ok'] is True


def test_query_endpoint_when_not_ready(client):
    response = client.post('/query', json={'question': 'teste'})
    assert response.status_code == 503
    assert 'error' in response.get_json()


def test_query_endpoint_no_question(client, mocker):
    api_module.FAISS_OK = True
    api_module.embeddings_model = object()
    api_module.vectorstore = object()
    response = client.post('/query', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_query_endpoint_success(client, mocker):
    api_module.FAISS_OK = True
    api_module.embeddings_model = object()
    api_module.vectorstore = object()
    answer_stub = {'answer': 'ok', 'citations': [], 'context_found': True}
    patched = mocker.patch('api.answer_question', return_value=answer_stub)
    response = client.post('/query', json={'question': 'Qual o horário de atendimento?'})
    assert response.status_code == 200
    assert response.get_json() == answer_stub
    patched.assert_called_once()


def test_agent_endpoint_not_ready(client):
    api_module.FAISS_OK = True
    api_module.embeddings_model = object()
    api_module.vectorstore = object()
    # LLM_OK permanece False, logo o agente não deve estar pronto.
    response = client.post('/agent/ask', json={'question': 'Teste'})
    assert response.status_code == 503
    assert 'error' in response.get_json()


def test_agent_endpoint_success(client, mocker):
    api_module.FAISS_OK = True
    api_module.LLM_OK = True
    api_module.embeddings_model = object()
    api_module.vectorstore = object()
    stub = {'answer': 'Resposta', 'citations': [], 'action': 'AUTO_RESOLVER'}
    patched = mocker.patch('api.run_agent', return_value=stub)
    response = client.post('/agent/ask', json={'question': 'Qual o e-mail do laboratório X?'})
    assert response.status_code == 200
    assert response.get_json() == stub
    patched.assert_called_once()
