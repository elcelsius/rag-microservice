"""Integration-style tests ensuring API caching hooks behave."""

from __future__ import annotations

from collections import Counter

import fakeredis
import pytest

import api
import cache_backend


@pytest.fixture
def fake_cache(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    cache_backend.reset_cache_client()
    monkeypatch.setenv("REDIS_URL", "redis://fake-host:6379/0")
    monkeypatch.setattr(cache_backend, "_get_client", lambda: fake)
    return fake


def test_query_endpoint_uses_cache(monkeypatch, fake_cache):
    monkeypatch.setattr(api, "FAISS_OK", True)
    monkeypatch.setattr(api, "embeddings_model", object())
    monkeypatch.setattr(api, "vectorstore", object())
    monkeypatch.setattr(api, "METRICS", Counter())

    calls = []

    def fake_answer(question, *args, **kwargs):
        calls.append((question, kwargs.get("confidence_min")))
        return {"answer": "ok", "citations": [], "context_found": True}

    monkeypatch.setattr(api, "answer_question", fake_answer)

    client = api.app.test_client()

    resp1 = client.post("/query", json={"question": "Onde fica a biblioteca?"})
    assert resp1.status_code == 200
    resp2 = client.post("/query", json={"question": "Onde fica a biblioteca?"})
    assert resp2.status_code == 200

    assert len(calls) == 1  # segunda chamada veio do cache
    assert calls[0][1] == api.CONFIDENCE_MIN_QUERY

    metrics = api.METRICS
    assert metrics["cache_hits_total"] == 1
    assert metrics["cache_misses_total"] == 1
    assert metrics["queries_total"] == 2


def test_agent_endpoint_uses_cache(monkeypatch, fake_cache):
    monkeypatch.setattr(api, "FAISS_OK", True)
    monkeypatch.setattr(api, "LLM_OK", True)
    monkeypatch.setattr(api, "embeddings_model", object())
    monkeypatch.setattr(api, "vectorstore", object())
    monkeypatch.setattr(api, "METRICS", Counter())

    calls = []

    def fake_run_agent(question, messages, embeddings, store, **kwargs):
        calls.append((question, kwargs.get("confidence_min"), kwargs.get("max_refine_attempts")))
        return {"answer": "Aqui", "citations": [], "action": "RESPONDER", "meta": {"refine_attempts": 0, "refine_success": True, "confidence": 0.8, "query_hash": "deadbeef", "refine_prompt_hashes": []}}

    monkeypatch.setattr(api, "run_agent", fake_run_agent)

    client = api.app.test_client()
    payload = {"question": "Onde fica a biblioteca?", "messages": [{"role": "user", "content": "Onde fica a biblioteca?"}]}

    resp1 = client.post("/agent/ask", json=payload)
    assert resp1.status_code == 200
    resp2 = client.post("/agent/ask", json=payload)
    assert resp2.status_code == 200

    assert len(calls) == 1
    assert calls[0][1] == api.CONFIDENCE_MIN_AGENT
    assert calls[0][2] == api.AGENT_REFINE_MAX_ATTEMPTS

    metrics = api.METRICS
    assert metrics["cache_hits_total"] == 1
    assert metrics["cache_misses_total"] == 1
    assert metrics["agent_queries_total"] == 2
