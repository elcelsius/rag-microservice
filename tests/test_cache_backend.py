"""Tests for Redis cache helper functions."""

from __future__ import annotations

import fakeredis
import pytest

import cache_backend


@pytest.fixture(autouse=True)
def _reset_cache_client(monkeypatch):
    cache_backend.reset_cache_client()
    monkeypatch.delenv("REDIS_URL", raising=False)


def test_cache_fetch_without_client():
    value, key, available = cache_backend.cache_fetch(cache_backend.QUERY_NAMESPACE, {"sample": 1})
    assert available is False
    assert value is None
    assert key.startswith(f"{cache_backend.QUERY_NAMESPACE}:")


def test_cache_store_and_fetch(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setenv("REDIS_URL", "redis://fake-host:6379/0")
    monkeypatch.setattr(cache_backend, "_get_client", lambda: fake)

    payload = {"question": "oi"}
    data = {"answer": "olá"}

    stored = cache_backend.cache_store(cache_backend.QUERY_NAMESPACE, payload, data)
    assert stored is True

    fetched, key, available = cache_backend.cache_fetch(cache_backend.QUERY_NAMESPACE, payload)
    assert available is True
    assert fetched == data
    assert fake.ttl(key) > 0


def test_invalidate_namespace(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setenv("REDIS_URL", "redis://fake-host:6379/0")
    monkeypatch.setattr(cache_backend, "_get_client", lambda: fake)

    payload_a = {"a": 1}
    payload_b = {"b": 2}

    cache_backend.cache_store(cache_backend.QUERY_NAMESPACE, payload_a, {"answer": "A"})
    cache_backend.cache_store(cache_backend.AGENT_NAMESPACE, payload_b, {"answer": "B"})

    removed = cache_backend.invalidate_namespace(cache_backend.QUERY_NAMESPACE)
    assert removed == 1

    fetched, _, available = cache_backend.cache_fetch(cache_backend.QUERY_NAMESPACE, payload_a)
    assert available is True
    assert fetched is None

    # other namespace untouched
    other, _, _ = cache_backend.cache_fetch(cache_backend.AGENT_NAMESPACE, payload_b)
    assert other == {"answer": "B"}
