"""Centralised Redis-backed cache helpers for response caching."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency at runtime
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


_LOCK = threading.Lock()
_REDIS_CLIENT: Optional["redis.Redis"] = None

QUERY_NAMESPACE = "rag:query"
AGENT_NAMESPACE = "rag:agent"


def _get_client() -> Optional["redis.Redis"]:
    """Return a cached Redis client if REDIS_URL and redis-py are available."""
    url = os.getenv("REDIS_URL")
    if not url or redis is None:
        return None

    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT

    with _LOCK:
        if _REDIS_CLIENT is None:
            try:
                _REDIS_CLIENT = redis.Redis.from_url(url, decode_responses=True)
            except Exception:
                _REDIS_CLIENT = None
        return _REDIS_CLIENT


def _cache_ttl() -> int:
    try:
        ttl = int(os.getenv("CACHE_TTL_SECONDS", "43200"))
    except Exception:
        ttl = 43200
    return max(ttl, 0)


def _stable_payload(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _make_key(namespace: str, payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_payload(payload).encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def cache_fetch(namespace: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """Fetch a cached JSON payload. Returns (value, key, cache_available)."""
    client = _get_client()
    key = _make_key(namespace, payload)

    if client is None:
        return None, key, False

    try:
        raw = client.get(key)
    except Exception:
        return None, key, True

    if raw is None:
        return None, key, True

    try:
        return json.loads(raw), key, True
    except Exception:
        return None, key, True


def cache_store(namespace: str, payload: Dict[str, Any], value: Dict[str, Any], *, key: Optional[str] = None) -> bool:
    client = _get_client()
    if client is None:
        return False

    if key is None:
        key = _make_key(namespace, payload)

    try:
        encoded = json.dumps(value, ensure_ascii=False)
    except Exception:
        return False

    ttl = _cache_ttl()
    try:
        if ttl > 0:
            client.setex(key, ttl, encoded)
        else:
            client.set(key, encoded)
    except Exception:
        return False
    return True


def invalidate_namespace(namespace: str) -> int:
    """Remove all keys for a namespace. Returns the number of keys deleted."""
    client = _get_client()
    if client is None:
        return 0

    pattern = f"{namespace}:*"
    removed = 0
    try:
        for batch in client.scan_iter(match=pattern, count=500):
            if not batch:
                continue
            if isinstance(batch, (list, tuple, set)):
                keys = list(batch)
            else:
                keys = [batch]
            if keys:
                removed += client.delete(*keys)
    except Exception:
        return removed
    return removed


def invalidate_all_responses() -> int:
    """Invalidate both query and agent response caches."""
    total = 0
    total += invalidate_namespace(QUERY_NAMESPACE)
    total += invalidate_namespace(AGENT_NAMESPACE)
    return total


def reset_cache_client() -> None:
    """Used in tests to drop the memoised Redis client."""
    global _REDIS_CLIENT
    with _LOCK:
        _REDIS_CLIENT = None


__all__ = [
    "AGENT_NAMESPACE",
    "QUERY_NAMESPACE",
    "cache_fetch",
    "cache_store",
    "invalidate_namespace",
    "invalidate_all_responses",
    "reset_cache_client",
]
