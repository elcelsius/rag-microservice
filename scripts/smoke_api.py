#!/usr/bin/env python3
"""Smoke test hitting /query, /agent/ask and /metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import requests

DEFAULT_BASE = "http://localhost:8080/api"


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str


def _http_post(url: str, payload: dict) -> requests.Response:
    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    return resp


def _http_get(url: str) -> requests.Response:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp


def check_query(base: str) -> SmokeResult:
    try:
        resp = _http_post(f"{base}/query", {"question": "onde fica a biblioteca?", "debug": False})
        data = resp.json()
        if "answer" not in data:
            return SmokeResult("/query", False, "faltou campo 'answer'")
        return SmokeResult("/query", True, "ok")
    except Exception as exc:  # pragma: no cover - smoke script
        return SmokeResult("/query", False, str(exc))


def check_agent(base: str) -> SmokeResult:
    try:
        payload = {
            "question": "onde fica a biblioteca?",
            "messages": [],
        }
        resp = _http_post(f"{base}/agent/ask", payload)
        data = resp.json()
        if "answer" not in data or "meta" not in data:
            return SmokeResult("/agent/ask", False, "faltou 'answer' ou 'meta'")
        return SmokeResult("/agent/ask", True, "ok")
    except Exception as exc:  # pragma: no cover
        return SmokeResult("/agent/ask", False, str(exc))


def check_metrics(base: str) -> SmokeResult:
    try:
        resp = _http_get(f"{base}/metrics")
        data = resp.json()
        counters = data.get("counters", {})
        expected = {"cache_hits_total", "cache_misses_total", "agent_refine_attempts_total"}
        missing = sorted(expected - counters.keys())
        if missing:
            return SmokeResult("/metrics", False, f"counters faltando: {', '.join(missing)}")
        return SmokeResult("/metrics", True, "ok")
    except Exception as exc:  # pragma: no cover
        return SmokeResult("/metrics", False, str(exc))


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE
    results = [
        check_query(base),
        check_agent(base),
        check_metrics(base),
    ]

    failed = [r for r in results if not r.ok]
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[SMOKE] {r.name:12s} {status} - {r.detail}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
