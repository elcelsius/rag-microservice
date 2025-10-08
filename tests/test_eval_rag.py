"""Tests for eval_rag helper functions."""

from __future__ import annotations

import asyncio

import eval_rag
from eval_rag import _compute_generation_metrics, evaluate_endpoint


class DummyScores:
    def __init__(self, data: dict[str, list[float]]):
        self._data = data

    def to_dict(self) -> dict[str, list[float]]:
        return self._data


class DummyResult:
    def __init__(self, data: dict[str, list[float]]):
        self.scores = DummyScores(data)


def _run(coro):
    return asyncio.run(coro)


def test_compute_generation_metrics_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    async def fake_evaluate(**kwargs):  # pragma: no cover - should not run
        raise AssertionError("evaluate should be skipped when API key is missing")

    metrics, available, message = _run(
        _compute_generation_metrics(
            dataset=object(),
            ragas_evaluate=fake_evaluate,
            ragas_metrics=[],
            llm_factory=lambda: object(),
        )
    )

    assert metrics == {}
    assert available is False
    assert "GOOGLE_API_KEY" in message


def test_compute_generation_metrics_with_stubbed_dependencies(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")

    async def fake_evaluate(**kwargs):
        return DummyResult({
            "faithfulness": [0.2, 0.4],
            "answer_relevancy": [1.0],
        })

    metrics, available, message = _run(
        _compute_generation_metrics(
            dataset=object(),
            ragas_evaluate=fake_evaluate,
            ragas_metrics=["m1"],
            llm_factory=lambda: object(),
        )
    )

    assert available is True
    assert message == ""
    assert metrics == {"faithfulness": 0.3, "answer_relevancy": 1.0}


def test_evaluate_endpoint_uses_generation_metrics(monkeypatch, capfd):
    dataset_records = [
        {"question": "Quem?", "ground_truth": ["foo"]},
    ]

    async def fake_compute(dataset):
        # validate that the helper receives a datasets.Dataset
        assert getattr(dataset, "num_rows", 0) == 1
        return {"faithfulness": 0.7}, False, "WARN: stub message"

    monkeypatch.setattr(eval_rag, "_compute_generation_metrics", fake_compute)
    monkeypatch.setattr(
        eval_rag,
        "query_api",
        lambda url, question: {"answer": "foo", "citations": [{"preview": "foo"}]},
    )

    result = _run(evaluate_endpoint(dataset_records, "http://stub", "stub"))
    output = capfd.readouterr().out

    assert result["ragas_available"] is False
    assert result["generation_metrics_ragas"] == {"faithfulness": 0.7}
    assert "WARN: stub message" in output
