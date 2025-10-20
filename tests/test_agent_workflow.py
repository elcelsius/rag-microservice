"""Tests for the auto-refine agent workflow."""

from __future__ import annotations

from collections import deque

import pytest

import agent_workflow


@pytest.fixture(autouse=True)
def restore_constants(monkeypatch):
    monkeypatch.setattr(agent_workflow, "AGENT_REFINE_ENABLED", True)
    monkeypatch.setattr(agent_workflow, "AGENT_REFINE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(agent_workflow, "AGENT_REFINE_CONFIDENCE", 0.6)


def test_auto_refine_happy_path(monkeypatch):
    calls = []

    def fake_answer(question, *_args, **_kwargs):
        calls.append((question, _kwargs.get("confidence_min")))
        if len(calls) == 1:
            return {
                "answer": "Ainda não encontrei",
                "citations": [],
                "context_found": False,
                "confidence": 0.2,
            }
        return {
            "answer": "Biblioteca Central fica no bloco A",
            "citations": [{"id": 1}],
            "context_found": True,
            "confidence": 0.82,
        }

    def fake_call_llm(system_prompt, user_prompt, expect_json=False, **_kwargs):
        if expect_json:
            return "", {"decisao": "AUTO_RESOLVER"}
        if "Diagnóstico" in user_prompt:
            return "biblioteca central bloco a", None
        return "", None

    monkeypatch.setattr(agent_workflow, "answer_question", fake_answer)
    monkeypatch.setattr(agent_workflow, "call_llm", fake_call_llm)

    result = agent_workflow.run_agent(
        "Onde fica a biblioteca?",
        messages=[],
        embeddings_model=object(),
        vectorstore=object(),
    )

    assert len(calls) == 2
    assert calls[0][1] == agent_workflow.AGENT_REFINE_CONFIDENCE
    assert result["action"] == "AUTO_RESOLVER"
    assert result["meta"]["refine_attempts"] == 1
    assert result["meta"]["refine_success"] is True
    assert "Biblioteca Central" in result["answer"]
    assert result["meta"]["query_hash"]
    assert len(result["meta"]["refine_prompt_hashes"]) >= 1
    assert result["meta"]["max_refine_allowed"] == agent_workflow.AGENT_REFINE_MAX_ATTEMPTS


def test_auto_refine_falls_back_to_pedir_info(monkeypatch):
    monkeypatch.setattr(agent_workflow, "AGENT_REFINE_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(agent_workflow, "AGENT_REFINE_CONFIDENCE", 0.9)

    calls = deque()

    def fake_answer(question, *_args, **_kwargs):
        calls.append((question, _kwargs.get("confidence_min")))
        return {
            "answer": "Sem dados",
            "citations": [],
            "context_found": False,
            "confidence": 0.1,
        }

    def fake_call_llm(system_prompt, user_prompt, expect_json=False, **_kwargs):
        if expect_json:
            return "", {"decisao": "AUTO_RESOLVER"}
        if "Diagnóstico" in user_prompt:
            return "Onde fica a biblioteca?", None  # mesma pergunta -> sem novo refinamento
        if "assistente prestativo" in system_prompt:
            return "Pode indicar o departamento?", None
        return "", None

    monkeypatch.setattr(agent_workflow, "answer_question", fake_answer)
    monkeypatch.setattr(agent_workflow, "call_llm", fake_call_llm)

    result = agent_workflow.run_agent(
        "Onde fica a biblioteca?",
        messages=[],
        embeddings_model=object(),
        vectorstore=object(),
    )

    assert len(calls) == 1
    assert calls[0][1] == agent_workflow.AGENT_REFINE_CONFIDENCE
    assert result["action"] == "PEDIR_INFO"
    assert result["meta"]["refine_attempts"] == 1
    assert result["meta"]["refine_success"] is False
    assert result["meta"]["max_refine_allowed"] == 1
    assert result["meta"]["query_hash"]
    assert "departamento" in result["answer"]


def test_run_agent_accepts_confidence_override(monkeypatch):
    captured = {}

    def fake_answer(question, *_args, **_kwargs):
        captured['confidence_min'] = _kwargs.get('confidence_min')
        return {
            'answer': 'Tudo certo',
            'citacoes': [],
            'context_found': True,
            'confidence': 0.95,
        }

    def fake_call_llm(system_prompt, user_prompt, expect_json=False, **_kwargs):
        if expect_json:
            return '', {'decisao': 'AUTO_RESOLVER'}
        return '', None

    monkeypatch.setattr(agent_workflow, 'answer_question', fake_answer)
    monkeypatch.setattr(agent_workflow, 'call_llm', fake_call_llm)

    result = agent_workflow.run_agent(
        'qual o horário?',
        messages=[],
        embeddings_model=object(),
        vectorstore=object(),
        confidence_min=0.87,
    )

    assert captured['confidence_min'] == 0.87
    assert result['meta']['refine_attempts'] == 0
    assert result['meta']['refine_success'] is True
    assert result['meta']['max_refine_allowed'] == agent_workflow.AGENT_REFINE_MAX_ATTEMPTS
    assert result['meta']['query_hash']


def test_max_refine_override_limits_attempts(monkeypatch):
    calls = []

    def fake_answer(question, *_args, **_kwargs):
        calls.append(question)
        return {
            "answer": "Sem contexto",
            "citacoes": [],
            "context_found": False,
            "confidence": 0.2,
        }

    def fake_call_llm(system_prompt, user_prompt, expect_json=False, **_kwargs):
        if expect_json:
            return "", {"decisao": "AUTO_RESOLVER"}
        return "", None

    monkeypatch.setattr(agent_workflow, "answer_question", fake_answer)
    monkeypatch.setattr(agent_workflow, "call_llm", fake_call_llm)

    result = agent_workflow.run_agent(
        "Onde fica a biblioteca?",
        messages=[],
        embeddings_model=object(),
        vectorstore=object(),
        max_refine_attempts=0,
    )

    assert len(calls) == 1
    assert result["meta"]["max_refine_allowed"] == 0
    assert result["meta"]["refine_attempts"] == 0
    assert result["action"] == "PEDIR_INFO"
