# triage.py
from __future__ import annotations

from typing import Dict

_GENERIC = {
    "oi", "ola", "olá", "hello", "hi", "teste", "ping",
    "bom dia", "boa tarde", "boa noite", "tudo bem", "como vai"
}

def run_triage(question: str) -> Dict[str, str]:
    q = (question or "").strip().lower()

    if not q or len(q) < 2:
        return {
            "action": "ask_clarification",
            "message": "Não entendi. Pode escrever sua pergunta?"
        }

    # só pede esclarecimento se for literalmente genérico
    if q in _GENERIC:
        return {
            "action": "ask_clarification",
            "message": "Pode detalhar melhor a sua pergunta?"
        }

    return {"action": "answer"}
