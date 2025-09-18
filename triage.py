# triage.py
# Este módulo contém uma função de triagem simples para classificar perguntas do usuário.
# Ele verifica se a pergunta é genérica ou muito curta, solicitando esclarecimentos se necessário.

from __future__ import annotations

from typing import Dict

# Conjunto de termos genéricos que, se a pergunta for exatamente um deles, indicam a necessidade de esclarecimento.
_GENERIC = {
    "oi", "ola", "olá", "hello", "hi", "teste", "ping",
    "bom dia", "boa tarde", "boa noite", "tudo bem", "como vai"
}

def run_triage(question: str) -> Dict[str, str]:
    """Realiza uma triagem inicial da pergunta do usuário.
    
    Verifica se a pergunta é muito curta ou se consiste em uma saudação genérica.
    
    Args:
        question (str): A pergunta original do usuário.
        
    Returns:
        Dict[str, str]: Um dicionário indicando a ação a ser tomada ("ask_clarification" ou "answer")
                       e uma mensagem, se for o caso.
    """
    q = (question or "").strip().lower()

    # Se a pergunta for vazia ou muito curta, pede esclarecimento.
    if not q or len(q) < 2:
        return {
            "action": "ask_clarification",
            "message": "Não entendi. Pode escrever sua pergunta?"
        }

    # Se a pergunta for uma das frases genéricas, pede esclarecimento.
    if q in _GENERIC:
        return {
            "action": "ask_clarification",
            "message": "Pode detalhar melhor a sua pergunta?"
        }

    # Caso contrário, a pergunta é considerada válida para ser respondida.
    return {"action": "answer"}


