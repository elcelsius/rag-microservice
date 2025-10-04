# triage.py
# Este módulo implementa uma função de triagem inicial para classificar a entrada do usuário.
# O objetivo é identificar rapidamente interações que não são perguntas reais (como saudações
# ou entradas muito curtas) antes de acionar o pipeline de RAG, que consome mais recursos.
# Isso ajuda a economizar custos de API e a melhorar a experiência do usuário, fornecendo
# respostas rápidas e pedindo esclarecimentos quando necessário.

from __future__ import annotations

from typing import Dict

# Um conjunto de termos e saudações comuns que não constituem uma pergunta.
# Se a entrada do usuário corresponder exatamente a um desses termos (ignorando maiúsculas/minúsculas),
# a triagem irá solicitar mais detalhes.
_GENERIC = {
    "oi", "ola", "olá", "hello", "hi", "teste", "ping",
    "bom dia", "boa tarde", "boa noite", "tudo bem", "como vai"
}

def run_triage(question: str) -> Dict[str, str]:
    """
    Executa uma verificação preliminar na pergunta do usuário.

    Esta função atua como um "portão de entrada", decidindo se a pergunta
    deve prosseguir para o pipeline de RAG ("action": "answer") ou se o
    sistema deve pedir ao usuário para reformular a pergunta ("action": "ask_clarification").

    Args:
        question (str): A pergunta original enviada pelo usuário.

    Returns:
        Um dicionário com duas chaves:
        - "action": A ação recomendada ('answer' ou 'ask_clarification').
        - "message": Uma mensagem para o usuário (presente apenas se a ação for 'ask_clarification').
    """
    # Normaliza a pergunta: remove espaços em branco e converte para minúsculas.
    q = (question or "").strip().lower()

    # Filtro 1: Verifica se a pergunta é vazia ou muito curta para ser significativa.
    if not q or len(q) < 2:
        return {
            "action": "ask_clarification",
            "message": "Não entendi. Pode escrever sua pergunta?"
        }

    # Filtro 2: Verifica se a pergunta é uma saudação genérica contida no conjunto _GENERIC.
    if q in _GENERIC:
        return {
            "action": "ask_clarification",
            "message": "Pode detalhar melhor a sua pergunta?"
        }

    # Se a pergunta passar por todos os filtros, é considerada válida e pode ser respondida.
    return {"action": "answer"}