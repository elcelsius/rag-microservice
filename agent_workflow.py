# agent_workflow.py
# Este módulo define o fluxo de trabalho do agente de IA usando LangGraph para processar perguntas.
# Ele orquestra a triagem de perguntas, a recuperação de informações (RAG) e a geração de respostas.

import os
from typing import TypedDict, Literal, List, Optional, Any

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# --- Importações Refatoradas ---
# Reutiliza o cliente LLM robusto e a função de carregar prompt do módulo central.
from llm_client import call_llm, load_prompt
# Importa a função RAG principal.
from query_handler import answer_question

# --- CARREGAMENTO DOS PROMPTS ---
# Carrega os prompts usando a função centralizada.
TRIAGEM_PROMPT = load_prompt("triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("pedir_info_prompt.txt")

# --- DEFINIÇÃO DO ESTADO DO GRAFO ---
# TypedDict que define a estrutura do estado que será passado entre os nós do grafo.
class AgentState(TypedDict, total=False):
    """Representa o estado atual do agente, contendo todas as informações relevantes para o fluxo de trabalho."""
    pergunta: str
    messages: List[BaseMessage]
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str
    # --- Injeção de Dependência ---
    # Os modelos necessários para o RAG são passados no estado inicial.
    embeddings_model: Optional[Any]
    vectorstore: Optional[Any]


# --- NÓS DO GRAFO (REFATORADOS) ---

def node_triagem(state: AgentState) -> AgentState:
    """Primeiro nó do grafo: classifica a pergunta do usuário usando o cliente LLM central."""
    print("--- Agente: Executando nó de triagem... ---")
    pergunta = state["pergunta"]
    
    # Usa a função call_llm central, que já tem retentativas e logging.
    # Espera uma resposta em JSON, que é parseada automaticamente.
    _, data = call_llm(
        system_prompt=TRIAGEM_PROMPT,
        user_prompt=pergunta,
        expect_json=True,
        max_tokens=150
    )

    # Fallback para garantir que a triagem sempre tenha um valor.
    if not isinstance(data, dict) or "decisao" not in data:
        data = {"decisao": "AUTO_RESOLVER", "campos_faltantes": []}

    print(f"--- Agente: Decisão da triagem -> {data} ---")
    return {"triagem": data}

def node_auto_resolver(state: AgentState) -> AgentState:
    """Nó que executa a lógica RAG para tentar responder a pergunta."""
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")

    # Verifica se os modelos foram injetados no estado.
    embeddings_model = state.get("embeddings_model")
    vectorstore = state.get("vectorstore")

    if not embeddings_model or not vectorstore:
        print("--- Agente: ERRO - Índice vetorial ou modelo de embeddings não disponível. ---")
        return {
            "resposta": "O sistema de busca ainda não está pronto. Por favor, tente novamente em alguns instantes.",
            "citacoes": [],
            "rag_sucesso": False,
        }

    # Lógica para criar uma pergunta autônoma a partir do histórico de conversa.
    if len(state.get("messages", [])) > 1:
        print("--- Agente: Condensando pergunta a partir do histórico...")
        condenser_prompt = (
            "Dada a conversa abaixo e a última pergunta do usuário, reformule a pergunta para que ela seja completa e autônoma, "
            "contendo todo o contexto necessário para ser entendida sem o histórico. Não responda à pergunta, apenas a reformule.\n\n"
            "HISTÓRICO DA CONVERSA:\n{chat_history}\n\nÚLTIMA PERGUNTA: {question}\n\nPERGUNTA AUTÔNOMA:"
        )
        chat_history_str = "\n".join(
            [f"{'Usuário' if isinstance(msg, HumanMessage) else 'IA'}: {msg.content}" for msg in state["messages"][:-1]]
        )
        prompt = condenser_prompt.format(chat_history=chat_history_str, question=state["pergunta"])
        
        # Usa o cliente LLM central para condensar a pergunta.
        standalone_question, _ = call_llm(system_prompt="Você é um assistente de reformulação de perguntas.", user_prompt=prompt)
        if not standalone_question:
            standalone_question = state["pergunta"] # Fallback
        print(f"--- Agente: Pergunta autônoma gerada -> '{standalone_question}' ---")
    else:
        standalone_question = state["pergunta"]

    # Chama a função RAG com os modelos injetados.
    resultado_rag = answer_question(standalone_question, embeddings_model, vectorstore, debug=False)

    rag_success = bool(resultado_rag.get("context_found"))
    return {
        "resposta": resultado_rag.get("answer", "Não foi possível encontrar uma resposta."),
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": rag_success,
    }

def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que usa o LLM para formular uma resposta pedindo mais informações."""
    print("--- Agente: Executando nó de pedir_info... ---")

    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])
    
    # Usa o cliente LLM central para gerar a pergunta de esclarecimento.
    clarification_text, _ = call_llm(
        system_prompt="Você é um assistente prestativo que pede informações adicionais quando uma pergunta é vaga.",
        user_prompt=prompt
    )

    if not clarification_text:
        clarification_text = "Não consegui entender completamente sua pergunta. Poderia fornecer mais detalhes?"

    return {
        "resposta": clarification_text,
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }


# --- LÓGICA CONDICIONAL (ARESTAS) ---

def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    """Decide qual caminho seguir após a triagem inicial."""
    print("--- Agente: Decidindo rota pós-triagem... ---")
    if state.get("triagem", {}).get("decisao") == "AUTO_RESOLVER":
        return "auto"
    return "info"


def decidir_pos_auto_resolver(state: AgentState) -> Literal["ok", "info"]:
    """Decide o que fazer após a tentativa de RAG."""
    print("--- Agente: Decidindo rota pós-RAG... ---")
    if state.get("rag_sucesso"):
        print("--- Agente: RAG bem-sucedido. Finalizando. ---")
        return "ok"
    print("--- Agente: RAG falhou. Solicitando mais informações. ---")
    return "info"


# --- CONSTRUÇÃO E COMPILAÇÃO DO GRAFO ---

def create_agent_workflow():
    """Cria e compila o grafo do agente."""
    workflow = StateGraph(AgentState)

    workflow.add_node("triagem_node", node_triagem)
    workflow.add_node("auto_resolver_node", node_auto_resolver)
    workflow.add_node("pedir_info_node", node_pedir_info)

    workflow.add_edge(START, "triagem_node")

    workflow.add_conditional_edges("triagem_node", decidir_pos_triagem, {
        "auto": "auto_resolver_node",
        "info": "pedir_info_node"
    })

    workflow.add_conditional_edges("auto_resolver_node", decidir_pos_auto_resolver, {
        "ok": END,
        "info": "pedir_info_node"
    })

    workflow.add_edge("pedir_info_node", END)

    return workflow.compile()

# Compila o grafo na inicialização do módulo para que esteja pronto para uso.
compiled_graph = create_agent_workflow()

# --- FUNÇÃO DE INVOCAÇÃO PRINCIPAL ---

def run_agent(pergunta: str, messages: List[BaseMessage], embeddings_model: Any, vectorstore: Any) -> dict:
    """
    Executa o fluxo de trabalho do agente para uma determinada pergunta e histórico.

    Args:
        pergunta (str): A pergunta do usuário.
        messages (List[BaseMessage]): O histórico da conversa.
        embeddings_model: O modelo de embeddings pré-carregado.
        vectorstore: O vectorstore FAISS pré-carregado.

    Returns:
        dict: Um dicionário contendo a resposta, citações e ação final.
    """
    # Monta o estado inicial, injetando as dependências (modelos).
    initial_state = {
        "pergunta": pergunta,
        "messages": messages,
        "embeddings_model": embeddings_model,
        "vectorstore": vectorstore,
    }
    
    # Invoca o grafo com o estado inicial.
    final_state = compiled_graph.invoke(initial_state)
    
    # Retorna a saída formatada.
    return {
        "answer": final_state.get("resposta"),
        "citations": final_state.get("citacoes", []),
        "action": final_state.get("acao_final", "UNKNOWN")
    }
