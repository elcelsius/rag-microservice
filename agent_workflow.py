# agent_workflow.py
# Este módulo define o fluxo de trabalho do agente de IA usando LangGraph para processar perguntas.
# Ele orquestra a triagem de perguntas, a recuperação de informações (RAG) e a geração de respostas.

import os
from typing import TypedDict, Literal, List, Optional, Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# --- Importações Refatoradas ---
# Reutiliza o cliente LLM robusto e a função de carregar prompt do módulo central.
from llm_client import call_llm, load_prompt
# Importa a função RAG principal.
from query_handler import answer_question, CONFIDENCE_MIN_AGENT

# --- CARREGAMENTO DOS PROMPTS ---
# Carrega os prompts usando a função centralizada.
TRIAGEM_PROMPT = load_prompt("triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("pedir_info_prompt.txt")
REFINE_PROMPT_TEMPLATE = load_prompt("refine_query_prompt.txt")


def _int_env(name: str, default: int, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except Exception:
        value = default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _float_env(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except Exception:
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    return value


AGENT_REFINE_ENABLED = os.getenv("AGENT_REFINE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
AGENT_REFINE_MAX_ATTEMPTS = _int_env("AGENT_REFINE_MAX_ATTEMPTS", 2, minimum=0, maximum=5)
AGENT_REFINE_CONFIDENCE = _float_env("AGENT_REFINE_CONFIDENCE", 0.55, minimum=0.0)

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
    rag_confidence: float
    acao_final: str
    # --- Injeção de Dependência ---
    # Os modelos necessários para o RAG são passados no estado inicial.
    embeddings_model: Optional[Any]
    vectorstore: Optional[Any]
    # --- Auto refine ---
    consulta_atual: Optional[str]
    rag_result: Optional[Dict[str, Any]]
    refine_attempts: int
    refine_history: List[str]
    refine_generated_new: bool
    confidence_override: Optional[float]
    meta: Dict[str, Any]


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

    confidence_override = state.get("confidence_override")
    if confidence_override is None:
        confidence_override = CONFIDENCE_MIN_AGENT

    consulta_atual = state.get("consulta_atual")

    # Lógica para criar uma pergunta autônoma a partir do histórico de conversa.
    if not consulta_atual:
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
            standalone_question, _ = call_llm(
                system_prompt="Você é um assistente de reformulação de perguntas.",
                user_prompt=prompt
            )
            if not standalone_question:
                standalone_question = state["pergunta"]  # Fallback
            consulta_atual = standalone_question.strip()
            print(f"--- Agente: Pergunta autônoma gerada -> '{consulta_atual}' ---")
        else:
            consulta_atual = state["pergunta"]

    # Chama a função RAG com os modelos injetados.
    resultado_rag = answer_question(
        consulta_atual,
        embeddings_model,
        vectorstore,
        debug=False,
        confidence_min=confidence_override,
    )

    rag_success = bool(resultado_rag.get("context_found"))
    rag_confidence = float(resultado_rag.get("confidence") or 0.0)

    next_state: AgentState = {
        "resposta": resultado_rag.get("answer", "Não foi possível encontrar uma resposta."),
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": rag_success,
        "consulta_atual": consulta_atual,
        "rag_confidence": rag_confidence,
        "rag_result": resultado_rag,
        "refine_generated_new": False,
        "confidence_override": confidence_override,
    }
    if rag_success:
        next_state["acao_final"] = "AUTO_RESOLVER"
    return next_state


def _build_refine_diagnostic(state: AgentState) -> str:
    reasons: List[str] = []
    confidence = float(state.get("rag_confidence") or 0.0)
    if not state.get("rag_sucesso"):
        reasons.append("Nenhum contexto relevante foi recuperado.")
    if confidence > 0:
        reasons.append(f"Confiança baixa ({confidence:.2f}).")
    if state.get("refine_attempts", 0) > 0:
        reasons.append("Refinamento anterior não retornou resposta satisfatória.")
    return "\n".join(reasons) or "Confiança insuficiente para responder com segurança."


def node_auto_refine(state: AgentState) -> AgentState:
    """Gera uma nova consulta quando o RAG não foi convincente."""
    attempts = int(state.get("refine_attempts", 0))
    consulta_anterior = state.get("consulta_atual") or state.get("pergunta") or ""

    print(f"--- Agente: Tentativa de auto_refine #{attempts + 1} ---")

    diagnostico = _build_refine_diagnostic(state)
    resposta = state.get("resposta") or "Sem resposta relevante."
    user_prompt = (
        f"{REFINE_PROMPT_TEMPLATE.strip()}"
        f"\n\nPergunta original: {state.get('pergunta', '').strip()}"
        f"\nConsulta anterior: {consulta_anterior.strip()}"
        f"\nResposta obtida: {resposta.strip()}"
        f"\nDiagnóstico: {diagnostico.strip()}\n"
    )

    new_query, _ = call_llm(
        system_prompt="Você gera consultas curtas, precisas e orientadas à busca documental.",
        user_prompt=user_prompt,
        max_tokens=64,
    )
    new_query = (new_query or "").strip()
    generated_new = bool(new_query) and new_query.lower() != consulta_anterior.strip().lower()
    consulta_final = new_query if generated_new else consulta_anterior

    history = list(state.get("refine_history", []))
    history.append(consulta_final)

    return {
        "consulta_atual": consulta_final,
        "refine_attempts": attempts + 1,
        "refine_history": history,
        "refine_generated_new": generated_new,
        "resposta": None,
        "citacoes": [],
        "rag_sucesso": False,
        "rag_confidence": 0.0,
        "rag_result": None,
        "confidence_override": state.get("confidence_override"),
    }

def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que usa o LLM para formular uma resposta pedindo mais informações."""
    print("--- Agente: Executando nó de pedir_info... ---")

    raw_options = None
    rag_result = state.get("rag_result") or {}
    if isinstance(rag_result, dict):
        raw_options = rag_result.get("clarification_options")
    if not raw_options and state.get("citacoes"):
        raw_options = [c.get("preview") for c in state.get("citacoes", []) if isinstance(c, dict)]

    if isinstance(raw_options, (list, tuple)):
        formatted_options = "\n".join(f"- {opt}" for opt in raw_options if opt)
    elif isinstance(raw_options, str) and raw_options.strip():
        formatted_options = raw_options.strip()
    else:
        formatted_options = "- (sem opcoes adicionais encontradas)"

    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(
        pergunta=state["pergunta"],
        opcoes=formatted_options,
    )

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
        "acao_final": "PEDIR_INFO",
        "rag_sucesso": False,
        "rag_confidence": 0.0,
    }


# --- LÓGICA CONDICIONAL (ARESTAS) ---

def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    """Decide qual caminho seguir após a triagem inicial."""
    print("--- Agente: Decidindo rota pós-triagem... ---")
    if state.get("triagem", {}).get("decisao") == "AUTO_RESOLVER":
        return "auto"
    return "info"


def decidir_pos_auto_resolver(state: AgentState) -> Literal["ok", "refine", "info"]:
    """Decide o que fazer após a tentativa de RAG."""
    print("--- Agente: Decidindo rota pós-RAG... ---")
    rag_success = bool(state.get("rag_sucesso"))
    confidence = float(state.get("rag_confidence") or 0.0)
    attempts = int(state.get("refine_attempts", 0))

    if rag_success and confidence >= AGENT_REFINE_CONFIDENCE:
        print("--- Agente: RAG bem-sucedido com confiança suficiente. ---")
        return "ok"

    can_refine = AGENT_REFINE_ENABLED and attempts < AGENT_REFINE_MAX_ATTEMPTS
    should_refine = (not rag_success) or (confidence < AGENT_REFINE_CONFIDENCE)

    if can_refine and should_refine:
        print("--- Agente: Tentará auto-refine antes de pedir mais informações. ---")
        return "refine"

    print("--- Agente: Refinamento não possível/suficiente. Solicitando mais informações. ---")
    return "info"


def decidir_pos_refine(state: AgentState) -> Literal["retry", "info"]:
    """Decide próximo passo após gerar uma reformulação."""
    attempts = int(state.get("refine_attempts", 0))
    generated_new = bool(state.get("refine_generated_new"))

    if attempts > AGENT_REFINE_MAX_ATTEMPTS:
        return "info"
    if not generated_new:
        print("--- Agente: Refinamento não gerou consulta nova. ---")
        return "info"
    return "retry"


# --- CONSTRUÇÃO E COMPILAÇÃO DO GRAFO ---

def create_agent_workflow():
    """Cria e compila o grafo do agente."""
    workflow = StateGraph(AgentState)

    workflow.add_node("triagem_node", node_triagem)
    workflow.add_node("auto_resolver_node", node_auto_resolver)
    workflow.add_node("pedir_info_node", node_pedir_info)
    workflow.add_node("auto_refine_node", node_auto_refine)

    workflow.add_edge(START, "triagem_node")

    workflow.add_conditional_edges("triagem_node", decidir_pos_triagem, {
        "auto": "auto_resolver_node",
        "info": "pedir_info_node"
    })

    workflow.add_conditional_edges("auto_resolver_node", decidir_pos_auto_resolver, {
        "ok": END,
        "refine": "auto_refine_node",
        "info": "pedir_info_node"
    })

    workflow.add_conditional_edges("auto_refine_node", decidir_pos_refine, {
        "retry": "auto_resolver_node",
        "info": "pedir_info_node",
    })

    workflow.add_edge("pedir_info_node", END)

    return workflow.compile()

# Compila o grafo na inicialização do módulo para que esteja pronto para uso.
compiled_graph = create_agent_workflow()

# --- FUNÇÃO DE INVOCAÇÃO PRINCIPAL ---

def run_agent(pergunta: str, messages: List[BaseMessage], embeddings_model: Any, vectorstore: Any, *, confidence_min: float | None = None) -> dict:
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
        "refine_attempts": 0,
        "refine_history": [],
        "rag_confidence": 0.0,
        "confidence_override": confidence_min,
    }
    
    # Invoca o grafo com o estado inicial.
    final_state = compiled_graph.invoke(initial_state)

    meta = {
        "refine_attempts": final_state.get("refine_attempts", 0),
        "refine_history": final_state.get("refine_history", []),
        "refine_success": bool(final_state.get("rag_sucesso")),
        "confidence": float(final_state.get("rag_confidence") or 0.0),
    }

    action = final_state.get("acao_final")
    if not action:
        action = "AUTO_RESOLVER" if final_state.get("rag_sucesso") else "UNKNOWN"

    # Retorna a saída formatada.
    return {
        "answer": final_state.get("resposta"),
        "citations": final_state.get("citacoes", []),
        "action": action,
        "meta": meta,
    }
