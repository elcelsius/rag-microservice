# agent_workflow.py
import os
import google.generativeai as genai
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from query_handler import find_answer_for_query

# --- CONFIGURAÇÃO DO MODELO ---
# Carrega as configurações do ambiente, com valores padrão
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# --- DEFINIÇÃO DA TRIAGEM (SEM "ABRIR_CHAMADO") ---

TRIAGEM_PROMPT = (
    "Você é um triador de perguntas para o assistente IA Compilot. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com a seguinte estrutura:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Use para perguntas claras e diretas que podem ser respondidas pela base de conhecimento (Ex: "Como funciona a política de home office?", "Qual o procedimento para reembolso de despesas?").\n'
    '- **PEDIR_INFO**: Use para mensagens vagas, incompletas ou que não fornecem contexto suficiente para uma busca (Ex: "Preciso de ajuda", "Tenho uma dúvida sobre uma política", "E sobre meu benefício?").'
    "Analise a mensagem e decida a ação mais apropriada."
)


class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(default_factory=list)


# --- INICIALIZAÇÃO DO LLM PARA TRIAGEM ---
llm_triagem = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)
triagem_chain = llm_triagem.with_structured_output(TriagemOut)


# --- DEFINIÇÃO DO ESTADO DO GRAFO ---

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str


# --- NÓS DO GRAFO ---

def node_triagem(state: AgentState) -> AgentState:
    """Primeiro nó: classifica a pergunta do usuário."""
    print("--- Agente: Executando nó de triagem... ---")
    pergunta = state["pergunta"]
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=pergunta)
    ])
    print(f"--- Agente: Decisão da triagem -> {saida.model_dump()} ---")
    return {"triagem": saida.model_dump()}


def node_auto_resolver(state: AgentState) -> AgentState:
    """Nó que executa a lógica RAG para tentar responder a pergunta."""
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")
    pergunta = state["pergunta"]

    # A função find_answer_for_query foi modificada para retornar um dicionário
    resultado_rag = find_answer_for_query(pergunta)

    update: AgentState = {
        "resposta": resultado_rag["answer"],
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": resultado_rag["context_found"],
    }

    if resultado_rag["context_found"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update


def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que formula uma resposta pedindo mais informações."""
    print("--- Agente: Executando nó de pedir_info... ---")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ", ".join(faltantes) if faltantes else "mais detalhes sobre sua dúvida"

    return {
        "resposta": f"Não consegui entender completamente sua pergunta. Por favor, forneça {detalhe} para que eu possa ajudar.",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }


# --- LÓGICA CONDICIONAL (ARESTAS) ---

def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    """Decide qual caminho seguir após a triagem inicial."""
    print("--- Agente: Decidindo rota pós-triagem... ---")
    if state["triagem"]["decisao"] == "AUTO_RESOLVER":
        return "auto"
    return "info"


def decidir_pos_auto_resolver(state: AgentState) -> Literal["ok", "info"]:
    """Decide o que fazer se o RAG teve sucesso ou falhou."""
    print("--- Agente: Decidindo rota pós-RAG... ---")
    if state.get("rag_sucesso"):
        print("--- Agente: RAG bem-sucedido. Finalizando. ---")
        return "ok"
    print("--- Agente: RAG falhou. Solicitando mais informações. ---")
    return "info"


# --- CONSTRUÇÃO E COMPILAÇÃO DO GRAFO ---

workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)

workflow.add_edge(START, "triagem")

workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "ok": END,
    "info": "pedir_info"
})

workflow.add_edge("pedir_info", END)

# Compila o grafo para que possa ser usado pela API
compiled_graph = workflow.compile()