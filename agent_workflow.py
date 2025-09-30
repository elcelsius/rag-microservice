# agent_workflow.py
# Este módulo define o fluxo de trabalho do agente de IA usando LangGraph para processar perguntas,
# incorporando um passo de auto-avaliação e correção para aumentar a confiabilidade.

import os
import asyncio
from typing import TypedDict, Literal, List, Optional

from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics import Faithfulness

# Importa o adaptador de LLM customizado
from llm_adapters import RagasGoogleApiLLM
from query_handler import answer_question

# Tenta reutilizar os modelos pré-carregados pela API para otimização.
try:
    from api import embeddings_model, vectorstore
except (ImportError, ModuleNotFoundError):
    embeddings_model = None
    vectorstore = None

# --- CONFIGURAÇÃO DO MODELO E AVALIAÇÃO ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", 0.8))

# --- FUNÇÃO AUXILIAR PARA CARREGAR PROMPTS ---
def load_prompt(file_path: str) -> str:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de prompt não encontrado em '{full_path}'")
        return ""

# --- CARREGAMENTO DOS PROMPTS ---
TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")
CORRECTION_PROMPT_TEMPLATE = load_prompt("prompts/correction_prompt.txt")

# --- DEFINIÇÃO DO ESQUEMA DE SAÍDA PARA A TRIAGEM ---
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(default_factory=list)

# --- INICIALIZAÇÃO DOS LLMS E MÉTRICAS ---
llm_triagem = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, api_key=GOOGLE_API_KEY)
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

# LLM para o RAGAs, usando nosso adaptador customizado
llm_for_ragas = RagasGoogleApiLLM(model_name=MODEL_NAME)
faithfulness_metric = Faithfulness(llm=llm_for_ragas)


# --- DEFINIÇÃO DO ESTADO DO GRAFO ---
class AgentState(TypedDict, total=False):
    pergunta: str
    messages: List[BaseMessage]
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    context: Optional[List[str]] # NOVO: Armazena o contexto recuperado
    rag_sucesso: bool
    faithfulness_score: Optional[float] # NOVO: Armazena o score de avaliação
    acao_final: str


# --- NÓS DO GRAFO ---
def node_triagem(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de triagem... ---")
    pergunta = state["pergunta"]
    saida: TriagemOut = triagem_chain.invoke([SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=pergunta)])
    print(f"--- Agente: Decisão da triagem -> {saida.model_dump()} ---")
    return {"triagem": saida.model_dump()}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")
    if embeddings_model is None or vectorstore is None:
        return {"resposta": "O índice vetorial não está pronto.", "citacoes": [], "rag_sucesso": False, "acao_final": "PEDIR_INFO"}

    standalone_question = state["pergunta"]
    
    resultado_rag = answer_question(query=standalone_question, embeddings_model=embeddings_model, vectorstore=vectorstore, debug=False)
    
    # Extrai o contexto (previews) das citações para a avaliação
    contexto_recuperado = [c.get("preview", "") for c in resultado_rag.get("citations", [])]

    rag_success = bool(resultado_rag.get("context_found"))
    return {
        "resposta": resultado_rag.get("answer", "Não foi possível encontrar uma resposta."),
        "citacoes": resultado_rag.get("citations", []),
        "context": contexto_recuperado, # Salva o contexto no estado
        "rag_sucesso": rag_success,
    }

# --- NOVOS NÓS: AUTO-AVALIAÇÃO E CORREÇÃO ---
def node_self_evaluate(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de auto-avaliação (faithfulness)... ---")
    if not state.get("context") or not state.get("resposta"):
        return {"faithfulness_score": 0.0}

    dataset = Dataset.from_dict({
        "question": [state["pergunta"]],
        "answer": [state["resposta"]],
        "contexts": [state["context"]],
    })

    score = evaluate(dataset, metrics=[faithfulness_metric])
    faithfulness_score = score.get("faithfulness", 0.0)
    print(f"--- Agente: Score de Faithfulness -> {faithfulness_score:.4f} ---")
    return {"faithfulness_score": faithfulness_score}

def node_correction(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de correção... ---")
    prompt = CORRECTION_PROMPT_TEMPLATE.format(
        pergunta=state["pergunta"],
        resposta_anterior=state["resposta"],
        contexto="\n".join(state.get("context", []))
    )
    # Reutiliza o LLM da triagem para a correção
    corrected_answer = llm_triagem.invoke(prompt).content
    print(f"--- Agente: Resposta corrigida -> {corrected_answer[:100]}... ---")
    # Mantém as citações originais, mas atualiza a resposta
    return {"resposta": corrected_answer, "acao_final": "AUTO_RESOLVER"}

def node_pedir_info(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de pedir_info... ---")
    # Reutiliza o LLM da triagem para gerar a pergunta de esclarecimento
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])
    clarification_text = llm_triagem.invoke(prompt).content
    return {"resposta": clarification_text, "citacoes": [], "acao_final": "PEDIR_INFO"}


# --- LÓGICA CONDICIONAL (ARESTAS) ---
def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    return "auto" if state["triagem"]["decisao"] == "AUTO_RESOLVER" else "info"

def decidir_pos_auto_resolver(state: AgentState) -> Literal["avaliar", "info"]:
    return "avaliar" if state.get("rag_sucesso") else "info"

def decidir_pos_self_evaluate(state: AgentState) -> Literal["corrigir", "fim"]:
    score = state.get("faithfulness_score", 0.0)
    print(f"--- Agente: Avaliando score {score:.4f} contra o limiar de {FAITHFULNESS_THRESHOLD} ---")
    return "corrigir" if score < FAITHFULNESS_THRESHOLD else "fim"


# --- CONSTRUÇÃO E COMPILAÇÃO DO GRAFO ---
workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("self_evaluate", node_self_evaluate)
workflow.add_node("correction", node_correction)

workflow.add_edge(START, "triagem")
workflow.add_edge("pedir_info", END)
workflow.add_edge("correction", END) # Após a correção, o fluxo termina

workflow.add_conditional_edges("triagem", decidir_pos_triagem, {"auto": "auto_resolver", "info": "pedir_info"})

# FLUXO MODIFICADO: Após o RAG, decide se avalia ou pede info
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {"avaliar": "self_evaluate", "info": "pedir_info"})

# NOVO FLUXO: Após a avaliação, decide se corrige ou finaliza
workflow.add_conditional_edges("self_evaluate", decidir_pos_self_evaluate, {"corrigir": "correction", "fim": END})

compiled_graph = workflow.compile()
