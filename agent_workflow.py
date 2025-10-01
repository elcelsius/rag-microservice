# agent_workflow.py
# Este mÃ³dulo define o fluxo do agente LangGraph com etapas de triagem, execuÃ§Ã£o RAG,
# autoavaliaÃ§Ã£o (RAGAs) e correÃ§Ã£o automÃ¡tica quando necessÃ¡rio.

import os
import google.generativeai as genai
from typing import TypedDict, Literal, List, Optional

from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from ragas import evaluate
from ragas.metrics import Faithfulness

from llm_adapters import RagasGoogleApiLLM
from query_handler import answer_question

# Tenta reutilizar os modelos prÃ©-carregados pela API para evitar reloads custosos.
try:
    from api import embeddings_model, vectorstore
except (ImportError, ModuleNotFoundError):
    embeddings_model = None
    vectorstore = None

# --- ConfiguraÃ§Ãµes globais ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", 0.8))

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        # Se jÃ¡ estiver configurado em outro ponto da aplicaÃ§Ã£o, ignore.
        pass


def load_prompt(file_path: str) -> str:
    """Carrega o conteÃºdo de um arquivo de prompt relativo a este arquivo."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as handler:
            return handler.read()
    except FileNotFoundError:
        print(f"ERRO CRÃTICO: Arquivo de prompt nÃ£o encontrado em '{file_path}'")
        return ""


TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")
CORRECTION_PROMPT_TEMPLATE = load_prompt("prompts/correction_prompt.txt")


class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(default_factory=list)


llm_triagem = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

llm_for_ragas = RagasGoogleApiLLM(model_name=MODEL_NAME)
faithfulness_metric = Faithfulness(llm=llm_for_ragas)


class AgentState(TypedDict, total=False):
    pergunta: str
    messages: List[BaseMessage]
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    context: Optional[List[str]]
    rag_sucesso: bool
    faithfulness_score: Optional[float]
    acao_final: str


def node_triagem(state: AgentState) -> AgentState:
    print("--- Agente: Executando nÃ³ de triagem... ---")
    pergunta = state["pergunta"]
    saida: TriagemOut = triagem_chain.invoke(
        [SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=pergunta)]
    )
    print(f"--- Agente: DecisÃ£o da triagem -> {saida.model_dump()} ---")
    return {"triagem": saida.model_dump()}


def _condensar_pergunta(state: AgentState) -> str:
    mensagens = state.get("messages") or []
    if len(mensagens) <= 1:
        return state["pergunta"]

    try:
        print("--- Agente: Condensando pergunta a partir do histÃ³rico... ---")
        condenser_prompt = (
            "Dada a conversa abaixo e a Ãºltima pergunta do usuÃ¡rio, reformule a pergunta para que ela seja completa e autÃ´noma, "
            "contendo todo o contexto necessÃ¡rio para ser entendida sem o histÃ³rico. NÃ£o responda Ã  pergunta, apenas a reformule.\n\n"
            "HISTÃ“RICO DA CONVERSA:\n{chat_history}\n\nÃšLTIMA PERGUNTA: {question}\n\nPERGUNTA AUTÃ”NOMA:"
        )
        chat_history = "\n".join(
            [
                f"{'UsuÃ¡rio' if isinstance(msg, HumanMessage) else 'IA'}: {msg.content}"
                for msg in mensagens[:-1]
            ]
        )
        prompt = condenser_prompt.format(chat_history=chat_history, question=state["pergunta"])
        model = genai.GenerativeModel(MODEL_NAME)
        resposta = model.generate_content(prompt)
        standalone_question = (resposta.text or "").strip()
        if standalone_question:
            print(f"--- Agente: Pergunta autÃ´noma gerada -> '{standalone_question}' ---")
            return standalone_question
    except Exception as exc:
        print(f"--- Agente: Falha ao gerar pergunta autÃ´noma ({exc}). Usando original. ---")

    return state["pergunta"]


def node_auto_resolver(state: AgentState) -> AgentState:
    print("--- Agente: Executando nÃ³ de auto_resolver (RAG)... ---")

    if embeddings_model is None or vectorstore is None:
        return {
            "resposta": "O Ã­ndice vetorial ainda nÃ£o estÃ¡ pronto para uso. Execute o ETL e reinicie a API.",
            "citacoes": [],
            "rag_sucesso": False,
            "acao_final": "PEDIR_INFO",
        }

    standalone_question = _condensar_pergunta(state)

    resultado_rag = answer_question(
        query=standalone_question,
        embeddings_model=embeddings_model,
        vectorstore=vectorstore,
        debug=False,
    )

    contexto_recuperado = [c.get("preview", "") for c in resultado_rag.get("citations", [])]
    rag_success = bool(resultado_rag.get("context_found"))

    update: AgentState = {
        "resposta": resultado_rag.get("answer", "NÃ£o foi possÃ­vel encontrar uma resposta."),
        "citacoes": resultado_rag.get("citations", []),
        "context": contexto_recuperado,
        "rag_sucesso": rag_success,
        "acao_final": "AUTO_RESOLVER" if rag_success else "PEDIR_INFO",
    }
    return update


def node_self_evaluate(state: AgentState) -> AgentState:
    print("--- Agente: Executando nÃ³ de auto-avaliaÃ§Ã£o (faithfulness)... ---")
    if not state.get("context") or not state.get("resposta"):
        return {"faithfulness_score": 0.0}

    dataset = Dataset.from_dict(
        {
            "question": [state["pergunta"]],
            "answer": [state["resposta"]],
            "contexts": [state["context"]],
        }
    )

    score = evaluate(dataset, metrics=[faithfulness_metric])
    faithfulness_score = score.get("faithfulness", 0.0)
    print(f"--- Agente: Score de Faithfulness -> {faithfulness_score:.4f} ---")
    return {"faithfulness_score": faithfulness_score}


def node_correction(state: AgentState) -> AgentState:
    print("--- Agente: Executando nÃ³ de correÃ§Ã£o... ---")
    prompt = CORRECTION_PROMPT_TEMPLATE.format(
        pergunta=state["pergunta"],
        resposta_anterior=state.get("resposta", ""),
        contexto="\n".join(state.get("context", [])),
    )
    corrected_answer = llm_triagem.invoke(prompt).content
    print(f"--- Agente: Resposta corrigida -> {corrected_answer[:100]}... ---")
    return {"resposta": corrected_answer, "acao_final": "AUTO_RESOLVER"}


def node_pedir_info(state: AgentState) -> AgentState:
    print("--- Agente: Executando nÃ³ de pedir_info... ---")
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])

    clarification_text = "NÃ£o consegui entender completamente sua pergunta. Poderia fornecer mais detalhes?"
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        if response and response.text:
            clarification_text = response.text
    except Exception as exc:
        print(f"--- Agente: Falha ao gerar pergunta de esclarecimento ({exc}). ---")
        try:
            clarification_text = llm_triagem.invoke(prompt).content
        except Exception:
            pass

    return {"resposta": clarification_text, "citacoes": [], "acao_final": "PEDIR_INFO"}


def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    return "auto" if state["triagem"]["decisao"] == "AUTO_RESOLVER" else "info"


def decidir_pos_auto_resolver(state: AgentState) -> Literal["avaliar", "info"]:
    return "avaliar" if state.get("rag_sucesso") else "info"


def decidir_pos_self_evaluate(state: AgentState) -> Literal["corrigir", "fim"]:
    score = state.get("faithfulness_score", 0.0)
    print(f"--- Agente: Avaliando score {score:.4f} vs limiar {FAITHFULNESS_THRESHOLD} ---")
    return "corrigir" if score < FAITHFULNESS_THRESHOLD else "fim"


workflow = StateGraph(AgentState)
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("self_evaluate", node_self_evaluate)
workflow.add_node("correction", node_correction)

workflow.add_edge(START, "triagem")
workflow.add_edge("pedir_info", END)
workflow.add_edge("correction", END)
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {"auto": "auto_resolver", "info": "pedir_info"})
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {"avaliar": "self_evaluate", "info": "pedir_info"})
workflow.add_conditional_edges("self_evaluate", decidir_pos_self_evaluate, {"corrigir": "correction", "fim": END})

compiled_graph = workflow.compile()
