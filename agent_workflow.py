# agent_workflow.py
# Este módulo define o fluxo do agente LangGraph com etapas de triagem, execução RAG,
# autoavaliação (RAGAs) e correção automática quando necessário.

import os
import json
from typing import TypedDict, Literal, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic.v1 import BaseModel, Field

from query_handler import answer_question, candidate_terms, sentence_hits_by_name

# Tenta reutilizar os modelos pré-carregados pela API para evitar reloads custosos.
# try:
#     from api import embeddings_model, vectorstore
# except (ImportError, ModuleNotFoundError):
#     embeddings_model = None
#     vectorstore = None

# --- Configurações globais ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# --- Carregamento de Prompts ---
def load_prompt(file_path: str) -> str:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as handler:
            return handler.read()
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de prompt não encontrado em '{file_path}'")
        return ""

TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")
CORRECTION_PROMPT_TEMPLATE = load_prompt("prompts/correction_prompt.txt")
EVALUATION_PROMPT_TEMPLATE = load_prompt("prompts/evaluation_prompt.txt")

# --- Definição de Esquemas e Estado ---
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "type": "array",
            "items": {"type": "string"}
        }
    )

class AgentState(TypedDict, total=False):
    pergunta: str
    messages: List[BaseMessage]
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    context: Optional[List[str]]
    rag_sucesso: bool
    evaluation: Optional[dict]  # Novo campo para o resultado da avaliação
    acao_final: str

# --- Inicialização Preguiçosa (Lazy Initialization) ---
_llm_triagem = None
_triagem_chain = None
def get_triagem_chain():
    global _llm_triagem, _triagem_chain
    if _triagem_chain is None:
        _llm_triagem = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, api_key=GOOGLE_API_KEY)
        _triagem_chain = _llm_triagem.with_structured_output(TriagemOut)
    return _triagem_chain, _llm_triagem

# --- Nós do Grafo ---
def node_triagem(state: AgentState) -> AgentState:
    from api import vectorstore # IMPORT LOCAL PARA EVITAR CICLO
    print("--- Agente: Executando nó de triagem... ---")
    pergunta = state["pergunta"]

    # --- Início da Triagem Inteligente ---
    terms = candidate_terms(pergunta)
    name_terms = [t for t in terms if t.isalpha() and len(t) > 3]
    print(f"--- Triagem: Termos de nome encontrados -> {name_terms} ---")

    if name_terms and vectorstore:
        try:
            all_docs = list(getattr(vectorstore.docstore, "_dict", {}).values())
            if all_docs:
                print(f"--- Triagem: Buscando pelo nome '{name_terms[0]}' em {len(all_docs)} documentos... ---")
                detailed_hits = []
                for doc in all_docs:
                    hits = sentence_hits_by_name(doc.page_content or "", name_terms)
                    if hits:
                        source = doc.metadata.get("source", "desconhecido")
                        detailed_hits.append({"source": source, "sentence": hits[0][0]})

                print(f"--- Triagem: Encontrados {len(detailed_hits)} hits brutos. ---")
                unique_sources = {hit['source'] for hit in detailed_hits}
                hit_count = len(unique_sources)
                print(f"--- Triagem: Encontrados {hit_count} documentos únicos com o nome: {list(unique_sources)} ---")

                if hit_count == 1:
                    print("--- Triagem: Hit único encontrado. Forçando AUTO_RESOLVER. ---")
                    return {"triagem": {"decisao": "AUTO_RESOLVER", "campos_faltantes": []}}
                
                if hit_count > 1:
                    print("--- Triagem: Múltiplos hits encontrados. Preparando para pedir esclarecimento contextual. ---")
                    return {"triagem": {"decisao": "PEDIR_INFO", "hits_contextuais": detailed_hits}}

        except Exception as e:
            print(f"--- Triagem: Falha na busca lexical rápida: {e}. Prosseguindo com LLM. ---")
    # --- Fim da Triagem Inteligente ---

    print("--- Triagem: Buscando decisão via LLM... ---")
    triagem_chain, _ = get_triagem_chain()
    saida: TriagemOut = triagem_chain.invoke([SystemMessage(content=TRIAGEM_PROMPT), HumanMessage(content=pergunta)])
    print(f"--- Agente: Decisão da triagem -> {saida.dict()} ---")
    dados = saida.dict()
    dados["campos_faltantes"] = dados.get("campos_faltantes") or []
    return {"triagem": dados}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")
    from api import embeddings_model, vectorstore # IMPORT LOCAL PARA EVITAR CICLO
    if embeddings_model is None or vectorstore is None:
        return {"resposta": "O índice vetorial não está pronto.", "citacoes": [], "rag_sucesso": False, "acao_final": "PEDIR_INFO"}
    
    resultado_rag = answer_question(question=state["pergunta"], embeddings_model=embeddings_model, vectorstore=vectorstore, debug=False)
    contexto_recuperado = [c.get("preview", "") for c in resultado_rag.get("citations", [])]
    rag_success = bool(resultado_rag.get("context_found"))
    return {"resposta": resultado_rag.get("answer"), "citacoes": resultado_rag.get("citations", []), "context": contexto_recuperado, "rag_sucesso": rag_success}

def node_self_evaluate(state: AgentState) -> AgentState:
    print("\n--- DEBUG: INICIANDO O CORPO INTEIRO DO NÓ DE AVALIAÇÃO ---")
    try:
        print("--- DEBUG: NÓ DE AUTO-AVALIAÇÃO (LÓGICA RESTAURADA) ---")
        if not state.get("context") or not state.get("resposta"):
            print("--- DEBUG: Contexto ou resposta ausentes. Reprovando automaticamente. ---")
            return {"evaluation": {"verdict": "Reprovado", "reason": "Contexto ou resposta ausentes."}}

        question = state["pergunta"]
        context = "\n---\n".join(state["context"])
        answer = state["resposta"]

        print("--- DEBUG: Dados enviados para o LLM-Juiz ---")
        print(f"  - Pergunta: {question}")
        print(f"  - Contexto: {context}")
        print(f"  - Resposta a ser avaliada: {answer}")
        print("--------------------------------------------")

        llm_evaluator = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, api_key=GOOGLE_API_KEY)
        
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer
        )

        response_str = llm_evaluator.invoke(prompt).content
        print(f"--- DEBUG: Resposta bruta do LLM-Juiz -> {response_str} ---")
        
        json_str = response_str.strip().replace("`", "")
        if json_str.startswith("json"):
            json_str = json_str[4:].strip()

        evaluation_result = json.loads(json_str)
        print(f"--- DEBUG: Resultado da avaliação (JSON) -> {evaluation_result} ---")
        return {"evaluation": evaluation_result}

    except Exception as e:
        # ESTE BLOCO DEVE PEGAR QUALQUER ERRO
        print(f"--- ERRO CRÍTICO CAPTURADO DENTRO DO NÓ: {type(e).__name__}: {e} ---")
        import traceback
        traceback.print_exc()
        return {"evaluation": {"verdict": "Reprovado", "reason": f"Falha crítica no nó de avaliação: {e}"}}


def node_correction(state: AgentState) -> AgentState:
    print("\n--- DEBUG: NÓ DE CORREÇÃO ---")
    
    previous_answer = state.get("resposta", "")
    evaluation = state.get("evaluation", {})

    print(f"--- DEBUG: Tentando corrigir resposta anterior devido ao veredito: {evaluation.get('verdict')} ---")
    print(f"--- DEBUG: Resposta anterior: {previous_answer}")
    print("---------------------------------")

    _, llm_triagem = get_triagem_chain()
    prompt = CORRECTION_PROMPT_TEMPLATE.format(
        pergunta=state["pergunta"], 
        resposta_anterior=previous_answer, 
        contexto="\n".join(state.get("context", []))
    )
    corrected_answer = llm_triagem.invoke(prompt).content
    print(f"--- DEBUG: Resposta corrigida -> {corrected_answer} ---")
    return {"resposta": corrected_answer, "acao_final": "AUTO_RESOLVER"}

def node_pedir_info(state: AgentState) -> AgentState:
    print("--- Agente: Executando nó de pedir_info... ---")
    contextual_hits = state.get("triagem", {}).get("hits_contextuais")

    if contextual_hits:
        print(f"--- Pedir Info: Gerando pergunta contextual com {len(contextual_hits)} hits. ---")
        opcoes_formatadas = [f"- {hit['sentence']}" for hit in contextual_hits[:3]]
        contextual_prompt_template = load_prompt("prompts/pedir_info_contextual_prompt.txt")
        if not contextual_prompt_template:
            contextual_prompt_template = "Encontrei estas opções:\n{opcoes}\n\nA qual você se refere?"

        _, llm_triagem = get_triagem_chain()
        prompt = contextual_prompt_template.format(
            pergunta=state["pergunta"], opcoes="\n".join(opcoes_formatadas)
        )
        clarification_text = llm_triagem.invoke(prompt).content
        return {"resposta": clarification_text, "citacoes": [], "acao_final": "PEDIR_INFO"}

    print("--- Pedir Info: Gerando pergunta genérica. ---")
    campos_faltantes = state.get("triagem", {}).get("campos_faltantes", [])
    _, llm_triagem = get_triagem_chain()
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(
        pergunta=state["pergunta"], campos_faltantes=", ".join(campos_faltantes)
    )
    clarification_text = llm_triagem.invoke(prompt).content
    return {"resposta": clarification_text, "citacoes": [], "acao_final": "PEDIR_INFO"}

# --- Lógica Condicional (Arestas) ---
def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    return "auto" if state["triagem"]["decisao"] == "AUTO_RESOLVER" else "info"

def decidir_pos_auto_resolver(state: AgentState) -> Literal["avaliar", "info"]:
    return "avaliar" if state.get("rag_sucesso") else "info"

def decidir_pos_self_evaluate(state: AgentState) -> Literal["corrigir", "fim"]:
    evaluation_result = state.get("evaluation", {})
    verdict = evaluation_result.get("verdict", "Reprovado")
    print(f"--- DEBUG: Decidindo o próximo passo com base no veredito: {verdict} ---")
    
    if verdict == "Aprovado":
        print("--- DEBUG: Veredito Aprovado. Finalizando o fluxo. ---")
        return "fim"
    else:
        print("--- DEBUG: Veredito Reprovado. Enviando para correção. ---")
        return "corrigir"

# --- Construção do Grafo ---
workflow = StateGraph(AgentState)
workflow.add_node("etapa_triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("self_evaluate", node_self_evaluate)
workflow.add_node("correction", node_correction)

workflow.add_edge(START, "etapa_triagem")
workflow.add_edge("pedir_info", END)
workflow.add_edge("correction", END)

workflow.add_conditional_edges("etapa_triagem", decidir_pos_triagem, {"auto": "auto_resolver", "info": "pedir_info"})
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {"avaliar": "self_evaluate", "info": "pedir_info"})
workflow.add_conditional_edges("self_evaluate", decidir_pos_self_evaluate, {"corrigir": "correction", "fim": END})

compiled_graph = workflow.compile()
