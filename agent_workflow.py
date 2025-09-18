# agent_workflow.py
# Este módulo define o fluxo de trabalho do agente de IA usando LangGraph para processar perguntas.
# Ele orquestra a triagem de perguntas, a recuperação de informações (RAG) e a geração de respostas.

import os
import google.generativeai as genai
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from query_handler import find_answer_for_query

# --- CONFIGURAÇÃO DO MODELO --- 
# Carrega as configurações do ambiente, com valores padrão para a chave da API e o nome do modelo Gemini.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# --- FUNÇÃO AUXILIAR PARA CARREGAR PROMPTS ---
def load_prompt(file_path: str) -> str:
    """Carrega o conteúdo de um arquivo de prompt de forma segura.
    
    Args:
        file_path (str): O caminho relativo para o arquivo de prompt.
        
    Returns:
        str: O conteúdo do prompt como uma string. Retorna uma string vazia em caso de erro.
    """
    try:
        # Garante que o caminho seja relativo ao diretório do script atual para portabilidade.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de prompt não encontrado em '{full_path}'")
        # Em um sistema de produção, seria ideal levantar uma exceção ou ter um tratamento de erro mais robusto.
        return "" # Retorna vazio para evitar que o programa quebre imediatamente.

# --- CARREGAMENTO DOS PROMPTS DAS FONTES EXTERNAS ---
# Carrega os prompts específicos para as etapas de triagem e solicitação de informações.
TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")

# --- DEFINIÇÃO DO ESQUEMA DE SAÍDA PARA A TRIAGEM ---
class TriagemOut(BaseModel):
    """Define o formato esperado da saída do modelo de triagem."""
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"] # A decisão do agente: resolver automaticamente ou pedir mais informações.
    campos_faltantes: List[str] = Field(default_factory=list) # Lista de campos que faltam, se a decisão for PEDIR_INFO.


# --- INICIALIZAÇÃO DO LLM PARA TRIAGEM ---
# Configura o modelo Gemini para a etapa de triagem, com temperatura 0.0 para respostas mais determinísticas.
llm_triagem = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)
# Cria uma cadeia que usa o LLM para triagem e formata a saída de acordo com TriagemOut.
triagem_chain = llm_triagem.with_structured_output(TriagemOut)


# --- DEFINIÇÃO DO ESTADO DO GRAFO ---
# TypedDict que define a estrutura do estado que será passado entre os nós do grafo.
class AgentState(TypedDict, total=False):
    """Representa o estado atual do agente, contendo todas as informações relevantes para o fluxo de trabalho."""
    pergunta: str # A pergunta original do usuário.
    messages: List[BaseMessage] # Histórico de mensagens da conversa.
    triagem: dict # Resultado da etapa de triagem.
    resposta: Optional[str] # A resposta final gerada pelo agente.
    citacoes: List[dict] # Citações ou fontes usadas para gerar a resposta.
    rag_sucesso: bool # Indica se a recuperação de informações (RAG) foi bem-sucedida.
    acao_final: str # A ação final tomada pelo agente (e.g., AUTO_RESOLVER, PEDIR_INFO).


# --- NÓS DO GRAFO ---

def node_triagem(state: AgentState) -> AgentState:
    """Primeiro nó do grafo: classifica a pergunta do usuário para determinar o próximo passo.
    
    Args:
        state (AgentState): O estado atual do agente.
        
    Returns:
        AgentState: O estado atualizado com o resultado da triagem.
    """
    print("--- Agente: Executando nó de triagem... ---")
    pergunta = state["pergunta"]
    # Invoca a cadeia de triagem com a pergunta do usuário e o prompt do sistema.
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=pergunta)
    ])
    print(f"--- Agente: Decisão da triagem -> {saida.model_dump()} ---")
    return {"triagem": saida.model_dump()} # Retorna o estado com a decisão da triagem.


def node_auto_resolver(state: AgentState) -> AgentState:
    """Nó que executa a lógica RAG para tentar responder a pergunta do usuário.
    Se houver histórico de conversa, a pergunta é condensada para ser autônoma.
    
    Args:
        state (AgentState): O estado atual do agente.
        
    Returns:
        AgentState: O estado atualizado com a resposta do RAG, citações e status de sucesso.
    """
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")

    # Lógica para criar uma pergunta autônoma a partir do histórico de conversa.
    if len(state.get("messages", [])) > 1:
        print("--- Agente: Condensando pergunta a partir do histórico...")

        # Prompt para o LLM reescrever a pergunta, tornando-a independente do contexto anterior.
        condenser_prompt_str = (
            "Dada a conversa abaixo e a última pergunta do usuário, reformule a pergunta para que ela seja completa e autônoma, "
            "contendo todo o contexto necessário para ser entendida sem o histórico. Não responda à pergunta, apenas a reformule.\n\n"
            "HISTÓRICO DA CONVERSA:\n{chat_history}\n\nÚLTIMA PERGUNTA: {question}\n\nPERGUNTA AUTÔNOMA:"
        )

        # Formata o histórico de chat para ser compreendido pelo LLM.
        chat_history_str = "\n".join(
            [f"{'Usuário' if isinstance(msg, HumanMessage) else 'IA'}: {msg.content}" for msg in
             state["messages"][:-1]])

        # Prepara o prompt com o histórico e a pergunta atual.
        prompt = condenser_prompt_str.format(chat_history=chat_history_str, question=state["pergunta"])

        # Usa o Gemini para gerar a nova pergunta autônoma.
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        standalone_question = response.text.strip()

        print(f"--- Agente: Pergunta autônoma gerada -> '{standalone_question}' ---")
    else:
        # Se não há histórico, a pergunta original já é autônoma.
        standalone_question = state["pergunta"]

    # Chama a função RAG para encontrar a resposta usando a pergunta (autônoma ou original).
    resultado_rag = find_answer_for_query(standalone_question)

    # Prepara o objeto de atualização do estado com a resposta do RAG.
    update: AgentState = {
        "resposta": resultado_rag["answer"],
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": resultado_rag["context_found"],
    }

    # Se o RAG encontrou contexto, define a ação final como AUTO_RESOLVER.
    if resultado_rag["context_found"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que usa o LLM para formular uma resposta pedindo mais informações ao usuário.
    
    Args:
        state (AgentState): O estado atual do agente.
        
    Returns:
        AgentState: O estado atualizado com a pergunta de esclarecimento.
    """
    print("--- Agente: Executando nó de pedir_info (versão inteligente)... ---")

    # Inicializa o modelo generativo dentro do nó.
    model = genai.GenerativeModel(MODEL_NAME)

    # Cria o prompt com a pergunta original do usuário para pedir esclarecimentos.
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])

    # Gera uma pergunta de esclarecimento usando o Gemini.
    try:
        response = model.generate_content(prompt)
        clarification_text = response.text
    except Exception as e:
        print(f"ERRO ao gerar pergunta de esclarecimento: {e}")
        # Fallback para uma mensagem genérica em caso de falha na API.
        clarification_text = "Não consegui entender completamente sua pergunta. Poderia fornecer mais detalhes?"

    return {
        "resposta": clarification_text,
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }


# --- LÓGICA CONDICIONAL (ARESTAS) ---

def decidir_pos_triagem(state: AgentState) -> Literal["auto", "info"]:
    """Decide qual caminho seguir após a triagem inicial, com base na decisão do nó de triagem.
    
    Args:
        state (AgentState): O estado atual do agente.
        
    Returns:
        Literal["auto", "info"]: "auto" se a pergunta pode ser auto-resolvida, "info" se precisa de mais informações.
    """
    print("--- Agente: Decidindo rota pós-triagem... ---")
    if state["triagem"]["decisao"] == "AUTO_RESOLVER":
        return "auto"
    return "info"


def decidir_pos_auto_resolver(state: AgentState) -> Literal["ok", "info"]:
    """Decide o que fazer se o RAG teve sucesso ou falhou na resolução da pergunta.
    
    Args:
        state (AgentState): O estado atual do agente.
        
    Returns:
        Literal["ok", "info"]: "ok" se o RAG foi bem-sucedido, "info" se falhou e precisa de mais informações.
    """
    print("--- Agente: Decidindo rota pós-RAG... ---")
    if state.get("rag_sucesso"):
        print("--- Agente: RAG bem-sucedido. Finalizando. ---")
        return "ok"
    print("--- Agente: RAG falhou. Solicitando mais informações. ---")
    return "info"


# --- CONSTRUÇÃO E COMPILAÇÃO DO GRAFO ---

# Inicializa o StateGraph com o tipo de estado definido.
workflow = StateGraph(AgentState)

# Adiciona os nós ao grafo.
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)

# Define a aresta inicial: o fluxo começa no nó de triagem.
workflow.add_edge(START, "triagem")

# Define as arestas condicionais após a triagem.
# A decisão do nó 'triagem' determina se vai para 'auto_resolver' ou 'pedir_info'.
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info"
})

# Define as arestas condicionais após a tentativa de auto-resolução (RAG).
# Se o RAG foi bem-sucedido, o fluxo termina; caso contrário, vai para 'pedir_info'.
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "ok": END,
    "info": "pedir_info"
})

# Define a aresta para o nó 'pedir_info': após pedir informações, o fluxo termina.
workflow.add_edge("pedir_info", END)

# Compila o grafo para que possa ser usado pela API.
compiled_graph = workflow.compile()


