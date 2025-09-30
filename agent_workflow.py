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

# --- CORREÇÃO: Importa a função correta do handler e os modelos pré-carregados ---
from query_handler import answer_question

# Tenta reutilizar o modelo de embeddings e o vectorstore já carregados pela API para otimização.
# Isso evita a sobrecarga de recarregar esses componentes pesados a cada chamada.
try:
    from api import embeddings_model, vectorstore
except (ImportError, ModuleNotFoundError):
    # Define como None se não puderem ser importados (ex: ao rodar testes fora do contexto da API).
    # O nó 'node_auto_resolver' tratará esse caso de forma segura.
    embeddings_model = None
    vectorstore = None

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
        return ""  # Retorna vazio para evitar que o programa quebre imediatamente.


# --- CARREGAMENTO DOS PROMPTS DAS FONTES EXTERNAS ---
# Carrega os prompts específicos para as etapas de triagem e solicitação de informações.
TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")


# --- DEFINIÇÃO DO ESQUEMA DE SAÍDA PARA A TRIAGEM ---
class TriagemOut(BaseModel):
    """Define o formato esperado da saída do modelo de triagem."""
    decisao: Literal[
        "AUTO_RESOLVER", "PEDIR_INFO"]  # A decisão do agente: resolver automaticamente ou pedir mais informações.
    campos_faltantes: List[str] = Field(
        default_factory=list)  # Lista de campos que faltam, se a decisão for PEDIR_INFO.


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
    pergunta: str  # A pergunta original do usuário.
    messages: List[BaseMessage]  # Histórico de mensagens da conversa.
    triagem: dict  # Resultado da etapa de triagem.
    resposta: Optional[str]  # A resposta final gerada pelo agente.
    citacoes: List[dict]  # Citações ou fontes usadas para gerar a resposta.
    rag_sucesso: bool  # Indica se a recuperação de informações (RAG) foi bem-sucedida.
    acao_final: str  # A ação final tomada pelo agente (e.g., AUTO_RESOLVER, PEDIR_INFO).


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
    return {"triagem": saida.model_dump()}  # Retorna o estado com a decisão da triagem.


def node_auto_resolver(state: AgentState) -> AgentState:
    """
    Nó que executa a lógica RAG para tentar responder a pergunta do usuário.
    Se houver histórico de conversa, a pergunta é condensada para ser autônoma.

    Args:
        state (AgentState): O estado atual do agente.

    Returns:
        AgentState: O estado atualizado com a resposta do RAG, citações e status de sucesso.
    """
    print("--- Agente: Executando nó de auto_resolver (RAG)... ---")

    # --- AJUSTE: Garante que o índice vetorial e os embeddings estejam prontos ---
    if embeddings_model is None or vectorstore is None:
        print("--- Agente: ERRO - Índice vetorial ou modelo de embeddings não carregado. ---")
        return {
            "resposta": "O índice vetorial ainda não está pronto para uso. Por favor, aguarde o carregamento ou contate o suporte.",
            "citacoes": [],
            "rag_sucesso": False,
            "acao_final": "PEDIR_INFO",
        }

    # Lógica para criar uma pergunta autônoma a partir do histórico de conversa.
    if len(state.get("messages", [])) > 1:
        print("--- Agente: Condensando pergunta a partir do histórico...")
        condenser_prompt_str = (
            "Dada a conversa abaixo e a última pergunta do usuário, reformule a pergunta para que ela seja completa e autônoma, "
            "contendo todo o contexto necessário para ser entendida sem o histórico. Não responda à pergunta, apenas a reformule.\n\n"
            "HISTÓRICO DA CONVERSA:\n{chat_history}\n\nÚLTIMA PERGUNTA: {question}\n\nPERGUNTA AUTÔNOMA:"
        )
        chat_history_str = "\n".join(
            [f"{'Usuário' if isinstance(msg, HumanMessage) else 'IA'}: {msg.content}" for msg in state["messages"][:-1]]
        )
        prompt = condenser_prompt_str.format(chat_history=chat_history_str, question=state["pergunta"])
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        standalone_question = response.text.strip()
        print(f"--- Agente: Pergunta autônoma gerada -> '{standalone_question}' ---")
    else:
        standalone_question = state["pergunta"]

    # --- CORREÇÃO: Chama a função RAG pública correta com os parâmetros necessários ---
    resultado_rag = answer_question(
        query=standalone_question,
        embeddings_model=embeddings_model,
        vectorstore=vectorstore,
        debug=False
    )

    # --- CORREÇÃO: Mapeia a saída do RAG de forma mais segura para o estado ---
    rag_success = bool(resultado_rag.get("context_found"))
    update: AgentState = {
        "resposta": resultado_rag.get("answer", "Não foi possível encontrar uma resposta."),
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": rag_success,
    }

    # --- AJUSTE: Define a ação final de forma explícita para clareza ---
    if rag_success:
        update["acao_final"] = "AUTO_RESOLVER"
    else:
        # Se o RAG falhou, a ação é pedir mais informações para o usuário.
        update["acao_final"] = "PEDIR_INFO"

    return update


def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que usa o LLM para formular uma resposta pedindo mais informações ao usuário.

    Args:
        state (AgentState): O estado atual do agente.

    Returns:
        AgentState: O estado atualizado com a pergunta de esclarecimento.
    """
    print("--- Agente: Executando nó de pedir_info (versão inteligente)... ---")

    model = genai.GenerativeModel(MODEL_NAME)
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])

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
    """
    Decide o que fazer após a tentativa de RAG. Se teve sucesso, finaliza. Se falhou, pede mais informações.

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

# Adiciona os nós ao grafo, cada um representando uma etapa do fluxo de trabalho.
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)

# Define o ponto de entrada do grafo.
workflow.add_edge(START, "triagem")

# Define as arestas condicionais que governam a transição entre os nós.
# Após a triagem, o fluxo pode ir para a resolução automática ou para pedir mais informações.
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info"
})

# Após a tentativa de auto-resolução, o fluxo pode terminar com sucesso ou ir para pedir mais informações.
workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "ok": END,
    "info": "pedir_info"
})

# Após pedir informações, o fluxo sempre termina, aguardando a próxima interação do usuário.
workflow.add_edge("pedir_info", END)

# Compila o grafo, criando um objeto executável.
compiled_graph = workflow.compile()