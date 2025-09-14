# agent_workflow.py
import os
import google.generativeai as genai
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from query_handler import find_answer_for_query

# --- CONFIGURAÇÃO DO MODELO ---
# Carrega as configurações do ambiente, com valores padrão
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# --- NOVA FUNÇÃO AUXILIAR PARA CARREGAR PROMPTS ---
def load_prompt(file_path: str) -> str:
    """Carrega o conteúdo de um arquivo de prompt de forma segura."""
    try:
        # Garante que o caminho seja relativo ao diretório do script atual
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERRO CRÍTICO: Arquivo de prompt não encontrado em '{full_path}'")
        # Em um sistema de produção, você poderia levantar uma exceção aqui
        # raise FileNotFoundError(f"Prompt não encontrado: {full_path}")
        return "" # Retorna vazio para evitar que o programa quebre imediatamente

# --- CARREGUE OS PROMPTS DAS FONTES EXTERNAS ---
TRIAGEM_PROMPT = load_prompt("prompts/triagem_prompt.txt")
PEDIR_INFO_PROMPT_TEMPLATE = load_prompt("prompts/pedir_info_prompt.txt")

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
    messages: List[BaseMessage]
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

    # --- MODIFICAÇÃO 3: Lógica para criar uma pergunta autônoma ---
    # Se houver mais de uma mensagem, significa que temos um histórico de conversa.
    if len(state.get("messages", [])) > 1:
        print("--- Agente: Condensando pergunta a partir do histórico...")

        # Cria um prompt para o LLM reescrever a pergunta
        condenser_prompt_str = (
            "Dada a conversa abaixo e a última pergunta do usuário, reformule a pergunta para que ela seja completa e autônoma, "
            "contendo todo o contexto necessário para ser entendida sem o histórico. Não responda à pergunta, apenas a reformule.\n\n"
            "HISTÓRICO DA CONVERSA:\n{chat_history}\n\nÚLTIMA PERGUNTA: {question}\n\nPERGUNTA AUTÔNOMA:"
        )

        # Formata o histórico para ser legível pelo LLM
        chat_history_str = "\n".join(
            [f"{'Usuário' if isinstance(msg, HumanMessage) else 'IA'}: {msg.content}" for msg in
             state["messages"][:-1]])

        # Prepara o prompt
        prompt = condenser_prompt_str.format(chat_history=chat_history_str, question=state["pergunta"])

        # Usa o Gemini para gerar a nova pergunta
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        standalone_question = response.text.strip()

        # --- DEBUG AGENTE: Verifique se este print está aqui ---
        print(f"\n--- DEBUG AGENTE: Pergunta autônoma enviada ao RAG -> '{standalone_question}' ---")
        # --- FIM DEBUG ---

        print(f"--- Agente: Pergunta autônoma gerada -> '{standalone_question}' ---")
    else:
        # Se não há histórico, a pergunta original já é autônoma
        standalone_question = state["pergunta"]
    # --- FIM DA MODIFICAÇÃO ---

    # A função find_answer_for_query agora usa a pergunta refinada
    resultado_rag = find_answer_for_query(standalone_question)

    update: AgentState = {
        "resposta": resultado_rag["answer"],
        "citacoes": resultado_rag.get("citations", []),
        "rag_sucesso": resultado_rag["context_found"],
    }

    if resultado_rag["context_found"]:
        update["acao_final"] = "AUTO_RESOLVER"

    # Adiciona a resposta da IA ao histórico de mensagens
    # (Opcional, mas boa prática para manter o estado consistente)
    # update["messages"] = state.get("messages", []) + [AIMessage(content=resultado_rag["answer"])]

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    """Nó que USA O LLM para formular uma resposta pedindo mais informações."""
    print("--- Agente: Executando nó de pedir_info (versão inteligente)... ---")

    # Inicializa o modelo generativo dentro do nó
    model = genai.GenerativeModel(MODEL_NAME)

    # Cria o prompt com a pergunta original do usuário
    prompt = PEDIR_INFO_PROMPT_TEMPLATE.format(pergunta=state["pergunta"])

    # Gera uma pergunta de esclarecimento usando o Gemini
    try:
        response = model.generate_content(prompt)
        clarification_text = response.text
    except Exception as e:
        print(f"ERRO ao gerar pergunta de esclarecimento: {e}")
        # Fallback para a mensagem antiga em caso de erro na API
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