# tools/evaluate.py
# Este script executa a avaliação do pipeline de RAG usando o framework RAGAs.

import os
import requests
import json
import asyncio
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# --- SOLUÇÃO DEFINITIVA: Injeção Explícita e Adaptador Robusto ---
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List, Optional

# Importa as classes das métricas para inicialização explícita
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

from langchain_huggingface import HuggingFaceEmbeddings

# Carrega variáveis de ambiente do arquivo .env (ex: GOOGLE_API_KEY)
load_dotenv()

# --- Configurações ---
API_URL = "http://localhost:5000/query"
EVAL_DATASET_PATH = "evaluation_dataset.jsonl"


# --- Adaptador de LLM para o RAGAs (Versão Final e Robusta) ---
class RagasGoogleApiLLM(LLM):
    model_name: str
    model: Any

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "ragas_google_api_llm"

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            # Adiciona lógica defensiva para lidar com formatos de prompt inesperados
            prompt_text = ""
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, tuple) and len(prompt) > 0 and isinstance(prompt[-1], str):
                prompt_text = prompt[-1]
            else:
                prompt_text = str(prompt)

            try:
                # Garante que estamos passando apenas os parâmetros que a API do Google aceita
                response = self.model.generate_content(prompt_text, generation_config=kwargs.get("generation_config"))
                generations.append([Generation(text=response.text)])
            except Exception as e:
                print(f"Erro na chamada da API do Google: {e}")
                generations.append([Generation(text="")])
        return LLMResult(generations=generations)

    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LLMResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, prompts, stop, **kwargs)


# --- Configurar o LLM e os Embeddings para o RAGAs ---
llm_eval = RagasGoogleApiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
embeddings_eval = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# --- Inicialização Explícita das Métricas ---
# Garante que cada métrica use o nosso LLM customizado.
faithfulness_metric = Faithfulness(llm=llm_eval)
answer_relevancy_metric = AnswerRelevancy(llm=llm_eval)
context_recall_metric = ContextRecall(llm=llm_eval)
context_precision_metric = ContextPrecision(llm=llm_eval)


def run_evaluation():
    """Carrega o dataset, chama a API, avalia com RAGAs e imprime os resultados."""
    
    print(f"Carregando dataset de avaliação de: {EVAL_DATASET_PATH}")
    eval_questions, ground_truths, ground_truth_contexts = [], [], []
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            eval_questions.append(data["question"])
            ground_truths.append(data["ground_truth"])
            ground_truth_contexts.append(data["contexts"])

    print(f"Executando {len(eval_questions)} perguntas contra a API em {API_URL}...")
    generated_answers, retrieved_contexts = [], []
    for question in eval_questions:
        try:
            response = requests.post(API_URL, json={"question": question}, timeout=90)
            response.raise_for_status()
            api_result = response.json()
            generated_answers.append(api_result.get("answer", ""))
            contexts = [c["preview"] for c in api_result.get("citations", [])]
            retrieved_contexts.append(contexts)
        except requests.RequestException as e:
            print(f"\nERRO: {e}")
            generated_answers.append("")
            retrieved_contexts.append([])

    ragas_dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths,
        "ground_truth_contexts": ground_truth_contexts
    })

    print("\nAvaliando as respostas com RAGAs...")
    result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_recall_metric,
            context_precision_metric,
        ],
        embeddings=embeddings_eval
    )

    print("\n--- Relatório de Avaliação RAG ---")
    print(result)
    print("-------------------------------------")

if __name__ == "__main__":
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"ERRO: Arquivo do dataset de avaliação não encontrado em '{EVAL_DATASET_PATH}'.")
    elif not os.getenv("GOOGLE_API_KEY"):
        print("ERRO: A variável de ambiente GOOGLE_API_KEY não foi encontrada.")
    else:
        run_evaluation()
