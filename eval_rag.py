#!/usr/bin/env python
# eval_rag.py
# Script abrangente para avaliação de sistemas RAG, combinando métricas de recuperação e de geração.

import os
import json
import math
import sys
import asyncio
from urllib.request import Request, urlopen

import pandas as pd
from datasets import Dataset

# --- Configuração da Avaliação ---
# A URL da API agora aponta para o endpoint do agente e é mais facilmente configurável.
API_URL = os.getenv("API_URL", "http://localhost:5000/agent/ask")
# O LLM usado para a avaliação com RAGAs deve ser configurado.
# Por padrão, usa o mesmo modelo que a aplicação para consistência.
EVAL_LLM_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest")

# --- Métricas de Recuperação (Retrieval) ---
# As funções para calcular Recall@k, MRR@k e nDCG@k permanecem, pois são valiosas.

def dcg_at_k(rels, k):
    """Calcula o Discounted Cumulative Gain."""
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels[:k]))

def ndcg_at_k(golds: list, preds_texts: list, k: int) -> float:
    """Calcula o Normalized Discounted Cumulative Gain @ k."""
    rels = [1 if any(gold.lower() in pred.lower() for gold in golds) else 0 for pred in preds_texts]
    ideal_rels = sorted([rel for rel in rels if rel > 0], reverse=True)
    return dcg_at_k(rels, k) / (dcg_at_k(ideal_rels, k) or 1.0)

def mrr_at_k(golds: list, preds_texts: list, k: int) -> float:
    """Calcula o Mean Reciprocal Rank @ k."""
    for i, pred in enumerate(preds_texts[:k]):
        if any(gold.lower() in pred.lower() for gold in golds):
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(golds: list, preds_texts: list, k: int) -> float:
    """Calcula o Recall @ k."""
    return 1.0 if any(gold.lower() in pred.lower() for pred in preds_texts[:k] for gold in golds) else 0.0

# --- Função de Chamada à API ---

def query_agent_api(question: str) -> dict:
    """Chama o endpoint do agente e retorna a resposta JSON."""
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"question": question}).encode("utf-8")
    req = Request(API_URL, data=data, headers=headers)
    try:
        with urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"ERROR: Falha ao chamar a API para a pergunta '{question}': {e}", file=sys.stderr)
        return {}

# --- Função Principal de Avaliação ---

async def main(csv_path: str):
    """Orquestra o processo de avaliação de ponta a ponta."""
    print(f"INFO: Iniciando avaliação com o arquivo: {csv_path}")
    print(f"INFO: API de destino: {API_URL}")

    # 1. Carregar o conjunto de dados de ground truth
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        # Renomeia colunas para o padrão esperado pelo RAGAs
        df = df.rename(columns={"query": "question", "expected": "ground_truth"})
        # A coluna 'ground_truth' pode conter múltiplos snippets esperados, separados por ||
        df["ground_truth"] = df["ground_truth"].apply(lambda x: [s.strip() for s in x.split("||")])
    except Exception as e:
        print(f"CRITICAL: Falha ao carregar ou processar o arquivo CSV de avaliação: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Executar as consultas na API e coletar os resultados
    results = []
    for _, row in df.iterrows():
        question = row["question"]
        print(f"INFO: Processando pergunta: '{question[:80]}...'")
        api_response = query_agent_api(question)
        
        # Extrai a resposta e os contextos (citações)
        answer = api_response.get("answer", "")
        contexts = [cit["preview"] for cit in api_response.get("citations", [])]
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row["ground_truth"],
        })
    
    # Converte os resultados para um Dataset do Hugging Face
    results_dataset = Dataset.from_list(results)
    print(f"SUCCESS: {len(results_dataset)} perguntas foram processadas.")

    # 3. Calcular Métricas de Recuperação (Retrieval)
    print("\nINFO: Calculando métricas de recuperação (Retrieval)...\n")
    retrieval_metrics = {
        "recall@5": [],
        "mrr@10": [],
        "ndcg@5": [],
    }
    for item in results_dataset:
        golds = item["ground_truth"]
        preds = item["contexts"]
        retrieval_metrics["recall@5"].append(recall_at_k(golds, preds, k=5))
        retrieval_metrics["mrr@10"].append(mrr_at_k(golds, preds, k=10))
        retrieval_metrics["ndcg@5"].append(ndcg_at_k(golds, preds, k=5))

    avg_retrieval_metrics = {
        k: round(sum(v) / len(v), 4) if v else 0.0
        for k, v in retrieval_metrics.items()
    }

    # 4. Calcular Métricas de Geração com RAGAs
    print("\nINFO: Calculando métricas de geração (RAGAs)...\n")
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Configura o LLM que o RAGAs usará para julgar as respostas
        ragas_llm = ChatGoogleGenerativeAI(model=EVAL_LLM_MODEL, temperature=0.0)

        generation_metrics = [faithfulness, answer_relevancy]
        
        # Executa a avaliação do RAGAs
        ragas_result = await evaluate(
            dataset=results_dataset,
            metrics=generation_metrics,
            llm=ragas_llm,
            raise_exceptions=False # Continua a avaliação mesmo que uma linha falhe
        )
        ragas_scores = ragas_result.scores.to_dict()
        avg_generation_metrics = {
            k: round(sum(v) / len(v), 4) if v else 0.0
            for k, v in ragas_scores.items()
        }
    except ImportError:
        print("WARN: Biblioteca 'ragas' não instalada. Pulando métricas de geração.", file=sys.stderr)
        avg_generation_metrics = {}
    except Exception as e:
        print(f"ERROR: Falha ao executar a avaliação com RAGAs: {e}", file=sys.stderr)
        avg_generation_metrics = {}

    # 5. Apresentar o relatório final
    final_report = {
        "dataset_size": len(results_dataset),
        "retrieval_metrics": avg_retrieval_metrics,
        "generation_metrics_ragas": avg_generation_metrics,
    }

    print("\n--- RELATÓRIO FINAL DE AVALIAÇÃO ---")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <caminho_para_o_csv_de_avaliacao>", file=sys.stderr)
        print("Exemplo: python eval_rag.py tests/eval_sample.csv", file=sys.stderr)
        sys.exit(1)
    # O RAGAs usa asyncio, então a função main é executada em um loop de eventos.
    asyncio.run(main(sys.argv[1]))
