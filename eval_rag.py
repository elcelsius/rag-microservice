
#!/usr/bin/env python
# eval_rag.py
# Script abrangente para avaliação de sistemas RAG, combinando métricas de recuperação e de geração.

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import unicodedata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any
from urllib.request import Request, urlopen

import pandas as pd
from datasets import Dataset

# --- Configuração da Avaliação ---
API_AGENT_DEFAULT = os.getenv("API_URL", "http://localhost:5000/agent/ask")
API_QUERY_DEFAULT = os.getenv("LEGACY_API_URL", "http://localhost:5000/query")
EVAL_LLM_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash-latest")
DEFAULT_DATASET = Path("evaluation_dataset.jsonl")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")



# --- Métricas de Recuperação (Retrieval) ---

def dcg_at_k(rels: List[int], k: int) -> float:
    """Calcula o Discounted Cumulative Gain."""
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(golds: List[str], preds_texts: List[str], k: int) -> float:
    """Calcula o Normalized Discounted Cumulative Gain @ k."""
    rels = [1 if any(gold.lower() in pred.lower() for gold in golds) else 0 for pred in preds_texts]
    ideal_rels = sorted([rel for rel in rels if rel > 0], reverse=True)
    return dcg_at_k(rels, k) / (dcg_at_k(ideal_rels, k) or 1.0)


def mrr_at_k(golds: List[str], preds_texts: List[str], k: int) -> float:
    """Calcula o Mean Reciprocal Rank @ k."""
    for i, pred in enumerate(preds_texts[:k]):
        if any(gold.lower() in pred.lower() for gold in golds):
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(golds: List[str], preds_texts: List[str], k: int) -> float:
    """Calcula o Recall @ k."""
    return 1.0 if any(gold.lower() in pred.lower() for pred in preds_texts[:k] for gold in golds) else 0.0


# --- Helpers de CLI e Dataset ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação do RAG Microservice")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Caminho do dataset (CSV ou JSONL). Pode ser omitido se --dataset for usado.",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_flag",
        help="Caminho do dataset (CSV ou JSONL).",
    )
    parser.add_argument(
        "--agent-url",
        "--agent-endpoint",
        dest="agent_url",
        default=API_AGENT_DEFAULT,
        help="Endpoint do agente (default: %(default)s)",
    )
    parser.add_argument(
        "--query-url",
        "--legacy-endpoint",
        dest="query_url",
        default=API_QUERY_DEFAULT,
        help="Endpoint legado /query (default: %(default)s)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compara agente e endpoint legado na mesma execução.",
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        default="reports",
        help="Diretório para salvar o relatório em JSON.",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Rótulo opcional para compor o nome do arquivo de saída.",
    )
    args = parser.parse_args()
    if getattr(args, "dataset_flag", None):
        args.dataset = args.dataset_flag
    if hasattr(args, "dataset_flag"):
        delattr(args, "dataset_flag")
    return args


def resolve_dataset_path(arg_path: str | None) -> Path:
    if arg_path:
        path = Path(arg_path)
        if not path.exists():
            raise SystemExit(f"Dataset não encontrado: {path}")
        return path
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    raise SystemExit("Informe o dataset (CSV ou JSONL) ou crie 'evaluation_dataset.jsonl'.")


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    suffix = dataset_path.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(dataset_path, encoding="utf-8")
        df = df.rename(columns={"query": "question", "expected": "ground_truth"})
        df["ground_truth"] = df["ground_truth"].apply(lambda x: [s.strip() for s in str(x).split("||")])
        records = df.to_dict(orient="records")
    elif suffix == ".jsonl":
        records = []
        for idx, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Falha ao ler JSONL na linha {idx}: {exc}") from exc
            records.append(payload)
    elif suffix == ".json":
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Falha ao ler JSON: {exc}") from exc
        if isinstance(payload, list):
            records = payload
        else:
            raise SystemExit("O arquivo JSON deve conter uma lista de exemplos.")
    else:
        raise SystemExit(f"Formato de dataset não suportado: {dataset_path.suffix}")

    normalized: List[Dict[str, Any]] = []
    for item in records:
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        ground_truth = _ensure_list(item.get("ground_truth"))
        normalized.append({
            "question": question,
            "ground_truth": ground_truth,
        })
    if not normalized:
        raise SystemExit("Dataset vazio após normalização.")
    return normalized


# --- Funções de chamada à API ---

def query_api(url: str, question: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"question": question}).encode("utf-8")
    req = Request(url, data=data, headers=headers)
    try:
        with urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"ERROR: Falha ao chamar a API '{url}' para a pergunta '{question[:80]}...': {exc}")
        return {}


# --- Avaliação principal ---

def _normalize_text(value: str) -> str:
    if not value:
        return ""
    return unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii').lower()


def _compute_retrieval_metrics(results_dataset: Dataset) -> Dict[str, float]:
    metrics = {"recall@5": [], "mrr@10": [], "ndcg@5": []}
    for item in results_dataset:
        golds_raw = item["ground_truth"]
        preds_raw = item["contexts"]
        golds = [_normalize_text(g) for g in golds_raw]
        preds = [_normalize_text(p) for p in preds_raw]
        metrics["recall@5"].append(recall_at_k(golds, preds, k=5))
        metrics["mrr@10"].append(mrr_at_k(golds, preds, k=10))
        metrics["ndcg@5"].append(ndcg_at_k(golds, preds, k=5))
    return {
        key: round(sum(values) / len(values), 4) if values else 0.0
        for key, values in metrics.items()
    }


async def evaluate_endpoint(dataset_records: List[Dict[str, Any]], url: str, label: str) -> Dict[str, Any]:
    print(f"\nINFO: Avaliando endpoint '{label}' -> {url}")
    results: List[Dict[str, Any]] = []
    for entry in dataset_records:
        question = entry["question"]
        print(f"INFO: Processando pergunta: '{question[:80]}...'")
        api_response = query_api(url, question)
        answer = api_response.get("answer", "")
        contexts = [cit.get("preview", "") for cit in api_response.get("citations", []) if isinstance(cit, dict)]
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": entry["ground_truth"],
        })

    results_dataset = Dataset.from_list(results)
    retrieval_metrics = _compute_retrieval_metrics(results_dataset)

    print("INFO: Calculando metricas de geracao (RAGAs)...")
    avg_generation_metrics: Dict[str, float] = {}
    ragas_available = True
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness, answer_relevancy  # type: ignore
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            ragas_available = False
            print("WARN: GOOGLE_API_KEY não configurado. Pulando métricas de geração (RAGAs).")
        else:
            ragas_llm = ChatGoogleGenerativeAI(model=EVAL_LLM_MODEL, temperature=0.0)
            generation_metrics = [faithfulness, answer_relevancy]
            ragas_result = await evaluate(
                dataset=results_dataset,
                metrics=generation_metrics,
                llm=ragas_llm,
                raise_exceptions=False,
            )
            ragas_scores = ragas_result.scores.to_dict()
            avg_generation_metrics = {
                key: round(sum(values) / len(values), 4) if values else 0.0
                for key, values in ragas_scores.items()
            }
    except ImportError:
        ragas_available = False
        print("WARN: Biblioteca ragas ou dependencias nao instaladas. Pulando metricas de geracao.")
    except Exception as exc:
        ragas_available = False
        print(f"ERROR: Falha ao executar avaliacao com RAGAs: {exc}")

    return {
        "endpoint": url,
        "dataset_size": len(results_dataset),
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics_ragas": avg_generation_metrics,
        "ragas_available": ragas_available,
    }


def _sanitize_label(label: str, default: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")
    return clean or default


async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(args.dataset)
    dataset_records = load_dataset(dataset_path)

    targets = {"agent": args.agent_url}
    if args.compare:
        targets["legacy"] = args.query_url

    results: Dict[str, Any] = {}
    for label, url in targets.items():
        results[label] = await evaluate_endpoint(dataset_records, url, label)

    timestamp = datetime.now(timezone.utc)
    final_report = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_size": len(dataset_records),
        "generated_at": timestamp.isoformat(),
        "targets": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_label = args.label or ("compare" if len(results) > 1 else next(iter(results)))
    file_label = _sanitize_label(base_label, "report")
    outfile = output_dir / f"eval-report-{file_label}-{timestamp.strftime('%Y%m%d-%H%M%S')}.json"
    outfile.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSUCCESS: Relatorio salvo em {outfile}")

    return final_report


def print_summary(report: Dict[str, Any]) -> None:
    print("\n--- RESUMO FINAL ---")
    print(f"Dataset: {report['dataset_path']} ({report['dataset_size']} itens)")
    for label, data in report["targets"].items():
        print(f"\n[{label}] -> {data['endpoint']}")
        print("  Metricas de Recuperacao:")
        for key, value in data["retrieval_metrics"].items():
            print(f"    {key}: {value}")
        metrics = data.get("generation_metrics_ragas") or {}
        if data.get("ragas_available") and metrics:
            print("  Metricas de Geracao (RAGAs):")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
        else:
            print("  Metricas de Geracao (RAGAs): indisponiveis")

if __name__ == "__main__":
    cli_args = parse_args()
    report = asyncio.run(run_evaluation(cli_args))
    print_summary(report)
