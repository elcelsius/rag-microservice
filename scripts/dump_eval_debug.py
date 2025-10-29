#!/usr/bin/env python
"""
dump_eval_debug.py
------------------

Percorre o arquivo evaluation_dataset.jsonl (ou outro informado) e dispara
requisições HTTP para o endpoint do RAG com `debug=true`, salvando o payload
completo (resposta + bloco de debug) em arquivos JSON individuais.

Uso básico:
    python scripts/dump_eval_debug.py

Parâmetros úteis:
    --dataset        Caminho do arquivo .jsonl (default: evaluation_dataset.jsonl)
    --mode           "agent" (default) ou "query" para escolher o endpoint
    --agent-url      URL do /agent/ask (default: http://127.0.0.1:5000/agent/ask)
    --query-url      URL do /query     (default: http://127.0.0.1:5000/query)
    --output         Pasta destino (default: reports/debug-dump-<timestamp>)
    --timeout        Timeout por requisição (default: 30s)

Requisitos: requests (já listado em requirements).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_AGENT_URL = "http://127.0.0.1:5000/agent/ask"
DEFAULT_QUERY_URL = "http://127.0.0.1:5000/query"
DEFAULT_DATASET = "evaluation_dataset.jsonl"


def _load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Linha {line_no}: JSON inválido ({exc}); ignorando.", file=sys.stderr)
                continue
            question = (obj.get("question") or "").strip()
            if not question:
                print(f"[WARN] Linha {line_no}: campo 'question' vazio; ignorando.", file=sys.stderr)
                continue
            yield line_no, question, obj


def _safe_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:60] or "question"


def dump_debug(
    dataset_path: Path,
    *,
    mode: str,
    agent_url: str,
    query_url: str,
    output_dir: Path,
    timeout: int,
) -> None:
    session = requests.Session()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok = 0
    errors: Dict[str, Any] = {}

    for idx, question, obj in _load_dataset(dataset_path):
        total += 1
        payload: Dict[str, Any] = {"question": question, "debug": True}
        url = query_url

        if mode == "agent":
            url = agent_url
            payload["messages"] = []

        try:
            resp = session.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            errors[question] = str(exc)
            err_path = output_dir / f"{idx:02d}_{_safe_filename(question)}_error.json"
            err_payload = {
                "question": question,
                "error": str(exc),
                "payload": payload,
                "original": obj,
            }
            err_path.write_text(json.dumps(err_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ERROR] {idx:02d} | {question[:60]} → {exc}", file=sys.stderr)
            continue

        out_path = output_dir / f"{idx:02d}_{_safe_filename(question)}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        ok += 1
        print(f"[OK] {idx:02d} | {question}")

    print("\nResumo:")
    print(f"  Total processado: {total}")
    print(f"  Sucesso:          {ok}")
    print(f"  Erros:            {len(errors)}")
    print(f"  Saída em:         {output_dir.resolve()}")

    if errors:
        print("\nErros encontrados:")
        for question, message in errors.items():
            print(f"  - {question[:80]} => {message}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta respostas com debug=true para cada pergunta do dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Caminho para o arquivo .jsonl (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--mode",
        choices=("agent", "query"),
        default="agent",
        help="Endpoint alvo: agent (/agent/ask) ou query (/query). Default: agent.",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_AGENT_URL,
        help=f"URL do endpoint /agent/ask (default: {DEFAULT_AGENT_URL})",
    )
    parser.add_argument(
        "--query-url",
        default=DEFAULT_QUERY_URL,
        help=f"URL do endpoint /query (default: {DEFAULT_QUERY_URL})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Diretório de saída (default: reports/debug-dump-<timestamp>)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout por requisição em segundos (default: 30)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        parser.error(f"dataset não encontrado: {dataset_path}")

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("reports") / f"debug-dump-{timestamp}"

    dump_debug(
        dataset_path,
        mode=args.mode,
        agent_url=args.agent_url,
        query_url=args.query_url,
        output_dir=output_dir,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
