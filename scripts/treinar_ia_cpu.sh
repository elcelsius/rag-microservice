#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-./data}"
OUT_DIR="${2:-/app/vector_store/faiss_index}"
EMB="${EMBEDDINGS_MODEL:-intfloat/multilingual-e5-large}"
EXTS="${EXTS:-txt,md,pdf,docx}"
LOADERS_DIR="${LOADERS_DIR:-./loaders}"

echo "[treinar_ia_cpu] data=$DATA_DIR out=$OUT_DIR emb=$EMB exts=$EXTS loaders=$LOADERS_DIR"
docker-compose -f docker-compose.cpu.yml build ai_etl
docker-compose -f docker-compose.cpu.yml run --rm -e EMBEDDINGS_MODEL="$EMB" ai_etl \
  python scripts/etl_build_index.py \
    --data "$DATA_DIR" \
    --out "$OUT_DIR" \
    --embeddings "$EMB" \
    --exts "$EXTS" \
    --loaders "$LOADERS_DIR"
