#!/usr/bin/env bash
# smoke_cpu.sh — smoke test CPU para rag-microservice
# Uso: ./smoke_cpu.sh [--with-etl] [--exts "txt,md,pdf,docx"] [--loaders ./loaders] [--question "sua pergunta"]
set -euo pipefail

Q="onde encontro informação de monitoria de computação?"
WITH_ETL=false
EXTS="txt,md,pdf,docx"
LOADERS="./loaders"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-etl) WITH_ETL=true; shift ;;
    --exts) EXTS="${2:-$EXTS}"; shift 2 ;;
    --loaders) LOADERS="${2:-$LOADERS}"; shift 2 ;;
    --question) Q="${2:-$Q}"; shift 2 ;;
    *) echo "arg desconhecido: $1"; exit 2 ;;
  esac
done

step() { echo -e "\n\033[1;34m[SMOKE]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERRO]\033[0m $*"; }

step "1/7 Build containers (API+Web${WITH_ETL:+ +ETL})"
docker-compose -f docker-compose.cpu.yml build ai_projeto_api ai_web_ui ${WITH_ETL:+ai_etl} >/dev/null

if $WITH_ETL; then
  step "2/7 (opcional) Rodando ETL — exts=${EXTS}, loaders=${LOADERS}"
  docker-compose -f docker-compose.cpu.yml run --rm ai_etl         python scripts/etl_build_index.py           --data ./data           --out /app/vector_store/faiss_index           --exts "${EXTS}"           --loaders "${LOADERS}"
  ok "ETL finalizado"
fi

step "3/7 Subindo API+Web (CPU)"
docker-compose -f docker-compose.cpu.yml up -d ai_projeto_api ai_web_ui >/dev/null

step "4/7 Healthz (8080)"
HZ=$(curl -fsS http://localhost:8080/api/healthz)
echo "$HZ" | jq . >/dev/null || { err "healthz não é JSON"; echo "$HZ"; exit 1; }
READY=$(echo "$HZ" | jq -r '.ready')
FAISS=$(echo "$HZ" | jq -r '.faiss')
[[ "$READY" == "true" && "$FAISS" == "true" ]] || { err "healthz não ready/faiss"; echo "$HZ"; exit 1; }
ok "ready:true faiss:true"

step "5/7 Consulta via 5000 (debug=true)"
R5000=$(curl -fsS -H "Content-Type: application/json" -d "{"question":"$Q","debug":true}" http://localhost:5000/query)
echo "$R5000" | jq '.context_found, .debug.route, .debug.timing_ms' || true

step "6/7 Consulta via 8080 (debug=false)"
R8080=$(curl -fsS -H "Content-Type: application/json" -d "{"question":"$Q","debug":false}" http://localhost:8080/api/query)
echo "$R8080" | jq '.answer, .citations' >/dev/null || { err "resposta inválida"; echo "$R8080"; exit 1; }
LEN=$(echo "$R8080" | jq -r '.answer | length')
[[ "${LEN:-0}" -gt 0 ]] || { err "answer vazio"; echo "$R8080"; exit 1; }
ok "answer ok (${LEN} chars)"

step "7/7 Verificações extra (reranker opcional)"
# Se debug.rerank existir, garantir que score é sempre float (não null)
if echo "$R5000" | jq '.debug.rerank' >/dev/null 2>&1; then
  BAD=$(echo "$R5000" | jq '[.debug.rerank.scored[]?.score] | map(type) | any(. != "number")')
  [[ "$BAD" != "true" ]] || { err "scores do rerank não são numéricos"; echo "$R5000" | jq '.debug.rerank'; exit 1; }
  ok "reranker com scores numéricos"
else
  ok "sem reranker (ou desativado)"
fi

ok "SMOKE CPU concluído com sucesso."
