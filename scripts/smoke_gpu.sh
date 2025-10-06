#!/usr/bin/env bash
# smoke_gpu.sh — smoke test GPU para rag-microservice
# Uso: ./smoke_gpu.sh [--with-etl] [--exts "txt,md,pdf,docx"] [--loaders ./loaders] [--question "sua pergunta"]
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

step() { echo -e "\n\033[1;34m[SMOKE-GPU]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERRO]\033[0m $*"; }

run() {
  printf '\033[0;36m   comando:\033[0m'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

if $WITH_ETL; then
  step "1/9 Derrubando containers existentes..."
else
  step "1/9 Derrubando containers existentes..."
fi
run docker-compose -f docker-compose.gpu.yml down --remove-orphans

if $WITH_ETL; then
  step "2/9 Build containers (API+Web +ETL)"
else
  step "2/9 Build containers (API+Web)"
fi
build_targets=(ai_projeto_api ai_web_ui)
if $WITH_ETL; then
  build_targets+=(ai_etl)
fi
run docker-compose -f docker-compose.gpu.yml build "${build_targets[@]}"

if $WITH_ETL; then
  step "3/9 (opcional) Rodando ETL — exts=${EXTS}, loaders=${LOADERS}"
  run docker-compose -f docker-compose.gpu.yml run --rm ai_etl \
    python scripts/etl_build_index.py \
      --data ./data \
      --out /app/vector_store/faiss_index \
      --exts "${EXTS}" \
      --loaders "${LOADERS}"
  ok "ETL finalizado"
fi

step "4/9 Subindo API+Web (GPU)"
run docker-compose -f docker-compose.gpu.yml up -d ai_projeto_api ai_web_ui

step "5/9 CUDA dentro do container"
printf '\033[0;36m   comando:\033[0m docker-compose -f docker-compose.gpu.yml exec -T ai_projeto_api python <<PY (truncado)\n'
docker-compose -f docker-compose.gpu.yml exec -T ai_projeto_api python - <<'PY' < /dev/null
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("num_gpus:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

step "NOVO: Aguardando container 'ai_projeto_api' ficar saudável"
for i in {1..300}; do # Timeout de 5 minutos (150 * 2s)
  STATUS=$(docker-compose -f docker-compose.gpu.yml ps ai_projeto_api | grep 'healthy' || true)
  if [[ -n "$STATUS" ]]; then
    ok "Container 'ai_projeto_api' está saudável!"
    break
  fi
  echo -n "."
  sleep 2
done
if [[ -z "$STATUS" ]]; then
  err "Container 'ai_projeto_api' não ficou saudável a tempo."
  docker-compose -f docker-compose.gpu.yml logs ai_projeto_api
  exit 1
fi

step "6/9 Aguardando Healthz (8080)"
echo "--- ATIVANDO MODO DEBUG (set -x) ---"
set -x # Ativa o modo de depuração para ver cada comando executado

for i in {1..30}; do
  HZ=$(curl -fsS http://localhost:8080/api/healthz || true)
  echo -e "\n   Tentativa $i/30: Verificando healthz... Resposta: ${HZ:-'(vazio)'}"
  READY=$(jq -r '.ready' <<< "$HZ" 2>/dev/null)
  FAISS=$(jq -r '.faiss' <<< "$HZ" 2>/dev/null)
  echo "   => ready='${READY}', faiss='${FAISS}'"
  if [[ "$READY" == "true" && "$FAISS" == "true" ]]; then
    ok "API pronta! ready:true faiss:true"
    break
  fi
  sleep 2
done

set +x # Desativa o modo de depuração
echo "--- MODO DEBUG DESATIVADO ---"

if [[ "$READY" != "true" || "$FAISS" != "true" ]]; then
  err "API não ficou pronta a tempo. Healthz: $HZ"
  docker-compose -f docker-compose.gpu.yml logs ai_projeto_api
  exit 1
fi

PAYLOAD_DEBUG=$(jq -n --arg question "$Q" '{question:$question, debug:true}')
PAYLOAD_NODEBUG=$(jq -n --arg question "$Q" '{question:$question, debug:false}')

step "7/9 Consulta via 5000 (debug=true)"
printf '\033[0;36m   comando:\033[0m curl POST /api/query (8080)\n'
R5000=$(curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD_DEBUG" http://localhost:8080/api/query)
echo "$R5000" | jq '.context_found, .debug.route, .debug.timing_ms' || true

step "8/9 Consulta via 8080 (debug=false)"
printf '\033[0;36m   comando:\033[0m curl POST /api/query (8080)\n'
R8080=$(curl -fsS -H "Content-Type: application/json" -d "$PAYLOAD_NODEBUG" http://localhost:8080/api/query)
echo "$R8080" | jq '.answer, .citations' >/dev/null || { err "resposta inválida"; echo "$R8080"; exit 1; }
LEN=$(echo "$R8080" | jq -r '.answer | length')
[[ "${LEN:-0}" -gt 0 ]] || { err "answer vazio"; echo "$R8080"; exit 1; }
ok "answer ok (${LEN} chars)"

step "9/9 Verificações extra (reranker opcional)"
if echo "$R5000" | jq '.debug.rerank' >/dev/null 2>&1; then
  BAD=$(echo "$R5000" | jq '[.debug.rerank.scored[]?.score] | map(type) | any(. != "number")')
  [[ "$BAD" != "true" ]] || { err "scores do rerank não são numéricos"; echo "$R5000" | jq '.debug.rerank'; exit 1; }
  ok "reranker com scores numéricos"
else
  ok "sem reranker (ou desativado)"
fi

ok "SMOKE GPU concluído com sucesso."
