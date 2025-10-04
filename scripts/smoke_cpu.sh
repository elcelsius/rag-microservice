#!/usr/bin/env bash
# smoke_cpu.sh — smoke test CPU para rag-microservice
# Uso: ./smoke_cpu.sh [--with-etl] [--exts "txt,md,pdf,docx"] [--loaders ./loaders] [--question "sua pergunta"] [--nocache] [--debug]
set -euo pipefail

Q="onde encontro informação de monitoria de computação?"
WITH_ETL=false
NOCACHE=false
DEBUG=false
EXTS="txt,md,pdf,docx"
LOADERS="./loaders"
COMPOSE_FILE="docker-compose.cpu.yml"
LOG_TAIL=${LOG_TAIL:-120}
LOG_TAIL_BRIEF=${LOG_TAIL_BRIEF:-40}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-etl) WITH_ETL=true; shift ;;
    --exts) EXTS="${2:-$EXTS}"; shift 2 ;;
    --loaders) LOADERS="${2:-$LOADERS}"; shift 2 ;;
    --question) Q="${2:-$Q}"; shift 2 ;;
    --nocache) NOCACHE=true; shift ;;
    --debug) DEBUG=true; shift ;;
    *) echo "arg desconhecido: $1"; exit 2 ;;
  esac
done

step() { echo -e "\n\033[1;34m[SMOKE]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERRO]\033[0m $*"; }

run_with_spinner() {
  local label="$1"
  shift
  if [[ "$DEBUG" == "true" ]]; then
    warn "executando (debug): $label"
    "$@"
    return $?
  fi
  "$@" >/dev/null &
  local pid=$!
  local spinner='|/-\'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r   [%c] %s" "${spinner:i%4:1}" "$label"
    sleep 0.4
    i=$(((i + 1) % 4))
  done
  wait "$pid"
  local status=$?
  if [[ $status -eq 0 ]]; then
    printf "\r   [ok] %s\n" "$label"
  else
    printf "\r   [err] %s\n" "$label"
  fi
  return $status
}

debug_context() {
  local services=($*)
  local tail="$LOG_TAIL"
  if [[ "$DEBUG" != "true" ]]; then
    tail="$LOG_TAIL_BRIEF"
    warn "exibindo resumo de logs (use --debug para mais detalhes)"
  fi
  warn "docker-compose ps (${COMPOSE_FILE})"
  if ! docker-compose -f "$COMPOSE_FILE" ps; then
    warn "falha ao executar docker-compose ps"
  fi
  if ((${#services[@]})); then
    warn "logs recentes (${services[*]})"
    if ! docker-compose -f "$COMPOSE_FILE" logs --tail "$tail" "${services[@]}"; then
      warn "falha ao coletar logs (${services[*]})"
    fi
  fi
}

json_payload() {
  local debug_value="$1"
  jq -Rn --arg question "$Q" --argjson debug "$debug_value" '{question:$question, debug:$debug}'
}

SERVICES=(ai_projeto_api ai_web_ui)
LABEL="API+Web"
if $WITH_ETL; then
  SERVICES+=(ai_etl)
  LABEL+=" +ETL"
fi

if $NOCACHE; then
  step "1/7 Build containers --no-cache (${LABEL})"
  run_with_spinner "docker-compose build --no-cache ${LABEL}" \
    docker-compose -f "$COMPOSE_FILE" build --no-cache "${SERVICES[@]}"
else
  step "1/7 Pull containers (${LABEL})"
  if ! run_with_spinner "docker-compose pull ${LABEL}" \
    docker-compose -f "$COMPOSE_FILE" pull "${SERVICES[@]}"; then
    warn "pull falhou, tentando build local (${LABEL})"
    run_with_spinner "docker-compose build ${LABEL}" \
      docker-compose -f "$COMPOSE_FILE" build "${SERVICES[@]}"
  fi
fi

if $WITH_ETL; then
  step "2/7 (opcional) Rodando ETL — exts=${EXTS}, loaders=${LOADERS}"
  if ! docker-compose -f "$COMPOSE_FILE" run --rm ai_etl         python scripts/etl_build_index.py           --data ./data           --out /app/vector_store/faiss_index           --exts "${EXTS}"           --loaders "${LOADERS}"; then
    err "ETL falhou"
    debug_context ai_etl
    exit 1
  fi
  ok "ETL finalizado"
fi

step "3/7 Subindo API+Web (CPU)"
run_with_spinner "docker-compose up -d API+Web" \
  docker-compose -f "$COMPOSE_FILE" up -d ai_projeto_api ai_web_ui

step "4/7 Healthz (8080)"
if ! HZ=$(curl -fsS http://localhost:8080/api/healthz); then
  err "falha ao consultar /api/healthz"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
if ! echo "$HZ" | jq . >/dev/null; then
  err "healthz não é JSON"
  echo "$HZ"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
READY=$(echo "$HZ" | jq -r '.ready')
FAISS=$(echo "$HZ" | jq -r 'if has("faiss") then .faiss else .faiss_ok end // "false"')
LLM=$(echo "$HZ" | jq -r '.llm_ok // "true"')
if [[ "$READY" != "true" || "$FAISS" != "true" || "$LLM" != "true" ]]; then
  err "healthz não ok (ready=$READY faiss=$FAISS llm=$LLM)"
  echo "$HZ"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
ok "ready:$READY faiss:$FAISS llm:$LLM"

step "5/7 Consulta via 5000 (debug=true)"
if ! R5000=$(curl -fsS -H "Content-Type: application/json" -d "$(json_payload true)" http://localhost:5000/query); then
  err "falha ao chamar porta 5000"
  debug_context ai_projeto_api
  exit 1
fi
echo "$R5000" | jq '.context_found, .debug.route, .debug.timing_ms' || true

step "6/7 Consulta via 8080 (debug=false)"
if ! R8080=$(curl -fsS -H "Content-Type: application/json" -d "$(json_payload false)" http://localhost:8080/api/query); then
  err "falha ao chamar porta 8080"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
if ! echo "$R8080" | jq '.answer, .citations' >/dev/null; then
  err "resposta inválida"
  echo "$R8080"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
LEN=$(echo "$R8080" | jq -r '.answer | length')
if [[ "${LEN:-0}" -le 0 ]]; then
  err "answer vazio"
  echo "$R8080"
  debug_context ai_projeto_api ai_web_ui
  exit 1
fi
ok "answer ok (${LEN} chars)"

step "7/7 Verificações extra (reranker opcional)"
# Se debug.rerank existir, garantir que score é sempre float (não null)
if echo "$R5000" | jq '.debug.rerank' >/dev/null 2>&1; then
  BAD=$(echo "$R5000" | jq '[.debug.rerank.scored[]?.score] | map(type) | any(. != "number")')
  if [[ "$BAD" == "true" ]]; then
    err "scores do rerank não são numéricos"
    echo "$R5000" | jq '.debug.rerank'
    debug_context ai_projeto_api
    exit 1
  fi
  ok "reranker com scores numéricos"
else
  ok "sem reranker (ou desativado)"
fi

ok "SMOKE CPU concluído com sucesso."
