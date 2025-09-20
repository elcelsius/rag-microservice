#!/usr/bin/env bash
set -e

echo "[1/5] Build/Up API+Web..."
docker-compose -f docker-compose.cpu.yml build ai_projeto_api ai_web_ui >/dev/null
docker-compose -f docker-compose.cpu.yml up -d ai_projeto_api ai_web_ui >/dev/null

echo "[2/5] Healthz (8080)..."
curl -fsS http://localhost:8080/api/healthz | jq .ready

echo "[3/5] Consulta via 5000..."
curl -fsS -H "Content-Type: application/json" \
  -d '{"question":"onde encontro informação de monitoria de computação?","debug":true}' \
  http://localhost:5000/query | jq '.context_found, .debug.timing_ms'

echo "[4/5] Consulta via 8080..."
curl -fsS -H "Content-Type: application/json" \
  -d '{"question":"onde encontro informação de monitoria de computação?","debug":false}' \
  http://localhost:8080/api/query | jq '.answer | length'

echo "[5/5] OK"
