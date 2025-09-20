#!/usr/bin/env bash
set -euo pipefail
echo "[inicia_site_cpu] subindo API+Web..."
docker-compose -f docker-compose.gpu.yml build ai_projeto_api ai_web_ui
docker-compose -f docker-compose.gpu.yml up -d ai_projeto_api ai_web_ui
echo "Healthz:"
curl -s http://localhost:8080/api/healthz | jq .
echo "UI: http://localhost:8080/"
