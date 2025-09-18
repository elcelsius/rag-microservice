#!/bin/bash
# Inicia o chat interativo no terminal em MODO CPU.

echo "🚀 Iniciando o Copiloto de IA (CPU)... Por favor, aguarde."
cd "$(dirname "$0")/.."
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm etl python3 query_handler.py

echo "✅ Sessão do copiloto (CPU) encerrada."