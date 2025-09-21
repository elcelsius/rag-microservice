#!/bin/bash
# Inicia o chat interativo no terminal em MODO CPU.

echo "🚀 Iniciando o Copiloto de IA (CPU) no terminal... Por favor, aguarde."
cd "$(dirname "$0")/.."

# Executa o script `query_handler.py` usando a configuração de CPU.
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm etl python3 query_handler.py

echo "✅ Sessão do copiloto (CPU) encerrada."