#!/bin/bash
# Inicia o chat interativo no terminal em MODO GPU.

echo "🚀 Iniciando o Copiloto de IA (GPU)... Por favor, aguarde."
cd "$(dirname "$0")"
docker-compose run --rm etl python3 query_handler.py

echo "✅ Sessão do copiloto (GPU) encerrada."