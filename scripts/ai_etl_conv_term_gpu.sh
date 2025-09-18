#!/bin/bash
# Inicia o chat interativo no terminal em MODO GPU.
# Este script é útil para testar o backend da IA diretamente, sem a interface web.

echo "🚀 Iniciando o Copiloto de IA (GPU) no terminal... Por favor, aguarde."
cd "$(dirname "$0")/.."

# Executa o script `query_handler.py` dentro de um contêiner temporário do serviço `etl`
# usando a configuração de GPU.
docker compose -f docker-compose.gpu.yml run --rm etl python3 query_handler.py

echo "✅ Sessão do copiloto (GPU) encerrada."