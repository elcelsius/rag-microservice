#!/bin/bash

# Este script executa o pipeline de ETL.
# Uso:
#   ./treinar_ia.sh            -> Limpa a base e retreina TUDO (processo mais demorado).
#   ./treinar_ia.sh --update   -> Adiciona apenas arquivos novos ou modificados (processo mais rápido).

MODE="rebuild"
ARG1="$1"

if [[ "$ARG1" == "--update" ]]; then
    MODE="update"
    echo "🚀 Iniciando o processo de ETL em modo de ATUALIZAÇÃO (rápido)..."
    echo "Verificando apenas arquivos novos ou modificados na pasta /data."
else
    echo "🧠 Iniciando o processo de ETL em modo de REBUILD COMPLETO (demorado)..."
    echo "A base de conhecimento será limpa e reconstruída do zero."
fi

# Garante que estamos executando a partir do diretório do script
cd "$(dirname "$0")"

# Executa o comando do Docker Compose, passando o modo de execução para o script Python
docker-compose run --rm etl python3 etl_orchestrator.py "$MODE"

echo ""
if [[ "$MODE" == "update" ]]; then
    echo "✅ Atualização concluída!"
else
    echo "✅ Treinamento completo concluído!"
fi