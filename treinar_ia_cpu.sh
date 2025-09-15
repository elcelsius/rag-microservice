#!/bin/bash
# Este script executa o pipeline de ETL em modo CPU.
# Uso:
#   ./treinar_ia_cpu.sh            -> Limpa a base e retreina TUDO.
#   ./treinar_ia_cpu.sh --update   -> Adiciona apenas arquivos novos.

MODE="rebuild"
ARG1="$1"

if [[ "$ARG1" == "--update" ]]; then
    MODE="update"
    echo "🚀 Iniciando o processo de ETL em modo de ATUALIZAÇÃO (CPU)..."
else
    echo "🧠 Iniciando o processo de ETL em modo de REBUILD COMPLETO (CPU)..."
fi

cd "$(dirname "$0")"
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm etl python3 etl_orchestrator.py "$MODE"

echo ""
if [[ "$MODE" == "update" ]]; then
    echo "✅ Atualização (CPU) concluída!"
else
    echo "✅ Treinamento completo (CPU) concluído!"
fi