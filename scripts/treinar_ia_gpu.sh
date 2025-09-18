#!/bin/bash

# --- Função de Ajuda ---
# Explica como usar o script e o que cada opção faz.
show_usage() {
    echo ""
    echo "Uso: ./treinar_ia_gpu.sh [OPÇÃO]"
    echo ""
    echo "Executa o pipeline de ETL para treinar a base de conhecimento da IA em modo GPU."
    echo ""
    echo "Opções Disponíveis:"
    echo "--------------------"
    echo ""
    echo "  (sem opção)     ->  REBUILD COMPLETO (Padrão)"
    echo "                  Apaga toda a base de conhecimento e a recria do zero com todos os"
    echo "                  arquivos da pasta /data. É o processo mais seguro para garantir"
    echo "                  consistência, porém mais demorado."
    echo ""
    echo "  --update        ->  ATUALIZAÇÃO INCREMENTAL"
    echo "                  Verifica e processa apenas arquivos novos ou modificados na pasta /data,"
    echo "                  preservando os dados existentes. É um processo significativamente mais rápido."
    echo ""
    echo "  --help, -h      ->  MOSTRAR ESTA AJUDA"
    echo "                  Exibe esta mensagem com as opções e explicações."
    echo ""
}

# --- Lógica Principal ---

# Verifica o argumento passado pelo usuário
ARG1="$1"

# Se o usuário pedir ajuda, mostra a mensagem e sai
if [[ "$ARG1" == "--help" || "$ARG1" == "-h" ]]; then
    show_usage
    exit 0
fi

MODE="rebuild" # Define o modo padrão como rebuild completo

# Define o modo de execução com base no argumento
if [[ "$ARG1" == "--update" ]]; then
    MODE="update"
elif [[ -n "$ARG1" ]]; then # Se um argumento foi passado, mas não é '--update'
    echo "Erro: Opção inválida '$ARG1'."
    show_usage
    exit 1
fi

# Mensagens de status para o usuário
if [[ "$MODE" == "update" ]]; then
    echo "🚀 Iniciando o processo de ETL em modo de ATUALIZAÇÃO (rápido)..."
    echo "Verificando apenas arquivos novos ou modificados na pasta /data."
else
    echo "🧠 Iniciando o processo de ETL em modo de REBUILD COMPLETO (demorado)..."
    echo "A base de conhecimento será limpa e reconstruída do zero."
fi

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker-compose -f docker-compose.gpu.yml run --rm ai_etl bash -lc 'python3 -u scripts/etl_build_index.py'


# Mensagem final
echo ""
if [[ "$MODE" == "update" ]]; then
    echo "✅ Atualização concluída!"
else
    echo "✅ Treinamento completo concluído!"
fi