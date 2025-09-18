#!/bin/bash
# Este script executa o pipeline de ETL em modo GPU para criar/atualizar
# a base de conhecimento da IA.

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
    echo "                  (Funcionalidade a ser implementada no script Python) Atualmente,"
    echo "                  esta opção se comporta como o rebuild, mas foi mantida para uso futuro."
    echo ""
    echo "  --help, -h      ->  MOSTRAR ESTA AJUDA"
    echo "                  Exibe esta mensagem com as opções e explicações."
    echo ""
}

# --- Lógica Principal ---

# Verifica o primeiro argumento passado para o script
ARG1="$1"

# Se o usuário pedir ajuda, mostra a mensagem e termina o script.
if [[ "$ARG1" == "--help" || "$ARG1" == "-h" ]]; then
    show_usage
    exit 0
fi

MODE="rebuild" # Define o modo padrão como rebuild completo

# Define o modo de execução com base no argumento
if [[ "$ARG1" == "--update" ]]; then
    MODE="update"
elif [[ -n "$ARG1" ]]; then # Se um argumento foi passado, mas não é um dos válidos
    echo "Erro: Opção inválida '$ARG1'."
    show_usage
    exit 1
fi

# Mensagens de status para informar o usuário sobre o que está acontecendo.
if [[ "$MODE" == "update" ]]; then
    echo "🚀 Iniciando o processo de ETL em modo de ATUALIZAÇÃO (GPU)..."
    echo "Verificando apenas arquivos novos ou modificados na pasta /data."
else
    echo "🧠 Iniciando o processo de ETL em modo de REBUILD COMPLETO (GPU)..."
    echo "A base de conhecimento será limpa e reconstruída do zero."
fi

# Garante que o script pare se houver erros.
set -euo pipefail
# Muda para o diretório raiz do projeto (um nível acima de onde o script está).
cd "$(dirname "$0")/.."

# Comando principal:
# - `docker compose -f ...`: Usa o arquivo de configuração específico para GPU.
# - `run --rm`: Executa um comando único em um novo contêiner para o serviço `ai_etl` e o remove ao final.
# - `bash -lc '...'`: Executa o comando python dentro de um shell bash de login, o `-u` garante que a saída do python não seja bufferizada.
docker compose -f docker-compose.gpu.yml run --rm ai_etl bash -lc 'python3 -u scripts/etl_build_index.py'

# Mensagem final de sucesso.
echo ""
if [[ "$MODE" == "update" ]]; then
    echo "✅ Atualização (GPU) concluída!"
else
    echo "✅ Treinamento completo (GPU) concluído!"
fi