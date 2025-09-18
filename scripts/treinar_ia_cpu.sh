#!/bin/bash
# Este script executa o pipeline de ETL em modo CPU para criar/atualizar
# a base de conhecimento da IA.

# --- Função de Ajuda ---
show_usage() {
    echo ""
    echo "Uso: ./treinar_ia_cpu.sh [OPÇÃO]"
    echo ""
    echo "Executa o pipeline de ETL para treinar a base de conhecimento da IA em modo CPU."
    echo ""
    echo "Opções Disponíveis:"
    echo "--------------------"
    echo "  (sem opção)     -> REBUILD COMPLETO (Padrão). Lento, mas seguro."
    echo "  --update        -> ATUALIZAÇÃO INCREMENTAL (Não implementado, se comporta como rebuild)."
    echo "  --help, -h      -> MOSTRAR ESTA AJUDA."
    echo ""
}

# --- Lógica Principal ---

ARG1="$1"

# Validação dos argumentos, igual à versão GPU para consistência.
if [[ "$ARG1" == "--help" || "$ARG1" == "-h" ]]; then
    show_usage
    exit 0
fi

MODE="rebuild"

if [[ "$ARG1" == "--update" ]]; then
    MODE="update"
elif [[ -n "$ARG1" ]]; then
    echo "Erro: Opção inválida '$ARG1'."
    show_usage
    exit 1
fi

# Mensagens de status para o usuário.
if [[ "$MODE" == "update" ]]; then
    echo "🚀 Iniciando o processo de ETL em modo de ATUALIZAÇÃO (CPU)..."
else
    echo "🧠 Iniciando o processo de ETL em modo de REBUILD COMPLETO (CPU)..."
fi

# Boas práticas de script: parar em caso de erro.
set -euo pipefail
# Navega para a raiz do projeto.
cd "$(dirname "$0")/.."

# Comando principal para executar o ETL em modo CPU.
# Aponta para os arquivos de compose corretos (`.yml` base e `.cpu.yml` para override).
docker compose -f docker-compose.yml -f docker-compose.cpu.yml run --rm ai_etl bash -lc 'python3 -u scripts/etl_build_index.py'

# Mensagem final de sucesso.
echo ""
if [[ "$MODE" == "update" ]]; then
    echo "✅ Atualização (CPU) concluída!"
else
    echo "✅ Treinamento completo (CPU) concluído!"
fi