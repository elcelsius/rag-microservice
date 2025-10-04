#!/usr/bin/env bash
# build_and_publish_images.sh
# Constrói (e opcionalmente publica) as imagens CPU/GPU com prefixo e tag customizáveis.
set -euo pipefail

show_help() {
  cat <<'USAGE'
Uso: ./scripts/build_and_publish_images.sh [opções]

Opções:
  --prefix PREFIX   Prefixo do repositório (ex.: ghcr.io/seu-org/rag). Também pode vir da env RAG_IMAGE_PREFIX.
  --tag TAG         Tag para as imagens (default: valor de RAG_IMAGE_TAG ou 'local').
  --push            Executa docker-compose push após o build (ou defina RAG_IMAGE_PUSH=true).
  -h, --help        Mostra esta mensagem.

Pré-requisitos:
  - docker login no registry desejado (se for usar --push).
  - docker-compose disponível (v1 ou v2).

As imagens geradas seguirão o padrão:
  ${RAG_IMAGE_PREFIX:-rag-microservice}-ai_projeto_api:${RAG_IMAGE_TAG:-local}
  ${RAG_IMAGE_PREFIX:-rag-microservice}-ai_etl:${RAG_IMAGE_TAG:-local}
E para GPU será acrescentado o sufixo 'gpu-' antes do nome do serviço.
USAGE
}

PREFIX=${RAG_IMAGE_PREFIX:-}
TAG=${RAG_IMAGE_TAG:-local}
PUSH=${RAG_IMAGE_PUSH:-false}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[ERRO] Argumento desconhecido: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PREFIX" ]]; then
  PREFIX="rag-microservice"
fi

export RAG_IMAGE_PREFIX="$PREFIX"
export RAG_IMAGE_TAG="$TAG"

bold=$'\033[1m'
reset=$'\033[0m'
step() { echo "${bold}[build-images]${reset} $*"; }

step "Prefixo: $RAG_IMAGE_PREFIX | Tag: $RAG_IMAGE_TAG"

docker_compose() {
  docker-compose "$@"
}

step "Build CPU (ai_etl, ai_projeto_api)"
docker_compose -f docker-compose.cpu.yml build ai_etl ai_projeto_api

step "Build GPU (ai_etl, ai_projeto_api)"
docker_compose -f docker-compose.gpu.yml build ai_etl ai_projeto_api

if [[ "$PUSH" == "true" ]]; then
  step "Push CPU images"
  docker_compose -f docker-compose.cpu.yml push ai_etl ai_projeto_api
  step "Push GPU images"
  docker_compose -f docker-compose.gpu.yml push ai_etl ai_projeto_api
  step "Push concluído"
fi

step "Concluído"



