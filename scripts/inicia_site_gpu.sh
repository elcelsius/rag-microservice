#!/bin/bash
# Inicia todo o ambiente em MODO GPU e abre o site no navegador.

echo "🚀 Iniciando todos os serviços (Postgres, API, Web UI) em modo GPU..."
# Navega para o diretório raiz do projeto.
cd "$(dirname "$0")/.."

# Sobe os contêineres definidos no arquivo de compose para GPU.
# --build: Reconstrói as imagens se o Dockerfile mudou.
# -d: Modo "detached" (roda em segundo plano).
docker compose -f docker-compose.gpu.yml up --build -d

echo ""
echo "⏳ Aguardando o servidor web ficar pronto na porta 8080..."

# Este loop verifica continuamente se o servidor web já está respondendo.
# `curl --silent --head --fail`: Envia uma requisição HEAD. Falha se o servidor não retornar status 2xx.
# A saída é redirecionada para /dev/null para não poluir o terminal.
while ! curl --silent --head --fail http://localhost:8080 > /dev/null; do
    echo -n "."
    sleep 2 # Espera 2 segundos entre as tentativas.
done

echo ""
echo "✅ Servidor web está no ar!"
echo "🌐 Abrindo o site no seu navegador padrão..."

# `explorer.exe` é um comando específico para quem usa WSL (Subsistema Windows para Linux)
# para abrir uma URL no navegador padrão do Windows.
explorer.exe http://localhost:8080

echo ""
echo "🎉 Tudo pronto! Seu ambiente (GPU) está no ar."