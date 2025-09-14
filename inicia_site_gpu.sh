#!/bin/bash

# Este script automatiza o processo de iniciar todo o ambiente do
# IA Compilot e abrir o site no navegador.

echo "🚀 Iniciando todos os serviços (Postgres, API, Web UI)..."
# O '--build' garante que quaisquer mudanças no Dockerfile sejam aplicadas.
# O '-d' (detached) roda tudo em segundo plano.
docker-compose up --build -d

echo ""
echo "⏳ Aguardando o servidor web ficar pronto na porta 8080..."

# Loop inteligente que espera o servidor web responder antes de continuar.
# Ele tenta acessar os cabeçalhos da URL a cada 2 segundos.
while ! curl --silent --head --fail http://localhost:8080 > /dev/null; do
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Servidor web está no ar!"
echo "🌐 Abrindo o site no seu navegador padrão..."

# Comando para abrir a URL no navegador padrão do Windows a partir do WSL2.
explorer.exe

# Para outros sistemas operacionais (deixar comentado):
# No macOS: open http://localhost:8080
# No Linux padrão com desktop: xdg-open http://localhost:8080

echo ""
echo "🎉 Tudo pronto! Seu ambiente está no ar."
