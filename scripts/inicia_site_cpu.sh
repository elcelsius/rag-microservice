#!/bin/bash
# Inicia todo o ambiente em MODO CPU e abre o site no navegador.

echo "🚀 Iniciando todos os serviços (Postgres, API, Web UI) em modo CPU..."
cd "$(dirname "$0")/.."

# Sobe os contêineres usando a configuração base e a de CPU.
# O arquivo docker-compose.cpu.yml sobrescreve ou adiciona configurações para rodar sem GPU.
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d

echo ""
echo "⏳ Aguardando o servidor web ficar pronto na porta 8080..."

# Loop de verificação para garantir que o site só seja aberto quando estiver pronto.
while ! curl --silent --head --fail http://localhost:8080 > /dev/null; do
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Servidor web está no ar!"
echo "🌐 Abrindo o site no seu navegador padrão..."

# Abre a URL no navegador do Windows a partir do WSL.
explorer.exe http://localhost:8080

echo ""
echo "🎉 Tudo pronto! Seu ambiente (CPU) está no ar."