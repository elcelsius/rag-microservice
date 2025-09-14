#!/bin/bash
# Inicia todo o ambiente em MODO CPU e abre o site no navegador.

echo "🚀 Iniciando todos os serviços em modo CPU (Postgres, API, Web UI)..."
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up --build -d

echo ""
echo "⏳ Aguardando o servidor web ficar pronto na porta 8080..."
while ! curl --silent --head --fail http://localhost:8080 > /dev/null; do
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Servidor web está no ar!"
echo "🌐 Abrindo o site no seu navegador padrão..."
explorer.exe http://localhost:8080

echo ""
echo "🎉 Tudo pronto! Seu ambiente (CPU) está no ar."