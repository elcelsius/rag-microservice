#!/bin/bash
# Inicia todo o ambiente em MODO GPU e abre o site no navegador.

echo "🚀 Iniciando todos os serviços (Postgres, API, Web UI)..."
docker-compose up --build -d

echo ""
echo "⏳ Aguardando o servidor web ficar pronto na porta 8080..."

# Loop que espera o servidor web responder.
while ! curl --silent --head --fail http://localhost:8080 > /dev/null; do
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Servidor web está no ar!"
echo "🌐 Abrindo o site no seu navegador padrão..."

# Comando para abrir a URL no navegador padrão do Windows a partir do WSL2.
explorer.exe http://localhost:8080

echo ""
echo "🎉 Tudo pronto! Seu ambiente (GPU) está no ar."