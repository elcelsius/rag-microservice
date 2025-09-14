# Passo 1: Usar a imagem de DESENVOLVIMENTO da NVIDIA, que contém o toolkit completo.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Passo 2: Instalar a versão padrão do Python e o pip para o Ubuntu 24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    apt-get clean

# Define o diretório de trabalho padrão
WORKDIR /app

# Copia primeiro o arquivo de dependências para aproveitar o cache do Docker
COPY requirements.txt ./requirements.txt

# Instala as dependências, quebrando a proteção do sistema (seguro dentro do Docker)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia todo o resto do código do projeto para o diretório de trabalho
COPY . .

# Define o comando padrão que será executado
CMD ["python3", "etl_orchestrator.py"]