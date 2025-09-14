# Passo 1: Usar uma imagem oficial da NVIDIA com o CUDA Toolkit
# Esta imagem é baseada em Ubuntu 24.04 e já vem com tudo que a GPU precisa
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# Passo 2: Instalar o Python e o pip dentro desta nova imagem
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Define o diretório de trabalho padrão
WORKDIR /app

# Copia primeiro o arquivo de dependências para aproveitar o cache do Docker
COPY requirements.txt ./requirements.txt

# Instala as dependências usando o pip que acabamos de instalar
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia todo o resto do código do projeto para o diretório de trabalho
COPY . .

# Define o comando padrão que será executado
CMD ["python3", "etl_orchestrator.py"]