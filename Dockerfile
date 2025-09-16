# syntax=docker/dockerfile:1

# ====== Base comum (CUDA + Python 3.11) ======
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.11 python3-pip python3.11-venv \
      python-is-python3 \
      build-essential curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# garante que "python" exista (além do pacote acima):
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# requirements + install
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# código
COPY . /app

# ====== Estágio: ETL ======
FROM base AS etl
# sem CMD aqui; o docker-compose define o command em ai_etl

# ====== Estágio: API ======
FROM base AS ai_api
EXPOSE 5000
ENV FLASK_ENV=production
# use python3 aqui para evitar o erro "python not found"
CMD ["python3", "-u", "api.py"]
