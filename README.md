# 🤖 AI Copilot - Serviço de ETL e RAG Genérico

Este projeto implementa um pipeline completo de **Retrieval-Augmented Generation (RAG)**, projetado para servir como o núcleo de um copiloto de IA para sistemas web complexos.

O objetivo é **ler, processar e indexar** uma base de conhecimento privada (documentação, código-fonte, etc.) e fornecer uma interface de consulta inteligente, capaz de responder perguntas complexas com alta precisão, utilizando a API do **Google Gemini**.

O sistema é otimizado para ambientes com **GPU NVIDIA**, mas é totalmente compatível com ambientes **apenas com CPU** através de scripts dedicados.

---

## 📋 Principais Funcionalidades
- **Pipeline de ETL Inteligente**: Suporta múltiplos formatos de arquivo e oferece dois modos de treinamento: rebuild completo ou atualização incremental (apenas para arquivos novos).
- **Base de Conhecimento Vetorial**: Utiliza *sentence-transformers* para gerar embeddings e **FAISS** para busca vetorial eficiente.
- **Portabilidade CPU/GPU**: Ambiente containerizado com suporte explícito para execução acelerada por **CUDA** ou em modo **CPU-only**.
- **Persistência de Metadados**: Armazenamento de chunks e rastreamento de arquivos processados em **PostgreSQL**.
- **Agente de IA com LangGraph**: Um agente inteligente avalia as perguntas antes de agir, decidindo entre responder ou pedir mais informações.
- **Geração de Respostas com LLM**: Integração com a API do **Google Gemini**, com modelo configurável via variáveis de ambiente.

---

## 🛠️ Stack de Tecnologias
- **Linguagem**: Python 3.11
- **Orquestração**: Docker & Docker Compose
- **IA & Machine Learning**:
  - LangChain, LangGraph
  - Sentence Transformers (*all-MiniLM-L6-v2*)
  - FAISS-GPU / FAISS-CPU
  - PyTorch
  - Google Generative AI (Gemma, Gemini)
- **Banco de Dados**: PostgreSQL 15
- **Ambiente Base**: Imagem NVIDIA CUDA (GPU) ou Python Slim (CPU)

---

## 🚀 Configuração do Ambiente

### ✅ Pré-requisitos
- Git
- Docker Desktop
- WSL2 (para usuários Windows)
- **Para modo GPU**: Drivers NVIDIA com suporte a CUDA instalados no host.

### 🔧 Instalação
1. Clone o repositório:
   ```bash
   git clone [https://github.com/elcelsius/ai_etl_project.git](https://github.com/elcelsius/ai_etl_project.git)
   cd ai_etl_project
Configure as variáveis de ambiente (copie .env.example para .env e preencha sua GOOGLE_API_KEY).

Adicione seus arquivos de documentação na pasta data/.

Dê permissão de execução para os scripts:

Bash

chmod +x *.sh
💡 Fluxo de Trabalho (Como Usar)
Escolha o ambiente de acordo com seu hardware.

Opção 1: Ambiente com GPU NVIDIA (Recomendado)
Use os scripts com o sufixo _gpu.

Para treinar a IA:

Bash

# Rebuild completo (lento, apaga tudo e refaz)
./treinar_ia_gpu.sh

# Atualização incremental (rápido, adiciona somente arquivos novos)
./treinar_ia_gpu.sh --update
Para iniciar o site e conversar pela interface web:

Bash

./inicia_site_gpu.sh
Para conversar pelo terminal:

Bash

./ai_etl_conv_term_gpu.sh
Opção 2: Ambiente Apenas com CPU
Use os scripts com o sufixo _cpu.

Para treinar a IA:

Bash

# Rebuild completo (lento, apaga tudo e refaz)
./treinar_ia_cpu.sh

# Atualização incremental (rápido, adiciona somente arquivos novos)
./treinar_ia_cpu.sh --update
Para iniciar o site e conversar pela interface web:

Bash

./inicia_site_cpu.sh
Para conversar pelo terminal:

Bash

./ai_etl_conv_term_cpu.sh
✍️ Autor: Celso Lisboa
📎 Repositório: github.com/elcelsius/ai_etl_project
