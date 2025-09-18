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
- Docker Desktop
- WSL2 (para usuários Windows)
- **Para modo GPU**: Drivers NVIDIA com suporte a CUDA instalados no host.

### 🔧 Instalação
1. Clone o repositório:
   ```bash
   git clone [https://github.com/elcelsius/ai_etl_project.git](https://github.com/elcelsius/ai_etl_project.git)
   cd ai_etl_project
   ```
2. Configure as variáveis de ambiente (copie `.env.example` para `.env` e preencha sua `GOOGLE_API_KEY`).
3. Adicione seus arquivos de documentação na pasta `data/`.
4. Dê permissão de execução para os scripts:
   ```bash
   chmod +x scripts/*.sh
   ```

---

## 💡 Fluxo de Trabalho (Como Usar)
Escolha o ambiente de acordo com seu hardware.

### Opção 1: Ambiente com GPU NVIDIA (Recomendado)
Use os scripts localizados em `scripts/` com o sufixo `_gpu`.

**Para treinar a IA (ETL):**
```bash
# Rebuild completo (lento, apaga tudo e refaz)
./scripts/treinar_ia_gpu.sh

# Atualização incremental (rápido, adiciona somente arquivos novos)
./scripts/treinar_ia_gpu.sh --update
```

**Para iniciar o site e conversar pela interface web:**
```bash
./scripts/inicia_site_gpu.sh
```

**Para conversar pelo terminal:**
```bash
./scripts/ai_etl_conv_term_gpu.sh
```

### Opção 2: Ambiente Apenas com CPU
Use os scripts localizados em `scripts/` com o sufixo `_cpu`.

**Para treinar a IA (ETL):**
```bash
# Rebuild completo (lento, apaga tudo e refaz)
./scripts/treinar_ia_cpu.sh

# Atualização incremental (rápido, adiciona somente arquivos novos)
./scripts/treinar_ia_cpu.sh --update
```

**Para iniciar o site e conversar pela interface web:**
```bash
./scripts/inicia_site_cpu.sh
```

**Para conversar pelo terminal:**
```bash
./scripts/ai_etl_conv_term_cpu.sh
```

---

## ⚙️ Como o Sistema Funciona

O projeto é dividido em três componentes principais: o pipeline de ETL, o serviço de API RAG e o agente de IA.

### 1. Pipeline de ETL (Extract, Transform, Load)

Responsável por processar a base de conhecimento e criar um índice vetorial para busca. Implementado em `etl_orchestrator.py`.

- **Extração**: Carrega documentos de diversos formatos (`.pdf`, `.docx`, `.md`, `.txt`, código) da pasta `data/` usando loaders específicos (definidos em `loaders/`).
- **Transformação**: Os documentos são divididos em pequenos pedaços (chunks) usando `RecursiveCharacterTextSplitter` para otimizar a busca. Metadados como `source_file` são associados a cada chunk.
- **Carregamento (Load)**:
    - **Embeddings**: Cada chunk é convertido em um vetor numérico (embedding) usando o modelo `sentence-transformers/all-MiniLM-L6-v2`.
    - **Vector Store (FAISS)**: Os embeddings são armazenados em um índice FAISS, que permite buscas de similaridade eficientes.
    - **Persistência de Metadados (PostgreSQL)**: Informações sobre os chunks e os arquivos processados (incluindo hashes para detecção de modificações) são armazenadas em um banco de dados PostgreSQL. Isso permite atualizações incrementais e rastreamento da base de conhecimento.

### 2. Serviço de API RAG

Uma API Flask (`api.py`) que expõe endpoints para consultas. Ela é responsável por receber as perguntas do usuário, buscar no índice vetorial e orquestrar a geração da resposta.

- **Health Checks e Métricas**: Inclui endpoints `/healthz` para verificar a prontidão da aplicação (FAISS e LLM) e `/metrics` para monitorar o tempo de atividade e o número de consultas.
- **Processamento de Consultas**: Ao receber uma pergunta, a API utiliza o modelo de embeddings e o vetorstore FAISS para encontrar os chunks de documentos mais relevantes.
- **Geração de Resposta**: Os chunks recuperados são passados para o função `answer_question` (em `query_handler.py`) que utiliza um LLM (Google Gemini) para gerar uma resposta coerente e citar as fontes.

### 3. Agente de IA (LangGraph)

Um agente inteligente (`agent_workflow.py`) construído com LangGraph que gerencia o fluxo de conversação.

- **Triagem**: O agente primeiro classifica a pergunta do usuário (`node_triagem`) para decidir se pode ser respondida diretamente ou se requer mais informações.
- **Auto-Resolução (RAG)**: Se a pergunta for clara, o agente tenta resolvê-la usando o pipeline RAG (`node_auto_resolver`). Se houver histórico de conversa, a pergunta é condensada para ser autônoma antes de ser enviada ao RAG.
- **Pedido de Informações**: Se a pergunta for ambígua ou o RAG não encontrar contexto suficiente, o agente formula uma pergunta de esclarecimento ao usuário (`node_pedir_info`) usando o LLM.
- **Tomada de Decisão**: A lógica condicional (`decidir_pos_triagem`, `decidir_pos_auto_resolver`) direciona o fluxo do grafo com base nos resultados da triagem e do RAG.

---

✍️ Autor: Celso Lisboa
📎 Repositório: github.com/elcelsius/ai_etl_project


