Este documento fornece um contexto completo para o projeto rag-microservice, detalhando sua arquitetura, componentes principais, configurações, fluxo de trabalho e estratégias de otimização. Ele serve como um guia centralizado para desenvolvedores e colaboradores, facilitando a compreensão, o desenvolvimento e a avaliação do projeto.

1. Visão Geral do Projeto
O rag-microservice é uma implementação de um sistema de Pergunta-Resposta (Question-Answering) baseado em Retrieval-Augmented Generation (RAG). A solução é orquestrada por um agente inteligente construído com LangGraph e é conteinerizada com Docker, promovendo modularidade e robustez.

O sistema é projetado para ser de alta performance e configurável, incorporando um cache de respostas com Redis para reduzir a latência, um framework de avaliação de qualidade com Ragas, e suporte a diferentes modelos de embeddings e LLMs (Large Language Models), como Google Gemini e OpenAI. Ele é capaz de responder a perguntas complexas utilizando um fluxo que pode incluir busca lexical e vetorial em uma base de conhecimento, reranking de resultados e geração de texto por um LLM.

2. Arquitetura e Fluxo de Dados
A arquitetura do sistema combina um pipeline de RAG híbrido com um fluxo de agente orquestrador, otimizado por uma camada de cache.

2.1. Diagrama do Fluxo de Dados (Mermaid)

Snippet de código

flowchart LR
    subgraph Client
        U[Usuário]
    end
    subgraph API
        A[Flask API<br/>/query]
        H[Health/Metrics]
        C[(Redis Cache)]
    end
    subgraph Retrieval
        VS[(FAISS Index)]
        EMB[HF Embeddings]
        MQ[Multi-Query<br/> + Sinônimos]
        LEX["Busca Lexical<br/>(sentenças + bônus de depto)"]
        RER[CrossEncoder<br/>(Reranker)]
    end
    subgraph LLM
        TRI[LLM Triagem]
        GEN[LLM Geração de Resposta]
    end
    subgraph ETL
        LD[Loaders<br/>(pdf, docx, md, txt, code, ...)]
        SPL[Chunking]
        EMB_E[HF Embeddings]
        VS_B[FAISS Build/Update]
        DB[(PostgreSQL<br/>hashes/chunks)]
        C_ETL[(Redis Cache Invalidation)]
    end

    U -->|Pergunta| A
    A -->|1. Cache?| C
    C -->|HIT| A
    C -->|MISS| TRI

    TRI --> AR[Auto Resolver]
    AR -->|Rota 1: Lexical| LEX
    LEX -->|Se encontrou| GEN
    AR -->|Rota 2: Vetorial| MQ --> VS --> RER --> GEN
    GEN -->|Resposta| A
    A -->|Salva no Cache| C

    H --- A

    %% ETL
    LD --> SPL --> EMB_E --> VS_B
    VS_B --> VS
    VS_B --> DB
    DB -->|incremental| VS_B
    VS_B -->|Invalida Cache| C_ETL
2.2. Fluxo do Agente (LangGraph)

O agente atua como um orquestrador inteligente que decide a melhor forma de responder a uma pergunta.

Pergunta do Usuário: Entrada inicial do sistema.

Nó de Triagem: Um LLM classifica a pergunta. Se puder ser respondida, encaminha para o nó Auto-Resolver; se precisar de mais informações, encaminha para o Nó de Pedir Informação.

Nó Auto-Resolver: Aciona o pipeline de RAG para buscar e gerar uma resposta.

Nó de Pedir Informação: Formula uma resposta solicitando esclarecimentos ao usuário.

2.3. Pipeline de RAG (Retrieval-Augmented Generation)

Este é o coração do sistema, acionado pelo agente.

Verificação de Cache (Redis): O sistema primeiro verifica se a pergunta já foi respondida e está no cache. Se houver um HIT, a resposta é retornada imediatamente.

Busca Híbrida (Cache MISS):

Rota Lexical (Prioritária): Uma busca lexical rápida é tentada, com bônus de pontuação para termos e departamentos relevantes definidos em terms.yml.

Rota Vetorial (Fallback): Se a busca lexical falhar, o pipeline continua com:

Multi-Query: Geração de múltiplas variações da pergunta para aumentar o recall.

Busca Vetorial (FAISS): As queries buscam documentos no índice vetorial FAISS.

CrossEncoder (Rerank): Os documentos recuperados são reranqueados por um modelo CrossEncoder para identificar os mais relevantes.

Seleção de Contexto: Os chunks de texto mais relevantes são selecionados para formar o contexto.

Geração e Cache da Resposta: Um LLM utiliza o contexto para gerar a resposta final, que é formatada (geralmente em Markdown com citações) e, antes de ser retornada, é salva no cache Redis para acelerar futuras requisições idênticas.

3. Componentes Principais
api.py: Servidor Flask que expõe os endpoints, carrega os modelos e o índice FAISS, e gerencia a conexão com o Redis para a lógica de cache.

agent_workflow.py: Define a lógica do agente LangGraph, seus nós e as regras de decisão.

query_handler.py: Implementa a lógica central do RAG, incluindo a busca híbrida, multi-queries, reranking e geração de resposta.

llm_client.py: Cliente unificado e robusto para interagir com LLMs (Google Gemini, OpenAI), abstraindo a comunicação e gerenciando retentativas.

etl_orchestrator.py: Script para o pipeline de ETL (Extract, Transform, Load) que constrói e atualiza a base de conhecimento vetorial (FAISS e PostgreSQL). Agora também é responsável por invalidar o cache do Redis após a atualização da base.

eval_rag.py: Script para avaliação de ponta a ponta do sistema RAG utilizando a biblioteca Ragas e um dataset de teste (ex: evaluation_dataset.jsonl).

telemetry.py: Módulo para logging de telemetria e métricas do sistema.

loaders/: Módulo com loaders unificados para diversos formatos de documentos.

prompts/: Diretório com os prompts utilizados pelos LLMs nas diferentes etapas.

config/ontology/terms.yml: Arquivo YAML que define termos, sinônimos, aliases e boosts para a busca.

docker-compose.cpu.yml / docker-compose.gpu.yml: Arquivos para orquestrar a execução dos serviços (API, UI, PostgreSQL, Redis) em ambientes CPU e GPU.

4. Configuração e Variáveis de Ambiente
O projeto é configurado via arquivo .env. As variáveis principais incluem chaves de API para LLMs (GOOGLE_API_KEY, OPENAI_API_KEY), configurações de modelos (EMBEDDINGS_MODEL, RERANKER_PRESET), parâmetros de busca (HYBRID_ENABLED, MQ_ENABLED) e as credenciais para os serviços de infraestrutura (POSTGRES_*, REDIS_*).

5. Como Executar o Projeto
O projeto é projetado para ser executado via Docker Compose.

Pré-requisitos: Docker, Docker Compose e uma chave de API de LLM (Google ou OpenAI).

Configuração: Copie .env.example para .env e preencha com suas chaves de API e outras configurações.

Execução:

CPU: docker-compose -f docker-compose.cpu.yml up --build

GPU: docker-compose -f docker-compose.gpu.yml up --build

Acesso:

API do RAG: http://localhost:5000

Interface Web: http://localhost:8080

PostgreSQL: localhost:5432

Redis: localhost:6379

6. Endpoints da API
POST /agent/ask: Endpoint principal para o fluxo do agente LangGraph.

POST /query: Endpoint legado para o pipeline de RAG direto.

GET /healthz: Verifica a prontidão dos serviços (FAISS, LLM).

GET /metrics: Retorna métricas de uptime e contadores de requisições.

7. Avaliação e Testes
Avaliação de Qualidade: O script eval_rag.py utiliza a biblioteca Ragas para calcular métricas de Recuperação (Recall, MRR) e Geração (Faithfulness, Answer Relevancy) a partir de um dataset de teste, permitindo a validação objetiva da qualidade do sistema.

Testes Automatizados: O projeto utiliza pytest para testes unitários e de integração.

8. Recomendações para Melhoria da Qualidade do RAG
Para otimizar a qualidade do sistema, é crucial analisar e refinar cada etapa do pipeline.

Embeddings e Vector Store: Avalie continuamente diferentes modelos de embeddings (ex: específicos de domínio) e otimize os parâmetros de indexação do FAISS para equilibrar velocidade e precisão.

Estratégias de Chunking: Experimente diferentes tamanhos de chunk e sobreposição. Considere o uso de chunking semântico para respeitar a estrutura dos documentos.

Recuperação (Retrieval): Ajuste os parâmetros da busca híbrida (LEXICAL_THRESHOLD, DEPT_BONUS) e enriqueça continuamente o arquivo terms.yml com sinônimos e aliases do domínio.

Reranking: Avalie os diferentes presets do reranker (fast, balanced, full) para encontrar o melhor trade-off entre qualidade, latência e custo computacional.

Geração de Respostas (LLM): Refine os prompts para guiar o LLM a gerar respostas mais precisas. Para casos complexos, considere modelos mais poderosos como gemini-1.5-pro ou gpt-4o.

Agente LangGraph e Orquestração: Melhore a precisão do nó de triagem com exemplos "few-shot" no prompt e monitore os fluxos de decisão para identificar gargalos ou classificações incorretas.

Avaliação Contínua: Invista na criação de um dataset de avaliação robusto e representativo. Automatize a execução dos testes de avaliação como parte de um pipeline de CI/CD para evitar regressões.

Otimização de Performance e Custo: Além do cache em Redis, explore a quantização de modelos para acelerar a inferência e monitore os custos das chamadas de API do LLM, otimizando os prompts e o contexto enviado.

Este documento unificado agora reflete a arquitetura completa e atualizada do rag-microservice, servindo como uma fonte única e confiável de informação para todos os envolvidos no projeto.

ATENÇÃO: Sempre responda em português brasileiro, independente do idioma do erro ou da pergunta.