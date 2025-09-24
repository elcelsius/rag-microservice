# Contexto do Projeto: RAG Microservice

## Visão Geral

Este projeto implementa um microserviço de RAG (Retrieval-Augmented Generation) em Python, utilizando Flask para a API, FAISS para a busca vetorial e LangChain para orquestração. O sistema é projetado para ser modular, configurável e de alta performance, incorporando **cache de respostas com Redis**, um **framework de avaliação de qualidade com RAGAs**, e suporte a diferentes modelos de embeddings e LLMs (Google Gemini e OpenAI).

## Arquitetura e Fluxo de Dados

O sistema segue um fluxo de RAG híbrido, otimizado com uma camada de cache.

1.  **Entrada e Cache**: Uma requisição `POST` é recebida no endpoint `/query`. O sistema primeiro verifica no cache **Redis** se já existe uma resposta para a pergunta. Se sim (HIT), a resposta é retornada imediatamente.
2.  **Triagem (Opcional com Agente)**: Em caso de MISS no cache, um grafo LangGraph pode classificar a intenção do usuário (`AUTO_RESOLVER` ou `PEDIR_INFO`).
3.  **Rota Lexical (Prioritária)**: O `query_handler.py` tenta uma busca lexical rápida, com bônus de pontuação para departamentos relevantes.
4.  **Rota Vetorial (Fallback)**: Se a busca lexical falhar, o sistema expande a pergunta (Multi-Query), consulta o índice **FAISS** e reordena os resultados com um **Cross-Encoder** para máxima relevância.
5.  **Geração e Cache**: A resposta final é gerada por um LLM. Antes de ser retornada ao usuário, ela é **salva no cache Redis** para acelerar futuras requisições idênticas.
6.  **Invalidação de Cache no ETL**: Ao final do processo de ETL, o cache do Redis é completamente invalidado para garantir que as respostas reflitam sempre a base de conhecimento mais recente.

### Diagrama (Mermaid)

```mermaid
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
      C_ETL[(Redis Cache)]
    end

    U -->|Pergunta| A
    A -->|1. Cache?| C
    C -->|HIT| A
    C -->|MISS| TRI

    TRI --> AR[Auto Resolver]
    AR -->|Rota 1| LEX
    LEX -->|se encontrou| GEN
    AR -->|Rota 2| MQ --> VS --> RER --> GEN
    GEN -->|Resposta| A
    A -->|Salva no Cache| C

    H --- A

    %% ETL
    LD --> SPL --> EMB_E --> VS_B --> VS
    VS_B --> DB
    DB -->|incremental| VS_B
    VS_B --> C_ETL
```

## Componentes Principais

*   `api.py`: Ponto de entrada da API Flask. Agora também gerencia a **conexão com o Redis** e implementa a lógica de leitura e escrita no cache.
*   `query_handler.py`: Coração da lógica de RAG (busca híbrida, rerank, geração).
*   `llm_client.py`: Abstrai a comunicação com os LLMs.
*   `etl_orchestrator.py`: Orquestra o ETL. Agora também é responsável por **invalidar o cache do Redis** após a atualização da base de conhecimento.
*   `agent_workflow.py`: Implementa o fluxo de agente com LangGraph.
*   `tools/evaluate.py`: **(NOVO)** Script para avaliação de qualidade do pipeline RAG usando a biblioteca `ragas` e um dataset de teste (`evaluation_dataset.jsonl`).
*   `config/ontology/terms.yml`: Arquivo de configuração central para a lógica de busca (sinônimos, departamentos, etc.).

## Tecnologias e Frameworks

*   **Backend**: Python 3.10+, Flask
*   **Cache**: **Redis**
*   **RAG**: LangChain, FAISS, HuggingFace Embeddings, Sentence Transformers.
*   **Avaliação de Qualidade**: **RAGAs**
*   **LLMs**: Google Gemini, OpenAI.
*   **Banco de Dados**: PostgreSQL.
*   **Containerização**: Docker, Docker Compose.

(O restante do documento permanece relevante e inalterado)
