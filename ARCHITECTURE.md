# Arquitetura & Fluxo do Sistema (RAG + Agente)

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
      LEX[Busca Lexical<br/> (hits por sentença + bônus de depto)]
      RER[CrossEncoder<br/> (Reranker)]
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
    subgraph Agent
      TG[Triagem]
      AR[Auto Resolver<br/>(chama RAG)]
      PD[Pedir Info]
    end

    U -->|Pergunta| A
    A -->|1. Cache?| C
    C -->|HIT| A
    C -->|MISS| TRI

    TRI -->|ação| TG
    TG -->|AUTO_RESOLVER| AR
    TG -->|PEDIR_INFO| PD

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
    
    %% Invalidação de Cache no ETL
    VS_B --> C_ETL
```

## Passo a passo (resumo)
1. **Entrada e Cache**: o usuário envia uma pergunta para `POST /query`. O sistema primeiro verifica no cache **Redis**. Se a resposta for encontrada (HIT), ela é retornada imediatamente.
2. **Triagem (em caso de MISS)**: se não houver cache, o LLM pode classificar a intenção (ex.: pedir esclarecimento).
3. **Rota Lexical**: se houver bons *hits* por sentença (com bônus por departamento), responde diretamente com trechos/citações.
4. **Rota Vetorial**: caso contrário, gera **multi-queries** (sinônimos do `terms.yml`), consulta o **FAISS**, reranqueia com **CrossEncoder** e calcula a **confiança**.
5. **Resgate/Resposta**: o texto final é gerado pelo LLM usando os trechos mais relevantes. A resposta é **salva no cache Redis** antes de ser retornada.
6. **ETL**: arquivos em `data/` são carregados, processados e indexados no FAISS. Ao final do processo, o **cache Redis é invalidado** para garantir que os dados não fiquem desatualizados.

## Notas de Configuração
- **Cache**: configure as variáveis `REDIS_HOST` e `REDIS_PORT` no `.env`.
- **Embeddings**: defina `EMBEDDINGS_MODEL` no `.env` para alinhar **ETL** e **API**.
- **Confiança**: o padrão atual no README é `CONFIDENCE_MIN=0.32`.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` no `/query` para inspeções.
