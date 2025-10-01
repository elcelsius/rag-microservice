# Arquitetura & Fluxo do Sistema (RAG + Agente)

```mermaid
flowchart LR
    subgraph Client
      U[UsuÃ¡rio]
    end
    subgraph API
      A[Flask API<br/>/query]
      H[Health/Metrics]
      C[(Redis Cache)]
    end
    subgraph Retrieval
      VS[(FAISS Index)]
      EMB[HF Embeddings]
      MQ[Multi-Query<br/> + SinÃ´nimos]
      LEX["Busca Lexical<br/>(sentenÃ§as + bÃ´nus de depto)"]
      RER[CrossEncoder<br/>(Reranker)]
    end
    subgraph LLM
      TRI[LLM Triagem]
      GEN[LLM GeraÃ§Ã£o de Resposta]
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

    TRI -->|aÃ§Ã£o| TG
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

    %% InvalidaÃ§Ã£o de Cache no ETL
    VS_B --> C_ETL
```

## Passo a passo (resumo)
1. **Entrada e Cache**: o usuÃ¡rio envia uma pergunta para `POST /query`. Antes de consultar o RAG, a API verifica o cache Redis; se houver HIT, a resposta Ã© retornada imediatamente.
2. **Triagem (em caso de MISS)**: se nÃ£o houver cache, o LLM pode classificar a intenÃ§Ã£o (ex.: pedir esclarecimento).
3. **Rota Lexical**: se houver bons *hits* por sentenÃ§a (com bÃ´nus por departamento), responde diretamente com trechos/citaÃ§Ãµes.
4. **Rota Vetorial**: caso contrÃ¡rio, gera *multi-queries* (sinÃ´nimos no `terms.yml`), consulta o FAISS, aplica o CrossEncoder e calcula a confianÃ§a.
5. **Resgate/Resposta**: a resposta final Ã© montada pelo LLM e salva no cache Redis antes de ser retornada.
6. **ETL**: arquivos em `data/` sÃ£o processados, indexados no FAISS e, ao final, o cache Redis Ã© invalidado para evitar respostas desatualizadas.

## Notas de ConfiguraÃ§Ã£o
- **Cache**: defina `REDIS_HOST`, `REDIS_PORT` e opcionalmente `CACHE_TTL_SECONDS` no `.env` para habilitar o armazenamento de respostas.
- **Embeddings**: mantenha ETL e API alinhados via `EMBEDDINGS_MODEL` (suportado pelo pacote `langchain-huggingface`).
- **ConfianÃ§a**: ajuste `CONFIDENCE_MIN` conforme o apetite de risco; com `REQUIRE_CONTEXT=true`, respostas sÃ³ sÃ£o liberadas acima do limiar.
- **SinÃ´nimos/Boosts**: mantenha o `terms.yml` atualizado para aliases, sinÃ´nimos e bÃ´nus por departamento.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` no `/query` para inspecionar fluxo e mÃ©tricas.
- **Compose GPU**: mesmo na pilha GPU, `ai_etl` roda em CPU (`CUDA_VISIBLE_DEVICES=""`) e depende de `sentencepiece==0.2.0`; a GPU fica reservada para a API.
