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
      JUDGE[LLM Juiz]
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
      SE[Auto AvaliaÃ§Ã£o]
      COR[CorreÃ§Ã£o]
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
    
    GEN -->|Resposta Gerada| SE
    SE -->|Veredito| JUDGE
    JUDGE -->|Aprovado| A
    JUDGE -->|Reprovado| COR
    COR -->|Resposta Corrigida| A

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
1. **Entrada e Cache**: O usuÃ¡rio envia uma pergunta para `POST /query`. A API verifica o cache Redis; se houver HIT, a resposta Ã© retornada imediatamente.
2. **Triagem (em caso de MISS)**: O agente classifica a intenÃ§Ã£o do usuÃ¡rio. Se a pergunta for clara, prossegue para a resolução; caso contrÃ¡rio, pede esclarecimentos.
3. **ResoluÃ§Ã£o (RAG)**: O sistema busca informaÃ§Ãµes relevantes usando uma rota lexical (busca por palavras-chave) ou vetorial (busca por similaridade semÃ¢ntica no FAISS com reranker).
4. **GeraÃ§Ã£o**: Com o contexto encontrado, o LLM gera uma resposta inicial.
5. **AutoavaliaÃ§Ã£o (LLM-as-a-Judge)**: A resposta gerada Ã© submetida a um "LLM Juiz", que a avalia com base em critÃ©rios de fidelidade e relevÃ¢ncia.
6. **CorreÃ§Ã£o ou Entrega**: Se a resposta for "Aprovada", ela Ã© enviada ao usuÃ¡rio e salva no cache. Se for "Reprovada", um nÃ³ de correÃ§Ã£o tenta reescrevê-la antes de ser enviada.
7. **ETL**: Um processo separado e assÃ­ncrono ingere documentos, os processa e atualiza o Ã­ndice vetorial, invalidando o cache ao final para manter a consistÃªncia.

## Notas de ConfiguraÃ§Ã£o
- **Cache**: defina `REDIS_HOST`, `REDIS_PORT` e opcionalmente `CACHE_TTL_SECONDS` no `.env` para habilitar o armazenamento de respostas.
- **Embeddings**: mantenha ETL e API alinhados via `EMBEDDINGS_MODEL` (suportado pelo pacote `langchain-huggingface`).
- **ConfianÃ§a**: ajuste `CONFIDENCE_MIN` conforme o apetite de risco; com `REQUIRE_CONTEXT=true`, respostas sÃ³ sÃ£o liberadas acima do limiar.
- **SinÃ´nimos/Boosts**: mantenha o `terms.yml` atualizado para aliases, sinÃ´nimos e bÃ´nus por departamento.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` no `/query` para inspecionar fluxo e mÃ©tricas.
- **Compose GPU**: mesmo na pilha GPU, `ai_etl` roda em CPU (`CUDA_VISIBLE_DEVICES=""`) e depende de `sentencepiece==0.2.0`; a GPU fica reservada para a API.
