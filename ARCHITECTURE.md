# Arquitetura & Fluxo do Sistema (RAG + Agente)

```mermaid

flowchart LR
    subgraph Client
      U[Usuário]
    end
    subgraph API
      A["Flask API\n/query"]
      H[Health/Metrics]
    end
    subgraph Retrieval
      VS[(FAISS Index)]
      EMB["HF Embeddings"]
      MQ["Multi-Query\n+ Sinônimos"]
      LEX["Busca Lexical\n(sentenças + bonus depto)"]
      RER["CrossEncoder\nReranker"]
    end
    subgraph LLM
      TRI["LLM Triagem"]
      GEN["LLM Geração de Resposta"]
    end
    subgraph ETL
      LD["Loaders\n(pdf, docx, md, txt, code, ...)"]
      SPL["Chunking"]
      EMB_E["HF Embeddings"]
      VS_B["FAISS Build/Update"]
      DB["PostgreSQL\nhashes/chunks"]
    end
    subgraph Agent
      TG["Triagem"]
      AR["Auto Resolver\nchama RAG"]
      PD["Pedir Info"]
    end

    U -->|Pergunta| A
    A -->|triagem opcional| TRI
    TRI -->|ação| TG
    TG -->|AUTO_RESOLVER| AR
    TG -->|PEDIR_INFO| PD

    AR -->|Rota 1| LEX
    LEX -->|se encontrou| GEN
    AR -->|Rota 2| MQ --> VS --> RER --> GEN
    GEN -->|Resposta + Citações + Confiança| A

    H --- A

    %% ETL
    LD --> SPL --> EMB_E --> VS_B --> VS
    VS_B --> DB
    DB -->|incremental| VS_B

    %% Embeddings em runtime
    A --- EMB
    A --- VS

```

## Passo a passo (resumo)
1. **Entrada**: o usuário envia uma pergunta para `POST /query`.
2. **Triagem opcional**: o LLM pode classificar a intenção (ex.: pedir esclarecimento).
3. **Rota Lexical**: se houver bons *hits* por sentença (com bônus por departamento), responde diretamente com trechos/citações.
4. **Rota Vetorial**: caso contrário, gera **multi-queries** (sinônimos do `terms.yml`), consulta o **FAISS**, reranqueia com **CrossEncoder** e calcula a **confiança**.
5. **Resgate/Resposta**: o texto final é gerado pelo LLM usando os trechos mais relevantes. `confidence >= CONFIDENCE_MIN` libera resposta “segura” quando `REQUIRE_CONTEXT=true`.
6. **ETL**: arquivos em `data/` são carregados pelos *loaders*, *chunkados*, embedded e indexados no FAISS. Metadados/hashes vão para o PostgreSQL para **atualizações incrementais**.

## Notas de Configuração
- **Embeddings**: defina `EMBEDDINGS_MODEL` no `.env` para alinhar **ETL** e **API**. A implementação usa `langchain_huggingface.HuggingFaceEmbeddings`, conforme a recomendação do LangChain 0.2+.
- **Confiança**: o padrão atual no README é `CONFIDENCE_MIN=0.32`.
- **Sinônimos/Boosts**: mantenha `terms.yml` para *aliases*, *synonyms* e *boosts*.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` no `/query` para inspeções.
- **Compose GPU**: mesmo na pilha GPU, `ai_etl` roda com `CUDA_VISIBLE_DEVICES=""` (CPU) e depende de `sentencepiece==0.2.0` para evitar o crash `free(): double free`; a GPU fica reservada para a API.
