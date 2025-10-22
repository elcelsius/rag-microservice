# Arquitetura & Fluxo do Sistema (RAG + Agente)

```mermaid
flowchart LR
  subgraph Client
    U[Usuário]
  end

  subgraph API
    A[Flask API<br/>/agent/ask]
    L[Endpoint legado<br/>/query]
    H[Health/Metrics]
  end

  subgraph Retrieval
    VS[(FAISS Index)]
    EMB[HF Embeddings]
    MQ[Multi-Query<br/>+ Sinônimos]
    LEX[Busca Lexical<br/>hits por sentença + bônus de depto]
    RER[CrossEncoder<br/>Reranker]
  end

  subgraph LLM
    TRI[LLM Triagem]
    GEN[LLM Geração de Resposta]
  end

  subgraph ETL
    LD[Loaders<br/>pdf, docx, md, txt, code, ...]
    SPL[Chunking]
    EMB_E[HF Embeddings]
    VS_B[FAISS Build/Update]
    DB[(PostgreSQL<br/>hashes/chunks)]
  end

  subgraph Agent
    TG[Triagem]
    AR[Auto Resolver<br/>chama RAG]
    PD[Pedir Info]
  end

  %% Entradas do usuário (LINHAS SEPARADAS!)
  U -->|Pergunta agente| A
  U -->|Pergunta legado| L

  %% Fluxo do agente
  A -->|triagem LangGraph| TRI
  TRI -->|ação| TG
  TG -->|AUTO_RESOLVER| AR
  TG -->|PEDIR_INFO| PD

  %% Rotas de recuperação
  L -->|bypass agente| LEX
  L -->|bypass agente| MQ

  AR -->|Rota 1| LEX
  LEX -->|se encontrou| GEN
  AR -->|Rota 2| MQ --> VS --> RER --> GEN

  %% Respostas
  GEN -->|Resposta + Citações + Confiança| A
  GEN -->|Resposta legado| L

  %% Saúde/telemetria
  H --- A

  %% ETL
  LD --> SPL --> EMB_E --> VS_B --> VS
  VS_B --> DB
  DB -->|incremental| VS_B

  %% Embeddings/índice em runtime
  A --- EMB
  A --- VS
  L --- VS
```

## Passo a passo (resumo)
1. **Entrada**: o usuário envia uma pergunta para `POST /agent/ask` (também exposto como `/api/ask` via proxy da UI). Há ainda o endpoint legado `POST /query` (roteado como `/api/query` na UI) que chama o pipeline direto.
> A interface Nginx (`ai_web_ui`) encaminha chamadas feitas para `/api/...` em `localhost:8080` para os mesmos endpoints expostos pela API Flask em `localhost:5000`.
2. **Triagem/roteamento**: o agente LangGraph decide a ação (responder direto, pedir esclarecimentos ou acionar o RAG).
3. **Rota Lexical**: se houver bons *hits* por sentença (com bônus por departamento), responde diretamente com trechos/citações.
4. **Rota Vetorial**: caso contrário, gera **multi-queries** (sinônimos do `terms.yml`), consulta o **FAISS**, reranqueia com **CrossEncoder** e calcula a **confiança**.
5. **Resgate/Resposta**: o texto final é gerado pelo LLM usando os trechos mais relevantes. `confidence >= CONFIDENCE_MIN` libera resposta “segura” quando `REQUIRE_CONTEXT=true`.
6. **ETL**: arquivos em `data/` são carregados pelos *loaders*, *chunkados*, embedded e indexados no FAISS. Metadados/hashes vão para o PostgreSQL para **atualizações incrementais**.

## Notas de Configuração
- **Embeddings**: defina `EMBEDDINGS_MODEL` no `.env` para alinhar **ETL** e **API**.
- **Confiança**: o padrão atual no README é `CONFIDENCE_MIN=0.32`.
- **Sinônimos/Boosts**: mantenha `terms.yml` para *aliases*, *synonyms* e *boosts*.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` no `/query` para inspeções.
