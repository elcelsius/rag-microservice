# RAG Microservice â€” README (atualizado)

Este projeto implementa um microserviÃ§o **RAG** (Retrieval-Augmented Generation) com cache de respostas (Redis), duas rotas de recuperaÃ§Ã£o (lexical e vetorial), **reranker** via CrossEncoder e orquestraÃ§Ã£o opcional por **Agente** (LangGraph). Aqui vocÃª encontra **como rodar**, **como depurar**, **como avaliar a qualidade**, e um **diagrama** do fluxo.

> Se preferir um documento dedicado de arquitetura, veja **ARCHITECTURE.md** (inclui o mesmo diagrama e explicaÃ§Ãµes detalhadas).

---

## SumÃ¡rio
- [Arquitetura (resumo + diagrama)](#arquitetura-resumo--diagrama)
- [Requisitos & Setup](#requisitos--setup)
- [ConfiguraÃ§Ã£o (.env)](#configuraÃ§Ã£o-env)
- [Executando o ETL (build do Ã­ndice)](#executando-o-etl-build-do-Ã­ndice)
- [Subindo a API](#subindo-a-api)
- [Testes rÃ¡pidos (Smoke)](#testes-rÃ¡pidos-smoke)
- [Consultas & Debug](#consultas--debug)
- [AvaliaÃ§Ã£o de Qualidade (RAGAs)](#avaliaÃ§Ã£o-de-qualidade-ragas)
- [Como funcionam os "pesos" e a confianÃ§a](#como-funcionam-os-pesos-e-a-confianÃ§a)
- [Boas prÃ¡ticas & Troubleshooting](#boas-prÃ¡ticas--troubleshooting)

---

## Arquitetura (resumo + diagrama)

**Fluxo alto nÃ­vel:**
1. O usuÃ¡rio chama `POST /query` com sua pergunta.
2. A API verifica o cache **Redis** para uma resposta existente. Se houver HIT, retorna imediatamente.
3. Se nÃ£o houver cache (MISS), o **Agente LangGraph** orquestra o fluxo, iniciando pela triagem.
4. A **Triagem** decide se a pergunta pode ser resolvida diretamente ou se precisa de esclarecimentos.
5. A **ResoluÃ§Ã£o (RAG)** busca o contexto via rota lexical ou vetorial (com reranker).
6. O **LLM** gera uma resposta com base no contexto.
7. A resposta passa por **AutoavaliaÃ§Ã£o (LLM-as-a-Judge)**. Se reprovada, uma etapa de **CorreÃ§Ã£o** Ã© acionada.
8. A resposta final (aprovada ou corrigida) Ã© retornada e salva no cache.

> Diagrama detalhado: veja tambÃ©m **ARCHITECTURE.md** (inclui invalidaÃ§Ã£o de cache no ETL).

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

---

## Requisitos & Setup

- Python 3.10+ (recomendado)
- Docker e Docker Compose
- Redis (para cache de respostas)
- FAISS (via LangChain/FAISS) + embeddings HuggingFace
- Chaves de LLM (Gemini/OpenAI) se for usar triagem/geraÃ§Ã£o

> O `requirements.txt` jÃ¡ inclui `langchain-huggingface` (LangChain 0.2+) e fixa `sentencepiece==0.2.0` para evitar o crash `free(): double free` em ambientes CUDA.

Instale dependÃªncias (exemplo):
```bash
pip install -r requirements.txt
```

---

## ConfiguraÃ§Ã£o (.env)

Crie um `.env` a partir de `.env.example`. Principais chaves:

```env
# Modelos
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large
CROSS_ENCODER=jinaai/jina-reranker-v2-base-multilingual

# Limiar de resposta segura
CONFIDENCE_MIN=0.32
REQUIRE_CONTEXT=true

# ExecuÃ§Ã£o
ROUTE_FORCE=auto   # auto | vector
TOP_K=6
PER_QUERY=4

# Provedores LLM (opcional)
OPENAI_API_KEY=...
GEMINI_API_KEY=...

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_SECONDS=86400
```

> O ETL e a API leem `EMBEDDINGS_MODEL` do mesmo `.env`, garantindo que o Ã­ndice e o runtime usem embeddings compatÃ­veis.

---

## Executando o ETL (build do Ã­ndice)

1. Coloque seus arquivos em `./data/` (pdf, docx, md, txt, csv, json, cÃ³digo, etc.).
2. Rode o build completo:
   ```bash
   python etl_orchestrator.py --rebuild
   ```
   ou use `python etl_build_index.py` para o fluxo simplificado.
3. O Ã­ndice fica em `./vector_store/faiss_index`.
4. Para uma atualizaÃ§Ã£o incremental:
   ```bash
   python etl_orchestrator.py update
   ```
5. Ao final de um rebuild/update, o orquestrador limpa o cache Redis (`rag_query:*`) para evitar dados defasados.

> No compose GPU (`docker-compose.gpu.yml`), o serviÃ§o `ai_etl` roda em CPU (`CUDA_VISIBLE_DEVICES=""`). Execute `docker compose -f docker-compose.gpu.yml run --rm ai_etl python3 -u scripts/etl_build_index.py` para reconstruir dentro do container.

---

## Subindo a API

Via Python (desenvolvimento):
```bash
flask --app api run --host 0.0.0.0 --port 5000
```

Via Docker Compose:
```bash
docker compose -f docker-compose.cpu.yml up --build
```

Endpoints Ãºteis:
- `POST /query` â€” consulta RAG + cache
- `GET /healthz` â€” health/readiness (inclui FAISS/LLM/Redis)
- `GET /metrics` â€” contadores simples (queries, cache hits, erros)
- `GET /debug/dict` e `GET /debug/env` â€” checagens rÃ¡pidas de dicionÃ¡rios e variÃ¡veis

---

## Testes rÃ¡pidos (Smoke)

- `./scripts/smoke_cpu.sh` â€” valida o stack CPU.
- `./scripts/smoke_gpu.sh` â€” valida o stack GPU (adicione `--with-etl` para regerar o Ã­ndice antes dos testes).

Ambos verificam healthz, fazem consultas com `debug=true` e destacam problemas comuns em FAISS, reranker ou GPU.

---

## Consultas & Debug

Exemplo de chamada com debug:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"question":"onde encontro informaÃ§Ã£o de monitoria de computaÃ§Ã£o?","debug":true}' \
  http://localhost:5000/query | jq
```

Campos Ãºteis no `debug`:
- `route`: `"lexical"` ou `"vector"`
- `mq_variants`: queries geradas com sinÃ´nimos do `terms.yml`
- `faiss.candidates[*].score`: similaridade vinda do FAISS (quando disponÃ­vel)
- `rerank.enabled`: indica se o CrossEncoder foi carregado
- `rerank.scored[*].score`: score **0â€“1** do CrossEncoder (ordenaÃ§Ã£o final)
- `confidence`: mÃ¡ximo dos scores reranqueados (apÃ³s normalizaÃ§Ã£o, se houver)

Respostas vindas do cache trazem o campo `"X-Cache-Status": "hit"` no corpo JSON.

---

## AvaliaÃ§Ã£o de Qualidade (RAGAs)

O projeto inclui um pipeline de avaliaÃ§Ã£o automatizada com `ragas`. Prepare `evaluation_dataset.jsonl` e execute:

```bash
python tools/evaluate.py
```

O relatÃ³rio retorna mÃ©tricas de `faithfulness`, `answer_relevancy`, `context_recall` e `context_precision` para acompanhar a qualidade do RAG.

---

## Como funcionam os "pesos" e a confianÃ§a

### Rota Lexical
- ExtraÃ­mos **termos candidatos** (palavras alfanumÃ©ricas â‰¥3, e-mails; stopwords sÃ£o ignoradas).
- Procuramos **sentenÃ§as** que batem forte (fuzzy/regex). Cada acerto soma pontos para o documento.
- Se a **fonte** do documento condiz com um **departamento** citado (via `terms.yml`), aplicamos um **bÃ´nus** (ex.: `+8`).
- Havendo hits suficientes, a resposta sai **sem** reranker (scores podem aparecer como `0.0` por design).

### Rota Vetorial (FAISS + Reranker)
- Geramos *multi-queries* com sinÃ´nimos/aliases (`terms.yml`) para ampliar a cobertura.
- Recuperamos candidatos no FAISS e registramos seus `score`s (quando disponÃ­veis).
- Aplicamos CrossEncoder (0â€“1) para ordenar por relevÃ¢ncia contextual.
- **ConfianÃ§a (`confidence`)** = **mÃ¡ximo** dos scores do reranker. Se `confidence >= CONFIDENCE_MIN` **e** `REQUIRE_CONTEXT=true`, respondemos como â€œcontexto suficienteâ€.

> Ajuste `CONFIDENCE_MIN` para respostas mais **conservadoras** (maior) ou mais **falantes** (menor).

---

## Boas prÃ¡ticas & Troubleshooting

- **Cache de Respostas**: Redis reduz latÃªncia e custo em perguntas recorrentes; certifique-se de invalidar apÃ³s ETL.
- **Alinhar Embeddings**: ETL e API devem usar o mesmo `EMBEDDINGS_MODEL`.
- **Tamanho dos chunks**: ajuste chunk size/overlap para equilibrar recall x precisÃ£o.
- **SinÃ´nimos atualizados**: mantenha `terms.yml` com aliases relevantes; remova termos ambÃ­guos.
- **Observabilidade**: use `/healthz`, `/metrics` e `debug=true` para diagnÃ³sticos rÃ¡pidos.
- **Rota forÃ§ada**: `ROUTE_FORCE=vector` ajuda a depurar FAISS/reranker sem interferÃªncia lexical.
- **Qualidade dos dados**: normalize fontes, remova duplicatas e preencha metadados; isso melhora o rerank.
- **Compose GPU**: no stack GPU, `ai_etl` roda em CPU (`sentencepiece==0.2.0`). Se ver `free(): double free`, refaÃ§a build com `--no-cache`.

---

## LicenÃ§a
MIT (ou a de sua preferÃªncia).
