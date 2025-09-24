# RAG Microservice — README (atualizado)

Este projeto implementa um microserviço **RAG** (Retrieval-Augmented Generation) com cache de respostas (Redis), duas rotas de recuperação (lexical e vetorial), **reranker** via CrossEncoder e orquestração opcional por **Agente** (LangGraph). Aqui você encontra **como rodar**, **como depurar**, **como avaliar a qualidade**, e um **diagrama** do fluxo.

> Se preferir um documento dedicado de arquitetura, veja **ARCHITECTURE.md** (inclui o mesmo diagrama e explicações detalhadas).

---

## Sumário
- [Arquitetura (resumo + diagrama)](#arquitetura-resumo--diagrama)
- [Requisitos & Setup](#requisitos--setup)
- [Configuração (.env)](#configuração-env)
- [Executando o ETL (build do índice)](#executando-o-etl-build-do-índice)
- [Subindo a API](#subindo-a-api)
- [Consultas & Debug](#consultas--debug)
- [Avaliação de Qualidade (RAGAs)](#avaliação-de-qualidade-ragas)
- [Como funcionam os "pesos" e a confiança](#como-funcionam-os-pesos-e-a-confiança)
- [Boas práticas & Troubleshooting](#boas-práticas--troubleshooting)

---

## Arquitetura (resumo + diagrama)

**Fluxo alto nível:**
1. O usuário chama `POST /query` com sua pergunta.
2. A API verifica o cache **Redis** para uma resposta existente. Se encontrada (HIT), retorna imediatamente.
3. Se não houver cache (MISS), a API (opcionalmente) faz triagem de intenção.
4. **Rota Lexical** (prioritária): busca por sentenças que batem com termos da pergunta (com **bônus** se a fonte pertencer a um **departamento** citado). Se for suficiente, responde.
5. **Rota Vetorial** (fallback/forçada): gera **multi-queries**, consulta **FAISS**, **reranqueia** com **CrossEncoder** e calcula uma **confiança**. Se for suficiente, responde.
6. O **Agente** (LangGraph) pode orquestrar o fluxo, tentando **AUTO_RESOLVER** ou **PEDIR_INFO**.
7. A resposta final é salva no cache **Redis** antes de ser retornada ao usuário.

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
    end

    U -->|Pergunta| A
    A -->|1. Cache?| C
    C -->|HIT| A
    C -->|MISS| TRI
    A -->|triagem opcional| TRI

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
```

---

## Requisitos & Setup

- Python 3.10+ (recomendado)
- Docker e Docker Compose
- Redis
- FAISS (via LangChain/FAISS) + HuggingFace Embeddings
- Chaves de LLM (Gemini/OpenAI) se for usar geração/triagem

Instale dependências (exemplo):
```bash
pip install -r requirements.txt
```

---

## Configuração (.env)

Crie um `.env` baseado em `.env.example`. Principais chaves:

```env
# Modelos
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large

# ... (outras configurações de RAG)

# Provedores LLM (opcional)
GEMINI_API_KEY=...

# Cache (opcional, mas recomendado)
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL_SECONDS=86400
```

---

## Executando o ETL (build do índice)

O ETL agora também limpa o cache do Redis para garantir que as respostas não fiquem desatualizadas após uma atualização da base de conhecimento.

---

## Subindo a API

Via Docker Compose (recomendado):
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

Endpoints úteis:
- `POST /query` — consulta RAG (com cache)
- `GET /healthz` — health/readiness (inclui status do Redis)
- `GET /metrics` — contadores simples (inclui hits de cache)

---

## Consultas & Debug

Uma resposta vinda do cache incluirá o cabeçalho `"X-Cache-Status": "hit"`.

---

## Avaliação de Qualidade (RAGAs)

O projeto inclui um framework de avaliação automatizado usando a biblioteca `ragas`. Para usá-lo, prepare o arquivo `evaluation_dataset.jsonl` e execute:

```bash
python tools/evaluate.py
```

O script imprimirá um relatório com métricas de `faithfulness`, `answer_relevancy`, `context_recall`, e `context_precision`.

---

## Boas práticas & Troubleshooting

- **Cache de Respostas**: O sistema utiliza Redis para cachear respostas de perguntas frequentes, reduzindo drasticamente a latência e o custo em consultas repetidas.
- **Alinhar Embeddings**: garanta que ETL **e** API usem o **mesmo** `EMBEDDINGS_MODEL`.
- **Observabilidade**: verifique `healthz`, counters/metrics e use `debug=true`.

---

## Licença
MIT (ou a de sua preferência).