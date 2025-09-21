# RAG Assist (UFEX) — README

> Sistema de pergunta–resposta (RAG) com **rota lexical**, **rota vetorial** (FAISS + CrossEncoder) e modo **híbrido**. Inclui telemetria JSONL e parâmetros ajustáveis por `.env`.

## Sumário
- [Arquitetura](#arquitetura)
- [Como rodar](#como-rodar)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Como os “pesos” funcionam](#como-os-pesos-funcionam)
- [Telemetria](#telemetria)
- [Dicas e troubleshooting](#dicas-e-troubleshooting)

---

## Arquitetura

```mermaid
flowchart TD
    Q[Usuário pergunta] -->|normaliza| NQ[Normalização & sinais]
    NQ -->|candidatos léxicos| LEX[Matcher de sentenças (fuzzy)]
    NQ --> MQ[Multi-Query]
    MQ -->|q1..qn| FAISS[(FAISS)]
    LEX -. opcional/híbrido .-> MERGE
    FAISS --> MERGE[Merge + Dedup + Cap por fonte]
    MERGE --> RERANK[CrossEncoder (rerank)]
    RERANK --> CTX[Seleção de contexto]
    CTX --> LLM[LLM - resposta final]
    LLM --> OUT[Markdown + Citações]
    OUT --> LOG[telemetry.jsonl]
```

- **LEX**: varre sentenças por _partial ratio_ / nomes aproximados. Usa `LEXICAL_THRESHOLD` e soma `DEPT_BONUS` quando a _source_ bate o departamento.
- **FAISS**: busca vetorial com _multi-query_.
- **Híbrido**: se `HYBRID_ENABLED=true`, une candidatos **lexicais + vetoriais** antes do **CrossEncoder**, com **cap por fonte** (`MAX_PER_SOURCE`).
- **Rerank**: CrossEncoder (ex.: `jinaai/jina-reranker-v2-base-multilingual`) decide a ordem final.
- **LLM**: sintetiza a resposta e formata em Markdown com citações.

---

## Como rodar

1. **ETL**: gere/atualize o índice FAISS (garanta o mesmo modelo de embeddings na API e no ETL).
2. **API**: exporte as variáveis do `.env` e inicie o serviço.
3. Faça uma requisição `POST /query` com `{"question": "...", "debug": true}` para inspecionar `debug`.

> Pré-requisitos: Python 3.10+, `langchain_community`, `sentence_transformers` (opcional, para o rerank), `PyYAML`, `rapidfuzz`.

---

## Variáveis de ambiente

Essenciais:
```
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large
CONFIDENCE_MIN=0.32
STRUCTURED_ANSWER=true
REQUIRE_CONTEXT=true
```

Lexical & híbrido:
```
HYBRID_ENABLED=true           # merge lexical+vetorial antes do rerank
LEXICAL_THRESHOLD=86          # corte para aceitar uma sentença lexical
DEPT_BONUS=8                  # bônus por “source” compatível com depto
MAX_PER_SOURCE=2              # diversidade no merge
```

Rerank:
```
RERANKER_ENABLED=true
RERANKER_NAME=jinaai/jina-reranker-v2-base-multilingual
RERANKER_CANDIDATES=30
RERANKER_TOP_K=5
RERANKER_MAX_LEN=512
RERANKER_DEVICE=cpu
```

Telemetria:
```
LOG_DIR=./logs
DEBUG_LOG=true
DEBUG_PAYLOAD=true
```

---

## Como os “pesos” funcionam

### 1) Peso **lexical**
- Cada sentença candidata recebe um escore `best` (0–100) por:
  - _match_ aproximado de nomes (Levenshtein),
  - `partial_ratio` da pergunta na sentença.
- A sentença só “entra no jogo” se `best >= LEXICAL_THRESHOLD`.
- Se a _source_ do documento aparenta o mesmo **departamento** da pergunta, soma-se `DEPT_BONUS` ao melhor escore do documento.
- Quando o **modo híbrido** está **desligado**, a rota lexical pode responder **sozinha** (retorno antecipado).
- Quando o **modo híbrido** está **ligado**, as passagens lexicais **não retornam sozinhas**: elas são **fundidas** com as vetoriais e seguem para o **reranker**.

### 2) Peso **vetorial + reranker**
- O FAISS traz top-K por similaridade de embeddings (não supervisionado).
- O **CrossEncoder** (supervisionado) reavalia **cada (pergunta, trecho)** e gera um **score 0..1**.  
  Este **score do CrossEncoder** é o “peso” final que decide a ordem e a confiança (`conf = max(score)`).
- Em modo **híbrido**, os trechos **lexicais** também passam pelo CrossEncoder. Assim, palavras/termos que “ajudam” de verdade **ganham peso** no **score supervisionado** do rerank.

---

## Telemetria

Arquivo: `LOG_DIR/queries.log` (JSONL).  
Campos úteis:
- `question`, `route` (`lexical|vector|hybrid`), `confidence` (0..1),  
- `timing_ms` (`retrieval`, `reranker`, `llm`, `total`),  
- `mq_variants`, `faiss_top` (amostra de candidatos), `ctx_docs`.

Basta importar e chamar:
```python
from telemetry import log_event
log_event(os.getenv("LOG_DIR","./logs"), payload_dict)
```

---

## Dicas e troubleshooting

- **Confiança baixa**: ajuste `RERANKER_NAME`, `RERANKER_CANDIDATES` e `CONFIDENCE_MIN`.
- **Muito “mais do mesmo”** nas fontes: diminua `MAX_PER_SOURCE` (ex.: `1`).
- **Lexical muito sensível**: aumente `LEXICAL_THRESHOLD` (ex.: `90`).
- **CrossEncoder pesado**: rode em `cuda` (`RERANKER_DEVICE=cuda`) ou desative (`RERANKER_ENABLED=false`).

---

> Dúvidas? Abra o `debug:true` na requisição para ver os detalhes da rota, candidatos, tempos e _scores_.
