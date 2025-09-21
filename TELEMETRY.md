# Telemetria

A telemetria registra um JSON por linha em `LOG_DIR/queries.log` com:
- `ts`, `ts_iso`, `question`, `route` (`lexical|vector|hybrid`)
- `mq_variants` (multi-queries geradas), `faiss_top` (candidatos após merge/dedupe)
- `confidence` (máximo do reranker)
- `timing` (`lexical_ms`, `faiss_ms`, `rerank_ms`, `total_ms`)
- `ctx_docs` (quantidade de citações usadas)

## Configuração (.env)
```
LOG_DIR=./logs
HYBRID_ENABLED=true
MAX_PER_SOURCE=2
LEXICAL_THRESHOLD=86
DEPT_BONUS=8
```

## Exemplo de linha
```json
{"ts": 1732200000.0, "question": "onde encontro monitoria?", "route":"hybrid", "confidence":0.71, "timing":{"lexical_ms":7.3,"faiss_ms":12.4,"rerank_ms":55.1,"total_ms":90.4}}
```
