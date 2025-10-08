# Monitoramento de métricas

## Endpoints úteis
- `/metrics`: counters JSON.
- `/healthz`: status da API (FAISS/LLM prontos).

## Consultas HTTP
```bash
curl -s http://localhost:5000/metrics | jq .
curl -s http://localhost:5000/metrics | jq '.counters.cache_hits_total'
```

## Recomendações
- Observar `cache_hits_total` vs `cache_misses_total` após smoke/tests.
- `queries_low_confidence_total` indica respostas com pouca confiança no RAG direto; `agent_low_confidence_total` aponta interações do agente que terminaram abaixo do limiar.
- `agent_refine_*` deve crescer em conjunto; um número alto em `agent_refine_exhausted_total` sinaliza refinamentos sem sucesso.
- Inspecionar `meta.confidence_threshold` e `low_confidence` nos logs para entender qual limiar estava ativo e se o alerta disparou.
```

