# Plano de Melhorias - RAG Microservice (snapshot 2025-09-24)

Documento vivo com prioridades, tarefas acionáveis, critérios de aceite e comandos. Foco imediato: **ligar as métricas do RAGAs**, **cache Redis**, e **auto-refine no agente**.

## 0) Foto atual do projeto (confirmado)

- Endpoints: **`/query`** (RAG, também exposto como `/api/query`), **`/agent/ask`** (LangGraph, alias `/api/ask`), **`/metrics`** (uptime + contadores básicos).
- ETL: loaders multi-formato, chunking `RecursiveCharacterTextSplitter`, embeddings HF, **FAISS** persistido.
- Recuperação: híbrida (lexical + vetorial), **multi-query opcional**, **reranker configurável**.
- Avaliação: `eval_rag.py` implementa fluxo e relata retrieval; **RAGAs presente no código**, porém relatórios recentes mostram `ragas_available: false` (faltam deps/chave).
- Docker: compose CPU/GPU, Web UI; `.env.example` bem preenchido.

---

## 1) Prioridades (ordem recomendada)

1. **RAGAs "valendo"** (métricas de geração preenchidas no relatório).
2. **Cache Redis** para respostas (queda de latência/custo + métricas de hit/miss).
3. **Agente: Auto-refine/rewrites** quando confiança baixa, com telemetria por iteração.
4. **Refinos RAG**: chunking, fusão RRF, metadados ricos.
5. **Observabilidade**: Prometheus/Grafana e logs estruturados ampliados.
6. **Portas de qualidade** (CI) e testes.

---

## 2) Avaliação contínua com RAGAs

### 2.1 O que fazer
- **Instalar dependências** (já presentes em `requirements*.txt`):
  ```
  ragas>=0.1.9
  datasets>=2.20.0
  langchain-google-genai>=1.0.5,<2.0.0
  google-generativeai>=0.7.2,<0.8
  langchain-huggingface==0.0.3
  sentence-transformers>=2.6.0,<2.7
  sentencepiece==0.2.0
  ```
- **Secret**: `GOOGLE_API_KEY` no `.env` e exportado no ambiente de execução do `eval_rag.py`.
- **Executar** (exemplo local):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  export GOOGLE_API_KEY="SUA_CHAVE"
  python eval_rag.py --dataset evaluation_dataset.jsonl     --agent-endpoint  http://localhost:5000/agent/ask     --legacy-endpoint http://localhost:5000/query     --out reports/
  ```
- O script já alerta quando `GOOGLE_API_KEY` não está configurado, evitando erros silenciosos nas métricas do RAGAs.

### 2.2 Critérios de aceite
- Relatórios em `reports/*.json` com:
  - `ragas_available: true`
  - `generation_metrics_ragas` preenchido (p.ex.: `faithfulness`, `context_precision`).
- Comando `python eval_rag.py ...` retorna **exit code 0**.
- Testes locais: `python -m pytest tests/test_eval_rag.py` garantem mensagens claras quando a execução do RAGAs não é possível (deps ou `GOOGLE_API_KEY`).

### 2.3 CI (gate rápido)
```yaml
# .github/workflows/eval.yml (resumo)
name: eval
on: [pull_request]
jobs:
  rag-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python eval_rag.py --dataset evaluation_dataset.jsonl --out reports/
        env:
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      - run: python tools/ci_assert_eval.py  # fail se métrica cair > tolerância
```
- ✅ Workflow `.github/workflows/eval.yml` criado (roda `eval_rag.py`, faz upload do relatório). Configure `secrets.GOOGLE_API_KEY` no repositório antes da primeira execução.

---

## 3) Cache de respostas com Redis

### 3.1 Compose e vars
```yaml
# docker-compose.cpu.yml (trecho)
services:
  ai_api:
    env_file: .env
    environment:
      - REDIS_URL=redis://ai_redis:6379/0
    depends_on: [ai_redis]
  ai_redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: ["redis-server", "--save", "", "--appendonly", "no"]
```

`.env.example`:
```
REDIS_URL=redis://ai_redis:6379/0
CACHE_TTL_SECONDS=43200  # 12h
```

### 3.2 Integração (esboço)
- Ponto: `answer_question()` (caminho do endpoint `/query`) e handler do `/agent/ask`.
- **Chave** = hash de: pergunta normalizada + top-k de chunks + versão de prompt + modelo LLM + flags de pipeline.
- **TTL**: `CACHE_TTL_SECONDS` (12h sugerido).
- **Métricas**: `cache_hits_total`, `cache_misses_total` em `/metrics`.

_Pseudocódigo_:
```python
import hashlib, json, redis
R = redis.from_url(os.environ["REDIS_URL"])

def cache_key(payload: dict) -> str:
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "resp:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()

def answer_question(..., debug=False):
    payload = {
      "q": normalize(question),
      "top_k": TOP_K,
      "llm": GOOGLE_MODEL,
      "preset": RERANKER_PRESET,
      "require_ctx": REQUIRE_CONTEXT,
      "prompt_v": PROMPT_VERSION,
    }
    key = cache_key(payload)
    if val := R.get(key):
        METRICS["cache_hits_total"] += 1
        return json.loads(val)
    METRICS["cache_misses_total"] += 1
    result = _slow_pipeline(...)
    R.setex(key, int(os.getenv("CACHE_TTL_SECONDS", "43200")), json.dumps(result, ensure_ascii=False))
    return result
```

### 3.3 Invalidação pós-ETL
- Ao final do `etl_orchestrator.py` bem-sucedido: **limpar prefixo** `resp:*` (ou incrementar **versão de índice** e compor no cache key).
- Alternativa mais segura: manter `INDEX_VERSION` em `.env` e **incluir no cache key**.

### 3.4 Critérios de aceite
- Latência média cai significativamente em re-queries idênticas.
- `/metrics` expõe acertos/erros do cache.
- ETL dispara invalidação/versão → sem respostas stale.

---

## 4) Agente LangGraph: auto-refine

### 4.1 Fluxo
- Após `auto_resolver`:
  1) **Self-check** (light): classificador/heurística de confiança (score interno + sinais do reranker).
  2) Se **baixo**: nó **`auto_refine`** gera 1-2 reformulações (query rewriting), refaz recuperação e tenta nova resposta.
  3) Se ainda insuficiente: seguir para **`pedir_info`**.

### 4.2 Telemetria do grafo
- Logar `rid`, nós percorridos, contagem de rewrites, tempo por nó, confiança final.
- Contadores: `agent_refine_attempts_total`, `agent_asked_for_more_info_total`.

### 4.3 Critérios de aceite
- Aumento de `faithfulness` e/ou `answer_relevancy` (RAGAs) em subset do dataset.
- Limite de iterações configurável (ex.: máx 2 rewrites).

> **Status:** Implementado (`meta.refine_history`, métricas `agent_refine_*`, testes `tests/test_agent_workflow.py`).

---

## 5) Refino do pipeline RAG

### 5.1 Chunking
- Testar grades: `chunk_size` {800, 1000, 1200, 1500} × `overlap` {100, 200}.
- Considerar **chunking semântico** (por headings/parágrafos) quando houver estrutura.

### 5.2 Fusão Híbrida (RRF)
- Em vez de fallback, aplicar **Reciprocal Rank Fusion** entre listas lexical e vetorial.
- Parâmetro `RRF_K` (ex.: 60) e mistura com pesos para priorizar precisão nas top posições.

### 5.3 Metadados & filtros
- Enriquecer metadados no ETL (autor, data, tipo). Usar como **boost**/filtro no retrieval e exposição ao usuário nas fontes.

### 5.4 Reranker
- Avaliar presets (`off|fast|balanced|full`) com RAGAs.
- Ajustar `RERANKER_CANDIDATES` e `RERANKER_TOP_K` visando qualidade vs latência.
- Usar GPU se disponível (`RERANKER_DEVICE=cuda`).

---

## 6) Observabilidade

### 6.1 `/metrics` → Prometheus
Expor em ambiente e criar dashboard de:
- `queries_total`, `queries_answered`, `queries_ambiguous`, `queries_not_found`, `errors_internal`
- `cache_hits_total`, `cache_misses_total`
- latência p50/p95/p99 (log/metrics)

_Exemplo de scrape:_
```yaml
scrape_configs:
  - job_name: 'rag_api'
    metrics_path: /metrics
    static_configs:
      - targets: ['ai_api:5000']
```

### 6.2 Logs estruturados
- Garantir JSON por request com `rid`, `status`, `took_ms`, rota, decisão do agente.
- Forward para ELK/Grafana Loki (futuro).

---

## 7) Portas de qualidade

- **`make eval`**: alvo que roda `eval_rag.py` e grava em `reports/`.
- **CI gate**: reprova PR se `faithfulness` cair > X% no golden set.
- **Testes**: manter `pytest -k api` e adicionar smoke do `/agent/ask`.

### 7.1 Validação local (workflow adotado)
Como o projeto é mantido localmente, seguimos um “gate” manual antes de publicar novas alterações:

1. **Preparar ambiente**  
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # PowerShell / Windows
   pip install -r requirements.txt
   ```
2. **Rodar avaliação RAGAs** (gera relatório em `reports/`)
   ```bash
   set GOOGLE_API_KEY=SEU_TOKEN
   python eval_rag.py ^
       --dataset evaluation_dataset.jsonl ^
       --agent-endpoint http://localhost:5000/agent/ask ^
       --legacy-endpoint http://localhost:5000/query ^
       --out reports/
   ```
   > O relatório deve sair com `ragas_available: true`. Guarde o JSON na pasta `reports/`.
3. **Executar smoke tests**  
   ```bash
   python scripts/smoke_api.py http://localhost:8080/api
   python scripts/smoke_api.py http://localhost:5000/api
   ```
4. **Checagem funcional de contatos**  
   ```bash
   curl -fsS http://localhost:8080/agent/ask ^
     -H "Content-Type: application/json" ^
     -d "{\"question\":\"Qual o telefone da Andreia da computacao?\",\"messages\":[],\"debug\":true}" ^
     | python -m json.tool
   ```
   > A resposta deve vir pela rota `contact_fallback`, exibindo o telefone específico.

Esses passos funcionam como “gate” manual antes de criar novas imagens Docker ou publicar documentação.

---

## 8) Checklist de implementação

- ✅ (RAGAs) Adicionar deps e chave; relatório com `ragas_available: true`.
- ⬜ (CI) Workflow mínimo executando `eval_rag.py` em subset. _(workflow disponível como `.github/workflows/eval.yml`; falta configurar o secret e validar o primeiro run)._
- ✅ (Redis) Serviço no compose + `REDIS_URL` e `CACHE_TTL_SECONDS` no `.env`.
- ✅ (Cache) Implementar no `/query` e `/agent/ask`; métricas de hit/miss.
- ✅ (ETL) Invalidação/versão do cache ao fim do ETL.
- ✅ (Agente) Nó `auto_refine` + telemetria; limite de iteração.
- ⬜ (RRF) Fusão de ranqueamento híbrido com `RRF_K`.
- ⬜ (Chunking) Experimentação com grade e, se possível, chunking semântico.
- ⬜ (Observabilidade) Exportar métricas para Prometheus; logs estruturados.
- ⬜ (Docs) Atualizar `README` com "Avaliação", "Cache" e "Observabilidade".
- ✅ (Smokes) Integrar `scripts/smoke_api.py` ao pipeline de CI (futuro).
- ⬜ (Manual) Executar checklist de validação local antes de publicar imagens/docs.

---

## 9) Apêndice - variáveis úteis

```
GOOGLE_API_KEY=...
EMBEDDINGS_MODEL=intfloat/multilingual-e5-large
RERANKER_PRESET=balanced   # off|fast|balanced|full
REQUIRE_CONTEXT=true
CONFIDENCE_MIN=0.6
TOP_K=5
REDIS_URL=redis://ai_redis:6379/0
CACHE_TTL_SECONDS=43200
INDEX_VERSION=1
```

---

### Notas finais
- Priorize **RAGAs** primeiro para ter um medidor objetivo de qualidade.
- Em produção, ajuste **TTL** do cache e **CONFIDENCE_MIN** conforme métricas e feedback.
