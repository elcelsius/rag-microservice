# 🧠 rag-microservice — README Completo (CPU + GPU)

Microserviço de **RAG (Retrieval-Augmented Generation)** com **Flask**, **FAISS**, **sentence-transformers** e UI estática via **nginx**. Suporta **CPU** e **GPU (CUDA)**. Inclui **reranker** opcional (CrossEncoder) e health-checks prontos para front-end.

---

## 📦 Visão geral da stack

- **Python 3.11**
- **Flask** — API (`/query`, `/healthz`)
- **LangChain (community + text-splitters)** — split de textos
- **FAISS** — índice vetorial persistido em `/app/vector_store/faiss_index`
- **Embeddings** — `intfloat/multilingual-e5-large` (Hugging Face)
- **Reranker opcional** — `jinaai/jina-reranker-v2-base-multilingual` (CrossEncoder)
- **nginx** — serve UI (8080) e proxy para API (`/api/*` → 5000)

Containers (CPU): `ai_etl`, `ai_projeto_api`, `ai_web_ui`, `ai_postgres` (opcional).

---

## 🗂 Estrutura de diretórios (essencial)

```
rag-microservice/
├─ config/
│  └─ ontology/terms.yml        # dicionário/ontologia de termos
├─ data/                        # base de documentos (TXT/MD/PDF/DOCX), com subpastas
├─ loaders/                     # leitores personalizados por tipo (code/docx/md/pdf/txt)
├─ prompts/                     # moldam o fluxo de resposta
│  ├─ pedir_info_prompt.txt
│  ├─ resposta_final_prompt.txt
│  └─ triagem_prompt.txt
├─ scripts/
│  ├─ etl_build_index.py        # ETL que gera índice FAISS
│  ├─ inicia_site_cpu.sh        # sobe tudo (CPU) + abre UI
│  ├─ inicia_site_gpu.sh        # sobe tudo (GPU) + abre UI
│  ├─ treinar_ia_cpu.sh         # executa ETL (CPU)
│  └─ treinar_ia_gpu.sh         # executa ETL (GPU)
├─ web_ui/
│  ├─ html/index.html           # UI: usa /api/query e /api/healthz
│  └─ conf.d/default.conf       # nginx: /api/* → ai_projeto_api:5000
├─ api.py                       # Flask app (endpoints)
├─ query_handler.py             # RAG + reranker + debug/telemetria
└─ docker-compose.*.yml         # orquestração CPU/GPU
```

**Papel das pastas-chave**  
- `config/ontology/terms.yml`: dicionário/ontologia de termos usados para triagem ou normalização de entidades/consultas.  
- `data/`: fontes de conhecimento; pode ter **subpastas**. O ETL lê recursivamente.  
- `loaders/`: leitores por tipo — cada `*_loader.py` implementa extração de texto para seu formato.  
- `prompts/`: textos dos prompts do pipeline (`triagem`, `pedir_info`, `resposta_final`).  
- `scripts/`: automações para iniciar serviços e executar ETL.

---

## 🔧 Variáveis de ambiente (API)

No serviço `ai_projeto_api`:

- `FAISS_STORE_DIR=/app/vector_store/faiss_index`
- `EMBEDDINGS_MODEL=intfloat/multilingual-e5-large`
- `RERANKER_ENABLED=true|false`
- `RERANKER_NAME=jinaai/jina-reranker-v2-base-multilingual`
- `RERANKER_TOP_K=5`
- `RERANKER_MAX_LEN=512`
- `REQUIRE_LLM_READY=false` (evita travar o healthz em LLM externo)

> Dica: se o reranker não for necessário (ou se a máquina é limitada), use `RERANKER_ENABLED=false` — o backend faz fallback seguro com `score=0.0`.

---

## 🏗 ETL (construção do índice)

O ETL percorre `./data`, lê arquivos suportados, divide em chunks e gera embeddings, salvando um índice **FAISS** persistente.  
- Script principal: `scripts/etl_build_index.py` (executado nos containers `ai_etl`).  
- Parâmetros (via env): `DATA_DIR` (padrão `/app/data`), `FAISS_OUT_DIR` (padrão `/app/vector_store/faiss_index`), `EMBEDDINGS_MODEL`.

**CPU**:
```bash
docker-compose -f docker-compose.cpu.yml build ai_etl
docker-compose -f docker-compose.cpu.yml run --rm ai_etl \
  python scripts/etl_build_index.py --data ./data --out /app/vector_store/faiss_index
```

**GPU** (se preferir rodar ETL igual, também funciona em CPU; GPU é opcional):
```bash
docker-compose -f docker-compose.gpu.yml build ai_etl
docker-compose -f docker-compose.gpu.yml run --rm ai_etl \
  python scripts/etl_build_index.py --data ./data --out /app/vector_store/faiss_index
```

---

## 🚀 Subir serviços

### CPU
Opção A (scripts prontos):
```bash
./scripts/treinar_ia_cpu.sh      # roda ETL
./scripts/inicia_site_cpu.sh     # sobe API+Web e abre navegador
```

Opção B (compose manual):
```bash
docker-compose -f docker-compose.cpu.yml build ai_projeto_api ai_web_ui
docker-compose -f docker-compose.cpu.yml up -d ai_projeto_api ai_web_ui
```

### GPU (CUDA)
Pré-requisitos: driver NVIDIA + NVIDIA Container Toolkit.

Opção A (scripts prontos):
```bash
./scripts/treinar_ia_gpu.sh      # roda ETL
./scripts/inicia_site_gpu.sh     # sobe API+Web (GPU) e abre navegador
```

Opção B (compose manual):
```bash
docker-compose -f docker-compose.gpu.yml build ai_projeto_api ai_web_ui
docker-compose -f docker-compose.gpu.yml up -d ai_projeto_api ai_web_ui
```

---

## 🧪 Smoke tests

**Via script**:
```bash
./smoke.sh               # CPU (consulta via 5000 e 8080)
```

**Manual rápido**:

- Healthz:
```bash
curl -s http://localhost:5000/healthz | jq .
curl -s http://localhost:8080/api/healthz | jq .
```

- Consulta (5000) com debug:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"question":"onde encontro informação de monitoria de computação?","debug":true}' \
  http://localhost:5000/query | jq '.context_found, .debug.route, .debug.rerank.enabled, .debug.timing_ms'
```

- Consulta (8080) via nginx:
```bash
curl -s -H "Content-Type: application/json" \
  -d '{"question":"onde encontro informação de monitoria de computação?","debug":false}' \
  http://localhost:8080/api/query | jq '.answer, .citations'
```

- UI: acesse `http://localhost:8080/`  
  - Botão **Perguntar** habilita somente quando `/api/healthz` retornar `ready:true`.

---

## 🔍 Validação funcional (checklist)

1. **ETL/FAISS**
   - Índice criado no volume (`/app/vector_store/faiss_index`).
   - `healthz` retorna `"faiss": true` e `faiss_store_dir` correto.
2. **Embeddings**
   - `healthz` mostra `embeddings_model` esperado.
   - `context_found: true` quando há documentos relevantes em `./data`.
3. **Reranker (opcional)**
   - Se ativo, `debug.rerank.enabled: true` e `name` correto.
   - Scores **sempre float** (0.0 em fallback).
4. **Telemetria**
   - `debug.timing_ms.retrieval` e `debug.timing_ms.reranker` (quando aplicável).
5. **nginx/UI**
   - `GET /api/healthz` (8080) → 200 com `ready:true`.
   - `POST /api/query` → 200 e resposta com `answer` + `citations`.
6. **Logs**
   - `docker logs -f ai_projeto_api` sem tracebacks.
   - Se o reranker falhar, WARN + fallback (sem quebrar).

---

## 🧩 Sobre *ontology*, *loaders* e *prompts*

- **Ontology (`config/ontology/terms.yml`)**: mantenha termos e aliases mapeados para normalização/triagem. Um rebuild do ETL **não** é obrigatório ao editar a ontologia, a menos que gere novos metadados que precisem ir ao índice.
- **Loaders (`loaders/*.py`)**: cada loader extrai **texto** de um tipo de arquivo; o ETL utiliza funções equivalentes internamente (TXT/MD/PDF/DOCX). Se ampliar tipos, adicione novo loader e ajuste o ETL, se necessário.
- **Prompts (`prompts/*.txt`)**:  
  - `triagem_prompt.txt` — ajuda a decidir a rota/estratégia de resposta.  
  - `pedir_info_prompt.txt` — pedido de dados adicionais ao usuário.  
  - `resposta_final_prompt.txt` — molda a resposta final.  
  Ajuste com cuidado; mudanças tendem a afetar estilo/estrutura das respostas.

---

## ⚡ GPU: checagens rápidas

Verifique CUDA dentro do container da API:
```bash
docker-compose -f docker-compose.gpu.yml exec ai_projeto_api python - <<'PY'
import torch
print("torch.cuda.is_available:", torch.cuda.is_available())
print("num_gpus:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Se faltar VRAM ou houver erro, reduza `RERANKER_TOP_K` ou desative o reranker.

---

## 🧰 Troubleshooting

- **`ready=false`/`faiss=false`** → rode o ETL; confirme `FAISS_STORE_DIR` na API.
- **Timeout via 8080** → confira `web_ui/conf.d/default.conf` (`location /api/` para a API).
- **Reranker lento/falhando** → `RERANKER_ENABLED=false` ou `TOP_K` menor; o backend já faz fallback seguro.
- **Comparação de `None`** → já mitigado (scores sempre float). Se aparecer, verifique se você alterou o front para não ordenar por campos inexistentes.
- **Sem internet p/ baixar modelos** → use cache local (`HF_HOME`/`TRANSFORMERS_CACHE`) ou desative o reranker.

---

## 📄 Licença
MIT (ou a política da sua organização).

---

## 🙌 Créditos
- Estrutura e ajustes do projeto: Celso Lisboa
- Patches de robustez (reranker, readiness, nginx/UI): colaboração assistida
