# 🧠 rag-microservice — README FINAL (CPU + GPU) + ETL/RAG explicados

Microserviço de **RAG (Retrieval-Augmented Generation)** com **Flask**, **FAISS**, **sentence-transformers** e UI via **nginx** (8080). Suporta **CPU** e **GPU (CUDA)**. Inclui **reranker** opcional (CrossEncoder).

---

## 📦 Stack e serviços

- **Python 3.11**
- **Flask** — API (`/query`, `/healthz`)
- **LangChain (community + text-splitters)**
- **FAISS** — índice vetorial persistente em `/app/vector_store/faiss_index`
- **Embeddings** — `intfloat/multilingual-e5-large`
- **Reranker (opcional)** — `jinaai/jina-reranker-v2-base-multilingual`
- **nginx** — UI (8080) + proxy `/api/*` → API (5000)
- Compose: `ai_etl` (ETL), `ai_projeto_api` (API), `ai_web_ui` (nginx/UI), `ai_postgres` (opcional)

---

## 🗂 Estrutura do repositório (essencial)

```
rag-microservice/
├─ config/ontology/terms.yml         # ontologia/dicionário para triagem/normalização
├─ data/                             # documentos (TXT/MD/PDF/DOCX...), subpastas ok
├─ loaders/                          # seus loaders: load(file_path)->list[Document]
│  ├─ code_loader.py                 # TextLoader com fallback de encoding
│  ├─ docx_loader.py                 # Docx2txtLoader / python-docx
│  ├─ md_loader.py                   # UnstructuredMarkdownLoader
│  ├─ pdf_loader.py                  # UnstructuredPDFLoader (mode="single")
│  └─ txt_loader.py                  # TextLoader
├─ prompts/
│  ├─ pedir_info_prompt.txt
│  ├─ resposta_final_prompt.txt
│  └─ triagem_prompt.txt
├─ scripts/
│  ├─ etl_build_index.py             # ETL (CLI: --data, --out, --exts, --loaders, ...)
│  ├─ treinar_ia_cpu.sh / treinar_ia_gpu.sh
│  ├─ inicia_site_cpu.sh / inicia_site_gpu.sh
│  ├─ smoke_cpu.sh / smoke_gpu.sh
│  └─ (outros auxiliares)
├─ web_ui/
│  ├─ html/index.html                # usa /api/query e /api/healthz
│  └─ conf.d/default.conf            # nginx mapeia /api/* -> ai_projeto_api:5000
├─ api.py                            # Flask app (endpoints)
├─ query_handler.py                  # RAG + reranker + debug/telemetria
└─ docker-compose.*.yml
```

---

## 🔧 Variáveis (API)

- `FAISS_STORE_DIR=/app/vector_store/faiss_index`
- `EMBEDDINGS_MODEL=intfloat/multilingual-e5-large`
- `RERANKER_ENABLED=true|false`
- `RERANKER_NAME=jinaai/jina-reranker-v2-base-multilingual`
- `RERANKER_TOP_K=5`
- `RERANKER_MAX_LEN=512`
- `REQUIRE_LLM_READY=false`

---

## ▶️ Executar scripts a partir da raiz

### Linux/macOS
```bash
chmod +x scripts/*.sh
./scripts/treinar_ia_cpu.sh
./scripts/inicia_site_cpu.sh
# GPU se aplicável
./scripts/treinar_ia_gpu.sh
./scripts/inicia_site_gpu.sh
# smokes
./smoke_cpu.sh
./smoke_gpu.sh
```

### Windows
- Preferível usar **WSL** (Ubuntu) e os comandos acima.
- PowerShell (fora do WSL): use `bash`:
```powershell
bash scripts/treinar_ia_cpu.sh
bash scripts/inicia_site_cpu.sh
bash smoke_cpu.sh
```

> Se aparecer **permission denied** → `chmod +x scripts/*.sh`  
> Se aparecer **bad interpreter / ^M** → `dos2unix scripts/*.sh` (CRLF → LF)  
> Se `docker-compose` não existir → use `docker compose` (v2).

---

## 🚀 Subir serviços

### CPU
```bash
./scripts/treinar_ia_cpu.sh         # roda ETL (gera FAISS a partir de ./data)
./scripts/inicia_site_cpu.sh        # sobe API+Web
curl -s http://localhost:8080/api/healthz | jq .
```

### GPU (CUDA)
```bash
./scripts/treinar_ia_gpu.sh
./scripts/inicia_site_gpu.sh
curl -s http://localhost:8080/api/healthz | jq .
```

---

## 🧪 Smokes (CPU/GPU) com flags

```bash
# CPU básico
./smoke_cpu.sh

# CPU com ETL e CSV/JSON (se você tiver loaders read_csv/read_json)
./smoke_cpu.sh --with-etl --exts "txt,md,pdf,docx,csv,json" --loaders ./loaders \
  --question "onde encontro informação de monitoria de computação?"

# GPU básico
./smoke_gpu.sh

# GPU com ETL e as mesmas extensões
./smoke_gpu.sh --with-etl --exts "txt,md,pdf,docx,csv,json" --loaders ./loaders
```

Os smokes validam:
- `ready:true` e `faiss:true` no `/api/healthz`
- resposta via 5000 e 8080
- (se reranker ativo) **scores numéricos** (sem `null`).

---

## 🧩 Como funciona o **ETL** neste projeto

O ETL é responsável por **preparar a base vetorial** usada nas buscas do RAG.

### Passo a passo
1. **Leitura de arquivos** (recursiva) em `./data` filtrando por extensões suportadas (`--exts`).  
2. **Loaders** (prioridade dupla):
   - **Estilo “read_\<ext\>”**: se existir uma função `read_<ext>(path)` em `loaders/`, ela é usada, retornando **texto** (`str`). Ex.: `read_csv`, `read_json`.
   - **Estilo “load(file_path)”**: se existir uma função `load(file_path) -> list[Document]` (seus loaders), o ETL **concatena** os `page_content` dos `Document` e segue.  
   - Se nenhum desses estiver disponível, usa **leitores nativos** de texto (txt/md/pdf/docx) como fallback.
3. **Chunking** com `RecursiveCharacterTextSplitter` (parâmetros `--chunk-size` e `--chunk-overlap`).  
4. **Embeddings**: cada chunk vira um vetor usando `intfloat/multilingual-e5-large` (padrão).  
5. **FAISS**: os vetores + metadados (`source`, `chunk`) são gravados em `/app/vector_store/faiss_index` (ou caminho passado com `--out`).

### Diagrama (alto nível)
```
./data  ──► (loaders) read_<ext> | load(file) | fallback ──► texto único
                                         │
                                  split em chunks
                                         │
                               embeddings (e5-large)
                                         │
                          FAISS (persistido em volume docker)
```

> Resultado: a API consegue fazer busca vetorial **rápida** sem depender do tempo de parsing/embedding a cada pergunta.

---

## 🔎 Como funciona o **RAG** (pipeline de consulta)

Quando você chama `POST /query` (via 5000) ou `POST /api/query` (via 8080):

1. **Triagem / Roteamento** (conforme seus prompts e regras internas):  
   Decide a rota (ex.: lexical vs. vetorial). No seu caso, a rota **lexical** vem aparecendo no debug; a rota vetorial usa FAISS.
2. **Busca (FAISS)**:  
   - A pergunta é embeddada com o mesmo modelo (`e5-large`).  
   - O FAISS retorna os **k** chunks mais próximos (candidatos).  
   - O tempo é registrado em `debug.timing_ms.retrieval` (quando `debug=true`).
3. **Reranker (opcional)**:  
   - Se `RERANKER_ENABLED=true`, o CrossEncoder pontua os pares `(pergunta, chunk)` e reordena.  
   - Se o modelo não estiver disponível ou falhar, o backend cai em **fallback** (scores `0.0`, `enabled=false`).  
   - O tempo é registrado em `debug.timing_ms.reranker`.
4. **Síntese/Resposta**:  
   - O sistema sintetiza um **resumo** usando os melhores trechos (com ou sem reranker).  
   - As **fontes** saem em `citations` (cada item com `source`, `chunk`, `preview`).  
   - `context_found` indica se havia contexto útil.
5. **Segurança de debug**:  
   - `debug.rerank.scored[*].score` é **sempre float** (0.0 no fallback), nunca `null` — evita erros no front.

### Diagrama
```
pergunta ─► (triagem) ─► (FAISS top-k) ─► (reranker?) ─► resposta + citações
                        │                │
                        └── timing_ms.retrieval   timing_ms.reranker
```

---

## ✅ Checklist rápido de validação

- `curl -s http://localhost:8080/api/healthz | jq .` → `ready:true`, `faiss:true`  
- `POST /api/query` retorna `answer` + `citations`  
- (se ativo) `debug.rerank.enabled:true` e `score` **numérico**  
- Logs limpos (`docker logs -f ai_projeto_api`)

---

## 🧰 Troubleshooting
- **`ready:false`/`faiss:false`** → rode ETL e verifique `FAISS_STORE_DIR` no container da API.  
- **Reranker lento** → `RERANKER_ENABLED=false` ou reduza `RERANKER_TOP_K`.  
- **Timeout via 8080** → confira `web_ui/conf.d/default.conf` (`location /api/`).  
- **CRLF em scripts** → `dos2unix scripts/*.sh`.  
- **Sem internet para modelos** → use cache local (`HF_HOME`/`TRANSFORMERS_CACHE`) ou desative o reranker.

---

## 📄 Licença
MIT (ou a da sua organização).

---

## 🙌 Créditos
- Projeto e organização: Celso Lisboa  
- Patches de robustez (scores, timing, readiness) + documentação: colaboração assistida
