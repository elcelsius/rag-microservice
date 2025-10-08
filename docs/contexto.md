# Contexto do Projeto RAG Microservice

Este documento fornece um panorama atualizado do projeto `rag-microservice`, descrevendo arquitetura, componentes, variáveis de ambiente e fluxos de trabalho. Ele serve como guia rápido para quem precisa entender a base do código antes de implementar novas funcionalidades.

## 1. Visão Geral do Projeto

O `rag-microservice` implementa um sistema de Pergunta-Resposta baseado em Retrieval-Augmented Generation (RAG) e coordenado por um agente construído com LangGraph. Todo o stack roda em Docker (CPU ou GPU), o que facilita a reprodução do ambiente. A aplicação consolida pipeline de ingestão (ETL), indexação FAISS, inferência com LLMs (Google Gemini ou OpenAI) e uma interface web simples.

## 2. Arquitetura do Sistema

### 2.1. Fluxo do Agente (LangGraph)

O agente definido em `agent_workflow.py` decide se precisa de mais contexto ou se pode acionar o pipeline de RAG diretamente:

1. Recebe a pergunta e o histórico de mensagens.
2. Executa o nó de triagem (`node_triagem`) com um prompt dedicado.
3. Se faltarem dados, envia uma pergunta de esclarecimento via `node_pedir_info`.
4. Caso contrário, segue para `node_auto_resolver`, que invoca o pipeline de RAG.
5. Se o RAG falhar (ex.: baixa confiança, sem contexto), o agente volta a pedir informações ao usuário.

Este ciclo torna a interação mais robusta para perguntas vagas ou múltiplos turnos.

### 2.2. Pipeline de RAG (Retrieval-Augmented Generation)

O pipeline em `query_handler.py` combina heurísticas lexicais e vetoriais. A sequência principal é:

1. **Normalização e pergunta autônoma**: limpeza da entrada e, quando há histórico, condensação feita pelo agente antes de chamar o pipeline.
2. **Multi-query**: geração opcional de variações da pergunta (`MQ_ENABLED`, `MQ_VARIANTS`) para ampliar a cobertura.
3. **Busca lexical**: ranking inicial com RapidFuzz sobre todos os documentos do `vectorstore.docstore`, ajudando a detectar nomes próprios (com variações e acentuação) e consultas muito específicas; quando encontra múltiplos candidatos, prepara as respostas de desambiguação que retornam com `needs_clarification`.
4. **Busca vetorial (FAISS)**: recuperação baseada em embeddings (`HuggingFaceEmbeddings`) com o índice FAISS persistido em disco.
5. **Fusão híbrida**: combinação dos resultados lexicais e vetoriais (`HYBRID_ENABLED`), com possibilidade de forçar rotas via `ROUTE_FORCE`.
6. **Reranqueamento**: aplicação de um CrossEncoder (quando disponível) ou fallback HuggingFace para refinar a ordem (`RERANKER_PRESET`, `RERANKER_*`).
7. **Avaliação de confiança**: cálculo de um score agregado; abaixo do limiar ativo (`CONFIDENCE_MIN` ou overrides por rota) e com `REQUIRE_CONTEXT=true` o sistema responde pedindo mais detalhes, sugerindo departamentos quando possível.
8. **Geração estruturada**: chamada ao LLM via `llm_client.call_llm`, montagem de resposta em Markdown com `### Resumo` e `### Fontes` e citações provenientes dos metadados.
9. **Telemetria e debug**: registro de métricas de tempo, rota escolhida e confiança por meio de `telemetry.log_event` quando `LOG_DIR` está configurado.

### 2.3. Telemetria e Observabilidade

- `/healthz` verifica se FAISS está carregado e, opcionalmente (`REQUIRE_LLM_READY`), se o LLM está acessível.
- `/metrics` expõe contadores de uso e uptime (ver `api.py`).
- Logs de telemetria são gravados no diretório apontado por `LOG_DIR`, consolidando rota escolhida, confiança e tempo por requisição.

## 3. Componentes Principais

- `api.py`: servidor Flask com endpoints `/agent/ask` (`/api/ask` via UI), `/query` (`/api/query`), `/healthz` e `/metrics`, além da inicialização de embeddings e FAISS.
- `agent_workflow.py`: define o grafo LangGraph do agente, prompts de triagem e fluxo de decisão.
- `query_handler.py`: implementa toda a recuperação híbrida, reranqueamento, cálculo de confiança, formatação da resposta e telemetria.
- `llm_client.py`: cliente resiliente que abstrai Gemini/OpenAI, retentativas (Tenacity) e carregamento de prompts.
- `etl_orchestrator.py`: orquestra o ETL incremental, gera embeddings, atualiza FAISS e persiste metadados no PostgreSQL.
- `loaders/`: coleções de loaders unificados para formatos como PDF, DOCX, Markdown, CSV, JSON etc.
- `prompts/`: prompts versionados usados pelo agente, triagem, esclarecimentos e resposta final.
- `scripts/`: utilitários (shell) para subir serviços, rodar smoke tests e pipelines de ETL (ex.: `scripts/smoke_cpu.sh`).
- `tools/check_data_and_loaders.py`: diagnóstico rápido dos dados e loaders usados pelo ETL.
- `tests/test_api.py`: suíte inicial de testes com pytest para validar endpoints principais.
- `web_ui/`: assets estáticos (HTML/JS/CSS) servidos pelo Nginx (`ai_web_ui`), que também faz proxy das rotas `/api/...` para a API Flask.
- `telemetry.py`, `triage.py`, `manage_rag.bat` e arquivos `Dockerfile.*`/`docker-compose.*` complementam a infraestrutura.

## 4. Configuração e Variáveis de Ambiente

Principais variáveis (veja `.env.example` para a lista completa):

- **LLM**: `GOOGLE_API_KEY`, `GOOGLE_MODEL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `LLM_PROVIDER`.
- **Embeddings/FAISS**: `EMBEDDINGS_MODEL`, `FAISS_STORE_DIR`, `FAISS_OUT_DIR` (ETL), `DATA_PATH_CONTAINER`.
- **Reranker**: `RERANKER_PRESET` (`off|fast|balanced|full`), `RERANKER_ENABLED`, `RERANKER_NAME`, `RERANKER_CANDIDATES`, `RERANKER_TOP_K`, `RERANKER_MAX_LEN`, `RERANKER_DEVICE`, `RERANKER_TRUST_REMOTE_CODE`.
- **Busca híbrida**: `HYBRID_ENABLED`, `LEXICAL_THRESHOLD`, `DEPT_BONUS`, `MAX_PER_SOURCE`, `ROUTE_FORCE` (diagnóstico).
- **Multi-query e confiança**: `MQ_ENABLED`, `MQ_VARIANTS`, `CONFIDENCE_MIN`, `REQUIRE_CONTEXT`.
- **Observabilidade**: `REQUIRE_LLM_READY`, `DEBUG_LOG`, `DEBUG_PAYLOAD`, `LOG_DIR`.
- **Banco de dados**: `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`. Nos ambientes Docker Compose o host utilizado é `ai_postgres` e o serviço expõe a porta interna `5432` para os demais containers.

## 5. Como Executar o Projeto

### 5.1. Pré-requisitos

- Docker e Docker Compose instalados.
- Chave API válida (Google Gemini ou OpenAI).
- Para GPU: NVIDIA Container Toolkit configurado.

### 5.2. Configuração

1. Copie `.env.example` para `.env`.
2. Preencha as credenciais de LLM, caso necessário ajuste `POSTGRES_HOST=ai_postgres`.
3. Disponibilize os documentos em `data/` antes de rodar o ETL.

### 5.3. Execução

Na raiz do projeto:

- Ambiente CPU: `docker-compose -f docker-compose.cpu.yml up --build`
- Ambiente GPU: `docker-compose -f docker-compose.gpu.yml up --build`

Após o start:
- API: `http://localhost:5000`
- UI: `http://localhost:8080`
- PostgreSQL (host): `localhost:5432` (porta configurável via `.env`)

## 6. Endpoints da API

- `POST /agent/ask`: fluxo completo com LangGraph e suporte a histórico de mensagens (acessível também como `/api/ask`).
- `POST /query`: pipeline RAG direto (sem agente), útil para testes e comparações (roteado como `/api/query` na UI).
- `GET /healthz`: probe de prontidão (FAISS + LLM quando `REQUIRE_LLM_READY=true`).
- `GET /metrics`: métricas agregadas (contadores e uptime).
- `GET /`: resposta simples para check rápido.

## 7. Avaliação e Testes

- `eval_rag.py`: executa consultas no endpoint do agente, calcula métricas de recuperação (Recall@5, MRR@10, nDCG@5) e, se `ragas` estiver instalado, gera `faithfulness` e `answer_relevancy`. O próximo passo é consolidar um `evaluation_dataset.jsonl` para padronizar os testes (ver melhorias).
- Testes automatizados: rodar `pytest -v` após instalar `requirements-cpu.txt` ou `requirements-gpu.txt`.
- Smoke tests: scripts em `scripts/smoke_cpu.sh`/`smoke_gpu.sh` automatizam um fluxo mínimo via Docker.

## 8. Estrutura de Diretórios

```text
.
|-- api.py
|-- agent_workflow.py
|-- contexto.md
|-- docker-compose.cpu.yml
|-- docker-compose.gpu.yml
|-- Dockerfile.cpu
|-- Dockerfile.gpu
|-- etl_orchestrator.py
|-- eval_rag.py
|-- llm_client.py
|-- manage_rag.bat
|-- query_handler.py
|-- telemetry.py
|-- triage.py
|-- requirements.txt
|-- requirements-cpu.txt
|-- requirements-gpu.txt
|-- config/
|   |-- ontology/
|       |-- terms.yml
|-- data/
|-- loaders/
|-- logs/
|-- prompts/
|-- scripts/
|   |-- smoke_cpu.sh
|   |-- smoke_gpu.sh
|   |-- etl_build_index.py
|-- tests/
|   |-- test_api.py
|-- tools/
|   |-- check_data_and_loaders.py
|-- web_ui/
|   |-- html/
|   |-- conf.d/
```

## 9. Notas Adicionais

- O ETL suporta execuções incrementais: detecta arquivos novos/alterados e remove entradas obsoletas.
- `REQUIRE_LLM_READY=true` impede a API de responder até que Gemini/OpenAI seja alcançado; útil em produção.
- `telemetry.py` grava logs estruturados (JSON) em `LOG_DIR`, facilitando ingestão em ferramentas externas.
- `ROUTE_FORCE=lexical|vector` força rotas específicas durante diagnósticos.
- As respostas do RAG já seguem o formato Markdown com seções `### Resumo` e `### Fontes`.
- Sempre responda ou pergunte em português brasileiro

## 10. Regras e diretrizes para o assistente (Codex)

1. **Linguagem**: sempre PT-BR.
2. **Sem "alucinar"**: basear mudanças nos arquivos reais do repositório; quando incerto, citar trecho/linha que pretende alterar.
3. **Assinaturas públicas**: não quebrar endpoints, shape de payloads ou contratos públicos.
4. **Embeddings consistentes**: ETL e API devem usar o mesmo `EMBEDDINGS_MODEL`. Se alterar um, alinhar o outro.
5. **Reranker presets**: mudar apenas via ENV (não hardcode em Python).
6. **`/app` path**: manter os caminhos existentes que usam `/app` (adiar refators globais).
7. **Estilo de código**: manter compatibilidade com `ruff` e testes; evitar imports múltiplos na mesma linha; sem `;` no fim da linha.
8. **Respostas do RAG**: manter formato Markdown com `### Resumo` e `### Fontes`.
9. **Prompts/LLM**: não versionar chaves; prompts vivem em `prompts/`. Evitar alterar a intenção dos prompts sem justificativa.
10. **Compose/DB**: ao criar strings de conexão em Docker, usar o serviço `ai_postgres` na porta interna `5432` (ver `docker-compose.*`).

---

## 11. Fluxo incremental com telemetria e verificações

- **Sempre por partes**: planeje uma entrega pequena, implemente, valide e observe os logs **antes** de iniciar a próxima.
- **Docstrings e comentários**: toda função/rota nova deve ter docstring descrevendo entradas/saídas; se houver lógica pouco óbvia, incluir comentário curto para facilitar manutenção futura.
- **Logs obrigatórios**: novos trechos devem registrar início/fim, status (`ok`/erro) e `elapsed_ms`. Prefira logs estruturados (JSON) e reutilize `telemetry.log_event` quando aplicável.
- **Checklist de validação**:
  1. `python -m pytest` (ou o subconjunto pertinente) no ambiente local.
  2. `./scripts/smoke_cpu.sh` (invoca `scripts/smoke_api.py` para validar `/query`, `/agent/ask`, `/metrics` real).
  3. `docker-compose -f docker-compose.cpu.yml up --build` seguido de `docker-compose -f docker-compose.cpu.yml logs -f ai_projeto_api` para confirmar inicialização limpa de FAISS, Redis, LLM e ausência de exceções.
  4. `curl http://localhost:5000/healthz` (status 200) e inspeção dos contadores em `/metrics` (`cache_hits_total`, `agent_refine_*`, etc.).
- **Observabilidade como gate**: se os logs indicarem regressão (latência alta, confiança zerada, cache sem hits), interromper novas features até diagnosticar.
- **Documentação/ENV**: sempre que criar um toggle ou ajuste, atualizar README/docs e espelhar o valor padrão em `.env.example`.
