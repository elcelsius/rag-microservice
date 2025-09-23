# Plano Consolidado de Melhorias para o RAG Microservice

Este documento integra as recomendações de melhoria para o RAG Microservice, conciliando o plano elaborado anteriormente com novas observações resultantes da revisão do código atual. A ideia é manter um roteiro vivo, priorizando correções imediatas e organizando evoluções de curto, médio e longo prazo.

## 0. Ajustes de Base (urgente)

- **`docker-compose.cpu.yml` e `docker-compose.gpu.yml` – porta interna do Postgres**: atualizar as variáveis `DATABASE_URL` para usar o serviço `ai_postgres` na porta interna `5432` (a porta mapeada no host pode variar). Hoje os arquivos referenciam `${POSTGRES_PORT}`, o que quebra a comunicação entre containers quando o valor exposto externamente é diferente. _Refs_: `docker-compose.cpu.yml:37`, `docker-compose.cpu.yml:59`, `docker-compose.gpu.yml:36`, `docker-compose.gpu.yml:58`.
- **`.env.example` – host padrão do banco**: alinhar `POSTGRES_HOST=ai_postgres` para refletir o host usado pelos containers. Isso evita configurações incorretas ao iniciar um projeto novo.

## 1. Avaliação Contínua e Monitoramento da Qualidade

_Status atual: o script `eval_rag.py` já mede métricas de recuperação e integra RAGAs, mas falta um dataset canônico e automação no pipeline._

### 1.1. Implementar Framework de Avaliação Contínua (RAGAs)

1. **Criar um "Golden Dataset"**  
   - Disponibilizar `evaluation_dataset.jsonl` (ou `.csv`) na raiz, com campos `question`, `ground_truth` e `contexts`.  
   - Incluir exemplos que cubram casos ambíguos, perguntas com múltiplas fontes e históricos multi-turno.

2. **Evoluir o `eval_rag.py`**  
   - Permitir carregar o dataset padrão automaticamente (flag `--dataset` opcional).  
   - Registrar métricas em arquivo (`reports/eval-YYYYMMDD.json`) para auditoria histórica.  
   - Adicionar opção para comparar `/agent/ask` e `/query` na mesma execução.

3. **Integrar ao fluxo de trabalho**  
   - Executar manualmente antes de merges relevantes (mudanças em prompt, embeddings, reranker).  
   - _Próximo passo_: adicionar job de CI que rode o dataset de regressão com um limite reduzido (foco em smoke de qualidade).

### 1.2. Métricas e Monitoramento

- Expandir o painel de métricas com as saídas do `METRICS` em `api.py` (ex.: expor `/metrics` em Prometheus).  
- Incluir métricas de `telemetry.log_event` (confiança média, rotas utilizadas, tempo de resposta).  
- Manter uma suíte de regressão manual/automática (scripts em `scripts/smoke_*.sh` podem ser reaproveitados).  
- Considerar A/B testing quando alternar modelos de embeddings ou rerankers.

## 2. Otimização de Performance e Custo

_Status atual: o pipeline usa reranker CPU e não possui cache das respostas; latência pode ser reduzida._

### 2.1. Implementar Cache de Respostas com Redis

1. Adicionar `redis` às dependências (`requirements*.txt`) e incluir o serviço nos `docker-compose.*`.
2. Na API (`api.py`), armazenar respostas chaveadas por `question` + hash do histórico; respeitar TTL configurável.
3. Invalidação: ao final do ETL (`etl_orchestrator.py`), limpar chaves relacionadas para evitar respostas desatualizadas.
4. Monitorar métricas de acerto/erro do cache e impactar nos contadores expostos em `/metrics`.

### 2.2. Otimizações Adicionais de Performance

- **Cache de embeddings**: persistir embeddings de chunks em disco (ou Redis) para acelerar reprocessamentos no ETL.  
- **Quantização/compactação**: avaliar versões quantizadas dos modelos (`e5-base`, INT8) para cenários CPU.  
- **Paralelização**: usar `asyncio` ou _thread pools_ no pipeline de multi-query para reduzir latência.  
- **Streaming de resposta**: expor endpoint que faça streaming parcial (útil para UI).  
- **Instrumentação**: integrar ferramentas como OpenTelemetry para medir tempo por etapa (lexical, vetorial, reranker, LLM).

## 3. Aprimoramento do Pipeline RAG

_Status atual: há multi-query, busca híbrida e reranker. O foco agora é refinar qualidade e recall._

### 3.1. Embeddings e Vector Store

- Avaliar modelos específicos de domínio (jurídico, médico etc.) disponíveis no Hugging Face Hub.  
- Comparar `intfloat/multilingual-e5-base` vs `-large` vs modelos alternativos com o dataset de avaliação.  
- Estudar particionamento do FAISS em shards por domínio ou departamento para escalabilidade.

### 3.2. Estratégias de Chunking

- Testar tamanhos de chunk dinâmicos (baseados em seções ou headings) usando `RecursiveCharacterTextSplitter`.  
- Marcar metadados com identificadores de seção para facilitar citações mais precisas.  
- Revisitar sobreposição (`chunk_overlap`) para reduzir duplicidade sem perder contexto.

### 3.3. Recuperação (Retrieval)

- Ajustar parâmetros de multi-query (`MQ_VARIANTS`) e thresholds lexicais (`LEXICAL_THRESHOLD`).  
- Implementar _Reciprocal Rank Fusion_ (RRF) para combinar as listas lexical e vetorial em vez de usar apenas fallback.  
- Automatizar a manutenção do `config/ontology/terms.yml` (identificar termos frequentes e sugerir novos aliases).  
- Enriquecer metadados no FAISS (ex.: tags de departamento, nível de documento) para filtros futuros.

### 3.4. Reranker

- Medir o custo/benefício dos presets atuais (`balanced` vs `full`) e avaliar modelos como `jinaai/jina-colbert-v1`.  
- Suportar execução em GPU quando disponível (`RERANKER_DEVICE=cuda`).  
- Cachear scores de reranker para combinações repetidas de (pergunta, documento) em Redis ou SQLite leve.

### 3.5. Geração de Respostas (LLM)

- Revisar prompts em `prompts/` com exemplos adicionais (few-shot) para melhorar consistência.  
- Oferecer modo _low-cost_ usando modelos menores (ex.: `gemini-1.5-flash-lite`) baseado em configuração de ambiente.  
- Implementar pós-processamento para remover redundâncias e garantir que citações referenciem os trechos corretos.  
- Guardar telemetria de tokens consumidos por requisição para acompanhar custos.

## 4. Aprimoramento do Agente LangGraph

_Status atual: grafo com triagem -> auto_resolver -> pedir_info. Há oportunidades para autocorreção e melhor monitoramento._

### 4.1. Implementar Agente de Auto-Correção

1. **Nó de auto-avaliação** (`node_self_evaluate`): usar RAGAs para calcular `faithfulness` com base no contexto retornado.  
2. **Nó de correção** (`node_correction`): quando o score for baixo, gerar nova resposta forçando aderência ao contexto recuperado.  
3. Ajustar arestas condicionais para encadear `auto_resolver` -> `self_evaluate` -> (`node_correction` ou `END`).  
4. Registrar métricas adicionais (score médio, taxa de correção) em `telemetry.log_event`.

### 4.2. Otimizações do Agente

- Aprimorar prompts (`triagem_prompt.txt`, `pedir_info_prompt.txt`) com exemplos reais coletados na produção.  
- Adicionar sinalizações no estado (`AgentState`) para registrar por que a triagem pediu mais informações.  
- Expor métricas de caminho (quantas iterações até `END`, quantas quedas em `PEDIR_INFO`).  
- Considerar banir loops longos com limite de tentativas para evitar ciclos infinitos.

## 5. Backlog de Otimizações Adicionais

- Expandir continuamente a ontologia (`config/ontology/terms.yml`) com base nas perguntas que falham.  
- Garantir retentativas com _backoff_ exponencial em chamadas críticas (LLM, Postgres, FAISS).  
- Suportar versionamento de índices FAISS para rollback rápido.  
- Adicionar modo de _dry-run_ no ETL para validar documentos sem persistir.  
- Explorar integração com ferramentas de monitoramento (Grafana, ELK) usando os logs estruturados.

## 6. Referências

1. [Ragas Documentation](https://docs.ragas.io/en/latest/)  
2. [Hugging Face Hub](https://huggingface.co/models)  
3. [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)  
4. [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_loaders/how_to/text_splitter)

## Próximos Passos Sugeridos

1. Concluir os ajustes de base listados na Seção 0.  
2. Construir o dataset de avaliação e integrar o `eval_rag.py` (Seção 1).  
3. Iniciar a implementação do cache de respostas com Redis (Seção 2.1).
