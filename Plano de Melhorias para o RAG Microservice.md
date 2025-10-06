# Plano de Ação para o RAG Microservice

Este documento define um roteiro prático e priorizado para aprimorar o microserviço RAG, com foco em melhorias de alto impacto. As ações são organizadas por prioridade, começando com as que trarão maior benefício em termos de qualidade e performance.

---

## Plano de Ação 1: Implementar Framework de Avaliação Contínua (RAGAs)

**Objetivo:** Substituir a avaliação manual e subjetiva por um processo automatizado e objetivo, permitindo medir o impacto de cada mudança no código e garantir a qualidade contínua das respostas.

**Status:** Concluído

### Passos de Implementação:

1.  **[x] Criar um "Golden Dataset" de Avaliação:**
    *   **Ação:** Desenvolver um arquivo `evaluation_dataset.jsonl` na raiz do projeto.
    *   **Estrutura:** Cada linha do arquivo conterá um objeto JSON com os seguintes campos:
        *   `question`: Uma pergunta de teste.
        *   `ground_truth`: A resposta ideal e factual para a pergunta.
        *   `contexts`: Uma lista de trechos de texto (os "chunks" exatos) que devem ser usados para formular a resposta.
    *   **Meta Inicial:** Começar com 20-30 exemplos que cubram diferentes tipos de perguntas (contato, procedimento, etc.).

2.  **[x] Desenvolver o Script de Avaliação (`evaluate.py`):**
    *   **Ação:** Criar um novo script `tools/evaluate.py`.
    *   **Lógica:**
        1.  Adicionar `ragas` ao `requirements.txt`.
        2.  O script deve carregar o `evaluation_dataset.jsonl`.
        3.  Para cada item no dataset, o script fará uma chamada `POST` para o endpoint `/query` da API local.
        4.  Com a pergunta, a resposta gerada pela API e o contexto esperado, o script usará a biblioteca **RAGAs** para calcular as seguintes métricas:
            *   `faithfulness`: Mede se a resposta alucina ou se baseia estritamente no contexto.
            *   `answer_relevancy`: Mede a relevância da resposta para a pergunta.
            *   `context_precision` e `context_recall`: Medem a qualidade do pipeline de recuperação (FAISS + Reranker).
        5.  Ao final, o script deve imprimir um relatório com a média de cada métrica.

3.  **[x] Integrar ao Fluxo de Trabalho:**
    *   **Ação:** Executar o script `evaluate.py` manualmente após qualquer mudança significativa (ex: alteração de prompts, ajuste de `CONFIDENCE_MIN`) para validar se a qualidade foi mantida ou melhorada.
    *   **Longo Prazo:** Integrar a execução do script em um pipeline de CI/CD para automatizar completamente a regressão de qualidade.

---

## Plano de Ação 2: Implementar Cache de Respostas com Redis

**Objetivo:** Reduzir drasticamente a latência e o custo para perguntas frequentes, melhorando a experiência do usuário e otimizando o uso de recursos.

**Status:** Concluído

### Passos de Implementação:

1.  **[x] Configurar o Ambiente:**
    *   **Ação:** Adicionar `redis` ao `requirements.txt`.
    *   **Ação:** Atualizar os arquivos `docker-compose.cpu.yml` and `docker-compose.gpu.yml` para incluir um serviço Redis.

2.  **[x] Implementar a Lógica de Cache no `api.py`:**
    *   **Ação:** Modificar o endpoint `@app.post("/query")`.
    *   **Lógica:**
        1.  No início da função, criar uma chave de cache a partir da pergunta do usuário (ex: `cache_key = f"rag_query:{hash(request.get_json()['question'].lower().strip())}"`).
        2.  Tentar buscar essa chave no Redis.
        3.  **Cache Hit:** Se a chave existir, decodificar o valor (JSON da resposta) e retorná-lo imediatamente, pulando todo o pipeline de RAG.
        4.  **Cache Miss:** Se a chave não existir, executar o pipeline de RAG normalmente.
        5.  Antes de retornar a resposta ao usuário, armazená-la no Redis usando a `cache_key` e definindo um tempo de expiração (TTL), por exemplo, 24 horas.

3.  **[x] Implementar a Invalidação do Cache:**
    *   **Ação:** Modificar o `etl_orchestrator.py`.
    *   **Lógica:** Ao final da execução bem-sucedida do ETL (tanto no modo `rebuild` quanto `update`), adicionar uma chamada para o Redis que limpa todas as chaves com o prefixo `rag_query:`. Isso garante que as respostas cacheadas não fiquem desatualizadas após uma atualização da base de conhecimento.

---

## Plano de Ação 3 (Avançado): Desenvolver Agente de Auto-Correção

**Objetivo:** Aumentar a confiabilidade das respostas em tempo real, criando um agente que avalia a si mesmo em busca de alucinações e tenta se corrigir antes de responder ao usuário.

**Status:** A Fazer

### Passos de Implementação:

1.  **[ ] Criar Nó de "Auto-Avaliação" no `agent_workflow.py`:**
    *   **Ação:** Após o nó `node_auto_resolver` gerar uma resposta, adicionar um novo nó chamado `node_self_evaluate`.
    *   **Lógica:** Este nó usará a biblioteca RAGAs para calcular a métrica `faithfulness` da resposta gerada em relação ao contexto que a suporta. O resultado (`faithfulness_score`) será adicionado ao `AgentState`.

2.  **[ ] Criar Nó de "Correção" no `agent_workflow.py`:**
    *   **Ação:** Adicionar um novo nó chamado `node_correction`.
    *   **Lógica:** Este nó receberá o estado contendo a resposta original e o contexto. Ele chamará o LLM com um prompt específico, como:
        > "A resposta anterior ('{resposta_anterior}') não foi totalmente fiel ao contexto fornecido ('{contexto}'). Reescreva a resposta garantindo que ela se baseie estritamente no contexto e corrija qualquer informação que não possa ser comprovada por ele."
    *   A nova resposta gerada por este nó substituirá a resposta original no `AgentState`.

3.  **[ ] Modificar as Arestas do Grafo:**
    *   **Ação:** Alterar a lógica condicional após o `node_auto_resolver`.
    *   **Lógica:**
        *   O fluxo agora irá para o `node_self_evaluate`.
        *   Após a auto-avaliação, uma nova aresta condicional decidirá:
            *   Se `faithfulness_score` for alto (ex: > 0.8), o fluxo vai para `END`.
            *   Se `faithfulness_score` for baixo, o fluxo vai para o `node_correction`.
        *   Após o `node_correction`, o fluxo vai para `END`.

---

## Backlog de Otimizações Adicionais

Estes são itens valiosos do plano original que devem ser considerados para implementação futura, após a conclusão dos planos de ação prioritários.

*   **[ ] Precisão: Fusão de Scores (Reciprocal Rank Fusion - RRF):** Em vez da abordagem de fallback, combinar os scores da busca lexical e vetorial usando RRF para melhorar a relevância dos resultados.
*   **[ ] Cobertura: Expansão Contínua da Ontologia:** Manter um processo contínuo de revisão e expansão do `config/ontology/terms.yml` com base na análise de perguntas que falharam.
*   **[ ] Cobertura: Enriquecimento de Metadados no ETL:** Expandir o `etl_orchestrator.py` para extrair e armazenar metadados adicionais (autor, data, tipo de documento) no PostgreSQL, usando-os para filtragem e boosting.
*   **[ ] Robustez: Retentativas com Backoff Exponencial:** Usar a biblioteca `tenacity` para implementar retentativas em chamadas de rede (LLMs, DB), tornando o sistema mais resiliente a falhas transitórias.
*   **[ ] Custo/Velocidade: Otimização de Modelos:** Avaliar sistematicamente modelos de embedding e LLMs mais leves/rápidos para encontrar o melhor equilíbrio entre custo, latência e precisão, usando o framework de avaliação (Plano 1) para medir o impacto.
