# Contexto do Projeto: RAG Microservice

## Visão Geral

Este projeto implementa um microserviço de RAG (Retrieval-Augmented Generation) em Python, utilizando Flask para a API, FAISS para a busca vetorial e LangChain para orquestração. O sistema é projetado para ser modular e configurável, com suporte a diferentes modelos de embeddings e LLMs (Google Gemini e OpenAI).

## Arquitetura e Fluxo de Dados

O sistema segue um fluxo de RAG híbrido, combinando busca lexical e vetorial para otimizar a relevância e a velocidade da resposta.

1.  **Entrada da API**: Uma requisição `POST` é recebida no endpoint `/query` (ou `/api/ask`) contendo a pergunta do usuário.
2.  **Triagem (Opcional com Agente)**: Se o modo `agent` estiver ativo (`agent_workflow.py`), um grafo LangGraph primeiro classifica a intenção do usuário. A decisão pode ser `AUTO_RESOLVER` (tentar responder com RAG) ou `PEDIR_INFO` (solicitar mais detalhes ao usuário).
3.  **Rota Lexical (Prioritária)**: O `query_handler.py` primeiro tenta uma busca lexical rápida. Ele procura por sentenças que contenham termos-chave da pergunta. Se um documento pertence a um "departamento" mencionado na pergunta (definido em `terms.yml`), ele recebe um bônus de pontuação. Se essa busca encontra resultados com alta confiança, a resposta é gerada diretamente a partir desses trechos, sem passar pela rota vetorial.
4.  **Rota Vetorial (Fallback/Forçada)**: Se a busca lexical falhar ou for desativada, o sistema parte para a busca vetorial.
    *   **Multi-Query**: A pergunta original é expandida em múltiplas variantes usando sinônimos e aliases definidos em `config/ontology/terms.yml` para aumentar a chance de encontrar documentos relevantes.
    *   **Busca Vetorial (FAISS)**: As variantes da pergunta são convertidas em embeddings e usadas para consultar o índice FAISS, que retorna os chunks de documentos mais similares.
    *   **Reranker (Cross-Encoder)**: Os documentos retornados pelo FAISS são reordenados por um modelo Cross-Encoder (ex: `jina-reranker`) para refinar a relevância contextual. O score do reranker (`0` a `1`) é usado para calcular a confiança da resposta.
5.  **Geração da Resposta**: Os chunks mais relevantes (após o rerank) são enviados para um LLM (Gemini ou OpenAI) junto com um prompt para gerar uma resposta final em linguagem natural.
6.  **Resposta da API**: A API retorna a resposta gerada, as citações dos documentos fonte, e um score de confiança.

### Diagrama (Mermaid)

```mermaid
flowchart LR
    subgraph Client
      U[Usuário]
    end
    subgraph API
      A[Flask API<br/>/query]
      H[Health/Metrics]
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
    subgraph Agent
      TG[Triagem]
      AR[Auto Resolver<br/>(chama RAG)]
      PD[Pedir Info]
    end

    U -->|Pergunta| A
    A -->|triagem opcional| TRI
    TRI -->|ação| TG
    TG -->|AUTO_RESOLVER| AR
    TG -->|PEDIR_INFO| PD

    AR -->|Rota 1| LEX
    LEX -->|se encontrou| GEN
    AR -->|Rota 2| MQ --> VS --> RER --> GEN
    GEN -->|Resposta + Citações + Confiança| A

    H --- A

    %% ETL
    LD --> SPL --> EMB_E --> VS_B --> VS
    VS_B --> DB
    DB -->|incremental| VS_B

    %% Embeddings em runtime
    A --- EMB
    A --- VS
```

## Componentes Principais

*   `api.py`: Ponto de entrada do microserviço. Define a aplicação Flask, os endpoints (`/query`, `/healthz`, `/metrics`, `/debug/*`), e carrega os modelos de embeddings e o índice FAISS na inicialização. Gerencia o estado de prontidão da aplicação.
*   `query_handler.py`: Coração da lógica de RAG. Contém as funções para:
    *   Processar a pergunta do usuário.
    *   Executar a busca lexical e vetorial.
    *   Aplicar o reranker.
    *   Chamar o LLM para gerar a resposta final (`call_llm`).
    *   Carregar e utilizar os termos da ontologia (`terms.yml`).
*   `llm_client.py`: Abstrai a comunicação com os LLMs (Google Gemini e OpenAI). Fornece uma função unificada `call_llm` que lida com as particularidades de cada provedor. A inicialização é *lazy*, ou seja, o cliente só é carregado na primeira chamada.
*   `etl_orchestrator.py`: Orquestra o processo de ETL para construir o índice FAISS. Ele lê os arquivos da pasta `data/`, os divide em chunks, gera os embeddings e salva o índice. Utiliza um banco de dados PostgreSQL para rastrear arquivos processados e seus hashes, permitindo atualizações incrementais.
*   `scripts/etl_build_index.py`: Script alternativo e mais simples para o ETL, focado em reconstruir o índice do zero. É mais flexível em termos de loaders, mas não possui a lógica de atualização incremental do `etl_orchestrator.py`.
*   `agent_workflow.py`: Implementa um fluxo de agente com LangGraph para orquestrar a conversa. Ele adiciona uma camada de triagem antes do RAG, permitindo que o sistema decida se deve tentar responder ou pedir mais informações, tornando a interação mais robusta.
*   `loaders/`: Diretório contendo módulos para carregar diferentes tipos de arquivos (`.pdf`, `.docx`, `.txt`, etc.) e transformá-los em objetos `Document` do LangChain.
*   `config/ontology/terms.yml`: Arquivo de configuração central para a lógica de busca. Define:
    *   `departments`: Mapeia slugs para nomes de departamentos, usado para dar bônus na busca lexical.
    *   `aliases`: Variações de termos.
    *   `synonyms`: Sinônimos e expansões de siglas, usados para gerar as multi-queries.
    *   `boosts`: Pesos para termos específicos.
*   `docker-compose.*.yml` e `Dockerfile.*`: Arquivos para containerização da aplicação com Docker, com variantes para CPU e GPU.





## Tecnologias e Frameworks

*   **Backend**: Python 3.10+, Flask
*   **RAG**: LangChain (para orquestração, loaders, text splitting), FAISS (vector store), HuggingFace Embeddings (modelos de embeddings), Sentence Transformers (Cross-Encoder para reranking).
*   **LLMs**: Google Gemini (via `google.generativeai`), OpenAI (via `openai` Python client).
*   **Orquestração de ETL**: Python, `psycopg2` (para PostgreSQL).
*   **Banco de Dados**: PostgreSQL (para metadados de arquivos processados e chunks).
*   **Containerização**: Docker, Docker Compose.
*   **Configuração**: `python-dotenv`, YAML (para `terms.yml`).
*   **Web UI**: Nginx, HTML, CSS, JavaScript (para a interface de usuário básica).
*   **Agente de IA**: LangGraph (para orquestração de fluxo de trabalho do agente).

## Fluxo de Dados Detalhado

### 1. Ingestão de Dados (ETL)

*   **Fonte**: Arquivos de dados (`.pdf`, `.docx`, `.md`, `.txt`, `.php`, `.sql`, `.json`, `.xml`, `.ini`, `.config`, `.example`, `.yml`, `.yaml`) localizados na pasta `data/`.
*   **Loaders**: Módulos em `loaders/` (ex: `pdf_loader.py`, `docx_loader.py`) carregam o conteúdo dos arquivos. O `etl_orchestrator.py` mapeia extensões de arquivo para os loaders apropriados.
*   **Text Splitting**: O `RecursiveCharacterTextSplitter` divide os documentos carregados em `chunks` menores (tamanho configurável, com sobreposição) para otimizar a busca e a relevância do contexto para o LLM.
*   **Embeddings**: O modelo `HuggingFaceEmbeddings` (configurável via `EMBEDDINGS_MODEL`, padrão `intfloat/multilingual-e5-large`) converte cada chunk de texto em um vetor numérico (embedding).
*   **Vector Store (FAISS)**: Os embeddings são armazenados no índice FAISS (`vector_store/faiss_index`), que permite buscas de similaridade vetorial eficientes.
*   **Metadados (PostgreSQL)**: Informações sobre os arquivos processados (caminho, hash) e os chunks (source_file, chunk_text, faiss_index) são armazenadas em um banco de dados PostgreSQL. Isso permite o rastreamento de arquivos e atualizações incrementais do índice FAISS.

### 2. Processamento de Consulta (API)

*   **Requisição**: O endpoint `/query` em `api.py` recebe a pergunta do usuário.
*   **LLM Client (`llm_client.py`)**: Gerencia a comunicação com os LLMs. Ele carrega o cliente (Google Gemini ou OpenAI) de forma *lazy* e formata as chamadas de API.
*   **Query Handler (`query_handler.py`)**: Contém a lógica central de RAG:
    *   **Pré-processamento**: A pergunta do usuário pode ser expandida em múltiplas variantes (`MQ_ENABLED`) usando sinônimos de `terms.yml`.
    *   **Busca Lexical**: Procura por termos exatos ou similares nos chunks, com bônus para documentos de departamentos relevantes.
    *   **Busca Vetorial**: Consulta o índice FAISS com os embeddings da pergunta (ou suas variantes) para recuperar os `TOP_K` documentos mais relevantes.
    *   **Reranking**: Um modelo `CrossEncoder` (ex: `jinaai/jina-reranker-v2-base-multilingual`) reordena os documentos recuperados, atribuindo um score de relevância contextual. A `CONFIDENCE_MIN` é usada para filtrar respostas de baixa confiança.
    *   **Geração de Resposta**: Os chunks reranqueados são passados para o LLM (via `llm_client.py`) para gerar a resposta final. O LLM também pode ser usado para triagem de intenção ou para gerar perguntas de esclarecimento.
*   **Resposta**: A resposta do LLM, juntamente com as citações e o score de confiança, é retornada ao usuário via API.

### 3. Fluxo do Agente (`agent_workflow.py`)

*   **Estado**: O `AgentState` mantém o contexto da conversa, incluindo a pergunta, histórico de mensagens, resultado da triagem, resposta e citações.
*   **Nó de Triagem (`node_triagem`)**: Usa um LLM para classificar a intenção da pergunta do usuário (`AUTO_RESOLVER` ou `PEDIR_INFO`).
*   **Nó de Auto-Resolução (`node_auto_resolver`)**: Se a triagem for `AUTO_RESOLVER`, este nó chama a função `answer_question` do `query_handler.py` para executar o RAG. Ele também pode condensar o histórico da conversa em uma pergunta autônoma para o RAG.
*   **Nó de Pedir Informações (`node_pedir_info`)**: Se a triagem for `PEDIR_INFO` (ou se o RAG falhar), este nó usa um LLM para formular uma pergunta de esclarecimento ao usuário.
*   **Arestas Condicionais**: O LangGraph define as transições entre os nós com base nas decisões da triagem e no sucesso do RAG.



## Dependências do Projeto

As dependências são gerenciadas via `requirements.txt` e arquivos específicos para CPU/GPU (`requirements-cpu.txt`, `requirements-gpu.txt`).

### `requirements.txt` (Principais)

*   **Web Framework**: `flask`, `flask-cors` (API RESTful e CORS).
*   **Configuração**: `python-dotenv` (carregamento de variáveis de ambiente), `PyYAML` (parsing de `terms.yml`).
*   **LangChain Ecosystem**: `langchain-community`, `langchain-core`, `langchain-text-splitters` (componentes para RAG, LLMs, divisão de texto).
*   **Embeddings & Transformers**: `sentence-transformers`, `huggingface-hub`, `transformers` (modelos de embeddings e reranker).
*   **Vector Store & Similaridade**: `rapidfuzz` (similaridade de strings para busca lexical), `numpy` (operações numéricas).
*   **Processamento de Documentos**: `pypdf` (PDFs), `python-docx` (DOCX), `nltk` (processamento de linguagem natural).
*   **LLM**: `google-generativeai` (SDK para Google Gemini).
*   **Banco de Dados**: `psycopg2` (driver PostgreSQL), `SQLAlchemy` (ORM).
*   **Outros**: `orjson` (JSON de alta performance), `requests`, `tenacity` (retentativas).

### `requirements-cpu.txt` / `requirements-gpu.txt`

Estes arquivos adicionam dependências específicas para o ambiente de execução:

*   **`faiss-cpu` / `faiss-gpu`**: A biblioteca FAISS para busca vetorial, otimizada para CPU ou GPU.
*   **`torch` / `torchvision` / `torchaudio`**: A biblioteca PyTorch, essencial para modelos de deep learning, com versões específicas para CPU ou GPU.

## Configurações Essenciais (`.env.example`)

O projeto utiliza variáveis de ambiente para configuração, carregadas via `python-dotenv`. Um arquivo `.env` deve ser criado com base em `.env.example`.

### Variáveis Chave

*   **`POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`**: Credenciais e detalhes de conexão para o banco de dados PostgreSQL, usado para metadados do ETL.
*   **`DATA_PATH_CONTAINER`**: Caminho dentro do container onde os documentos de entrada para o ETL estão localizados (padrão: `/app/data`).
*   **`GOOGLE_API_KEY` / `GEMINI_API_KEY`**: Chave da API do Google para acesso ao Gemini.
*   **`GOOGLE_MODEL` / `GEMINI_MODEL`**: Nome do modelo Gemini a ser utilizado (padrão: `models/gemini-2.5-flash-lite`).
*   **`OPENAI_API_KEY` / `OPENAI_MODEL`**: (Opcional) Chave e nome do modelo OpenAI para uso como fallback ou alternativa.
*   **`EMBEDDINGS_MODEL`**: Nome do modelo de embeddings do Hugging Face (padrão: `intfloat/multilingual-e5-large`). É crucial que o ETL e a API usem o mesmo modelo.
*   **`FAISS_STORE_DIR` / `FAISS_OUT_DIR`**: Caminho onde o índice FAISS é salvo e carregado (padrão: `/app/vector_store/faiss_index`).
*   **`REQUIRE_LLM_READY`**: Booleano (`true`/`false`) que indica se a API deve aguardar o LLM estar pronto para se considerar `ready` no endpoint `/healthz`.
*   **`RERANKER_ENABLED`**: Ativa/desativa o uso do reranker (padrão: `true`).
*   **`RERANKER_NAME`**: Nome do modelo Cross-Encoder para reranking (padrão: `jinaai/jina-reranker-v2-base-multilingual`).
*   **`RERANKER_CANDIDATES` / `RERANKER_TOP_K`**: Número de candidatos para o reranker e quantos serão retornados.
*   **`RERANKER_DEVICE`**: Dispositivo para o reranker (`cpu` ou `cuda`).
*   **`MQ_ENABLED`**: Ativa/desativa a geração de multi-queries (padrão: `true`).
*   **`MQ_VARIANTS`**: Número de variantes de pergunta a serem geradas.
*   **`CONFIDENCE_MIN`**: Limiar mínimo de confiança (0.0 a 1.0) para considerar uma resposta válida (padrão: `0.20`).
*   **`REQUIRE_CONTEXT`**: Se `true`, exige que um contexto seja encontrado para gerar uma resposta.
*   **`STRUCTURED_ANSWER` / `MAX_SOURCES`**: Configurações para a formatação da resposta e o número máximo de citações.
*   **`DEBUG_LOG` / `DEBUG_PAYLOAD`**: Ativa/desativa logs de debug detalhados.
*   **`ROUTE_FORCE`**: (Diagnóstico) Força uma rota específica (`auto`, `lexical`, `vector`) para debug.

## Ontologia e Sinônimos (`config/ontology/terms.yml`)

Este arquivo YAML é fundamental para a inteligência de busca do RAG, fornecendo um dicionário de termos que enriquece tanto a busca lexical quanto a vetorial.

### Estrutura

*   **`departments`**: Um mapeamento de `slug` (identificador curto) para o nome canônico de departamentos. Usado para aplicar bônus na busca lexical se a pergunta mencionar um departamento e o documento for associado a ele.
    *   Exemplo: `computacao: "Departamento de Computação"`
*   **`aliases`**: Mapeia termos comuns ou grafias alternativas para suas formas canônicas. Ajuda a normalizar a entrada do usuário.
    *   Exemplo: `andreia: ["andrea", "andria", "andréia", "andréa", "andreá"]`
*   **`synonyms`**: Mapeia siglas, abreviações ou termos equivalentes para suas expansões ou sinônimos. Essencial para a geração de multi-queries, garantindo que a busca vetorial explore diferentes formulações da pergunta.
    *   Exemplo: `bcc: "Bacharelado em Ciência da Computação"`
*   **`keywords`**: Uma lista de palavras-chave importantes que podem aparecer em buscas. Não diretamente usado para boosting, mas pode ser útil para análise ou expansão futura.
*   **`boosts`**: Define pesos opcionais por departamento para influenciar a relevância da busca. Permite que certos departamentos tenham seus documentos priorizados.
    *   Exemplo: `department: computacao: 1.15`

### Uso

O `query_handler.py` carrega este arquivo na inicialização para popular os dicionários `DEPARTMENTS`, `ALIASES`, `SYNONYMS` e `BOOSTS`. Estes são então usados para:

1.  **Expansão de Query**: `synonyms` e `aliases` são usados para gerar variações da pergunta original, que são então usadas para consultar o FAISS (multi-query).
2.  **Bônus Lexical**: `departments` e `boosts` são usados para aplicar bônus de pontuação a documentos que correspondem a departamentos relevantes na busca lexical.

Este arquivo é um ponto central para ajustar a precisão e o recall do sistema RAG sem modificar o código-fonte, permitindo uma adaptação rápida a novos domínios de conhecimento ou requisitos de busca.



## Estrutura de Diretórios e Convenções de Código

O projeto segue uma estrutura de diretórios lógica, separando as diferentes funcionalidades em módulos e pastas dedicadas. Isso facilita a organização, a manutenção e a escalabilidade do microserviço.

### Estrutura de Diretórios

```
.env.example
.gitignore
ARCHITECTURE.md
README.md
agent_workflow.py
api.py
config/
├── ontology/
│   └── terms.yml
docker-compose.cpu.yml
docker-compose.gpu.yml
Dockerfile.cpu
Dockerfile.gpu
etl_orchestrator.py
llm_client.py
loaders/
├── __init__.py
├── code_loader.py
├── csv_loader.py
├── docx_loader.py
├── json_loader.py
├── md_loader.py
├── pdf_loader.py
└── txt_loader.py
manage_rag.bat
prompts/
├── pedir_info_prompt.txt
├── resposta_final_prompt.txt
└── triagem_prompt.txt
query_handler.py
rag-microservice.zip
requirements-cpu.txt
requirements-gpu.txt
requirements.txt
scripts/
├── ai_etl_conv_term_cpu.sh
├── ai_etl_conv_term_gpu.sh
├── etl_build_index.py
├── inicia_site_cpu.sh
├── inicia_site_gpu.sh
├── smoke_cpu.sh
├── smoke_gpu.sh
├── treinar_ia_cpu.sh
└── treinar_ia_gpu.sh
tools/
└── check_data_and_loaders.py
triage.py
web_ui/
├── conf.d/
│   └── default.conf
├── html/
│   ├── index.html
│   ├── logo_final.png
│   ├── script.js
│   └── style.css
├── nginx.conf
├── nginx.default.conf
└── web_ui_script.js
```

*   **`./` (Raiz do Projeto)**: Contém os arquivos principais da aplicação (`api.py`, `query_handler.py`, `llm_client.py`, `etl_orchestrator.py`, `agent_workflow.py`), documentação (`README.md`, `ARCHITECTURE.md`), arquivos de configuração de ambiente (`.env.example`, `.gitignore`) e Docker (`Dockerfile.*`, `docker-compose.*.yml`).
*   **`config/`**: Armazena arquivos de configuração. Atualmente, contém a ontologia do sistema em `ontology/terms.yml`.
*   **`loaders/`**: Módulos Python responsáveis por carregar e pré-processar diferentes tipos de documentos (PDF, DOCX, Markdown, texto simples, código, CSV, JSON) para o processo de ETL. Cada arquivo `_loader.py` implementa a lógica específica para seu formato.
*   **`prompts/`**: Contém arquivos de texto com os prompts utilizados pelos LLMs para tarefas como triagem de perguntas e geração de respostas. A separação dos prompts em arquivos facilita a manutenção e o versionamento.
*   **`scripts/`**: Scripts shell e Python para automação de tarefas como a construção do índice ETL, inicialização do site e testes de fumaça. Existem versões específicas para ambientes CPU e GPU.
*   **`tools/`**: Utilitários auxiliares, como `check_data_and_loaders.py`, que podem ser usados para validação ou manutenção do sistema.
*   **`web_ui/`**: Contém os arquivos para a interface de usuário web, incluindo configurações do Nginx (`nginx.conf`, `conf.d/default.conf`), arquivos HTML, CSS e JavaScript (`html/`).

### Convenções de Código

O projeto adota as seguintes convenções para garantir a legibilidade e a manutenibilidade do código:

*   **Python**: Segue as diretrizes do PEP 8 para formatação de código. Utiliza `from __future__ import annotations` para anotações de tipo *forward references*.
*   **Tipagem Estática**: Uso extensivo de *type hints* (`typing` module) para melhorar a clareza do código e permitir a detecção de erros em tempo de desenvolvimento. Exemplos incluem `TypedDict`, `Literal`, `Optional`, `List`, `Tuple`, `Dict`, `Set`, `Iterable`.
*   **Variáveis de Ambiente**: As configurações são carregadas preferencialmente via variáveis de ambiente (`os.getenv`), com valores padrão definidos no código ou em `.env.example`. Isso promove a configuração *12-factor app*.
*   **Logging**: Utiliza `print()` com `flush=True` para logs em tempo real, especialmente em ambientes conteinerizados. Mensagens de log são prefixadas com `[API]`, `[ETL]`, `[LLM]`, `[DICT]`, `[DEBUG]` e `[INFO]`, `[WARN]`, `[ERROR]` para facilitar a filtragem e o monitoramento.
*   **Lazy Loading**: Componentes pesados, como o cliente LLM (`llm_client.py`) e o modelo de reranker (`query_handler.py`), são carregados sob demanda (lazy loading) para otimizar o tempo de inicialização da aplicação e o uso de recursos.
*   **Tratamento de Erros**: Blocos `try-except` são usados para lidar com exceções de forma graciosa, especialmente durante o carregamento de modelos, arquivos e chamadas de API externas. Em casos críticos, as exceções são relançadas (`raise`) para indicar falha.
*   **Modularização**: O código é dividido em módulos lógicos (`api.py`, `query_handler.py`, `llm_client.py`, etc.), cada um com uma responsabilidade clara, promovendo o princípio de responsabilidade única.
*   **Comentários e Docstrings**: O código é bem comentado e utiliza *docstrings* para explicar a finalidade de classes, funções e parâmetros, facilitando a compreensão por outros desenvolvedores.
*   **Deserialização Segura**: Ao carregar o índice FAISS, `allow_dangerous_deserialization=True` é usado, com um aviso explícito sobre a necessidade de cautela ao carregar dados de fontes não confiáveis. Isso é uma consideração importante de segurança.

Essas convenções e a estrutura do projeto são projetadas para criar um sistema robusto, escalável e fácil de manter, que pode ser rapidamente adaptado e estendido para diferentes casos de uso de RAG.



## Como Usar Este Contexto com Assistentes de IA (Codex/Gemini Code Assist)

Este documento foi elaborado para fornecer um contexto abrangente e estruturado para assistentes de IA, como o Codex e o Gemini Code Assist, a fim de otimizar sua capacidade de entender, gerar e depurar código neste projeto RAG Microservice. Ao interagir com o assistente, considere as seguintes diretrizes:

### 1. Forneça o Contexto Relevante

Sempre que possível, inclua seções específicas deste documento em seus prompts. Por exemplo, se estiver trabalhando no `query_handler.py`, mencione a seção 'Componentes Principais' e 'Fluxo de Dados Detalhado' que descrevem esse módulo. Isso ajuda o assistente a focar nas informações mais pertinentes.

### 2. Seja Explícito sobre o Arquivo e a Função

Ao pedir ajuda com um trecho de código, especifique o arquivo (`api.py`, `query_handler.py`, etc.) e a função ou classe relevante. Por exemplo:

"No arquivo `query_handler.py`, na função `_apply_rerank`, estou tentando entender como os scores são normalizados. Explique o trecho de código responsável por isso."

### 3. Utilize a Estrutura de Diretórios

Se precisar de ajuda para localizar um arquivo ou entender a relação entre módulos, faça referência à seção 'Estrutura de Diretórios'.

"Onde eu encontraria os loaders para arquivos `.json`? Qual arquivo em `loaders/` é responsável por isso?"

### 4. Consulte as Configurações e Variáveis de Ambiente

Ao depurar problemas de configuração ou entender o comportamento do sistema, mencione as variáveis de ambiente relevantes ou o arquivo `terms.yml`.

"Estou vendo que o reranker não está sendo ativado. Quais variáveis de ambiente em `.env.example` controlam isso e como devo configurá-las?"

### 5. Peça por Geração de Código com Base em Padrões Existentes

Se precisar adicionar uma nova funcionalidade, instrua o assistente a seguir os padrões de código e as convenções já estabelecidas no projeto.

"Preciso adicionar um novo loader para arquivos `.xml` na pasta `loaders/`. Crie o arquivo `xml_loader.py` seguindo o padrão dos outros loaders, como o `json_loader.py`."

### 6. Use o Diagrama Mermaid para Entender o Fluxo

Se tiver dúvidas sobre o fluxo geral do sistema ou a interação entre componentes, o diagrama Mermaid pode ser uma referência útil. Você pode até pedir ao assistente para explicar uma parte específica do diagrama.

"Explique o que acontece no fluxo de dados quando a `Rota Vetorial` é ativada, com base no diagrama de arquitetura."

### 7. Peça por Explicações Detalhadas

Não hesite em pedir ao assistente para detalhar conceitos ou tecnologias específicas mencionadas neste documento. Por exemplo:

"Explique o conceito de 'Multi-Query' e como ele é implementado neste projeto, referenciando o `query_handler.py` e o `terms.yml`."

### 8. Mantenha o Contexto Atualizado

Se o projeto evoluir, certifique-se de atualizar este arquivo `context.md` para que o assistente de IA continue a fornecer a assistência mais precisa e relevante.

Ao seguir estas diretrizes, você maximizará a eficácia dos assistentes de IA, transformando-os em colaboradores poderosos no desenvolvimento e manutenção do seu RAG Microservice.

### 9. Idioma padrão Português Brasil

Sempre responda em portugues brasileiro.