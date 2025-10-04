# Guia para Publicar as Imagens Docker

Este passo a passo mostra, em detalhes, como gerar e publicar as imagens do projeto (API + ETL) no registro Docker da sua escolha. O processo foi pensado para ser fácil: basta seguir cada etapa na ordem.

## 1. Requisitos

1. Ter Docker e docker-compose configurados.
2. Estar logado no registry onde você quer guardar as imagens:
   - Docker Hub: `docker login`
   - GitHub Container Registry (GHCR): `docker login ghcr.io`
3. Estar na raiz do repositório (`rag-microservice`).

## 2. Escolha do prefixo e da tag

- **Prefixo** é o nome do repositório onde as imagens serão gravadas. Exemplos:
  - Docker Hub: `seu-usuario/rag`
  - GHCR: `ghcr.io/seu-usuario/rag`
- **Tag** identifica a versão da imagem (por exemplo, a data e hora: `20251001-104516`).

## 3. Rodando o script

Use o comando abaixo substituindo pelo seu prefixo e tag. O parâmetro `--push` faz o upload automático após o build.

```bash
./scripts/build_and_publish_images.sh --prefix SEU_PREFIXO --tag SUA_TAG --push
```

Exemplo real (Docker Hub):
```bash
./scripts/build_and_publish_images.sh --prefix elcelsius/rag --tag $(date +%Y%m%d-%H%M%S) --push
```

> ⚠️ O script monta as imagens para CPU e GPU. A primeira execução pode levar vários minutos porque baixa dependências grandes (Hugging Face, PyTorch etc.).

## 4. Conferindo o resultado

Ao final você verá logs parecidos com:
```
[build-images] Prefixo: elcelsius/rag | Tag: 20251001-104516
[build-images] Build CPU (ai_etl, ai_projeto_api)
...
[build-images] Build GPU (ai_etl, ai_projeto_api)
...
[build-images] Push CPU images
...
[build-images] Push GPU images
...
[build-images] Concluído
```
Se aparecer alguma mensagem de erro (por exemplo falta de login ou problema de rede nos pacotes APT), corrija e rode novamente.

## 5. Usando as imagens publicadas

Depois que as imagens estiverem no registry, escolha **uma** das opções abaixo:

### Opção A – Variáveis de ambiente

```bash
export RAG_IMAGE_PREFIX=elcelsius/rag
export RAG_IMAGE_TAG=20251001-104516
```
Cada novo terminal requer exportar novamente ou definir no shell init.

### Opção B – Editando `.env`

Adicione (ou descomente) as linhas ao `.env`:
```ini
RAG_IMAGE_PREFIX=elcelsius/rag
RAG_IMAGE_TAG=20251001-104516
```

> Sem definir prefixo/tag, o Docker Compose continua construindo imagens locais (`rag-microservice-...:local`).

## 6. Subindo os containers com as imagens novas

```bash
docker-compose -f docker-compose.cpu.yml pull
# ou para GPU
docker-compose -f docker-compose.gpu.yml pull

docker-compose -f docker-compose.cpu.yml up -d ai_projeto_api ai_web_ui
```

A partir daí, o Compose já utiliza `elcelsius/rag-ai_projeto_api:20251001-104516` (e o equivalente para o ETL). O serviço de UI continua usando a imagem oficial `nginx:1.27-alpine`, portanto não precisa de push.

## 7. Outras dicas

- Para publicar novamente, basta escolher outra tag (por exemplo com data/hora diferente) e repetir o comando de build.
- Se quiser voltar às imagens locais, remova/limpe `RAG_IMAGE_PREFIX` e `RAG_IMAGE_TAG` ou comente esses valores no `.env`.
- Caso o build demore muito, certifique-se de estar com boa conexão ou prepare-se para usar o cache local das imagens.
- Se está usando GHCR, lembre-se de criar um Personal Access Token (PAT) com escopo `write:packages` antes de executar `docker login ghcr.io`.

Pronto! Assim você evita rebuilds demorados toda vez que rodar os smoke tests ou pipelines, e mantém as imagens disponíveis para qualquer ambiente.
