# etl_orchestrator.py
# Este script orquestra o processo de ETL (Extract, Transform, Load) para construir e atualizar a base de conhecimento vetorial.
# Ele suporta tanto a reconstrução completa do índice quanto atualizações incrementais.

import os
import re
import torch
import psycopg2
import psycopg2.extras
import sys
import hashlib
import yaml
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
# Importa o módulo de loaders unificado
import loaders

try:
    from cache_backend import invalidate_all_responses
except Exception:  # pragma: no cover - cache opcional
    invalidate_all_responses = None  # type: ignore

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÕES GLOBAIS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("DATA_PATH_CONTAINER", "data/")
if not os.path.exists(DATA_PATH):
    local_data = os.path.join(os.getcwd(), "data")
    if os.path.exists(local_data):
        print(f"WARN: Caminho de dados '{DATA_PATH}' não encontrado. Usando fallback local '{local_data}'.")
        DATA_PATH = local_data
    else:
        print(f"WARN: Nenhum diretório de dados encontrado em '{DATA_PATH}' ou '{local_data}'.")
VECTOR_STORE_PATH = "vector_store/faiss_index"
BATCH_SIZE = 32

DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")

def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except Exception:
        value = default
    if minimum is not None and value < minimum:
        value = minimum
    return value

CHUNK_SIZE = _int_env("CHUNK_SIZE", 300, minimum=1)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 60, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    CHUNK_OVERLAP = max(0, CHUNK_SIZE - 1)

CHUNK_SEPARATORS_RAW = os.getenv("CHUNK_SEPARATORS", "\\n\\n|\\n| |")
CHUNK_SEPARATORS = [sep.replace("\\n", "\n") for sep in CHUNK_SEPARATORS_RAW.split("|") if sep]

# Lista de extensões suportadas pelo loader unificado. Usada para otimizar a busca de arquivos.
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".md", ".txt", ".php", ".sql", ".json", ".xml",
    ".ini", ".config", ".example", ".yml", ".yaml", ".csv"
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{2}\s*)?(?:\(?\d{2}\)?\s*)?(?:\d{4,5}[-\s]?\d{4})")
URL_RE = re.compile(r"https?://[^\s]+")
SIGLA_RE = re.compile(r"\b[A-ZÀ-Ü]{2,6}(?:/[A-ZÀ-Ü]{2,6})*\b")

ABBREVIATIONS = {
    "sr",
    "sra",
    "srta",
    "prof",
    "profa",
    "profª",
    "prof.",
    "dra",
    "dr",
    "d.",
    "etc",
    "dti",
    "staepe",
    "stpg",
}

LINE_BREAK_RE = re.compile(r"\n{2,}")

TERMS_PATH = os.getenv("TERMS_YAML", "config/ontology/terms.yml")

def _load_terms_dictionary(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            return {}
        data.setdefault("departments", {})
        data.setdefault("synonyms", {})
        return data
    except FileNotFoundError:
        print(f"WARN: Dicionário de termos não encontrado em '{path}'. Prosseguindo sem enriquecimento de setor.", flush=True)
        return {}
    except Exception as exc:
        print(f"WARN: Falha ao carregar termos de '{path}': {exc}", flush=True)
        return {}

TERMS_DICT = _load_terms_dictionary(TERMS_PATH)
DEPARTMENTS = {}
DEPARTMENT_LABEL_TO_SLUG: Dict[str, str] = {}
DEPARTMENT_SYNONYMS: Dict[str, str] = {}
for slug, label in (TERMS_DICT.get("departments") or {}).items():
    if slug is None:
        continue
    slug_norm = str(slug).strip().lower()
    label_str = str(label).strip()
    if not slug_norm or not label_str:
        continue
    DEPARTMENTS[slug_norm] = label_str
    DEPARTMENT_LABEL_TO_SLUG[label_str.lower()] = slug_norm
    DEPARTMENT_SYNONYMS[slug_norm] = label_str
    DEPARTMENT_SYNONYMS[label_str.lower()] = label_str
synonyms_dict = TERMS_DICT.get("synonyms") or {}
for slug, expansions in synonyms_dict.items():
    slug_norm = str(slug).strip().lower()
    if slug_norm not in DEPARTMENTS:
        continue
    label = str(DEPARTMENTS[slug_norm])
    for exp in expansions or []:
        exp_norm = str(exp).strip().lower()
        if exp_norm:
            DEPARTMENT_SYNONYMS[exp_norm] = label

PROTECTED_PATTERNS = (EMAIL_RE, PHONE_RE, URL_RE, SIGLA_RE)

def _invalidate_response_cache(reason: str) -> None:
    if invalidate_all_responses is None:
        return
    try:
        removed = invalidate_all_responses()
        print(f"INFO: Cache de respostas invalidado ({removed} chave(s)) após {reason}.")
    except Exception as exc:
        print(f"WARN: Falha ao invalidar cache de respostas: {exc}")

# --- FUNÇÕES DE BANCO DE DADOS E ARQUIVOS ---

def get_db_connection():
    """Estabelece e retorna uma conexão com o banco de dados PostgreSQL."""
    hosts: List[str] = [DB_HOST]
    if DB_HOST not in {"localhost", "127.0.0.1"}:
        hosts.append("localhost")

    last_error: Exception | None = None
    for idx, host in enumerate(hosts):
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=host,
                port=DB_PORT,
            )
            if idx > 0:
                print(f"WARN: Conexão ao Postgres usando host '{DB_HOST}' falhou; utilizando fallback '{host}'.")
            return conn
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except psycopg2.OperationalError as exc:
            last_error = exc
            continue

    raise last_error if last_error is not None else RuntimeError(
        "Não foi possível estabelecer conexão com o Postgres."
    )

def get_file_hash(file_path):
    """Calcula o hash SHA256 de um arquivo para detectar modificações."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def setup_database():
    """Garante que as tabelas necessárias existam no banco de dados."""
    print("INFO: Conectando ao banco de dados para configurar as tabelas...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        source_file VARCHAR(1024) NOT NULL,
                        chunk_text TEXT NOT NULL,
                        faiss_index BIGINT UNIQUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS processed_files (
                        id SERIAL PRIMARY KEY,
                        source_file VARCHAR(1024) UNIQUE NOT NULL,
                        file_hash VARCHAR(64) NOT NULL,
                        processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        print("SUCCESS: Tabelas verificadas/criadas com sucesso.")
    except Exception as e:
        print(f"ERROR: Não foi possível configurar o banco de dados: {e}")
        raise

def get_processed_files_from_db(conn) -> Dict[str, str]:
    """Recupera os arquivos já processados e seus hashes do banco de dados."""
    with conn.cursor() as cur:
        cur.execute("SELECT source_file, file_hash FROM processed_files")
        return {row[0]: row[1] for row in cur.fetchall()}

def get_files_on_disk(path: str) -> Dict[str, str]:
    """Mapeia todos os arquivos suportados no disco e seus hashes."""
    disk_files = {}
    for root, _, files in os.walk(path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_EXTENSIONS:
                file_path = os.path.join(root, filename)
                try:
                    disk_files[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"WARN: Não foi possível calcular o hash de {file_path}: {e}")
    return disk_files


def _token_count(text: str) -> int:
    return len([tok for tok in text.split() if tok])


def _has_protected_span(text: str) -> bool:
    snippet = text or ""
    if not snippet:
        return False
    for pattern in PROTECTED_PATTERNS:
        if pattern.search(snippet):
            return True
    return False


def _guess_sector(text: str, source: str) -> Tuple[str, str]:
    payload = f"{source or ''}\n{ text or '' }".lower()
    for token, label in DEPARTMENT_SYNONYMS.items():
        if token and token in payload:
            slug = DEPARTMENT_LABEL_TO_SLUG.get(label.lower(), "")
            return label, slug
    return "", ""


def _split_long_sentence(text: str) -> List[str]:
    if _has_protected_span(text):
        trimmed = text.strip()
        return [trimmed] if trimmed else []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["; ", ", ", " ", ""],
    )
    docs = splitter.create_documents([text])
    return [doc.page_content.strip() for doc in docs if doc.page_content.strip()]


def _is_sentence_boundary(paragraph: str, idx: int) -> bool:
    prev_idx = idx - 1
    while prev_idx >= 0 and paragraph[prev_idx].isspace():
        prev_idx -= 1
    if prev_idx < 0:
        return False

    end = prev_idx
    start = end
    while start >= 0 and (paragraph[start].isalnum() or paragraph[start] in "-_"):
        start -= 1
    word = paragraph[start + 1 : end + 1].lower()
    if word in ABBREVIATIONS:
        return False

    next_idx = idx + 1
    length = len(paragraph)
    while next_idx < length and paragraph[next_idx].isspace():
        next_idx += 1
    if next_idx >= length:
        return False
    return paragraph[next_idx].isupper()


def _split_paragraph_sentences(paragraph: str) -> List[str]:
    sentences: List[str] = []
    length = len(paragraph)
    start = 0
    idx = 0
    while idx < length:
        char = paragraph[idx]
        if char in ".?!":
            if _is_sentence_boundary(paragraph, idx):
                sentence = paragraph[start : idx + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = idx + 1
        idx += 1
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_sentences_pt(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    blocks = [
        block.strip()
        for block in LINE_BREAK_RE.split(text)
        if block and block.strip()
    ]
    paragraphs: List[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if lines:
            paragraphs.append(" ".join(lines))

    sentences: List[str] = []
    for para in paragraphs:
        sentences.extend(_split_paragraph_sentences(para))

    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def _collect_overlap_sentences(sentences: List[str]) -> List[str]:
    if not sentences or CHUNK_OVERLAP <= 0:
        return []
    overlap_tokens = 0
    selected: List[str] = []
    for sentence in reversed(sentences):
        token_count = _token_count(sentence)
        selected.insert(0, sentence)
        overlap_tokens += token_count
        if overlap_tokens >= CHUNK_OVERLAP:
            break
    return selected


def _normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11:
        return f"({digits[0:2]}) {digits[2:7]}-{digits[7:11]}"
    if len(digits) == 10:
        return f"({digits[0:2]}) {digits[2:6]}-{digits[6:10]}"
    return (value or "").strip()


def _extract_metadata_fields(text: str, source: str) -> Dict[str, object]:
    emails = sorted({match.lower() for match in EMAIL_RE.findall(text)})
    phones_raw = {_normalize_phone(match) for match in PHONE_RE.findall(text)}
    phones = sorted({phone for phone in phones_raw if phone})
    urls = sorted(set(URL_RE.findall(text)))
    siglas = sorted({token for token in SIGLA_RE.findall(text)})
    sector_label, sector_slug = _guess_sector(text, source)
    return {
        "emails": emails,
        "phones": phones,
        "urls": urls,
        "siglas": siglas,
        "sector": sector_label,
        "sector_slug": sector_slug,
    }


def _guess_section(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    header = lines[0]
    if len(header) > 120:
        return header[:117] + "..."
    return header


def _chunk_document(doc: Document) -> List[Document]:
    content = doc.page_content or ""
    sentences = _split_sentences_pt(content)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        token_count = _token_count(sentence)
        if token_count == 0:
            continue

        if token_count > CHUNK_SIZE:
            if current:
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current = []
                current_tokens = 0
            for segment in _split_long_sentence(sentence):
                if segment:
                    chunks.append(segment)
            continue

        if current_tokens + token_count <= CHUNK_SIZE:
            current.append(sentence)
            current_tokens += token_count
            continue

        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
        overlap_sentences = _collect_overlap_sentences(current)
        current = overlap_sentences + [sentence]
        current_tokens = sum(_token_count(s) for s in current)

    if current:
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    if not chunks:
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS or ["\n\n", "\n", " ", ""],
        )
        fallback_docs = fallback_splitter.split_documents([doc])
        return fallback_docs

    total = len(chunks)
    base_metadata = dict(doc.metadata or {})
    source = base_metadata.get("source") or base_metadata.get("file") or ""

    output: List[Document] = []
    for idx, chunk_text in enumerate(chunks):
        metadata = dict(base_metadata)
        metadata["chunk"] = idx
        metadata["chunk_total"] = total
        if source:
            metadata["chunk_source"] = source
        metadata["section"] = _guess_section(chunk_text)
        metadata["chunk_tokens"] = _token_count(chunk_text)
        meta_fields = _extract_metadata_fields(chunk_text, source)
        sector = meta_fields.get("sector")
        sector_slug = meta_fields.get("sector_slug")
        if sector and isinstance(sector, str):
            metadata["sector"] = sector
        if sector_slug and isinstance(sector_slug, str):
            metadata["sector_slug"] = sector_slug
        for key in ("emails", "phones", "urls", "siglas"):
            value = meta_fields.get(key)
            if isinstance(value, list):
                metadata[key] = value
        output.append(Document(page_content=chunk_text, metadata=metadata))

    return output


def _split_documents_portuguese(documents: List[Document]) -> List[Document]:
    results: List[Document] = []
    for doc in documents:
        results.extend(_chunk_document(doc))
    return results

# --- FUNÇÕES DO PIPELINE DE ETL ---

def process_and_embed_documents(docs_to_process: List[Document], embeddings_model) -> Tuple[List[Document], FAISS]:
    """Divide documentos em chunks e gera embeddings para eles."""
    if not docs_to_process:
        return [], None

    print(
        "INFO: %d documentos serão divididos (chunk_size=%d, overlap=%d)."
        % (len(docs_to_process), CHUNK_SIZE, CHUNK_OVERLAP)
    )
    split_chunks = _split_documents_portuguese(docs_to_process)
    print(f"SUCCESS: Documentos divididos em {len(split_chunks)} chunks.")

    if not split_chunks:
        return [], None

    print("INFO: Gerando embeddings...")
    vector_store = FAISS.from_documents(split_chunks, embeddings_model)
    print("SUCCESS: Embeddings gerados com sucesso.")
    return split_chunks, vector_store

def remove_docs_from_store(files_to_remove: Set[str], vector_store: FAISS, conn):
    """Remove os dados de arquivos específicos do índice FAISS e do banco de dados."""
    if not files_to_remove:
        return

    print(f"INFO: Removendo {len(files_to_remove)} arquivo(s) obsoleto(s)/modificado(s)...")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT faiss_index FROM document_chunks WHERE source_file = ANY(%s)",
            (list(files_to_remove),)
        )
        faiss_indices_to_remove = [row[0] for row in cur.fetchall() if row[0] is not None]

        if faiss_indices_to_remove:
            # A remoção de vetores do FAISS pode ser complexa. A biblioteca FAISS não suporta remoção direta por ID de forma eficiente em todos os cenários.
            # A abordagem mais segura para consistência total é um rebuild. No entanto, para muitos casos, a remoção funciona.
            try:
                removed_count = vector_store.delete(faiss_indices_to_remove)
                if removed_count:
                     print(f"INFO: Removidos {len(faiss_indices_to_remove)} vetores do índice FAISS.")
            except Exception as e:
                print(f"WARN: Falha ao remover vetores do FAISS: {e}. Recomenda-se um rebuild para consistência total.")

        cur.execute("DELETE FROM document_chunks WHERE source_file = ANY(%s)", (list(files_to_remove),))
        cur.execute("DELETE FROM processed_files WHERE source_file = ANY(%s)", (list(files_to_remove),))
        conn.commit()
    print(f"SUCCESS: Entradas de {len(files_to_remove)} arquivo(s) removidas do banco de dados.")

def add_docs_to_store(files_to_add: Set[str], vector_store: FAISS, conn, embeddings_model):
    """Processa e adiciona novos documentos ao índice FAISS e ao banco de dados."""
    if not files_to_add:
        return

    print(f"INFO: Adicionando {len(files_to_add)} arquivo(s) novo(s)/modificado(s)...")
    
    docs_to_process = []
    file_hashes = {}
    for file_path in files_to_add:
        # A função unificada load_document lida com os diferentes tipos de arquivo e seus erros.
        loaded_docs = loaders.load_document(file_path)
        if loaded_docs:
            docs_to_process.extend(loaded_docs)
            try:
                file_hashes[file_path] = get_file_hash(file_path)
            except Exception as e:
                 print(f"WARN: Não foi possível calcular o hash de {file_path} após o carregamento: {e}")

    if not docs_to_process:
        return

    split_chunks, new_vector_store = process_and_embed_documents(docs_to_process, embeddings_model)

    if not new_vector_store:
        return

    vector_store.merge_from(new_vector_store)
    print(f"INFO: {len(split_chunks)} novos vetores mesclados ao índice FAISS.")

    with conn.cursor() as cur:
        cur.execute("SELECT max(id) FROM document_chunks")
        last_id = (cur.fetchone()[0] or 0)

        chunk_data = [
            (doc.metadata.get('source', 'desconhecido'), doc.page_content, last_id + 1 + i)
            for i, doc in enumerate(split_chunks)
        ]
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO document_chunks (source_file, chunk_text, faiss_index) VALUES (%s, %s, %s)",
            chunk_data
        )

        file_data = list(file_hashes.items())
        psycopg2.extras.execute_batch(
            cur,
            "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s) ON CONFLICT (source_file) DO UPDATE SET file_hash = EXCLUDED.file_hash, processed_at = CURRENT_TIMESTAMP",
            file_data
        )
        conn.commit()
    print("SUCCESS: Banco de dados atualizado com os novos metadados.")

# --- LÓGICAS DE EXECUÇÃO PRINCIPAIS (REFATORADAS) ---

def run_full_rebuild():
    """Executa um rebuild completo da base de conhecimento de forma eficiente em memória."""
    print("\n--- INICIANDO REBUILD COMPLETO ---")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            print("INFO: Limpando dados antigos (tabelas document_chunks e processed_files)...")
            cur.execute("TRUNCATE TABLE document_chunks, processed_files RESTART IDENTITY;")
            conn.commit()

    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
    
    vector_store = None
    all_files_on_disk = list(get_files_on_disk(DATA_PATH).keys())
    
    for i in range(0, len(all_files_on_disk), BATCH_SIZE):
        batch_paths = all_files_on_disk[i:i+BATCH_SIZE]
        print(f"\n--- Processando lote {i//BATCH_SIZE + 1}/{(len(all_files_on_disk) + BATCH_SIZE - 1)//BATCH_SIZE} ---")
        
        docs_to_process = []
        file_hashes = {}
        for file_path in batch_paths:
            loaded_docs = loaders.load_document(file_path)
            if loaded_docs:
                docs_to_process.extend(loaded_docs)
                try:
                    file_hashes[file_path] = get_file_hash(file_path)
                except Exception as e:
                    print(f"WARN: Não foi possível calcular o hash de {file_path} após o carregamento: {e}")
        
        if not docs_to_process:
            continue

        split_chunks, batch_vector_store = process_and_embed_documents(docs_to_process, embeddings_model)

        if not batch_vector_store:
            continue
            
        if vector_store is None:
            vector_store = batch_vector_store
        else:
            vector_store.merge_from(batch_vector_store)

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                chunk_data = [(doc.metadata.get('source', 'desconhecido'), doc.page_content) for doc in split_chunks]
                psycopg2.extras.execute_batch(cur, "INSERT INTO document_chunks (source_file, chunk_text) VALUES (%s, %s)", chunk_data)
                
                file_data = list(file_hashes.items())
                psycopg2.extras.execute_batch(cur, "INSERT INTO processed_files (source_file, file_hash) VALUES (%s, %s)", file_data)
                conn.commit()

    if vector_store:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, source_file, chunk_text FROM document_chunks ORDER BY id")
                db_chunks = cur.fetchall()
                update_data = [(i, db_chunks[i][0]) for i in range(len(db_chunks))]
                psycopg2.extras.execute_batch(cur, "UPDATE document_chunks SET faiss_index = %s WHERE id = %s", update_data)
                conn.commit()

        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"\nSUCCESS: Novo índice FAISS salvo em '{VECTOR_STORE_PATH}' com {vector_store.index.ntotal} vetores.")
        _invalidate_response_cache("rebuild completo")
    else:
        print("WARNING: Nenhum documento foi processado. O índice não foi criado.")
        _invalidate_response_cache("rebuild sem documentos")

def run_incremental_update():
    """Executa uma atualização incremental real, lidando com arquivos novos, modificados e excluídos."""
    print("\n--- INICIANDO ATUALIZAÇÃO INCREMENTAL ---")

    if not os.path.exists(VECTOR_STORE_PATH):
        print("WARNING: Nenhum índice FAISS existente encontrado. Executando um rebuild completo como fallback.")
        run_full_rebuild()
        return

    with get_db_connection() as conn:
        processed_files_db = get_processed_files_from_db(conn)
        files_on_disk = get_files_on_disk(DATA_PATH)

        disk_path_set = set(files_on_disk.keys())
        db_path_set = set(processed_files_db.keys())

        new_files = disk_path_set - db_path_set
        deleted_files = db_path_set - disk_path_set
        
        potential_modified = disk_path_set.intersection(db_path_set)
        modified_files = {path for path in potential_modified if processed_files_db[path] != files_on_disk[path]}

        if not new_files and not deleted_files and not modified_files:
            print("SUCCESS: Nenhuma alteração detectada. A base de conhecimento já está atualizada.")
            return

        print(f"INFO: Detectado: {len(new_files)} novo(s), {len(modified_files)} modificado(s), {len(deleted_files)} excluído(s).")

        embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={'device': DEVICE})
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)

        files_to_remove = modified_files.union(deleted_files)
        if files_to_remove:
            remove_docs_from_store(files_to_remove, vector_store, conn)

        files_to_add = new_files.union(modified_files)
        if files_to_add:
            add_docs_to_store(files_to_add, vector_store, conn, embeddings_model)

        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"\nSUCCESS: Índice FAISS atualizado e salvo. Total de vetores: {vector_store.index.ntotal}")
        _invalidate_response_cache("atualização incremental")


if __name__ == "__main__":
    setup_database()

    execution_mode = "rebuild"
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        execution_mode = "update"

    if execution_mode == "update":
        run_incremental_update()
    else:
        run_full_rebuild()

    print("\n--- PROCESSO DE ETL CONCLUÍDO ---")
