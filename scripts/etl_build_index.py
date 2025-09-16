import os, glob, io
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "/app/data/txt"
OUT_DIR  = os.environ.get("FAISS_OUT_DIR", "/app/vector_store/faiss_index")
os.makedirs(OUT_DIR, exist_ok=True)

def read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "ignore")

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
texts, metas = [], []

if paths:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    for p in paths:
        content = read_text(p)
        chunks = splitter.split_text(content)
        for i, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"source": p, "chunk": i + 1})
else:
    texts = ["Base sem documentos. Adicione .txt em /data/txt e recrie o índice."]
    metas = [{"source": "dummy", "chunk": 1}]

print(f"[ETL] Gerando embeddings (sentence-transformers) para {len(texts)} chunks ...", flush=True)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)
vs.save_local(OUT_DIR)
print(f"[ETL] Índice FAISS salvo em: {OUT_DIR}", flush=True)
