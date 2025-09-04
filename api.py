# api.py
import os
import torch
import faiss
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from query_handler import find_answer_for_query # Importamos nossa lógica

app = Flask(__name__)

# --- CARREGAMENTO DOS MODELOS (FEITO UMA SÓ VEZ) ---
# Para alta performance, carregamos os modelos na memória quando a API inicia.
print("--- Carregando modelos... a API estará pronta em breve. ---")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store/faiss_index"

embeddings_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': DEVICE}
)
index = faiss.read_index(f"{VECTOR_STORE_PATH}/index.faiss")
print(f"--- Modelos carregados no dispositivo: {DEVICE}. API pronta! ---")
# ---------------------------------------------------------

@app.route("/query", methods=["POST"])
def handle_query():
    """Endpoint que recebe uma pergunta e retorna a resposta da IA."""
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "A 'question' é obrigatória no corpo do JSON."}), 400

    question = request.json['question']
    
    print(f"INFO: Recebida nova pergunta: '{question}'")
    
    # Chama nossa função de RAG com os modelos já carregados
    final_answer = find_answer_for_query(question, embeddings_model, index)

    return jsonify({"answer": final_answer})

if __name__ == "__main__":
    # Roda o servidor Flask, acessível por outros containers na rede Docker
    app.run(host="0.0.0.0", port=5000)