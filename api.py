# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from agent_workflow import compiled_graph  # Importa o agente compilado

app = Flask(__name__)
CORS(app)

print("--- API IA Compilot pronta! ---")


@app.route("/query", methods=["POST"])
def handle_query():
    """Endpoint que recebe uma pergunta, passa para o agente e retorna a resposta final."""
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "A 'question' é obrigatória no corpo do JSON."}), 400

    question = request.json['question']

    print(f"INFO: Recebida nova pergunta: '{question}'")

    # Invoca o agente com a pergunta do usuário
    final_state = compiled_graph.invoke({"pergunta": question})

    # Imprime o estado final do agente para depuração
    print(f"DEBUG: Estado final do agente -> {final_state}")

    # Prepara a resposta final a partir do estado do agente
    response_data = {
        "answer": final_state.get("resposta", "Ocorreu um erro ao processar sua solicitação."),
        "citations": final_state.get("citacoes", []),
        "final_action": final_state.get("acao_final", "ERRO")
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)