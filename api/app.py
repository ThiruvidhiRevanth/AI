from flask import Flask, request, jsonify
from llama_cpp import Llama  # Import Llama model library

app = Flask(__name__)

# Load the model
llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF",
    filename="Llama-3.1-Nemotron-70B-Instruct-HF-IQ1_M.gguf"
)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    response = llm.generate(prompt)
    return jsonify({'response': response})
