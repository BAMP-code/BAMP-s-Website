from flask import Flask, request, jsonify
from chatbot import Chatbot
from util import load_together_client, DEFAULT_STOP, stream_llm_to_console

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import Chatbot
from util import load_together_client, DEFAULT_STOP, stream_llm_to_console

app = Flask(__name__)
CORS(app)

# Initialize chatbot and LLM client
chatbot = Chatbot(llm_enabled=True)
llm_client = load_together_client()
llm_history = [{
    "role": "system",
    "content": chatbot.llm_system_prompt(),
}]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Please send a message!"})

    # Add user message to LLM history
    llm_history.append({"role": "user", "content": user_message})

    # Call LLM
    response = stream_llm_to_console(messages=llm_history, client=llm_client, stop=DEFAULT_STOP)

    # Add assistant response to history
    llm_history.append({"role": "assistant", "content": response})

    return jsonify({"reply": response})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
