from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import Chatbot
from util import load_together_client, DEFAULT_STOP, stream_llm_to_console

app = Flask(__name__)
CORS(app)

# Initialize chatbot in programming mode
chatbot = Chatbot(llm_enabled=True)
llm_client = load_together_client()
llm_history = [{
    "role": "system",
    "content": chatbot.llm_system_prompt(),
}]

# Track user movies
user_movies = []

def run_programming_mode(user_input):
    """
    Mimics REPL LLM programming mode:
    - Extract emotions
    - Check relevance
    - Manage personality-based responses
    """
    # Preprocess input if chatbot supports it
    preprocessed_input = chatbot.preprocess(user_input) if hasattr(chatbot, 'preprocess') else user_input

    # 1. Extract emotion (optional, but REPL programming mode does this)
    if hasattr(chatbot, 'extract_emotion'):
        emotions = chatbot.extract_emotion(preprocessed_input)
    else:
        emotions = []

    # 2. Check if input is irrelevant (arbitrary)
    if hasattr(chatbot, 'is_arbitrary') and chatbot.is_arbitrary(preprocessed_input):
        response = chatbot.llm_mode_missing_title_message(user_input)
        return response

    # 3. Collect movies until we have 5
    if len(user_movies) < 5:
        user_movies.append(user_input)
        if len(user_movies) < 5:
            return f"Got it! Please tell me {5 - len(user_movies)} more movies you liked or disliked."
        else:
            return "Awesome! Now I can start giving you recommendations based on your movies."

    # 4. Generate personality-based response
    if hasattr(chatbot, 'llm_personality_response'):
        response = chatbot.llm_personality_response(user_input)
    else:
        # fallback: direct LLM call
        llm_history.append({"role": "user", "content": user_input})
        response = stream_llm_to_console(messages=llm_history, client=llm_client, stop=DEFAULT_STOP)
        llm_history.append({"role": "assistant", "content": response})

    # 5. Optionally, include emotion info in debug (if you want)
    if emotions:
        response += f" (Detected emotions: {', '.join(emotions)})"

    return response


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Please send a message!"})

    response = run_programming_mode(user_message)
    return jsonify({"response": response})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
