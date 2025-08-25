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

def run_programming_mode(user_input):
    """
    Handles a single user message:
    - Preprocess input
    - Extract movie title
    - Detect emotions
    - Check relevance
    - Respond with personality and immediate movie recommendation
    """

    # 1. Preprocess the input
    preprocessed_input = chatbot.preprocess(user_input) if hasattr(chatbot, 'preprocess') else user_input

    # 2. Extract movie title (if function exists)
    if hasattr(chatbot, 'extract_title'):
        movie_title = chatbot.extract_title(preprocessed_input)
    else:
        movie_title = preprocessed_input  # fallback: assume entire input is a title

    # 3. Detect emotions
    emotions = chatbot.extract_emotion(preprocessed_input) if hasattr(chatbot, 'extract_emotion') else []

    # 4. Check if input is irrelevant (arbitrary)
    if hasattr(chatbot, 'is_arbitrary') and chatbot.is_arbitrary(preprocessed_input):
        response = chatbot.llm_mode_missing_title_message(user_input)
        return response

    # 5. Compose the message for the LLM
    llm_message = f"I liked the movie: {movie_title}"

    # Add to LLM history
    llm_history.append({"role": "user", "content": llm_message})

    # 6. Get personality-based recommendation
    response = chatbot.llm_personality_response(llm_message)

    # Append detected emotions (optional)
    if emotions:
        response += f" (Detected emotions: {', '.join(emotions)})"

    # 7. Add bot response to history
    llm_history.append({"role": "assistant", "content": response})

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
