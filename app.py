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

user_movies = []

def run_programming_mode(user_input):
    preprocessed_input = chatbot.preprocess(user_input) if hasattr(chatbot, 'preprocess') else user_input

    # Extract movie title
    movie_title = chatbot.extract_title(preprocessed_input) if hasattr(chatbot, 'extract_title') else None

    # Detect emotions
    emotions = chatbot.extract_emotion(preprocessed_input) if hasattr(chatbot, 'extract_emotion') else []

    # Check if input is irrelevant
    if hasattr(chatbot, 'is_arbitrary') and chatbot.is_arbitrary(preprocessed_input):
        return chatbot.llm_mode_missing_title_message(user_input)

    # Track movies
    if movie_title and movie_title not in user_movies:
        user_movies.append(movie_title)

    # Decide whether to give a recommendation
    if len(user_movies) >= 1:
        response = chatbot.llm_personality_recommendation(preprocessed_input, user_movies)
    else:
        response = chatbot.llm_personality_response(preprocessed_input)

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
