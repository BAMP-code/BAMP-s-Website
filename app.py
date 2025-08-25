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
    preprocessed_input = chatbot.preprocess(user_input) if hasattr(chatbot, 'preprocess') else user_input
    emotions = chatbot.extract_emotion(preprocessed_input) if hasattr(chatbot, 'extract_emotion') else []

    if hasattr(chatbot, 'is_arbitrary') and chatbot.is_arbitrary(preprocessed_input):
        # Irrelevant input: personality + nudge
        response = chatbot.llm_mode_missing_title_message(user_input)
    else:
        # Input is a movie: force immediate recommendation
        llm_history.append({"role": "user", "content": f"I liked the movie: {user_input}"})

        personality_prompt = f"""
        You are a movie-recommending AI with the personality of Bryan Alexis Pineda.
        Be kind, playful, sometimes humorous. 
        **The user just gave a movie they liked or disliked. Immediately give a movie recommendation.**
        Do NOT ask any follow-up questions about the movie or their preferences.  
        Keep the response concise and in Bryan's tone.
        """

        response = stream_llm_to_console(
            messages=llm_history, 
            client=llm_client, 
            stop=DEFAULT_STOP, 
            system_prompt=personality_prompt
        )

        llm_history.append({"role": "assistant", "content": response})

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
