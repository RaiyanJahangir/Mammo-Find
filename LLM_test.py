from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize chatbot model
template = """
Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="mistral")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Global conversation context
conversation_context = ""

# Initialize Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot interaction."""
    global conversation_context
    data = request.json
    user_message = data.get('prompt', '')  # Get user input from request
    
    if not user_message:
        return jsonify({"response": "No message provided!"}), 400

    try:
        result = chain.invoke({"context": conversation_context, "question": user_message})
        conversation_context += f"\nUser: {user_message}\nAI: {result}"  # Update context
        return jsonify({"response": result}), 200
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

def handle_terminal_chat():
    """Allows chatting with the AI directly from the terminal."""
    global conversation_context
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot session ended.")
            break

        result = chain.invoke({"context": conversation_context, "question": user_input})
        print("Bot:", result)

        # Update conversation context
        conversation_context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI chatbot in terminal or as an API.")
    parser.add_argument("--mode", choices=["terminal", "api"], default="terminal",
                        help="Run chatbot in 'terminal' mode or start Flask API with 'api' mode.")

    args = parser.parse_args()

    if args.mode == "terminal":
        handle_terminal_chat()
    else:
        app.run(port=5000, debug=True)

# # Terminal run
# python3 LLM_test.py --mode terminal

# # API run (flask server)
# python3 LLM_test.py --mode api