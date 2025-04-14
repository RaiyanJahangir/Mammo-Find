import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from pyvis.network import Network
from flask import Flask, request, jsonify
from flask_cors import CORS


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """ 
You are an AI assistant that provides information about mammogram datasets. 
You will answer them from the relevant document to ensure the response is precise, medically relevant, and well-organized.

# Relevant information from the document: {context}

# Question: {question}
"""

# Load the sentence transformer model
extract_model= SentenceTransformer("all-MiniLM-L6-v2")

allowable_models = [
    "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", 
    "llama2:latest", "llama2:13b-chat", "tinyllama", "mistral", "mistral-nemo:latest",  
    "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", 
    "deepseek-r1:1.5b", "deepseek-r1:7b"
]

def query_rag(query_text: str, model_name: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model=model_name)
    response_text = model.invoke(prompt)

    return response_text

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Get the JSON data sent from the frontend
    user_message = data.get('prompt')  # Extract the 'prompt' sent from the frontend
    # print("Welcome to the Mammogram Dataset Chatbot! Type /bye to exit.")
    # print("Available models:")
    # for i, model in enumerate(allowable_models):
    #     print(f"{i+1}. {model}")
    
    # model_choice = input("Select a model by entering the corresponding number: ")
    # try:
    #     model_choice = int(model_choice) - 1
    #     if model_choice < 0 or model_choice >= len(allowable_models):
    #         raise ValueError
    # except ValueError:
    #     print("Invalid selection. Using default model: mistral-nemo.")
    #     model_choice = allowable_models.index("mistral-nemo:latest")
    
    if user_message:
      # model_choice="mistral-nemo:latest"
    
      # model_name = allowable_models[model_choice]
      model_name="mistral-nemo:latest"
      print(f"Using model: {model_name}\n")      
          
      response = query_rag(user_message, model_name)
      print(f"AI: {response}\n")
      
      return jsonify({"response": response})
    
    else:
        return jsonify({"response": "Error: No message received"}), 400
        

if __name__ == "__main__":
     app.run(debug=True,host='127.0.0.1', port=8080)