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
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading


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
  
  
def generate_graph(response, model_name):
    
    prompt = """
        Extract the most important keywords, key phrases, or ideas from the following text.
        Do not just split sentences; extract the concepts that capture the meaning.
        Then output a mindmap as a JSON object with two arrays: 'nodes' and 'edges'.
        The 'nodes' array should include:
        - A central node with id "root" and label with the main idea or concept in 1 or 2 or 3 words (use shape "box" and color "#ffd700").
        - For each key phrase, add a node with a unique id ("node_0", "node_1", ...) and label set to the phrase.
            Use shape "ellipse" and color "#87CEEB" for these nodes.
        The 'edges' array should include an edge from the central node "root" to each key phrase node.
        Output only valid JSON.
        
        Example JSON format:
        {
            "nodes": "[
                {"id": "root", "label": "Main Idea", "shape": "box", "color": "#ffd700"},
                {"id": "node_0", "label": "key phrase 1", "shape": "ellipse", "color": "#87CEEB"},
                ...
            ]",
            "edges": "[
                {"from": "root", "to": "node_0", "color": "#898980", "weight": 1},
                ...
            ]"
        }

        Text:
    """
    
    # HTML template for displaying the mindmap using vis-network.
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Mindmap Display</title>
        <!-- Vis-Network CSS and JS -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.css"
            integrity="sha512-WgxfT5LWjfszlPHXRmBWHkV2eceiWTOBvrKCNbdgDYTHrT2AeLCGbF4sZlZw3UMN3WtL0tGUoIAKsu8mllg/XA=="
            crossorigin="anonymous" referrerpolicy="no-referrer" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"
                integrity="sha512-LnvoEWDFrqGHlHmDD2101OrLcbsfkrzoSpvtSQtxK3RMnRV0eOkhhBN2dXHKRrUU8p2DGRTk35n4O8nWSVe1mQ=="
                crossorigin="anonymous" referrerpolicy="no-referrer"></script>
        <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
        }
        #mynetwork {
            width: 100vw;
            height: 100vh;
            border: 1px solid lightgray;
        }
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script>
        // Mindmap data passed from Flask.
        var mindmap = {{ mindmap|tojson }};
        var container = document.getElementById("mynetwork");
        var options = {
            physics: { stabilization: { enabled: true, iterations: 1000 }},
            interaction: { hover: true }
        };
        var network = new vis.Network(container, mindmap, options);
        </script>
    </body>
    </html>
    """
    
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format( response=response)
    prompt+=response
    model = OllamaLLM(model=model_name)
    json_result = model.invoke(prompt)
    print("Raw JSON result:", repr(json_result))
    
    # Remove the markdown code block markers if they exist:
    if json_result.startswith("```json"):
        json_result = json_result[len("```json"):]

    if json_result.endswith("```"):
        json_result = json_result[:-len("```")]

    # Also, strip any leading/trailing whitespace
    json_result = json_result.strip()
    
    print("Cleaned JSON result:", repr(json_result))
    
    mindmap_data = json.loads(json_result)
    
    with app.app_context():
        rendered_html = render_template_string(html_template, mindmap=mindmap_data)
        # Save the rendered HTML to a file or further processing.
        with open("mindmap.html", "w", encoding="utf-8") as html_file:
            html_file.write(rendered_html)
    
    
    
    
    
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
      
      # Start the background task concurrently before returning the response
      threading.Thread(target=generate_graph, args=(response,model_name)).start()
      
      return jsonify({"response": response})
    
    else:
        return jsonify({"response": "Error: No message received"}), 400
        

if __name__ == "__main__":
     app.run(debug=True,host='127.0.0.1', port=8080)