import os
import json
import time
import threading
import webbrowser
from flask import Flask, request, jsonify, render_template_string, render_template, Response
from flask_cors import CORS
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from sentence_transformers import SentenceTransformer
import subprocess

# Constants and configuration
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE_ZERO_SHOT = """ 
You are an AI assistant that provides information about mammogram datasets. 
You will answer them from the relevant document to ensure the response is precise, medically relevant, and well-organized.

# Relevant information from the document: {context}

# Question: {question}
"""

PROMPT_TEMPLATE_N_SHOT = """ 
You are an AI assistant that provides information about mammogram datasets. 
You will answer them from the relevant document to ensure the response is precise, medically relevant, and well-organized.

Here are some provided questions and how to answer them:

"question": "What is the largest mammogram dataset available?",
"answer": "The largest mammogram dataset available is the EMory BrEast Imaging Dataset (EMBED)."

"question": "Which is the oldest dataset?",
"answer": "The oldest dataset is the Mammographic Image Analysis Society (MIAS) database, which was created in 1994."

"question": "What is the most common format of mammogram datasets?",
"answer": "The most common format of mammogram datasets is DICOM."

"question": "List me all the publicly available mammogram datasets.",
"answer": "The list of datasets that are publicly available are: DDSM, CBIS-DDSM, RBIS-DDSM, Inbreast, Vindr-Mammo, MIAS, RSNA, CMMD, KAU-BCMD, BCS-DBT, and DMID"

"question": "List me all the datasets that are available upon signing agreements.",
"answer": "The list of datasets that are available upon signing agreements are: EMBED, OPTIMAM, and Vindr-Mammo. However, Vindr-Mammo is also publicly available in another site."

"question": "Give me a comparison between EMBED and DMID dataset.",
"answer": "The EMBED dataset is the largest mammogram dataset available, with 3.4 million images, while the DMID dataset is the most recent, published in 2024, with only 510 images. The EMBED dataset has a racially diverse data with mammogram images, clinical data, and metadata, while the DMID dataset contains mammogram images, metadata, and radiological reports."

"question": "Tell something about EMBED dataset.",
"answer": "Stands for EMory BrEast imaging Dataset. This dataset contains 3,383,659 screening and diagnostic mammogram images from 115,910 patients. Among these, 20% of the total 2D and C-view dataset is available for research use. This 20% contains Total 480,606 dicom images, Total 676,009 png images (20%) and Total 34,004 spot magnified images. It also has 4 files of clinical data and metadata."

"question": "What type of task can I perform with the EMBED dataset?",
"answer": "Breast_Cancer_Detection, Breast_Cancer_Risk_Prediction, Mammographic_Report_Generation, Breast_Cancer_Type_Classification, Breast_Tumor_Classification, Tumor_Localization, Breast_Density_Estimation, Synthetic_Data_Generation"

"question": "Is the DDSM dataset available?",
"answer": "Yes, the DDSM dataset is publicly available at http://www.eng.usf.edu/cvprg/Mammography/Database.html and a mini version is available at https://www.kaggle.com/datasets/skooch/ddsm-mammography"

"question": "Which datasets were collected from the USA?",
"answer": "EMBED, DREAM, DDSM, CBIS-DDSM, RBIS-DDSM, RSNA, BCS-DBT, LLNL datasets were collected from the USA."


# Relevant information from the document: {context}

# Question: {question}

"""

# Load the sentence transformer model for extraction (if needed)
extract_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define allowable models (adjust as needed)
allowable_models = [
    "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", 
    "llama2:latest", "llama2:13b-chat", "tinyllama", "mistral", "mistral-nemo:latest",  
    "mistrallite:latest", "mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", 
    "deepseek-r1:1.5b", "deepseek-r1:7b"
]

def query_rag(query_text: str, model_name: str, prompt_type: str = "zero_shot"):
    """Queries the retrieval-augmented generation (RAG) pipeline and returns the response text."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    if(prompt_type == "zero_shot"):
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ZERO_SHOT)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_N_SHOT)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model=model_name)
    response_text = model.invoke(prompt)
    return response_text


# Global flag to indicate when mindmap.html has been updated.
graph_updated = False

def generate_graph(response, model_name):
    """
    Uses the provided response from the RAG query and generates a mind map JSON  
    by invoking the Ollama model. The generated JSON is then rendered into an HTML  
    file (mindmap.html) using vis-network and includes SSE-based auto-refresh code.
    """
    prompt = """
        Extract the most important keywords, key phrases, or ideas from the following text.
        Do not just split sentences; extract the concepts that capture the meaning.
        Then output a mindmap as a JSON object with two arrays: 'nodes' and 'edges'.
        Do not output anything other than a json object. No extra words outside the json object.
        The 'nodes' array should include:
        - A central node with id "root" and a label representing the main idea in 1-3 words (use shape "box" and color "#ffd700").
        - For each key phrase, add a node with a unique id ("node_0", "node_1", ...) and the label set to the phrase.
        - The other nodes may also be connected to each other if they are related.
          Use shape "box" and color "#87CEEB" for these nodes.
        The 'edges' array should include an edge from the central node "root" to each key phrase node.
        Output only valid JSON.
        
        During describing datasets, try to make nodes out of total number of images in dataset, filetypes, datatypes, country of origin, special properties and so on. For comparative analysis, you may make separate root nodes for each dataset with their features and feature values as other nodes or may connect the dataset name nodes with a main root node. The other nodes may be connected to feature nodes. 

        Example JSON format:
        {
            "nodes": [
                {"id": "root", "label": "Main Idea", "shape": "box", "color": "#ffd700"},
                {"id": "node_0", "label": "key phrase 1", "shape": "box", "color": "#87CEEB"},
                ...
            ],
            "edges": [
                {"from": "root", "to": "node_0", "color": "#898980", "weight": 1},
                ...
            ]
        }
        
        Here is an example given for your understanding:
        {
            "nodes": [
                {"id": "root", "label": "EMBED", "shape": "box", "color": "#ffd700"},
                {"id": "node_0", "label": "3,383,659 Mammogram Images", "shape": "box", "color": "#87CEEB"},
                {"id": "node_1", "label": "115,910 Patients", "shape": "box", "color": "#87CEEB"},
                {"id": "node_2", "label": "20% available for research", "shape": "box", "color": "#87CEEB"},
                {"id": "node_3", "label": "dicom images", "shape": "box", "color": "#87CEEB"},
                {"id": "node_4", "label": "png images", "shape": "box", "color": "#87CEEB"},
                {"id": "node_5", "label": "spot magnified images", "shape": "box", "color": "#87CEEB"},
                {"id": "node_6", "label": "clinical data and metadata", "shape": "box", "color": "#87CEEB"}
                
                ...
            ],
            "edges": [
                {"from": "root", "to": "node_0", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_1", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_2", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_3", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_4", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_5", "color": "#898980", "weight": 1},
                {"from": "root", "to": "node_6", "color": "#898980", "weight": 1},
                {"from": "node_0", "to": "node_3", "color": "#898980", "weight": 1},
                {"from": "node_0", "to": "node_4", "color": "#898980", "weight": 1},
                {"from": "node_0", "to": "node_5", "color": "#898980", "weight": 1},
                ...
            ]
        }
        

        Text:
    """
    prompt += response
    model = OllamaLLM(model=model_name)
    json_result = model.invoke(prompt)
    print("Raw JSON result:", repr(json_result))
    
    # Remove markdown code block markers if they exist
    if json_result.startswith("```json"):
        json_result = json_result[len("```json"):]
    if json_result.endswith("```"):
        json_result = json_result[:-len("```")]
    json_result = json_result.strip()
    print("Cleaned JSON result:", repr(json_result))
    
    # Parse JSON into a Python dictionary
    mindmap_data = json.loads(json_result)
    
    # HTML template for the mindmap page with SSE auto-refresh
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
            var mindmap = {{ mindmap|tojson }};
            var container = document.getElementById("mynetwork");
            var options = {
                physics: { stabilization: { enabled: true, iterations: 1000 } },
                interaction: { hover: true }
            };
            var network = new vis.Network(container, mindmap, options);
        </script>
        
    </body>
    </html>
    """
    
    # Render the HTML using Flask's template engine with the mindmap data,
    # then save it to the file "mindmap.html"
    with app.app_context():
        rendered_html = render_template_string(html_template, mindmap=mindmap_data)
        with open("static/mindmap.html", "w", encoding="utf-8") as html_file:
            html_file.write(rendered_html)
    print("mindmap.html has been created and saved.")

    # Set the global flag so that SSE clients can be notified
    global graph_updated
    graph_updated = True
    
    
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Enable CORS for all routes

def sse_updates():
    global graph_updated
    # Loop continuously, yielding an event only when the file has updated
    while True:
        if graph_updated:
            yield "data: changed\n\n"
            graph_updated = False  # Reset after notifying
        time.sleep(1)  # Check every second


@app.route('/file_updates')
def file_updates():
    return Response(sse_updates(), mimetype='text/event-stream')


def ensure_model_installed(model_name):
    """
    Check if the given model is installed locally.
    If not, pull/install it using the Ollama CLI.
    """
    try:
        list_result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        installed_models = list_result.stdout
        if model_name not in installed_models:
            print(f"Model '{model_name}' not found locally. Pulling model...")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Pull output: {pull_result.stdout}")
        else:
            print(f"Model '{model_name}' is already installed.")
    except subprocess.CalledProcessError as e:
        print("Error checking or pulling model:", e)
    except Exception as ex:
        print("Unexpected error:", ex)
        



@app.route('/')
def index():
    # Serve the index.html file from the static folder.
    return app.send_static_file("index.html")


def open_browser():
    webbrowser.open("http://127.0.0.1:8080")


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()  # Get JSON data from the frontend
    user_message = data.get('prompt')
    model_name = data.get('model') or "llama3.2:3b-instruct-q4_K_M"
    prompt_type = data.get('prompt_type') or "zero_shot"
    if user_message:
        print(f"Using model: {model_name}\n")
        ensure_model_installed(model_name)
        response = query_rag(user_message, model_name, prompt_type)
        print(f"AI: {response}\n")
        threading.Thread(target=generate_graph, args=(response, model_name)).start()
        return jsonify({"response": response})
    else:
        return jsonify({"response": "Error: No message received"}), 400


if __name__ == "__main__":
    # Use a Timer to open the browser only once after a short delay.
    threading.Timer(1, open_browser).start()
    # Disable the reloader to prevent duplicate tabs (if using debug mode).
    app.run(debug=True, host='127.0.0.1', port=8080, threaded=True, use_reloader=False)