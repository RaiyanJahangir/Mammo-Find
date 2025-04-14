import json
import subprocess
from flask import Flask, jsonify, render_template_string
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

def extract_key_phrases(text):
    """
    Uses an Ollama model via the CLI to extract meaningful key phrases from the text.
    The model is expected to return a JSON with two arrays: 'nodes' and 'edges'.
    Example JSON format:
      {
          "nodes": [
              {"id": "root", "label": "Main Idea", "shape": "box", "color": "#ffd700"},
              {"id": "node_0", "label": "key phrase 1", "shape": "ellipse", "color": "#87CEEB"},
              ...
          ],
          "edges": [
              {"from": "root", "to": "node_0", "color": "#898980", "weight": 1},
              ...
          ]
      }
    """
    prompt = f"""
Extract the most important keywords, key phrases, or ideas from the following text.
Do not just split sentences; extract the concepts that capture the meaning.
Then output a mindmap as a JSON object with two arrays: 'nodes' and 'edges'.
The 'nodes' array should include:
  - A central node with id "root" and label "Main Idea" (use shape "box" and color "#ffd700").
  - For each key phrase, add a node with a unique id ("node_0", "node_1", ...) and label set to the phrase.
    Use shape "ellipse" and color "#87CEEB" for these nodes.
The 'edges' array should include an edge from the central node "root" to each key phrase node.
Output only valid JSON.

Text: {text}
"""
    # Change the model name to match your installed Ollama model.
    model_name = "mistral-nemo:latest"
    
    # Run the Ollama CLI with the specified prompt.
    # The command might vary; here we assume a command line interface like:
    #     ollama chat <model_name> --prompt "<prompt>"
    try:
        # model = OllamaLLM(model=model_name)
        # response_text = model.invoke(prompt)
        result = subprocess.run(
            ["ollama", "run", model_name, "--prompt", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        # Attempt to load the output as JSON.
        mindmap_data = json.loads(output)
    except Exception as e:
        print("Error calling Ollama or parsing its output:", e)
        # Fallback to a static example.
        mindmap_data = {
            "nodes": [
                {"id": "root", "label": "Main Idea", "shape": "box", "color": "#ffd700"},
                {"id": "node_0", "label": "Example Key Phrase", "shape": "ellipse", "color": "#87CEEB"}
            ],
            "edges": [
                {"from": "root", "to": "node_0", "color": "#898980", "weight": 1}
            ]
        }
    return mindmap_data

# Example input text to be processed.
text = """
Artificial Intelligence (AI) has rapidly transformed industries.
Machine Learning is a subset of AI that provides systems the ability to automatically learn.
Natural Language Processing enables machines to understand human language.
Deep Learning techniques have revolutionized pattern recognition.
Ethics and transparency are critical as AI continues to expand.
"""

# Generate the mindmap data using the Ollama model.
mindmap_data = extract_key_phrases(text)

# Initialize the Flask application.
app = Flask(__name__)

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

# Main route renders the HTML with the embedded mindmap.
@app.route('/')
def index():
    return render_template_string(html_template, mindmap=mindmap_data)

# Optional route to fetch the mindmap data as JSON.
@app.route('/mindmap')
def get_mindmap():
    return jsonify(mindmap_data)

if __name__ == '__main__':
    app.run(debug=True)
