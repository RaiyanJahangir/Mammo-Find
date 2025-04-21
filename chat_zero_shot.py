import os
import json
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from get_embedding_function import ensure_model_installed
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from pyvis.network import Network


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

def extract_meaningful_nodes(text, top_n=10):
    """Extract key phrases as nodes using sentence similarity."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]  # Split into sentences and clean up
    if len(sentences) == 0:
        return []  # If no sentences found, return empty list

    embeddings = extract_model.encode(sentences, convert_to_tensor=True)

    # Compute similarity between all pairs
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

    # Select the top N most distinct sentences as key nodes
    scores = similarity_matrix.mean(dim=1)  # Average similarity per sentence
    top_indices = scores.argsort(descending=True)[:min(top_n, len(sentences))]  # Prevent out-of-range

    nodes = [sentences[i] for i in top_indices]
    
    # print("Extracted Nodes:", nodes)  # Debugging: Print extracted nodes
    return list(set(nodes))  # Remove duplicates

def generate_edges(nodes):
    """Creates edges based on co-occurrence of nodes."""
    if len(nodes) < 2:
        return []  # No edges possible if fewer than 2 nodes
    edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    return edges

def create_graph_dataframe(text):
    """Processes text, extracts nodes and edges, and creates a DataFrame."""
    nodes = extract_meaningful_nodes(text)
    
    if not nodes:
        print("No meaningful nodes found.")  # Debugging output
        return pd.DataFrame(columns=["Node1", "Node2"])  # Return empty DataFrame

    edges = generate_edges(nodes)

    if not edges:
        print("Only one node found, no edges can be formed.")  # Debugging output
        return pd.DataFrame(columns=["Node1", "Node2"])  # Return empty DataFrame

    df = pd.DataFrame(edges, columns=["Node1", "Node2"])
    return df

def chat():
    print("Welcome to the Mammogram Dataset Chatbot! Type /bye to exit.")
    print("Available models:")
    for i, model in enumerate(allowable_models):
        print(f"{i+1}. {model}")
    
    model_choice = input("Select a model by entering the corresponding number: ")
    try:
        model_choice = int(model_choice) - 1
        if model_choice < 0 or model_choice >= len(allowable_models):
            raise ValueError
    except ValueError:
        print("Invalid selection. Using default model: mistral.")
        model_choice = allowable_models.index("mistral")
    
    model_name = allowable_models[model_choice]
    print(f"Using model: {model_name}\n")
    ensure_model_installed(model_name)
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "/bye":
            print("Goodbye!")
            break
        
        response = query_rag(user_input, model_name)
        print(f"AI: {response}\n")
        
        
        # Generate graph DataFrame
        df_graph = create_graph_dataframe(response)
        # print(df_graph)
        
        # Create the network
        net = Network(directed=False, height="100%", width="100%")
        
        nodes_1 = df_graph['Node1'].values
        nodes_2 = df_graph['Node2'].values

        
        # Add nodes to the network with fixed positions
        for i in range(0, len(nodes_1)):
            # Add the node for data_Name
            net.add_node(nodes_1[i], title=nodes_1[i], color='#87CEEB' )
            
            # Add the node for Task
            # net.add_node(nodes_2[i], label=nodes_2[i], color="#FFDF00")
            net.add_node(nodes_2[i], title=nodes_2[i], color="#87CEEB")
            
            # Add the edges between nodes
            net.add_edge(nodes_1[i], nodes_2[i], weight=1, color="#898980")
            
        # Generate the HTML file with the network graph
        net.write_html('Sample_Graph.html')
        

if __name__ == "__main__":
    chat()