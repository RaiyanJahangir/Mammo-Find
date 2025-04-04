import os
import numpy as np
import pandas as pd
import json
import faiss
import logging
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# ==============================
# 1. LOAD AND PROCESS dataset.json
# ==============================
# def load_datasets_from_json(file_path):
#     """Reads dataset descriptions from a JSON file with variable structures and stores them in a list."""
#     with open(file_path, "r", encoding="utf-8") as file:
#         datasets = json.load(file)

#     dataset_list = []

#     for dataset in datasets:
#         # Extracting the 'Name' field, assuming it's the identifier for the dataset
#         dataset_name = dataset.get("Name", "Unknown Name").strip()  # Default to "Unknown Name" if not present

#         # Extracting other fields if available
#         dataset_info = {
#             "name": dataset_name,
#             "information": dataset.get("Information", "No information available").strip(),
#             "can_be_used_for": dataset.get("Can be used for", "No usage information").strip(),
#             "is_data_available": dataset.get("Is Data Available?", "Unknown availability").strip(),
#             "data_link": dataset.get("Data Link", "No link provided").strip(),
#             "associated_article": dataset.get("Associated Article", "No associated article").strip(),
#             "data_article_published_on": dataset.get("Data/Article published on", "Unknown date").strip(),
#             "article_available_at": dataset.get("Article available at", "No article link provided").strip(),
#             "types_of_data": dataset.get("Types of data in dataset", "No data types specified").strip(),
#             "types_of_files": dataset.get("Types of files in dataset", "No file types specified").strip(),
#             "data_collected_from": dataset.get("Data collected from", "Unknown location").strip(),
#         }

#         # If the dataset contains 'Derived from', include that field if it's available
#         derived_from = dataset.get("Derived from")
#         if derived_from:
#             dataset_info["derived_from"] = derived_from.strip()

#         # Append the dataset information to the dataset list
#         dataset_list.append(dataset_info)

#     return dataset_list

# # Load dataset descriptions
# dataset_file_path = "/mnt/data1/raiyan/Mammo-Find/dataset.json"  # Change this if needed
# datasets = load_datasets_from_json(dataset_file_path)

# Load the dataset
df = pd.read_excel("Mammography_Dataset.xlsx")

# Extract all the columns 
documents = df.values.astype(str).tolist()  # Converts all columns to a list of strings


# ==============================
# 2. EMBEDDING & FAISS INDEXING
# ==============================

# # Extract all dataset fields and concatenate them into a single string for each dataset
# documents = []

# for dataset in datasets:
#     # Concatenate all available fields as a string (excluding None values)
#     dataset_info = []
    
#     # Loop over all keys in the dataset
#     for key, value in dataset.items():
#         if value:  # Only add non-empty values
#             dataset_info.append(f"{key}: {value}")
    
#     # Join all fields into a single string and add it to the documents list
#     documents.append("\n".join(dataset_info))

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the documents (dataset entries)
document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Initialize FAISS index
dimension = document_embeddings.shape[1]  
index = faiss.IndexFlatL2(dimension)  
index.add(np.array(document_embeddings))

# Document Retrieval
def retrieve_documents(query, top_k=15):
    """Retrieves the top-k most relevant documents using FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the most relevant documents based on the indices from FAISS
    retrieved_docs = [documents[i] for i in indices[0]]
    
    # # Flatten the list if necessary
    # flat_retrieved_docs = [item for sublist in retrieved_docs for item in sublist]
    # return flat_retrieved_docs
    return retrieved_docs


# ==============================
# 3. LLM SETUP & PROMPT
# ==============================

# Initialize the LLM model
temp=0
model = OllamaLLM(model="mistral-nemo:latest",temperature=temp)

# Define the chat template
template = """ 
You are an AI assistant that provides information about mammogram datasets. 
You will answer them from the relevant document to ensure the response is precise, medically relevant, and well-organized.

Here is the conversation history: {conversation_context}  

Question: {question}

Answer: {document_context}

"""

# Here is the conversation history: {conversation_context}  #conversion context is actually not important here.

# Relevant information from the document: {document_context}

# Question: {question}

# Question: {question}

# Answer: {conversation_context}


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Global conversation context
conversation_context = ""

# ==============================
# 4. TERMINAL CHAT FUNCTION
# ==============================

# Terminal chat function
def handle_terminal_chat():
    """Runs the chatbot in the terminal."""
    global conversation_context
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot session ended.")
            break

        # Retrieve relevant documents based on the user's question
        retrieved_documents = retrieve_documents(user_input, top_k=10)  # Retrieve related data
        context= "\n".join(retrieved_documents)
        # Use both conversation context and the document content in the prompt
        result = chain.invoke({
            "conversation_context": conversation_context,
            "document_context": context,
            "question": user_input
        })


        # Output the result from the model
        print("Bot:", result)

        # Update conversation context to include the new question-answer pair
        conversation_context += f"\nUser: {user_input}\nAI: {result}"

# ==============================
# 5. FLASK API SETUP
# ==============================

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot interaction."""
    global conversation_context
    data = request.json
    user_message = data.get('prompt', '')

    # Validate user message
    if not user_message:
        return jsonify({"response": "No message provided!"}), 400

    # Retrieve relevant documents based on the user's message
    retrieved_context = retrieve_documents(user_message, top_k=10)

    # Invoke the model with both conversation context and document-based context
    result = chain.invoke({
        "conversation_context": conversation_context,  # Add conversation history
        "document_context": retrieved_context,  # Add retrieved document content
        "question": user_message  # User's current question
    })

    # Update the global conversation context with the new Q&A pair
    conversation_context += f"\nUser: {user_message}\nAI: {result}"

    # Return the response to the user
    return jsonify({"response": result})

# ==============================
# 6. ENTRY POINT & MODE SELECTION
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI chatbot in terminal or API mode.")
    parser.add_argument("--mode", choices=["terminal", "api"], default="terminal",
                        help="Run chatbot in 'terminal' mode or start API with 'api' mode.")

    args = parser.parse_args()

    if args.mode == "terminal":
        handle_terminal_chat()
    else:
        app.run(port=8080, debug=True)
