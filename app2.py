import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from flask import Flask, request, jsonify
from flask_cors import CORS


# Load the dataset
df = pd.read_excel("Mammography_Dataset.xlsx")

# Extract all the columns 
documents = df.values.astype(str).tolist()  # Converts all columns to a list of strings


# Initialize the SentenceTransformer model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the documents
document_embeddings = embedding_model.encode(documents, convert_to_numpy=True)


# Initialize a FAISS index for fast retrieval
dimension = document_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean) metric

# Add the document embeddings to the FAISS index
index.add(np.array(document_embeddings))

# Function to retrieve the top-k most relevant documents
def retrieve_documents(query, top_k=15):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the top-k documents based on the indices
    retrieved_docs = [documents[i] for i in indices[0]]

    # Flatten the list if necessary
    flat_retrieved_docs = [item for sublist in retrieved_docs for item in sublist]
    return flat_retrieved_docs



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the Mistral model for generation (you can use the model you have locally)
model = OllamaLLM(model="llama3.2")

# Define the conversation template for the prompt
template = """ 
If the users do not ask anything about mammograms or simply say Hi/Hello, just act like a normal model.

Question: Tell me about EMBED dataset?
Answer: It's a racially diverse data. 3,383,659  screening and diagnostic mammogram images from 115 910 patients. Among these, 20 percent of the total 2D and C-view dataset and is available for research use. Its available based on signing agreement.

Here is the conversation history: {context}

Question: {question}

Please provide a concise answer to the following question, using no more than 70 words:

Answer:
"""
# Want answers about: {retrieved_context}

# Global conversation context
context=""

# Create the prompt from the template
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# This is a POST route for chat, which you already have
@app.route('/chat', methods=['POST'])
def chat():
    global context
    data = request.get_json()  # Get the JSON data sent from the frontend
    user_message = data.get('prompt')  # Extract the 'prompt' sent from the frontend
    
    if user_message:
        # Retrieve relevant documents for the query
        retrieved_docs = retrieve_documents(user_message, top_k=10)  # You can adjust the number of retrieved docs
        retrieved_context = "\n".join(retrieved_docs)  # Combine the retrieved documents to create context\
        # context = "\n".join(retrieved_docs)  # Combine the retrieved documents to create context

        # result = chain.invoke({"context": retrieved_context, "question": user_message})
        result = chain.invoke({"context": retrieved_context, "question": user_message,})
        context += f"\nUser: {user_message}\nAI: {result}"  # Update context
        return jsonify({"response": result})
    else:
        return jsonify({"response": "Error: No message received"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
