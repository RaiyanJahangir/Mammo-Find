import os
import json
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
OUTPUT_DIR = "output/zero-shot"

PROMPT_TEMPLATE = """ 
You are an AI assistant that provides information about mammogram datasets. 
You will answer them from the relevant document to ensure the response is precise, medically relevant, and well-organized.

# Relevant information from the document: {context}

# Question: {question}

"""

allowable_models = [ "medllama2:latest", "llama3.1:latest", "gemma:7b-instruct", "mistral:7b-instruct", 
         "llama2:latest",  "llama2:13b-chat", "llama2:13b-chat", "tinyllama", "mistral", "mistral-nemo:latest",  "mistrallite:latest","mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b", "deepseek-r1:7b"]

# allowable_models = [ "tinyllama", "mistral", "mistral-nemo:latest",  "mistrallite:latest","mistral-nemo:12b-instruct-2407-q4_K_M", "llama3.2:3b-instruct-q4_K_M", "deepseek-r1:1.5b", "deepseek-r1:7b"]


# allowable_models=[  "mixtral:latest"]

# "vanilj/llama-3-8b-instruct-32k-v0.1:latest",  "qordmlwls/llama3.1-medical:latest",

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def query_rag(query_text: str, model_name: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model=model_name)
    response_text = model.invoke(prompt)

    return response_text

def main():
    input_file = "question.json"
    questions = load_json(input_file)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for model_name in allowable_models:
        print(f"Processing model: {model_name}")
        output_file = os.path.join(OUTPUT_DIR, f"{model_name.replace(':', '_')}.json")
        
        for i, item in enumerate(questions):
            item["generated_answer"] = query_rag(item["question"], model_name)
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1} questions for {model_name}")
        
        save_json(output_file, questions)
        print(f"Answers saved to {output_file}\n")

if __name__ == "__main__":
    main()