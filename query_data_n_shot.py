import os
import json
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function
from get_embedding_function import ensure_model_installed

CHROMA_PATH = "chroma"
OUTPUT_DIR = "output/n-shot"

PROMPT_TEMPLATE = """ 
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

    ensure_model_installed(model_name)
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