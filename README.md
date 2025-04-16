# MammoFind, the Dataset Recommender
An LLM-based Web Visualization tool for Mammogram dataset recommendations 

## 1) Installations

- Install Python 3.11.7

- Install Ollama by visiting [https://ollama.com/](https://ollama.com/) and follow the installation instructions.

- You can pull the following models beforehand from your command prompt since they are used in the given version of the tool:

### For embedding
```
ollama pull nomic-embed-text
```

### For chatting
```
ollama pull mistral-nemo:latest
```

The other models are: 
- llama3.2:3b-instruct-q4_K_M
- tinyllama
- medllama2:latest
- gemma:7b-instruct
- Other models you want to use
  
## 2) Clone the git repository
```
git clone https://github.com/RaiyanJahangir/Dataset-Recommender.git
```

## 3) Go to the root directory of the project
```
cd Dataset-Recommender
```

## 4) Create a virtual environment 

For Windows
```
py -3.11 -m venv myenv  
```

For Linux
```
python3.11 -m venv myenv 
```

## 5) Activate the virtual environment 

For Windows
```
myenv/Scripts/activate
```

For Linux
```
source myenv/bin/activate
```

## 6) Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

## 7) Create the knowledge base
For Windows
```
python populate_database.py
```

For Linux
```
python3 populate_database.py
```

## 8) Run and check the tool. 
If you want to directly evaluate results, skip this step. 

### Start the Flask Server 
This code activates the Flask server and connects the LLM to the web interface for chatting.

For Windows
```
python app.py
```

For Linux
```
python3 app.py
```

After running the code, the tool will open in your browser.


## 9) Passing queries to models and generating answers
### 9a) Zero-shot prompting
For Windows
```
python query_data_zero_shot.py
```

For Linux
```
python3 query_data_zero_shot.py
```

### 9b) N-shot prompting
For Windows
```
python query_data_n_shot.py
```

For Linux
```
python3 query_data_n_shot.py
```

Check the output folder.

## 10) Record performance
### 10a) Zero-shot prompting
For Windows
```
python evaluate_model_zero_shot.py
```

For Linux
```
python3 evaluate_model_zero_shot.py
```

### 10b) N-shot prompting
For Windows
```
python evaluate_model_n_shot.py
```

For Linux
```
python3 evaluate_model_n_shot.py
```

Check the evaluation folder

## 11) Deactivate Virtual Environment and wrap up
```
deactivate
```