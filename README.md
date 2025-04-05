# MammoFind, the Dataset Recommender
An LLM-based Web Visualization tool for Mammogram dataset recommendations 

## To run the code, follow the instructions below

- You will need Python 3.11.7

- You will need Ollama installed on your pc and have some LLMs downloaded

### Clone the git repository
```
git clone https://github.com/RaiyanJahangir/Mammo-Find.git
```

### Go to the root directory of the project
```
cd Mammo-Find
```

### Create a virtual environment 

For Windows
```
py -3.11 -m venv myenv  
```

For Linux
```
python3.11 -m venv myenv 
```

### Activate the virtual environment 

For Windows
```
myenv\Scripts\activate
```

For Linux
```
source myenv/bin/activate
```

### Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

### Now run the Graph_Generator.ipynb

For Windows
```
jupyter nbconvert --to notebook --execute --inplace Graph_Generator.ipynb
```

For Linux
```
jupyter nbconvert --to notebook --execute --inplace Graph_Generator.ipynb
```

### After running the notebook, check index.html to see the generated graph

### Then start the Flask Server by running app.py
This code activates the Flask server and connects the LLM to the web interface for chatting.

For Windows
```
python app.py
```

For Linux
```
python3 app.py
```

