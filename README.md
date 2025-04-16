# MammoFind, the Dataset Recommender
An LLM-based Web Visualization tool for Mammogram dataset recommendations 

## To run the code, follow the instructions below

- Install Python 3.11.7

- Install Ollama
  
## 1) Clone the git repository
```
git clone https://github.com/RaiyanJahangir/Dataset-Recommender.git
```

## 2) Go to the root directory of the project
```
cd Dataset-Recommender
```

## 3) Create a virtual environment 

For Windows
```
py -3.11 -m venv myenv  
```

For Linux
```
python3.11 -m venv myenv 
```

## 4) Activate the virtual environment 

For Windows
```
myenv/Scripts/activate
```

For Linux
```
source myenv/bin/activate
```

## 5) Install all the necessary packages and libraries
```
pip install -r requirements.txt
```

## 6) Run and check the tool. 
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

