from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import subprocess

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



def get_embedding_function():
    model_name = "nomic-embed-text"
    ensure_model_installed(model_name)
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings