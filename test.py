import json

def load_datasets_from_json(file_path):
    """Reads dataset descriptions from a JSON file and ensures missing keys don't cause errors."""
    with open(file_path, "r", encoding="utf-8") as file:
        dataset_list = json.load(file)  # Load JSON into a list of dictionaries
    
    return dataset_list

# Load dataset descriptions
dataset_file_path = "/mnt/data1/raiyan/Mammo-Find/dataset.json"
datasets = load_datasets_from_json(dataset_file_path)

# Example usage with missing key handling
for dataset in datasets:
    print(f"Dataset Name: {dataset.get('Name', 'Unknown Name')}")
    print(f"Information: {dataset.get('Information', 'No description available')}")
    print(f"Is Data Available?: {dataset.get('Is Data Available?', 'Unknown')}")
    print(f"Data Link: {dataset.get('Data Link', 'No link provided')}")
    print(f"Types of Data: {dataset.get('Types of data in dataset', 'Not specified')}")
    print("-" * 50)
