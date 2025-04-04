import json

def txt_to_json(input_txt_file, output_json_file):
    """Converts a colon-separated text file into a list of JSON objects, handling empty lines as separate objects."""
    data_list = []
    current_obj = {}
    
    with open(input_txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  # If the line is empty, store the current object and reset
                if current_obj:
                    data_list.append(current_obj)
                    current_obj = {}
            else:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    current_obj[key] = value
    
    if current_obj:  # Add the last object if it's not empty
        data_list.append(current_obj)
    
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)
    
    print(f"JSON file saved as {output_json_file}")

# Example usage
input_file = "dataset.txt"  
output_file = "dataset.json"
txt_to_json(input_file, output_file)
