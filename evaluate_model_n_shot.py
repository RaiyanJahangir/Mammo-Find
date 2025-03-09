import os
import json
import csv
from evaluate import load

# Load metrics
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")
meteor = load("meteor")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_metrics(actuals, predictions):
    bleu_score = bleu.compute(predictions=predictions, references=[[act] for act in actuals])
    rouge_score = rouge.compute(predictions=predictions, references=actuals)
    bert_score = bertscore.compute(predictions=predictions, references=actuals, lang="en")
    meteor_score = meteor.compute(predictions=predictions, references=actuals)
    
    return {
        "BLEU": bleu_score["bleu"],
        "ROUGE-L": rouge_score["rougeL"],
        "BERTScore_Precision": sum(bert_score["precision"]) / len(bert_score["precision"]),
        "BERTScore_Recall": sum(bert_score["recall"]) / len(bert_score["recall"]),
        "BERTScore_F1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        "METEOR": meteor_score["meteor"]
    }

def save_results_csv(csv_file, model_name, results):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "BLEU", "ROUGE-L", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1", "METEOR"])
        writer.writerow([model_name, results["BLEU"], results["ROUGE-L"], results["BERTScore_Precision"], results["BERTScore_Recall"], results["BERTScore_F1"], results["METEOR"]])

def main():
    input_dir = "/mnt/data1/raiyan/Mammo-Find/output/n-shot"
    output_dir = "/mnt/data1/raiyan/Mammo-Find/evaluation/n-shot"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "evaluation_results.csv")
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            data = load_json(file_path)
            
            actuals = [item["answer"] for item in data]
            predictions = [item["generated_answer"] for item in data]
            
            results = evaluate_metrics(actuals, predictions)
            
            model_name = os.path.splitext(file_name)[0]  # Extract model name
            output_file = os.path.join(output_dir, f"{model_name}_evaluation.json")
            
            with open(output_file, 'w', encoding='utf-8') as out_file:
                json.dump(results, out_file, indent=4)
            
            save_results_csv(csv_file, model_name, results)
            
            print(f"Evaluation saved for {file_name} -> {output_file} and updated in {csv_file}")

if __name__ == "__main__":
    main()


