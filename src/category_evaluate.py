import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

load_dotenv()

def main(file_path, dataset_name):
    try:
        df_model = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Il file '{file_path}' non è stato trovato.")

    dataset = load_dataset(dataset_name, split="train")
    df_hf = pd.DataFrame(dataset)

    merged_df = pd.merge(df_model, df_hf[['id', 'difficulty', 'category']], on='id', how='left')
    
    os.makedirs("./out", exist_ok=True)
    model_name = file_path.split('/')[2] 
    parts = file_path.split('/')
    
    out_path = './out/category_accuracies.txt'
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("### Models Category accuracy file ###\n")

    with open(out_path, 'a') as f:
        if parts[4] == 'tir' or parts[4] == 'cot' or parts[4] == 'vision':
            f.write(f"\n\nModel: {model_name}\nMode: {parts[3]}, {parts[4]}\n")
        else:
            f.write(f"\n\nModel: {model_name}\nMode: {parts[3]}\n")
        
        yes_count = merged_df[merged_df['model_response'] == 'yes'].shape[0]
        accuracy = yes_count / merged_df.shape[0]
        accuracy_str = f"Global Accuracy: {accuracy:.2%}"   
        f.write(accuracy_str + '\n')
        
        easy_df= merged_df[merged_df['difficulty'] == 'easy']
        shape = easy_df.shape[0]
        yes_count = easy_df[easy_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Easy Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        
        medium_df= merged_df[merged_df['difficulty'] == 'medium']
        shape = medium_df.shape[0]
        yes_count = medium_df[medium_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Medium Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        
        hard_df= merged_df[merged_df['difficulty'] == 'hard']
        shape = hard_df.shape[0]
        yes_count = hard_df[hard_df['model_response'] == 'yes'].shape[0]
        
        accuracy = yes_count / shape
        accuracy_str = f"Hard Difficulty Accuracy: {accuracy:.2%}"
        
        f.write(accuracy_str + '\n')
        
        for category in ['CE', 'C1', 'C2', 'L1', 'L2', 'GP', 'HC']:
            category_df = merged_df[merged_df['category'].str.contains(category, na=False)]
            shape = category_df.shape[0]
            yes_count = category_df[category_df['model_response'] == 'yes'].shape[0]
            
            accuracy = yes_count / shape
            accuracy_str = f"{category} Category Accuracy: {accuracy:.2%}"
            
            f.write(accuracy_str + '\n')

if __name__ == "__main__":
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    parser = argparse.ArgumentParser(description="Calcola l'accuratezza per categoria da un dataset.")
    parser.add_argument("--file_path", type=str, required=True, help="Percorso del file CSV contenente i dati del modello.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Nome del dataset su Hugging Face.")
    args = parser.parse_args()

    main(args.file_path, args.dataset_name)
