import json
import os
from transformers import AutoTokenizer, HfArgumentParser
from typing import Optional
from dataclasses import dataclass, field
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

def defy_prompt_messages(model_name, mode, question):
    if "Qwen2.5-Math" or "Mathstral" or "Llama" in model_name:
            if mode == "cot":
                return [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question}
                ]
            elif mode == "tir":
                return [
                    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": question}
                ]
        
    if "deepseek-math" in model_name:
            if mode == "cot":
                return [
                    {"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                ]
            elif mode == "tir":
                return [
                    {"role": "user", "content": question + "\n\nYou are an expert programmer. Solve the above mathematical problem by writing a Python. Express your answer as a numeric type or a SymPy object."}
                ]
                
    if "NuminaMath" or "tora" in model_name:
            if mode == "cot":
                return [
                    {"role": "user", "content": question},
                ]
            elif mode == "tir":
                return [
                    {"role": "user", "content": question},
                ]

def print_chat_messages(input_filename: str, model_tokenizer_id: str, out_dir: str = "out/logs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Using tokenizer: {model_tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_id)
    model_name = model_tokenizer_id.split("/")[-1]
    out_file_path = os.path.join(out_dir, f"messages_{model_name}.txt")

    if not os.path.isfile(input_filename):
        print(f"Error: Input file '{input_filename}' does not exist.")
        return

    if ".jsonl" in input_filename:
        with open(input_filename) as f:
            lines = [json.loads(line) for line in f]

        if hasattr(tokenizer, "apply_chat_template"):
            with open(out_file_path, 'a') as f:
                for line in lines:
                    text = tokenizer.apply_chat_template(line['messages'], tokenize=False, add_generation_prompt=False)
                    f.write(text + "\n" + "*" * 100 + "\n")
        else:
            print("Error: Tokenizer does not have an 'apply_chat_template' method.")

if __name__ == "__main__":
    
    @dataclass
    class ScriptArguments:
        # Arguments for the print_chat_messages function
        print_chat_messages: Optional[bool] = field(default=False, metadata={"help": "Whether to print chat messages from the input filename."})
        input_filename: Optional[str] = field(default="out/completions/Qwen2.5-Math-7B-Instruct/completions_tir.jsonl")
        model_tokenizer_id: Optional[str] = field(default="Qwen/Qwen2.5-Math-7B-Instruct")

    # Login with Hugging Face token
    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Error: HF_TOKEN not found in environment variables.")

    # Parse input arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.print_chat_messages:
        print_chat_messages(args.input_filename, args.model_tokenizer_id)
