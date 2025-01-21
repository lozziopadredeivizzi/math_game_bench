from openai import OpenAI
import torch
import json
import os
import re
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 

#deepseek-chat #gpt-4o-2024-08-06

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="deepseek-reasoner", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="alecocc/mathematic_games_dataset_en_2024_def")
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=10, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question without images.'})
    
#"gpt-4o-2024-08-06"

def make_completion(instruction):
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages = [
                {"role": "system","content": "You are a mathematical expert. Solve the user's problem by reasoning step by step, and enclose the final answer in \\boxed{}."},
                {"role": "user", "content": f"Problem:\n{instruction}"}
            ],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            seed=42
        )
        
        return response
    except Exception as e:
        print(e)
        return ""

def make_completion_deepseek_r(instruction):
    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages = [
                {"role": "user", "content": f"Problem:\n{instruction}"}
            ],
        )
        
        return response
    except Exception as e:
        print(e)
        return ""

def extract_answer(text):
    start = text.rfind('\\boxed{')
    offset = 7
    text = text[start+offset:]
    end = text.rfind("}")
    return text[:end] if start >= 0 and end >= 0 else ""


if __name__ == "__main__":

    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="out_bench.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if "deepseek" in args.model_name.lower():
        client = OpenAI(
            api_key=DEEPSEEK_KEY,
            base_url="https://api.deepseek.com"
        )
    else:
        client = OpenAI(
            api_key=OPENAI_KEY
        )

    MODEL_NAME = args.model_name 

    dataset = load_dataset(args.dataset_name, split="train")
    if args.text_only: # to use to ignore images from data
        dataset = dataset.filter(lambda example: example['image'] == None)
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, args.max_samples))
    
    if args.start_idx > 0 and args.max_samples < 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))
    
    os.makedirs('out/completions/openai', exist_ok=True)
    
    for i, item in enumerate(tqdm(dataset)): 
        prompt = item['question']
        response = make_completion(prompt)
        completion = response.choices[0].message.content.strip()
        if args.model_name  == "deepseek-reasoner":
            reasoning_content = response.choices[0].message.reasoning_content.strip()

        model = response.model
        usage = dict(response.usage) if args.model_name  != "deepseek-reasoner" else ""
        with open(f'out/completions/openai/completion_{MODEL_NAME}_cot.jsonl', 'a') as f:    
            result = {
                "model": model,
                "id": item['id'],
                "gold_answer": item['answer'],
                "final_answer": extract_answer(completion),
                "reasoning": reasoning_content if args.model_name  == "deepseek-reasoner" else completion,
                "usage": usage if completion else {},
            }
            if args.model_name  == "deepseek-reasoner":
                result['answer'] = completion

            json.dump(result, f, ensure_ascii=False)
            f.write('\n')
