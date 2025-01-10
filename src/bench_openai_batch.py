from openai import OpenAI
import torch
import json
import os
import base64
import re
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 
from datetime import datetime

#codellama/CodeLlama-34b-Instruct-hf #bigcode/starcoder2-15b-instruct-v0.1 #mistralai/Codestral-22B-v0.1 #deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct #TheBloke/CodeLlama-70B-Instruct-AWQ #casperhansen/llama-3-70b-instruct-awq #hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-4o-mini-2024-07-18", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="alecocc/mathematic_games_dataset_en_2024_def")
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of prompts to sample for each question"})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=False, metadata={"help": 'whether to consider only textual question without images.'})
    img_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question combined with images.'})

    def __post_init__(self):
        if self.text_only and self.img_only:
            raise ValueError("The options 'text_only' and 'img_only' cannot both be True at the same time.")
        
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":
    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    client = OpenAI(
        api_key=OPENAI_KEY
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/batch_api/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/batch_api/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    MODEL_NAME =  args.model_name 

    dataset = load_dataset(args.dataset_name, split="train")
    if args.text_only: # to use to ignore images from data
        dataset = dataset.filter(lambda example: example['image'] == None)
    
    if args.img_only: # to use to ignore images from data
        dataset = dataset.filter(lambda example: example['image'] != None)
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, args.max_samples))
    
    if args.start_idx > 0 and args.max_samples < 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    logger.info(f"First sample:\n{dataset[0]}")
    #######################################
    #### 1. Preparing Your Batch File #####
    #######################################
    
    total_promtps = 0
    
    for i, item in enumerate(tqdm(dataset)): 
        
        batch_request = {"custom_id": "", "method": "POST", "url": "/v1/chat/completions", "body": {"model": args.model_name, "messages": [{"role": "system", "content": "You are a mathematical expert. Solve the user's problem by reasoning step by step, and enclose the final answer in \\boxed{}."},], "temperature": args.temperature, "max_tokens": 2048}}
        prompt = item['question']
        id = item['id']
        
        if args.text_only:
            batch_request['body']["messages"].append({"role": "user", "content": prompt})

        if args.img_only:
            image_path = f"jpg_images/image_{id}.jpg"
            base64_image = encode_image(image_path)
            batch_request['body']["messages"].append({
                "role": "user",
                "content":[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }]
            })

        for k in range(args.n_sampling):
            batch_request["custom_id"] = f"request-{id}-{k}"
            with open(f'out/batch_api/{output_dir}/input_batch.jsonl', 'a') as f:
                json.dump(batch_request, f, ensure_ascii=False)
                f.write("\n")
                total_promtps+=1
        
    logger.info(f"UNIQUE PROMPTS: {total_promtps / args.n_sampling}")
    logger.info(f"TOTAL PROMPTS: {total_promtps}")


    batch_input_file = client.files.create(
    file=open(f"out/batch_api/{output_dir}/input_batch.jsonl", "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Running batch inference for Math benchmark."
        }
    )
    logger.info(batch_obj)

    batch_id = batch_obj.id
    logger.info(f"BATCH ID: {batch_id}")

    with open(f'out/batch_api/{output_dir}/batch_id.txt', 'w') as f:
        f.write(batch_id)

    