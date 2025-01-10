from tqdm import tqdm 
import json
import time
from datetime import datetime


# from openai import OpenAI
import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
import random
import PIL.Image

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import HfArgumentParser
from datasets import load_dataset
from typing import Optional
from dataclasses import dataclass, field


# gpt-4o-mini-2024-07-18 # gpt-4o-2024-08-06 #gemini-1.5-pro #gemini-2.0-flash-exp

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gemini-1.5-flash", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="alecocc/mathematic_games_dataset_en_2024_def")
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    top_k: Optional[float] = field(default=500, metadata={"help": "Top k sampling."})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of prompts to sample for each question"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question without images.'})
    img_only: Optional[bool] = field(default=False, metadata={"help": 'whether to consider only textual question combined with images.'})

    def __post_init__(self):
        if self.text_only and self.img_only:
            raise ValueError("The options 'text_only' and 'img_only' cannot both be True at the same time.")
        if self.model_name == "gemini-2.0-flash-exp" and self.img_only:
            raise ValueError("The model 'gemini-2.0-flash-exp' does not currently support image inputs.")
        

def extract_answer(text):
    start = text.rfind('\\boxed{')
    offset = 7
    text = text[start+offset:]
    end = text.rfind("}")
    return text[:end] if start >= 0 and end >= 0 else ""

if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    now = datetime.now()
    # Format the date and time as a string
    output_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'out/batch_api/gemini/{output_dir}', exist_ok=True)
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename=f"out/batch_api/gemini/{output_dir}/batch.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logger.info(args)
    MODEL_NAME =  args.model_name 

    if MODEL_NAME == "gemini-2.0-flash-exp":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction="You are a mathematical expert. Solve the user's problem by reasoning step by step, and enclose the final answer in \\boxed{}.",
            generation_config=genai.GenerationConfig(
            max_output_tokens=2048,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        ))

        model_info = genai.get_model(f"models/{MODEL_NAME}")
        logger.info(f"DEFAULT PARAMS SETTING:\n{model_info}")
    
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

    for i, item in enumerate(tqdm(dataset)): 
        prompt = "Problem: " + item['question']
        id = item['id']
        answer = item['answer']

        for k in range(args.n_sampling):
            response = ""
            if args.text_only:

                if MODEL_NAME == "gemini-2.0-flash-exp":
                    response = client.models.generate_content(
                        model=MODEL_NAME, 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                        system_instruction="You are a mathematical expert. Solve the user's problem by reasoning step by step, and enclose the final answer in \\boxed{}.",
                        temperature=args.temperature,
                        max_output_tokens=2048,
                        top_p=args.top_p,
                        )
                    )
                else:
                    response = model.generate_content(prompt)
            
            elif args.img_only:

                if MODEL_NAME == "gemini-2.0-flash-exp":
                    raise ValueError("The model 'gemini-2.0-flash-exp' does not support image-only inputs.")
                else:
                    try:
                        input_img = PIL.Image.open(f"jpg_images/image_{id}.jpg")
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Image file 'jpg_images/image_{id}.jpg' not found.")
                    except Exception as e:
                        raise RuntimeError(f"An error occurred while opening the image: {e}")
                    
                    response = model.generate_content([prompt, input_img])

            with open(f"out/batch_api/gemini/{output_dir}/completions_{MODEL_NAME}_{args.mode}.jsonl", 'a') as f:
                json.dump({"id": id, "gold_answer": answer, "final_answer": extract_answer(response.text), "reasoning": response.text}, f, ensure_ascii=False)
                f.write('\n')
            
            time.sleep(5)
    
