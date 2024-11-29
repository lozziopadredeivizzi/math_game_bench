import torch
import json
import os
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
from typing import Optional, List, NamedTuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict 
from huggingface_hub import hf_hub_download
from PIL.Image import Image

import re
import sys
import io
import traceback
import multiprocessing
import signal
import warnings

# Load variables from the .env file
load_dotenv()


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="microsoft/Phi-3.5-vision-instruct", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="lozziopadredeivizzi/mathematic_games_dataset_en")
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=32, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default=None, metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=-1, metadata={"help": "Maximum input sequence length"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    n_sampling: Optional[int] = field(default=1, metadata={"help": "Number of solutions to generate for a given prompt"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=True, metadata={"help": 'whether to consider only textual question without images.'})
    n_gpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to use for inference."})
    n_rounds: Optional[int] = field(default=3, metadata={"help": "Number of gpus to use for inference."})
    gguf_filename: Optional[str] = field(default='', metadata={"help": "gguf filename to download from HuggingFace"})
    original_model_name: Optional[str] = field(default='', metadata={"help": "orginal name of the model gguf quantized. es "})

class ModelRequestData(NamedTuple):
    llm: LLM
    prompt: str
    stop_token_ids: Optional[List[str]]
    image_data: List[Image]
    chat_template: Optional[str]

def load_phi3v(dataset):
    
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        #max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"num_crops": 16},
    )
    
    requests = []
    stop_token_ids = None
    for item in dataset:
        img = item['image']
        question = item['question']
        placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate([1], start=1))
        prompt = f"<|user|>\n{placeholders}\nYou are a mathematical expert. Solve the given problem by reasoning step by step.\n\nProblem: {question.strip()}\n\nPlease, make sure to enclose your final answer within \\boxed{{}}.<|end|>\n<|assistant|>\n"
        requests.append(
            ModelRequestData(
            llm=llm,
            prompt=prompt,
            stop_token_ids=stop_token_ids,
            image_data=[img],#[fetch_image(url) for url in image_urls],
            chat_template=None,
            )
        )
    
    return requests

model_example_map = {
    "Phi-3.5-vision-instruct": load_phi3v,
}

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="output.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    dataset = load_dataset('alecocc/mathematic_games_dataset_en', split="train")
    dataset = dataset.filter(lambda example: example['image'] != None)
    dataset = dataset.select(range(32))

    req_data = model_example_map[args.model_name.split("/")[-1]](dataset)
    
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=2048,
        stop_token_ids=req_data[0].stop_token_ids)

    batches = [req_data[i:i+args.batch_size] for i in range(0, len(req_data), args.batch_size)]

    for batch in tqdm(batches):
        
        requests_in_batch = [{
            "prompt": req.prompt,
            "multi_modal_data": {
                "image": req.image_data
            },
        } for req in batch]

        outputs = req_data[0].llm.generate(
            requests_in_batch,
            sampling_params=sampling_params,
            use_tqdm=False)

        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
            print("*"*50)
