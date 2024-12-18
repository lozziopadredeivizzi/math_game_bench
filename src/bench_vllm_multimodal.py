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
from transformers import AutoTokenizer, HfArgumentParser, AutoProcessor
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
    text_only: Optional[bool] = field(default=False, metadata={"help": 'whether to consider only textual question without images.'})
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

def extract_answer(text):
    start = text.rfind('\\boxed{')
    offset = 7
    text = text[start+offset:]
    end = text.rfind("}")
    return text[:end] if start >= 0 and end >= 0 else ""

def load_qwen2_vl(dataset):
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = "Qwen/Qwen2-VL-7B-Instruct"

    # Tested on L40
    llm = LLM(
        model=model_name,
        max_model_len=32768 if process_vision_info is None else 4096,
        #max_num_seqs=5,
        limit_mm_per_prompt={"image": 1},
    )

    requests = []
    for item in dataset:
        img = item['image']
        question = item['question']

        placeholders = [{"type": "image", "image": img}]
        messages = [{
            "role": "system",
            "content": "You are a mathematical expert. Solve the given problem by reasoning step by step. Please, make sure to enclose your final answer within \\boxed{{}}."
        }, {
            "role":
            "user",
            "content": [
                *placeholders,
                {
                    "type": "text",
                    "text": f"Problem: {question.strip()}\n\nLet's think step by step. Remember to enclose your final answer within \\boxed{{}}."
                },
            ],
        }]
        

        processor = AutoProcessor.from_pretrained(model_name)

        prompt = processor.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

        stop_token_ids = None

        if process_vision_info is None:
            image_data = [img]
        else:
            image_data, _ = process_vision_info(messages)

        requests.append({
            "id": item['id'], 
            "answer": item['answer'],
            "request":  ModelRequestData(
                llm=llm,
                prompt=prompt,
                stop_token_ids=stop_token_ids,
                image_data=image_data,#[fetch_image(url) for url in image_urls],
                chat_template=None,
            )}
        )

    return requests

def load_phi3v(dataset):
    
    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"num_crops": 16},
    )
    
    requests = []
    stop_token_ids = None
    for item in dataset:
        img = item['image']
        question = item['question']
        gold_answer =  item['answer'],
        placeholders = "\n".join(f"<|image_{i}|>"
                             for i, _ in enumerate([1], start=1))
        prompt = f"<|user|>\n{placeholders}\nYou are a mathematical expert. Solve the given problem by reasoning step by step.\n\nProblem: {question.strip()}\n\nPlease, make sure to enclose your final answer within \\boxed{{}}.<|end|>\n<|assistant|>\n"
        requests.append({
            "id": item['id'], 
            "answer": item['answer'],
            "request":  ModelRequestData(
                llm=llm,
                prompt=prompt,
                stop_token_ids=stop_token_ids,
                image_data=[img],#[fetch_image(url) for url in image_urls],
                chat_template=None,
            )}
        )
    
    return requests

model_example_map = {
    "Phi-3.5-vision-instruct": load_phi3v,
    "Qwen2-VL-7B-Instruct": load_qwen2_vl,
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

    dataset = load_dataset(args.dataset_name, split="train")
    
    dataset = dataset.filter(lambda example: example['image'] != None)
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, args.max_samples))
    
    if args.start_idx > 0 and args.max_samples < 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    req_data = model_example_map[args.model_name.split("/")[-1]](dataset)

    sampling_params = SamplingParams(
        n=args.n_out_sequences,
        temperature=args.temperature,
        top_p = args.top_p,
        max_tokens=2048,
        stop_token_ids=req_data[0]['request'].stop_token_ids,
        seed=0)


    batches = [req_data[i:i+args.batch_size] for i in range(0, len(req_data), args.batch_size)]

    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/completions/multimodal/{model_name}", exist_ok=True)
    eval_mode_str = "pass_1" if args.n_out_sequences == 1 else f"maj_{args.n_out_sequences}"

    with open(args.out_dir + f'/prompts/{model_name}_example_prompts.txt', 'w') as f:
        for i in range(5):
            prompt = req_data[i]['request'].prompt
            f.write(f"ID: {req_data[i]['id']}\n")
            f.write(prompt)
            f.write("*"*100+'\n')


    for batch in tqdm(batches):
        ids = [el['id'] for el in batch]
        batch_requests = [el['request'] for el in batch]
        gold_answers = [el['answer'] for el in batch]

        requests_in_batch = [{
            "prompt": req.prompt,
            "multi_modal_data": {
                "image": req.image_data
            },
        } for req in batch_requests]

        outputs = req_data[0]['request'].llm.generate(
            requests_in_batch,
            sampling_params=sampling_params,
            use_tqdm=False)

        for id_out, output in enumerate(outputs):
            for out in output.outputs:
                completion = out.text
                with open(args.out_dir + f"/completions/multimodal/{model_name}/completions_{args.mode}_{eval_mode_str}.jsonl", 'a') as f:
                    json.dump({"id": ids[id_out], "gold_answer": gold_answers[id_out], "final_answer": extract_answer(completion), "reasoning": completion}, f, ensure_ascii=False)
                    f.write('\n')
