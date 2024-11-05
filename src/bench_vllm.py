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
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 
import re

# Load variables from the .env file
load_dotenv()

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-Math-7B-Instruct", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="medqa", metadata={"help": "the dataset name", "choices":["medmcqa", "medqa", "mmlu"]})
    split: Optional[str] = field(default="test", metadata={"help": "splits to consider seprated by comma"})
    templates_dir: Optional[str] =  field(default="./templates", metadata={"help": "prompt templates directory"})
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    out_name: Optional[str] =  field(default=None, metadata={"help": "output filename"})
    max_samples: Optional[int] = field(default=32, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default="./hf_cache", metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=16000, metadata={"help": "Maximum input sequence length"})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    eval_modality: Optional[str] = field(default='first', metadata={"help": "One of ['keep_correct', 'majority', 'first']. keep_correct: keep all correct predictions, 'majority': consider the prediction obtained from majority voting, 'first': take the first generation and discard the rest"})
    mode: Optional[str] = field(default='cot', metadata={"help": "Inference mode: CoT or TIR", "choices":["cot", "tir"]})
    text_only: Optional[bool] = field(default=False, metadata={"help": 'whether to consider only textual question without images.'})


def exec_code(code):
    # Dictionary to store variables from exec
    exec_locals = {}

    # Execute the code and store variables in exec_locals
    exec(code, {}, exec_locals)

    # Regular expression to capture text within print() statements
    out_variable = re.findall(r'print\((.*?)\)', code)[-1]
    
    return exec_locals[out_variable]

def extract_answer(text):
    #match = re.search(r"\\boxed\{(.+?)\}", text)
    #return match.group(1) if match else ""
    start = text.rfind('\\boxed{')
    offset = 7
    text = text[start+offset:]
    end = text.rfind("}")
    return text[:end] if start >= 0 and end >= 0 else ""

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "llama" in args.model_name.lower():
        terminators = [
            tokenizer.eos_token,
            "<|eot_id|>"
        ]
    elif "Qwen2.5-Math" in args.model_name:
        terminators = ['```output']
    else:
        terminators = None

    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=args.temperature, 
        top_p=1.0, 
        max_tokens=2048, 
        #use_beam_search=False,
        stop=terminators,
        #logprobs=5,
        seed=0
    )
    
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq" if "awq" in args.model_name.lower() else None,
        #download_dir= args.cache_dir,
        enforce_eager=True,
        #max_model_len=,
        trust_remote_code=True
        #max_num_seqs
    )

    dataset = load_dataset('lozziopadredeivizzi/mathematic_games_dataset_en', split="train")
    if args.text_only:
        dataset = dataset.filter(lambda example: example['image'] == None)

    if args.max_samples > 0: 
        dataset = dataset.select(range(args.max_samples))
    
    prompts = []
    for i, item in enumerate(dataset):

        if "Qwen2.5-Math" in args.model_name:
            if args.mode == "cot":
                messages = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": item['question']}
                ]
            elif args.mode == "tir":
                messages = [
                    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": item['question']}
                ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append((item['id'], text))
    
    # save first 10 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    with open(args.out_dir + '/prompts/example_prompts.txt', 'w') as f:
        for i in range(10):
            f.write(prompts[i][1])
            f.write("*"*100+'\n')
       

    batches = [prompts[i:i+args.batch_size] for i in range(i, len(prompts), args.batch_size)]

    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/completions/{model_name}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):
        
        ids = [el[0] for el in batch]
        input_prompts = [el[1] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            #prompt = out.prompt
            completions = [o.text.strip() for o in out.outputs]
            for completion in completions:
                with open(args.out_dir + f"/completions/{model_name}/completions_{args.mode}.jsonl", 'a') as f:
                    json.dump({"id": ids[id_out], "final_answer": extract_answer(completion), "reasoning": completion}, f, ensure_ascii=False)
                    f.write('\n')
