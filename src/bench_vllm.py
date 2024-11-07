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

 # 11:03:46 llm_engine.py:161] Initializing an LLM engine (v0.5.0) with config: model='Qwen/Qwen2.5-Math-7B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-Math-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=Qwen/Qwen2.5-Math-7B-Instruct)

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-Math-7B-Instruct", metadata={"help": "model's HF directory or local path"})
    dataset_name: Optional[str] = field(default="lozziopadredeivizzi/mathematic_games_dataset_en")
    out_dir: Optional[str] =  field(default="./out", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=32, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
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

def exec_code(code):
    # Dictionary to store variables from exec
    exec_locals = {}
    try:
        # Execute the code and store variables in exec_locals
        exec(code, {}, exec_locals)
    except Exception as e:
        # Capture and return the error message in JSON format
        return json.dumps({"error": str(e)})
   
    return exec_locals

def extract_answer(text):
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
        top_p=args.top_p, 
        max_tokens=2048 if args.mode == "cot" else 1024, 
        stop=terminators,
        seed=None
    )
    
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq" if "awq" in args.model_name.lower() else None,
        #download_dir=args.cache_dir,
        enforce_eager=True,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus,
    )

    dataset = load_dataset(args.dataset_name, split="train")
    if args.text_only: # to use to ignore images from data
        dataset = dataset.filter(lambda example: example['image'] == None)
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.max_samples))
    
    

    prompts = []
    for i, item in enumerate(dataset):
        # currenlty only Qwen2.5-Math is handled. This part must be adapted for each LLM considered in our tests. Maybe a separate function in a utils folders might help.
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

        prompts.append({
            "id": item['id'], 
            "answer": item['answer'],
            "prompt": text, 
            "chat_history": messages
        })
        #prompts.append((item['id'], text, messages))
    
    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 5 if args.max_samples > 5 else args.max_samples
    with open(args.out_dir + '/prompts/example_prompts.txt', 'w') as f:
        for i in range(n_prompts_to_stamp):
            f.write(f"ID: {prompts[i]['id']}\n")
            f.write(prompts[i]['prompt'])
            f.write("*"*100+'\n')
    
    if args.n_sampling > 0:
        import copy
        batches = [[copy.deepcopy(el) for _ in range(args.n_sampling)] for el in prompts]
    else:
        batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")

    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/completions/{model_name}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):

        if args.mode == "cot":
            ids = [el['id'] for el in batch]
            input_prompts = [el['prompt'] for el in batch]

            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

            for id_out, out in enumerate(outputs):
                completions = [o.text.strip() for o in out.outputs]
                for completion in completions:
                    with open(args.out_dir + f"/completions/{model_name}/completions_{args.mode}.jsonl", 'a') as f:
                        json.dump({"id": ids[id_out], "final_answer": extract_answer(completion), "reasoning": completion}, f, ensure_ascii=False)
                        f.write('\n')

        elif args.mode == "tir":
            batch_data = [batch,[],[],[]]
            id_prompt = batch[0]['id']
            gold_answer = batch[0]['answer']
            for n_round in range(args.n_rounds+1):
                input_prompts = [el['prompt'] for el in batch_data[n_round]]
                messages = [el['chat_history'] for el in batch_data[n_round]]
                
                outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
                for id_out, out in enumerate(outputs):
                    completion = out.outputs[0].text
                    if extract_answer(completion).strip() or n_round == args.n_rounds: # answer found or reached max possible rounds
                        
                        messages[id_out].append({"role": "assistant", "content": completion})
                        
                        with open(args.out_dir + f"/completions/{model_name}/completions_{args.mode}.jsonl", 'a') as f:
                            json.dump({"id": id_prompt, "gold_answer": gold_answer, "final_answer": extract_answer(completion), "messages": messages[id_out]}, f, ensure_ascii=False)
                            f.write('\n')

                    elif "```python" in completion:
                        
                        response = completion.split("```python")[1].split("```")[0]
                        if response.strip():
                            output = exec_code(response)
                            output = tuple(output.values()) if isinstance(output, dict) else output
                            
                        
                        messages[id_out].append({"role": "assistant", "content": completion})
                        messages[id_out].append({"role": "user", "content": f"```output\n{output}\n```"})
                        
                        if n_round < args.n_rounds and messages[id_out]:
                            text = tokenizer.apply_chat_template(
                                messages[id_out],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                            
                            batch_data[n_round+1].append({
                                "id": id_prompt,
                                "prompt": text,
                                "chat_history": messages[id_out]}
                            )
                            
        elif args.mode == "tir_test":  ### IGNORE THIS, only for quick testing
            # generate N sampling for each prompt
            ids = [el['id'] for el in batch]
            input_prompts = [el['prompt'] for el in batch]
            messages = [el['chat_history'] for el in batch]
            

            outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)
            for id_out, out in enumerate(outputs):
                completion = out.outputs[0].text
                current_message = messages[id_out]
                print("Current message FIRST :", current_message)
                current_message.append({"role": "assistant", "content": completion})
                print("Current message THEN:", current_message)
                print("_______________________________________")
                current_message = []
                with open(args.out_dir + f"/completions/{model_name}/completions_prova.jsonl", 'a') as f:
                    json.dump({"id": ids[id_out], "completion": completion, "messages": current_message}, f, ensure_ascii=False)
                    f.write('\n')