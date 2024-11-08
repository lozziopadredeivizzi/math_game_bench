# International Mathematical Games Championships (IMGC)
This repository contains code for evaluating Large Language Models (LLMs) on MathGB, the first benchmark designed specifically for the International Mathematical Games Championships.

# Evaluation
### add env variable
create an `.env` file and add one line with your HuggingFace token inside: `HF_TOKEN=hf_....`

### setup inference parameters
There are two types of inference: `cot` and `tir`

```python
python3 -m src.bench_vllm \
    --model_name Qwen/Qwen2.5-Math-7B-Instruct \
    --dataset_name lozziopadredeivizzi/mathematic_games_dataset_en \
    --out_dir "./out" \
    --max_samples -1 \ # if > 0 you set a maximun prompt to consider for the execution, useful for debug
    --batch_size 2 \ # this is considered only for COT
    --cache_dir None \
    --n_out_sequences 1 \ #this is always 1 for tir while 8 or 16 for COT
    --n_sampling 8 \ # used only for tir
    --temperature 0.7 \
    --top_p 0.8 \
    --mode tir \
    --text_only True \ # ignoring images
    --n_gpus 1 \
    --n_rounds 3 # number of rounds to run ONLY for TIR
```
