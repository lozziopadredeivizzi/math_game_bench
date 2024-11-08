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

### Example of TIR conversation
```
<|im_start|>system
Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
Taxes, what a passion! Carla paved her entire rectangular room using square tiles (of the same size). Counting the used tiles, she realizes that on the edge of the room there are as many as inside. How many tiles did Carla use altogether (knowing that they are less than 50)?<|im_end|>
<|im_start|>assistant
To solve this problem, let's break it down step-by-step:

1. Let the dimensions of the room be \(m \times n\) tiles.
2. The total number of tiles used is \(m \times n\).
3. The number of tiles on the edge of the room is \(2m + 2n - 4\).
4. The number of tiles inside the room is \((m-2) \times (n-2)\).
5. According to the problem, the number of tiles on the edge is equal to the number of tiles inside, so we can set up the equation:
   \[
   2m + 2n - 4 = (m-2)(n-2)
   \]
6. Simplifying the equation, we get:
   \[
   2m + 2n - 4 = mn - 2m - 2n + 4
   \]
   \[
   4m + 4n - 8 = mn
   \]
   \[
   mn - 4m - 4n + 8 = 0
   \]
7. Adding 16 to both sides, we get:
   \[
   mn - 4m - 4n + 16 = 8
   \]
   \[
   (m-4)(n-4) = 8
   \]
8. We need to find pairs \((m-4, n-4)\) that multiply to 8. The factor pairs of 8 are \((1,8)\), \((2,4)\), \((-1,-8)\), and \((-2,-4)\). Since \(m\) and \(n\) are positive integers, the valid pairs are \((1,8)\) and \((2,4)\).
9. Converting back to \((m,n)\), the pairs are \((5,12)\) and \((6,8)\).
10. We need to find the total number of tiles for each pair and check which one is less than 50.
   - For \((m,n) = (5,12)\), the total number of tiles is \(5 \times 12 = 60\).
   - For \((m,n) = (6,8)\), the total number of tiles is \(6 \times 8 = 48\).

Since 48 is less than 50, the answer is 48. Let's confirm this with Python code.
```python
# Define the function to check the number of tiles
def find_tiles():
    # Possible factor pairs of 8
    factor_pairs = [(1, 8), (2, 4)]
    # List to store the total number of tiles
    total_tiles = []
    for (a, b) in factor_pairs:
        m, n = a + 4, b + 4
        total_tiles.append(m * n)
    return total_tiles

# Find the total number of tiles
total_tiles = find_tiles()
# Filter the tiles that are less than 50
valid_tiles = [tiles for tiles in total_tiles if tiles < 50]
print(valid_tiles[0])
'''
<|im_end|>
<|im_start|>user
'''output
(<function find_tiles at 0x7dd7f4089360>, [60, 48], [48])
'''<|im_end|>
<|im_start|>assistant
The total number of tiles Carla used altogether, knowing that they are less than 50, is \(\boxed{48}\).<|im_end|>
```

### TODO
- [ ] Solve bug during TIR of infinite pending `exec_code` function. See: [Qwen2.5-Math evaluation code](https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/python_executor.py)
- [ ] Adapt the script for multiple LLMs
- [ ] Solve issue for >70B models such as QWEN 2.5 math with GGUF and 4bit bitsandbytes 
