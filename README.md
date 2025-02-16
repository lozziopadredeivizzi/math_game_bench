# International Mathematical Games Championships (IMGC)
This repository contains code for evaluating Large Language Models (LLMs) on MathGB, the first benchmark designed specifically for the International Mathematical Games Championships.

# Performances
## Without images (text-generation only)
| Model                          | Method      | Pass@1 | Maj@8 | Easy (525 it) | Medium (424 it) | Hard (430 it) |
|--------------------------------|-------------|--------|-------|---------------|-----------------|---------------|
| **Qwen2.5-Math-7B-Instruct**   | CoT         | 35.71% | -     | 44.30%        | 33.96%          | 26.74%        |
|                                | TIR         | 42.84% | -     | 49.91%        | 40.57%          | 36.28%        |
| **DeepSeek-Math-7B-Instruct**  | CoT         | 20.03% | -     | 20.16%        | 16.88%          | 22.33%        |
|                                | TIR         | 13.89% | -     | 18.13%        | 15.09%          | 7.44%         |
| **NumminaMath-7B-Instruct**    | CoT         | 24.98% | -     | 24.65%        | 25.19%          | 25.15%        |
|                                | TIR         | 30.60% | -     | 38.13%        | 31.13%          | 20.70%        |
| **MathStral-7B-Instruct**      | CoT         | 24.55% | -     | 31.96%        | 20.75%          | 19.07%        |
| **ToRA-7B**                    | TIR         | 11.16% | -     | 12.15%        | 12.97%          | 8.14%         |
| **GPT-4**                      | CoT         | 42.80% | -     | 50.10%        | 43.73%          | 33.41%        |  
| **Qwen2.5-Math-7B-Instruct**   | CoT         | -      | 35.57%| 43.18%        | 38.44%          | 23.26%        |
|                                | TIR         | -      | 46.06%| 51.21%        | 48.46%          | 37.30%        |
| **DeepSeek-Math-7B-Instruct**  | CoT         | -      | 20.09%| 25.05%        | 21.23%          | 12.79%        |
|                                | TIR         | -      | 20.16%| 24.49%        | 21.23%          | 13.72%        |
| **NumminaMath-7B-Instruct**    | CoT         | -      | 25.05%| 33.08%        | 27.36%          | 12.79%        |
|                                | TIR         | -      | 36.89%| 43.26%        | 38.68%          | 16.98%        |
| **MathStral-7B-Instruct**      | CoT         | -      | 25.27%| 31.40%        | 25.94%          | 16.35%        |
| **ToRA-7B**                    | TIR         | -      | 11.38%| 13.83%        | 12.97%          | 6.76%         |

# Prerequisites 
### 1) Clone github repository (or pull request if already done)
```
git clone https://github.com/lozziopadredeivizzi/math_game_bench.git
```

### 2) Build Docker Image
```
docker build -t math-bench-img .
```
⚠️ In case of build errors with the image, consider using `FROM nvidia/cuda:12.3.2-devel-ubuntu20.04` as an alternative base image

### 3) Run the container 
```
srun -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --pty bash # to get the lock
docker run  -v /home/<YOUR_USERNAME>/math_game_bench:/math_game_bench --rm --gpus "device=$GPU_VISIBLE_DEVICES" -it math-bench-img  /bin/bash # to run container
```

### 4) Open byobu (run within container)
```
byobu
```

### 5) Save requiremens (run within container)
```
pip freeze > requirements.txt
```

# Evaluation

1) ### Add env variable
create an `.env` file and add one line with your HuggingFace token inside: `HF_TOKEN=hf_....`

2) ### Setup inference parameters
There are two types of inference: 

#### CoT:

```python
python3 -m src.bench_vllm \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --dataset_name "lozziopadredeivizzi/mathematic_games_dataset_en" \
    --out_dir "./out" \
    --max_samples -1 \ # if > 0 you set a maximun prompt to consider for the execution, useful for debug
    --batch_size 8 \ # this is considered only for COT
    --cache_dir None \
    --n_out_sequences 8 \ # for maj@8
    --temperature 0.7 \
    --top_p 0.8 \
    --mode cot \
    --text_only True \ # ignoring images
    --n_gpus 1 \
```

#### TIR:

```python
python3 -m src.bench_vllm \
    --model_name "Qwen/Qwen2.5-Math-7B-Instruct" \
    --dataset_name "lozziopadredeivizzi/mathematic_games_dataset_en" \
    --out_dir "./out" \
    --max_samples -1 \ # if > 0 you set a maximun prompt to consider for the execution, useful for debug
    --cache_dir None \
    --n_out_sequences 1 \ 
    --n_sampling 8 \ # used only for tir for maj@8
    --temperature 0.7 \
    --top_p 0.8 \
    --mode tir \
    --text_only True \ # ignoring images
    --n_gpus 1 \
    --n_rounds 3 # number of rounds to run ONLY for TIR
```

3) ### Run the code
```
./src/bench_vllm.sh
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
- [ ] Solve issue for >70B models such as QWEN 2.5 math with GGUF and 4bit bitsandbytes
