python3 -m src.bench_vllm_multimodal \
    --model_name "microsoft/Phi-3.5-vision-instruct" \
    --dataset_name "alecocc/mathematic_games_dataset_en" \
    --out_dir "./out" \
    --max_samples 32 \
    --batch_size 8 \
    --cache_dir None \
    --n_out_sequences 8 \
    --temperature 0.8 \
    --top_p 1.0 \
    --mode cot \
    --text_only False \
    --n_gpus 1 