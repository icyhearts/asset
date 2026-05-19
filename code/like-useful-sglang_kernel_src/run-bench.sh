while true; do
    #python3 -m sglang.bench_serving --backend sglang  --dataset-path /mnt/yrfs/users/like/package/share-gpt/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name random  --random-input 128 --random-output 128 --random-range-ratio 1 --num-prompts 1024 --host localhost --port 30000 --max-concurrency 100
    NUM_PROMPT=100; python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30123 --num-prompts $NUM_PROMPT --dataset-name random 
done
