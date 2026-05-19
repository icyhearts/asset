python3 -m sglang.bench_serving --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json --dataset-name random  --random-input 128 --random-output 128 --num-prompts 1000 --request-rate 128 --random-range-ratio 1.0
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 128 --random-output 128 --random-range-ratio 1 --num-prompts 1000 --host 10.157.101.163 --port 3000 --output-file "deepseekv3_multinode.jsonl"


