set -x
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30123 --num-prompts 1000 
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30123 --num-prompts 1000 --dataset-name random 
NUM_PROMPT=100; python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30123 --num-prompts $NUM_PROMPT --dataset-name random > temp/bench_$NUM_PROMPT.log 2>&1 &

