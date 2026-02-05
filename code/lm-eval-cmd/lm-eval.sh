#curl http://localhost:30103/v1/completions  -H "Content-Type: application/json"  -d '{ "model": "/share_data/users/like/package/hf-models/Meta-Llama-3.1-8B/", "prompt": ["The capital of france is ", "intel corp is "], "max_tokens": 20, "temperature": 0 }'




# Please use vLLM to host the model firstly.
# See simo/extensions/vllm/README.md for more details.
set -x
export HF_DATASETS_CACHE=/share/users/like/huggingface_cache/
MODEL_PATH="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
SERVING_URL=http://localhost:30103/
#SERVING_URL=http://localhost:30113/
lm_eval --model local-completions --tasks wikitext --model_args model=${MODEL_PATH},base_url=${SERVING_URL}/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False

