#!/usr/bin/env bash
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_CACHE=/share/users/like/huggingface_cache/

# lm-eval serve deepseek model
#lm_eval --model local-chat-completions --tasks mmlu --model_args model=/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/,base_url=http://0.0.0.0:30113/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048  --batch_size auto --apply_chat_template --num_fewshot 0
lm_eval --model local-completions --tasks mmlu --model_args model=/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/,base_url=http://0.0.0.0:30113/v1/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048  --batch_size auto --num_fewshot 0

