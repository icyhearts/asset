#!/usr/bin/env bash
set -x

HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_CACHE=/share/users/like/huggingface_cache/ lm_eval --model local-chat-completions --tasks mmlu_pro --model_args model=/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/,base_url=http://0.0.0.0:30113/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048  --batch_size 128 --apply_chat_template --num_fewshot 0

