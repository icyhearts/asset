set -x
which python
nowstr=$(date +%Y_%m_%d___%H_%M_%S)
export SIMO_SGLANG_REGISTER=1
export CUDA_VISIBLE_DEVICES=4
tp_size=1
model_path="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
model_args_base="\"pretrained\": \"${model_path}\",  \"tp_size\": ${tp_size}, \"dtype\": \"auto\", \"mem_fraction_static\": 0.75, \"max_total_tokens\": 32768, \"chunked_prefill_size\": 8192"
lm_eval --model sglang \
    --model_args "{${model_args_base}}" \
    --tasks mmlu \
    --batch_size auto \
    --output_path ./results/mmlu_deepseek_v2_lite_simo-fp16.$nowstr > temp/mmlu_deepseek_v2_lite-offline_simo-fp16.$nowstr.log 2>&1 &
