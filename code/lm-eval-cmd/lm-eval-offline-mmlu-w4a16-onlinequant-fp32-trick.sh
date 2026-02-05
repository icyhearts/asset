set -x
# task name: mmlu mmlu_pro  wikitext
task_name=$1
nowstr=$(date +%Y_%m_%d___%H_%M_%S)
export CUDA_VISIBLE_DEVICES=4
export SIMO_SGLANG_REGISTER=1
tp_size=1
model_path="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
config_file="quant_config_w4a16_int4_per_group-exclude-kv_b_proj-promote-scale-precision.json"
model_args_base="\"pretrained\": \"${model_path}\", \"quantization\": \"simo\", \"json_model_override_args\": \"{\\\"quantization_config_file\\\": \\\"/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/${config_file}\\\"}\", \"tp_size\": ${tp_size}, \"dtype\": \"auto\", \"mem_fraction_static\": 0.75, \"max_total_tokens\": 32768, \"chunked_prefill_size\": 8192"
lm_eval --model sglang \
    --model_args "{${model_args_base}}" \
    --tasks $task_name \
    --batch_size auto \
    --output_path ./results/${task_name}_deepseek_v2_lite_simo-w4a16.$config_file.$nowstr > temp/${task_name}_deepseek_v2_lite-offline_w4a16.log.$config_file.$nowstr 2>&1 &
