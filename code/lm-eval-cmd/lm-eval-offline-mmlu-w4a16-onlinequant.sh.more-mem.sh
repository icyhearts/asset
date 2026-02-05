set -x
#pretrained=/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/,quantization=simo,json_model_override_args='{\"quantization_config_file\": \"/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group-exclude-kv_b_proj.json\"}'   \
export CUDA_VISIBLE_DEVICES=5
export SIMO_SGLANG_REGISTER=1
tp_size=1
model_path="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
config_file="quant_config_w4a16_int4_per_group-exclude-kv_b_proj.json"
model_args_base="\"pretrained\": \"${model_path}\", \"quantization\": \"simo\", \"json_model_override_args\": \"{\\\"quantization_config_file\\\": \\\"/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/${config_file}\\\"}\", \"tp_size\": ${tp_size}, \"dtype\": \"auto\",  \"disable_cuda_graph\": true, \"mem_fraction_static\": 0.5, \"max_total_tokens\": 8192, \"max_prefill_tokens\": 8192"
lm_eval --model sglang \
    --model_args "{${model_args_base}}" \
    --tasks mmlu \
    --batch_size auto \
    --output_path ./results/mmlu_deepseek_v2_lite_w4a16 > temp/mmlu_deepseek_v2_lite-offline_w4a16.log 2>&1 &
