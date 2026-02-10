set -x
# task name: mmlu mmlu_pro  wikitext
task_name=mmlu
nowstr=$(date +%Y_%m_%d___%H_%M_%S)
export CUDA_VISIBLE_DEVICES=6
export SIMO_SGLANG_REGISTER=1
tp_size=1

model_path="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
config_file="quant_config_w8a8_fp8_per_block.json"
model_args_base="\"pretrained\": \"${model_path}\", \"quantization\": \"simo\", \"json_model_override_args\": \"{\\\"quantization_config_file\\\": \\\"/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/${config_file}\\\"}\", \"tp_size\": ${tp_size}, \"dtype\": \"auto\", \"mem_fraction_static\": 0.5"
lm_eval --model sglang \
    --model_args "{${model_args_base}}" \
    --tasks $task_name \
    --batch_size auto \
    --output_path ./results/${task_name}_deepseek_v2_lite_simo-w8a8.$config_file.$nowstr > temp/${task_name}_deepseek_v2_lite-offline_w8a8-per_block.log.$config_file.$nowstr 2>&1 
echo "lm eval done"
