set -x
# task name: mmlu mmlu_pro  wikitext
task_name=mmlu
nowstr=$(date +%Y_%m_%d___%H_%M_%S)
export SIMO_SGLANG_REGISTER=1
tp_size=1

model_path="/data_gpu/models/share_data/modelzoo/weights/llm/llama/llama3.1/llama3.1-8B-Instruct/safetensor_weights/"
model_args_base="\"pretrained\": \"${model_path}\",  \"tp_size\": ${tp_size}, \"dtype\": \"auto\", \"mem_fraction_static\": 0.5"
lm_eval --model sglang \
    --model_args "{${model_args_base}}" \
    --tasks $task_name \
    --batch_size auto \
    --output_path ./results/${task_name}_llama_simo-bf16.$config_file.$nowstr > temp/${task_name}_llama-offline_bf16.log.$config_file.$nowstr 2>&1
echo "lm eval done"
