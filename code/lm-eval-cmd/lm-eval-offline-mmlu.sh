set -x
export CUDA_VISIBLE_DEVICES=6
lm_eval --model sglang \
    --model_args pretrained=/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/ \
    --tasks mmlu \
    --batch_size auto \
    --output_path ./results/mmlu_deepseek_v2_lite > temp/mmlu_deepseek_v2_lite-offline.log 2>&1 &
