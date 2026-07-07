#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
set -x
export CUDA_VISIBLE_DEVICES=6
. /data/like/miniconda3/bin/activate simo_sglang
which python
cd /data/like/package/simo_conda_sglang/

#/softhome/like/opt/cuda-12.8/bin/nsys profile --force-overwrite=true  -o /softhome/like/package/h100/package/sglang_kernel_src/temp/load_gptq_awq.$CUDA_VISIBLE_DEVICES.bscale_dtype-$bscale_dtype.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl'  python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"   
/share_data/users/like/opt/cuda-12.8/bin/nsys profile --force-overwrite=true  -o /softhome/like/package/h100/package/sglang_kernel_src/temp/offline_batch_inference.sgl.$nows.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl' python examples/runtime/engine/offline_batch_inference.py  --model-path /data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/ --quantization simo --json-model-override-args='{"quantization_config_file": "/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group-exclude-kv_b_proj.json"}'



