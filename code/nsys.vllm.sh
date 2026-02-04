
###
set -x
export SIMO_SGLANG_REGISTER=1 
export CUDA_VISIBLE_DEVICES=6
. /softhome/like/miniconda3/bin/activate simo_vllm
which python
cd /softhome/like/package/h100/package/vllm-for-conda-simo

nows=`date +%Y_%m_%d___%H_%M_%S`
export PROTECT_USER=likf
export MEMG=77
/bin/bash /share_data/users/like/bash-bin/bin/kgp.sh $CUDA_VISIBLE_DEVICES $PROTECT_USER &&  /softhome/like/opt/cuda-12.8/bin/nsys profile --force-overwrite=true  -o /softhome/like/package/h100/package/sglang_kernel_src/temp/offline_batch_inference.vllm.$nows.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl'  python examples/offline_inference/basic/generate.py --model /data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/ --tensor-parallel-size 1 --max-num-seqs 256 --gpu-memory-utilization 0.8 --max-tokens 256 --quantization simo --hf-overrides='{"quantization_config_file": "/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json"}'    &&    /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  



