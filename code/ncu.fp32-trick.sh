#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
# --launch-count   1 --launch-skip  3

set -x
. /softhome/like/miniconda3/bin/activate simo_sglang
which python
cd /softhome/like/package/h100/package/sglang_kernel_src
export CUDA_VISIBLE_DEVICES=7
/softhome/like/opt/cuda-12.8/bin/ncu --set full    --export temp/fused_moe_kernel_gptq_awq_fp32_scale.py.$CUDA_VISIBLE_DEVICES.ncu python temp/fused_moe_kernel_gptq_awq_fp32_scale.py   --round 3 --warmup 2 
