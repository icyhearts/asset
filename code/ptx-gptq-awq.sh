set -x
nows=`date +%Y_%m_%d___%H_%M_%S`
which python
export CUDA_VISIBLE_DEVICES=7
cd /softhome/like/package/h100/package/sglang_kernel_src

export bscale_dtype="bf16"
TRITON_PRINT_AUTOTUNING=1 MLIR_ENABLE_DUMP=1   python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype" 
