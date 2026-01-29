#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
# --launch-count   1 --launch-skip  3
set -x
. /softhome/like/miniconda3/bin/activate simo_sglang
which python
cd /softhome/like/package/h100/package/sglang_kernel_src
export CUDA_VISIBLE_DEVICES=3; export bscale_dtype="";  /softhome/like/opt/cuda-12.8/bin/nsys --force-overwrite=true     python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"
