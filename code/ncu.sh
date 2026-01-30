#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
# --launch-count   1 --launch-skip  3
set -x
. /softhome/like/miniconda3/bin/activate simo_sglang
which python
cd /softhome/like/package/h100/package/sglang_kernel_src
#export CUDA_VISIBLE_DEVICES=3; export bscale_dtype="";  /softhome/like/opt/cuda-12.8/bin/ncu --set full    --export temp/fused_moe_kernel_gptq_awq.$CUDA_VISIBLE_DEVICES.bscale_dtype-$bscale_dtype.ncu python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"
export CUDA_VISIBLE_DEVICES=4

ncu --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct python temp/load_gptq_awq.py

export bscale_dtype="";  /softhome/like/opt/cuda-12.8/bin/ncu  --metrics  dram__bytes_read.sum,lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct \
    python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"
