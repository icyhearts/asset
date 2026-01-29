#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
# --launch-count   1 --launch-skip  3
set -x
. /softhome/like/miniconda3/bin/activate simo_sglang
which python
cd /softhome/like/package/h100/package/sglang_kernel_src
#export CUDA_VISIBLE_DEVICES=3; export bscale_dtype="";  /softhome/like/opt/cuda-12.8/bin/nsys profile --force-overwrite=true  -o temp/load_gptq_awq.$CUDA_VISIBLE_DEVICES.bscale_dtype-$bscale_dtype.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl'  python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"
nows=`date +%Y_%m_%d___%H_%M_%S`
bash /share_data/users/like/bash-bin/bin/kgp.sh 6 likf &&  && CUDA_VISIBLE_DEVICES=6  MEMG=75 /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench > temp/out.log.$nows 2>&1 &




