#~/opt/cuda-12.8/bin/ncu --set full --launch-count   --launch-skip --export temp 
# --launch-count   1 --launch-skip  3
set -x
. /softhome/like/miniconda3/bin/activate simo_sglang
which python
cd /softhome/like/package/h100/package/sglang_kernel_src
export CUDA_VISIBLE_DEVICES=3; export bscale_dtype="";  /softhome/like/opt/cuda-12.8/bin/nsys --force-overwrite=true  -o temp/load_gptq_awq.$CUDA_VISIBLE_DEVICES.bscale_dtype-$bscale_dtype.nsys\
    python temp/load_gptq_awq.py --round 3 --warmup 2 --bscale_dtype "$bscale_dtype"


--trace='cuda,cublas,cudnn,nvtx,osrt,opengl'

	   Possible values are 'cuda', 'nvtx', 'cublas', 'cublas-verbose', 'cusolver', 
	   'cusolver-verbose', 'cusparse', 'cusparse-verbose', 'mpi', 'oshmem', 'ucx', 
	   'osrt', 'cudnn', 'opengl', 'opengl-annotations', 'openacc', 'openmp', 
	   'nvvideo', 'vulkan', 'vulkan-annotations', 'python-gil', 'syscall' or 'none'.
	   Select the API(s) to trace. Multiple APIs can be selected, separated by commas only
	   (no spaces).
	   If '<api>-annotations' is selected, the corresponding API will also be traced.
	   If 'none' is selected, no APIs are traced.

