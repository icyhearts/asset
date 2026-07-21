set -x
export CUDA_VISIBLE_DEVICES=1
#. /data/like/miniconda3/bin/activate simo_sglang
. /share/users/like/miniconda3/bin/activate simo_sglang
which python


cd /share/users/like/package/simo_conda_sglang
mkdir -p temp

#/share_data/users/like/opt/cuda-13.0/bin/nsys profile --force-overwrite=true  -o temp/torch_mm_batch_varian.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl' python like-useful/split-k.py  
/share_data/users/like/opt/cuda-13.0/bin/nsys profile --force-overwrite=true  -o temp/torch_rmsnorm.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl' python ../sglang_kernel_src/like-useful/native.py
