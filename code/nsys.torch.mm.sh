set -x
export CUDA_VISIBLE_DEVICES=6
. /data/like/miniconda3/bin/activate simo_sglang
which python
cd /data/like/package/simo_conda_sglang/
mkdir -p temp

/share_data/users/like/opt/cuda-12.8/bin/nsys profile --force-overwrite=true  -o temp/torch_mm_batch_varian.nsys --trace='cuda,cublas,cudnn,nvtx,osrt,opengl' python like-useful/split-k.py  
