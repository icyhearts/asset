memg=1
CUDA_VISIBLE_DEVICES=0 SLMS=50 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=1 SLMS=50 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=2 SLMS=50 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=3 SLMS=50 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
