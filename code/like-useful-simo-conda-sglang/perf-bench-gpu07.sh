memg=72
CUDA_VISIBLE_DEVICES=0 SLMS=10 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=4 SLMS=10 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=6 SLMS=10 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
CUDA_VISIBLE_DEVICES=7 SLMS=10 MEMG=$memg /share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench  &
