set -x
# sgl:
cd /data/like/package//sglang_kernel_src/
git push -f gitlab main-local-dep
# cutlass
cd /data/like/package/cutlass/
git push -f gitlab like

# asset
cd /softhome/like/asset && bash temp/add.sh && git commit -m "update"  && git push gitlab master

# cute-gemm reed
cd /data/like/package/cute-gemm
git push -f gitlab like

cd /data/like/package/hpc-ops
git push -f gitlab like
