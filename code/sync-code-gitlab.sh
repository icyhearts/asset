set -x
# sgl:
cd /share/users/like/package//sglang_kernel_src/
git push -f gitlab release/v0.5.18-local-dep

# cutlass
cd /share/users/like/package//cutlass
git push -f gitlab like

# asset
cd /softhome/like/asset && bash temp/add.sh && git commit -m "update"  && git push gitlab master

# cute-gemm reed
cd /share/users/like/package//cute-gemm
git push -f gitlab like

cd /share/users/like/package//hpc-ops
git push -f gitlab like
