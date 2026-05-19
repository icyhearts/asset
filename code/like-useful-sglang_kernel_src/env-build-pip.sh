export NCCL_NVLS_ENABLE=0
export MAX_JOBS=60
export CUDA_ROOT_DIR=/share_data/users/like/opt/cuda-12.8/
export CUDA_HOME=$CUDA_ROOT_DIR
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin/:$PATH
export DG_JIT_CACHE_DIR=/tmp/deep_gemm_cache_like
mkdir -p $DG_JIT_CACHE_DIR
export DG_JIT_NVCC_COMPILER=$CUDA_HOME/bin/nvcc
export TRITON_CACHE_DIR=/tmp/triton_cache_like
mkdir -p $TRITON_CACHE_DIR
# for pip install
export NVCC_VERBOSE=1
export CUDA_VERBOSE_BUILD=1
export CMAKE_VERBOSE_MAKEFILE=ON
# for scikit-build-core verbose build
export SKBUILD_VERBOSE=1
