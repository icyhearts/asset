export NCCL_NVLS_ENABLE=0
export MAX_JOBS=60
#export CUDA_ROOT_DIR=/share_data/users/like/opt/cuda-12.8/
export CUDA_ROOT_DIR=/share_data/users/like/opt/cuda-13.0
export CUDA_HOME=$CUDA_ROOT_DIR
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin/:$PATH
export DG_JIT_NVCC_COMPILER=$CUDA_HOME/bin/nvcc
# for pip install
export NVCC_VERBOSE=1
export CUDA_VERBOSE_BUILD=1
export CMAKE_VERBOSE_MAKEFILE=ON
# for scikit-build-core verbose build
export SKBUILD_VERBOSE=1


if [[ -d /data/like ]]; then
    export LIKE_CACHE_ROOT_DIR=/data/like
elif [[ -d /local_data/like ]]; then
    export LIKE_CACHE_ROOT_DIR=/local_data/like
else
    export LIKE_CACHE_ROOT_DIR=/tmp/like
    echo "WARNING: /data/like and /local_data/like do not exist; using ${LIKE_CACHE_ROOT_DIR} for build caches." >&2
fi

export DG_JIT_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/deep_gemm_cache_dir
export TVM_FFI_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/tvm_ffi_cache_dir
export TRITON_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/triton_cache_like

export SGLANG_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/sglang
export SGLANG_JIT_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/sglang_jit
export SGLANG_DG_CACHE_DIR=${LIKE_CACHE_ROOT_DIR}/cache/deep_gemm_cache_dir


mkdir -p $DG_JIT_CACHE_DIR
mkdir -p $TVM_FFI_CACHE_DIR
mkdir -p $TRITON_CACHE_DIR


mkdir -p $SGLANG_CACHE_DIR
mkdir -p $SGLANG_JIT_CACHE_DIR
mkdir -p $SGLANG_DG_CACHE_DIR



export SGLANG_JIT_CACHE_DEBUG=1
