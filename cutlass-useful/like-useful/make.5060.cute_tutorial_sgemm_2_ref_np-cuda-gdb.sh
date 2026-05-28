set -x
#（--fmad=false 可避免指令融合影响断点）

cd /softhome/like/package/h100/package/cutlass/build-rtx-5060ti/examples/cute/tutorial && /share_data/users/like/opt/cuda-13.0/bin//nvcc -forward-unknown-to-host-compiler  --options-file CMakeFiles/cute_tutorial_sgemm_2_ref_np.dir/includes_CUDA.rsp -DCUTLASS_VERSIONS_GENERATED -Xcudafe=--diag_suppress=550 --fmad=false -O0 -G -g  -std=c++17  --generate-code=arch=compute_120a,code=[sm_120a] --generate-code=arch=compute_120a,code=[compute_120a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -MD -MT examples/cute/tutorial/CMakeFiles/cute_tutorial_sgemm_2_ref_np.dir/sgemm_2_ref_np.cu.o -MF CMakeFiles/cute_tutorial_sgemm_2_ref_np.dir/sgemm_2_ref_np.cu.o.d -x cu -c /softhome/like/package/h100/package/cutlass/examples/cute/tutorial/sgemm_2_ref_np.cu -o CMakeFiles/cute_tutorial_sgemm_2_ref_np.dir/sgemm_2_ref_np.cu.o

echo "ret=$?"
cd /softhome/like/package/h100/package/cutlass/build-rtx-5060ti/

VERBOSE=1 make cute_tutorial_sgemm_2_ref_np

mv examples/cute/tutorial/cute_tutorial_sgemm_2_ref_np examples/cute/tutorial/cute_tutorial_sgemm_2_ref_np.cuda-gdb-v3
