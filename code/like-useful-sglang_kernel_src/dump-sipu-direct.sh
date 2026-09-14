
export PYTHONUNBUFFERED=1
export SGL_KERNEL_LOG=0
export TORCHINDUCTOR_SIZE_ASSERTS=0
export SGLANG_PROFILE_FORWARD=0
export TORCH_COMPILE_DEBUG=0
export SGLANG_SIPU_USE_TILED_WEIGHTS=0
export SGLANG_SIPU_USE_TILED_W_VC=0
export SGLANG_SIPU_USE_TILED_FP8_LINEAR=0
export PYTHONPATH=/sgl-workspace/sglang/test/srt/sipu/test_utils${PYTHONPATH:+:$PYTHONPATH}

cd /sgl-workspace/sgl-kernel-sipu
source setup.sh
source setup_triton.sh
source setup_tilelang.sh
unset TORCH_DEVICE_BACKEND_AUTOLOAD
cd /sgl-workspace/sglang/test/srt/sipu
python3 -u test_utils/run_test_job.py   --config-yaml /sgl-workspace/sglang/test/srt/sipu/configs/deepseek/ds_v32_2layer.yaml   --launch-config deepep_deepgemm_text   --test-case text-only   --device sipu   --dump-base /share_data/sglang_sipu/accuracy_verify/like   --log-base /sgl-workspace/sglang/test/srt/sipu/logs
'
