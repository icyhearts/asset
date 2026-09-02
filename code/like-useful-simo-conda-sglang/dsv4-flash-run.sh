set -x
source /share/users/like/package/sglang_kernel_src/like-useful/env-build-pip.sh
CUDA_VISIBLE_DEVICES=0,1,2,3
# SGLANG_LOGGING_CONFIG_PATH=/share_data/users/like/package/h100/package/sglang_kernel_src/like-useful/custom_sglang.json
SGLANG_LOGGING_CONFIG_PATH=/share_data/users/like/package/h100/package/sglang_kernel_src/like-useful/custom_sglang.simple.json SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  sglang serve \
  --trust-remote-code \
  --model-path /data/like/hf-models/deepseek-v4-flash/ \
  --mem-fraction-static 0.7 \
   --cuda-graph-max-bs 16 \
  --log-level debug \
  --tp 4 \
  --moe-runner-backend marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30121 > temp/sgl.dsv4-flash.log.`nowstr.sh` 2>&1 &
