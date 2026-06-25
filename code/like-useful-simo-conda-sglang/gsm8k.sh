set -x
lm-eval \
  --model local-completions \
  --model_args '{"model": "default", "base_url": "http://127.0.0.1:30121/v1/completions", "num_concurrent": 1}' \
  --tasks gsm8k \
  --batch_size auto \
  > temp/lm-eval-gsm8k-dsv4-flash.sglang-serve-api.`nowstr.sh`.log 2>&1 &
