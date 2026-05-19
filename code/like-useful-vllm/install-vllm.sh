set -x
source temp/build-pip-env-bjh100-v0.17.0-like-dev-v2026-03-26.sh
pip install --no-build-isolation -e . > temp/pip-vllm-src.log.txt.`nowstr.sh` 2>&1 &
