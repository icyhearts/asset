set -x
source temp/build-pip-env-bjh100-v0.20.0-like-dev-v2026_05_25.sh
#--no-build-isolation 
# pip install --config-settings=build.verbose=true -vvv -e "sgl-kernel" --no-build-isolation
pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation  --index-url https://download.pytorch.org/whl/cu128  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple > temp/pip-vllm-src.log.txt.`nowstr.sh` 2>&1 &
