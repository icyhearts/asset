set -x
set -e
source temp/build-pip-env-bjh100-v0.27.1-like-dev-v2026_08_17.sh
# pip install torch==2.11.0
pip install torch==2.13.0
#--no-build-isolation 
# pip install --config-settings=build.verbose=true -vvv -e "sgl-kernel" --no-build-isolation
#pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation  --index-url https://download.pytorch.org/whl/cu128  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple > temp/pip-vllm-src.log.txt.`nowstr.sh` 2>&1 &
pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation   > temp/pip-vllm-src.log.txt.`nowstr.sh` 2>&1 &
