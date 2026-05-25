set -x
LOG=temp/pip-sglang-log.main-local-dep.txt
source install-scripts-sgl/env-build-pip.sh
pip install --config-settings=build.verbose=true -vvv -e "python" --no-build-isolation  --index-url https://download.pytorch.org/whl/cu128  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple > $LOG 2>&1 &
