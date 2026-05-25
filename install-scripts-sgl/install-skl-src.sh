set -x
source install-scripts-sgl/env-build-pip.sh
#pip install --config-settings=build.verbose=true -vvv -e "sgl-kernel" --no-build-isolation > temp/pip-sgl-kernel.log.main-local-dep.txt 2>&1 &
pip install --config-settings=build.verbose=true -vvv -e "sgl-kernel" --no-build-isolation --index-url https://download.pytorch.org/whl/cu128  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple > temp/pip-sgl-kernel.log.main-local-dep.txt 2>&1 &
