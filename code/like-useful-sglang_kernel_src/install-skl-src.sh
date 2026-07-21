set -x
source like-useful/env-build-pip.sh
pip install --config-settings=build.verbose=true -vvv -e "sgl-kernel" --no-build-isolation > temp/pip-sgl-kernel.log.main-local-dep.txt.`nowstr.sh` 2>&1 &
