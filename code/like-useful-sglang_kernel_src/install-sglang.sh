set -x
LOG=temp/pip-sglang-log.main-local-dep.txt.`nowstr.sh`
source like-useful/env-build-pip.sh
pip install --config-settings=build.verbose=true -vvv -e "python" --no-build-isolation > $LOG 2>&1 &
