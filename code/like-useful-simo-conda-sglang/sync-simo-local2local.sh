set -x
set -e
cd /data/like/package/simo_conda_sglang
src_commit=$(git rev-parse HEAD)
current_branch=$(git branch --show-current) # main-local-dep
git push -f gitlab $current_branch




cd /data/like/package/simo_conda_vllm
git fetch gitlab $current_branch
git checkout -B $current_branch gitlab/$current_branch
dst_commit=$(git rev-parse HEAD)
[[ "$src_commit" == "$dst_commit" ]] && echo "equal" || echo "not equal"
