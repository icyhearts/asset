set -x
set -e
cd /data/like/package/sglang_kernel_src
git branch
src_commit=$(git rev-parse HEAD)
current_branch=$(git branch --show-current) # main-local-dep
git push -f gitlab $current_branch
ssh bjh15 "
  set -x
  cd /data/like/package/sglang_kernel_src
  git reset --hard HEAD
  git fetch gitlab $current_branch
  git checkout -B $current_branch gitlab/$current_branch
  git branch
  git rev-parse HEAD
"
dst_commit=$(ssh -o StrictHostKeyChecking=no bjh15 "cd /data/like/package/sglang_kernel_src; git rev-parse HEAD")
[[ "$src_commit" == "$dst_commit" ]] && echo "equal" || echo "not equal"
