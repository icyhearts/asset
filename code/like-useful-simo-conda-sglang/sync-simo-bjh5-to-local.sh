set -x
set -e
proj_dir="/data/like/package/simo_conda_sglang"
ssh_out=$(ssh bjh5 "cd $proj_dir; git push -f gitlab HEAD >/dev/null 2>&1; git rev-parse HEAD; git branch --show-current")

echo "|$ssh_out|"
{ read -r src_commit; read -r current_branch; } <<< "$ssh_out"

echo "src_commit: $src_commit"
echo "current_branch: $current_branch"

if [[ -n "$src_commit" ]] && [[ -n "$current_branch" ]]; then
  echo "src_commit and current_branch not empty"
  cd $proj_dir
  git reset --hard HEAD
  git fetch gitlab $current_branch
  git checkout -B $current_branch gitlab/$current_branch
  dst_commit=$(git rev-parse HEAD)
  [[ "$src_commit" == "$dst_commit" ]] && echo "equal" || echo "not equal"

  cd /data/like/package/simo_conda_vllm
  git reset --hard HEAD
  git fetch gitlab $current_branch
  git checkout -B $current_branch gitlab/$current_branch
  dst_commit=$(git rev-parse HEAD)
  [[ "$src_commit" == "$dst_commit" ]] && echo "equal" || echo "not equal"

fi

