#!/bin/bash
# ============================================================================
# run_demo.sh - 启动多进程 demo, 支持调试多个 Worker
#
# 用法:
#   bash run_demo.sh                # 只调试 Worker-0 (端口 5678)
#   bash run_demo.sh "0,1"          # 调试 Worker-0 和 Worker-1 (端口 5678, 5679)
#   bash run_demo.sh "0,1" 6000     # 自定义端口基数 (端口 6000, 6001)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="/share_data/users/like/miniconda3/envs/simo_sglang"
DEBUG_WORKERS="${1:-0}"
DEBUG_PORT_BASE="${2:-5678}"

echo "============================================"
echo " Demo: debugpy + nvim-dap multi-process debug"
echo " CONDA_ENV:       ${CONDA_ENV}"
echo " DEBUG_WORKERS:   ${DEBUG_WORKERS}"
echo " DEBUG_PORT_BASE: ${DEBUG_PORT_BASE}"
echo "============================================"

# 激活 conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "[INFO] Python: $(which python)"
echo "[INFO] debugpy version: $(python -c 'import debugpy; print(debugpy.__version__)')"

cd "${SCRIPT_DIR}"

echo ""
echo "[INFO] Starting main_server.py"
echo "[INFO] After you see 'waiting for debugger to attach...':"
echo "       1. Open nvim"
echo "       2. :lua require'dap'.continue()   (or <leader>dc)"
echo "       3. Select the worker you want to attach"
echo ""

python main_server.py \
    --num-workers 2 \
    --debug-workers "${DEBUG_WORKERS}" \
    --debug-port-base "${DEBUG_PORT_BASE}"
