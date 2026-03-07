#!/bin/bash
#
# kill_high_mem.sh - 当指定进程的总内存占用超过阈值时，杀掉所有匹配的进程
#
# 用法:
#   bash kill_high_mem.sh <mem_gib> <program> <log_file>
#
# 参数:
#   mem_gib  - 内存阈值，单位 GiB。所有匹配进程的 RSS 内存之和超过此值时触发 kill
#   program  - 进程名关键字。用 grep 匹配 ps 输出的 CMD 列（部分匹配即可）
#   log_file - 日志文件路径。执行 kill 前会向该文件追加当前时间、实际内存占用、阈值
#
# 示例:
#   bash kill_high_mem.sh 4 chrome /tmp/kill.log
#   bash kill_high_mem.sh 10 python /tmp/kill.log
#
# 工作原理:
#   1. 用 ps -eo pid,rss,cmd 列出所有进程的 PID、RSS（单位 KiB）、命令行
#   2. 用 grep 筛选出 CMD 中包含 program 的行（排除 grep 自身和本脚本自身）
#   3. 提取所有匹配进程的 RSS 值并求和，得到总内存占用（KiB）
#   4. 将阈值 mem_gib 换算为 KiB（乘以 1048576）
#   5. 如果总内存 > 阈值，先向日志文件写入时间、实际内存、阈值，再用 kill -9 杀掉所有匹配进程
#

set -euo pipefail

# ============================================================
# 参数检查
# ============================================================
if [ $# -ne 3 ]; then
  echo "用法: $0 <mem_gib> <program> <log_file>"
  echo "  mem_gib  - 内存阈值 (GiB)"
  echo "  program  - 进程名关键字"
  echo "  log_file - 日志文件路径"
  exit 1
fi

MEM_GIB="$1"    # 内存阈值 (GiB)
PROGRAM="$2"    # 进程名关键字
LOG_FILE="$3"   # 日志文件路径

# 校验 mem_gib 是数字（整数或小数）
if ! echo "$MEM_GIB" | grep -qE '^[0-9]+\.?[0-9]*$'; then
  echo "错误: mem_gib 必须是正数，当前值: $MEM_GIB"
  exit 1
fi

# ============================================================
# 第1步: 找出所有匹配 program 的进程，提取 PID 和 RSS
# ============================================================
# ps -eo pid,rss,cmd 输出格式:
#   PID     RSS CMD
#   1234  523264 /opt/google/chrome/chrome ...
#
# RSS 单位是 KiB (参考 ps 手册: rss - resident set size in kilobytes)
#
# grep 过滤:
#   - grep "$PROGRAM"       : 匹配包含进程名的行
#   - grep -v "grep"        : 排除 grep 自身的进程
#   - grep -v "$0"          : 排除本脚本自身的进程
# ============================================================
PS_OUTPUT=$(ps -eo pid,rss,cmd | grep "$PROGRAM" | grep -v "grep" | grep -v "$0" || true)

# 如果没有匹配到任何进程，直接退出
if [ -z "$PS_OUTPUT" ]; then
  echo "没有找到包含 '$PROGRAM' 的进程"
  exit 0
fi

# 统计匹配的进程数量
PROC_COUNT=$(echo "$PS_OUTPUT" | wc -l)

# 提取所有 PID（第1列）
PIDS=$(echo "$PS_OUTPUT" | awk '{print $1}')

# ============================================================
# 第2步: 计算所有匹配进程的 RSS 内存总和 (KiB)
# ============================================================
# awk 提取第2列 (RSS)，求和
TOTAL_RSS_KIB=$(echo "$PS_OUTPUT" | awk '{sum += $2} END {print sum}')

# ============================================================
# 第3步: 将阈值从 GiB 换算为 KiB，并比较
# ============================================================
# 1 GiB = 1024 MiB = 1024 * 1024 KiB = 1048576 KiB
# 用 awk 做浮点运算，因为 bash 不支持浮点比较
THRESHOLD_KIB=$(awk "BEGIN {printf \"%.0f\", $MEM_GIB * 1048576}")

# 将 KiB 转为可读的 GiB 用于打印（保留2位小数）
TOTAL_RSS_GIB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_RSS_KIB / 1048576}")

echo "进程名匹配: '$PROGRAM'"
echo "匹配进程数: $PROC_COUNT"
echo "总内存占用: ${TOTAL_RSS_GIB} GiB (${TOTAL_RSS_KIB} KiB)"
echo "内存阈值:   ${MEM_GIB} GiB (${THRESHOLD_KIB} KiB)"

# ============================================================
# 第4步: 如果总内存超过阈值，kill 所有匹配的进程
# ============================================================
# 用 awk 做浮点比较: 总内存 > 阈值 则返回 1，否则返回 0
EXCEEDED=$(awk "BEGIN {print ($TOTAL_RSS_KIB > $THRESHOLD_KIB) ? 1 : 0}")

if [ "$EXCEEDED" -eq 1 ]; then
  # ============================================================
  # 第5步: 在执行 kill 前，向日志文件追加记录
  # ============================================================
  # 记录内容: 当前时间 | 进程名 | 实际总内存占用 | 内存阈值 | 匹配进程数
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${TIMESTAMP}] program='${PROGRAM}' total_rss=${TOTAL_RSS_GIB}GiB threshold=${MEM_GIB}GiB proc_count=${PROC_COUNT} action=KILL" >> "$LOG_FILE"

  echo ""
  echo ">>> 总内存 ${TOTAL_RSS_GIB} GiB 超过阈值 ${MEM_GIB} GiB，开始杀进程 <<<"
  echo ">>> 已写入日志: ${LOG_FILE}"
  echo ""

  # 逐个 kill，打印每个被杀进程的信息
  echo "$PS_OUTPUT" | while read -r line; do
    PID=$(echo "$line" | awk '{print $1}')
    RSS=$(echo "$line" | awk '{print $2}')
    # 将单个进程的 RSS 转为 MiB 方便阅读
    RSS_MIB=$(awk "BEGIN {printf \"%.1f\", $RSS / 1024}")
    echo "  kill -9 $PID (RSS: ${RSS_MIB} MiB)"
    kill -9 "$PID" 2>/dev/null || echo "  警告: 无法杀掉进程 $PID (可能已退出或权限不足)"
  done

  echo ""
  echo "完成: 已杀掉 $PROC_COUNT 个 '$PROGRAM' 进程"
else
  echo ""
  echo "总内存 ${TOTAL_RSS_GIB} GiB 未超过阈值 ${MEM_GIB} GiB，不做任何操作"
fi
