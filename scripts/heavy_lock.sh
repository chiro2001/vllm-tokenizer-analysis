#!/usr/bin/env bash
# 全局"重活锁" —— 保证任意时刻全项目只有一个 CPU/内存密集任务在跑。
#
# 为什么需要：本机是共享开发机（12 核 / 29 GiB）。4 条并行线各自跑基准，
# 会互相污染测量结果，并且可能把别人的会话拖垮。
#
# 用法:
#   scripts/heavy_lock.sh <命令...>       # 拿锁执行；拿不到就等（默认等 30 分钟）
#   WAIT=0 scripts/heavy_lock.sh <命令>   # 拿不到立刻失败（用于"只试一次"）
#   TIMEOUT=600 scripts/heavy_lock.sh ... # 自定义等待上限（秒）
#   scripts/heavy_lock.sh --status        # 看当前谁持有
#   scripts/heavy_lock.sh --release       # 强制释放（仅在确认持有者已死时用）
#
# 建议与 limit.sh 组合使用：
#   scripts/heavy_lock.sh scripts/limit.sh cargo bench -p vllm-tokenizer
#
# 持有期间会写 lease 文件，记录 owner / 目的 / PID / 开始时间，便于排查。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCKDIR="$ROOT/.locks/heavy"
LEASE="$LOCKDIR/lease.json"
WAIT="${WAIT:-1800}"
OWNER="${OWNER:-${USER:-unknown}:$$}"

status() {
  if [[ -d "$LOCKDIR" ]]; then
    echo "HELD"
    [[ -f "$LEASE" ]] && cat "$LEASE"
  else
    echo "FREE"
  fi
}

acquire() {
  local deadline=$(( $(date +%s) + WAIT ))
  mkdir -p "$(dirname "$LOCKDIR")"   # 父目录必须先存在，否则下面 mkdir 永远失败
  while true; do
    if mkdir "$LOCKDIR" 2>/dev/null; then
      python3 - "$LEASE" "$OWNER" "$*" <<'PY'
import json, sys, time, socket
path, owner, purpose = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({
    "owner": owner,
    "host": socket.gethostname(),
    "purpose": purpose,
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "start_epoch": int(time.time()),
}, open(path, "w"), ensure_ascii=False, indent=1)
PY
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      echo "[heavy_lock] 等待超时（${WAIT}s），当前持有者：" >&2
      status >&2
      return 1
    fi
    echo "[heavy_lock] 被占用，等待中… $(status | head -1)" >&2
    sleep 15
  done
}

release() { rm -rf "$LOCKDIR"; }

case "${1:-}" in
  --status) status ;;
  --release) release; echo "released" ;;
  -h|--help|"")
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    acquire "$@"
    trap 'release' EXIT INT TERM
    echo "[heavy_lock] acquired by $OWNER; running: $*" >&2
    "$@"
    ;;
esac
