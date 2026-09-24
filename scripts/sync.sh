#!/usr/bin/env bash
# 把本仓库同步到 a3-22（aarch64 测量机）。默认只读推送，不删远端文件。
#
# 用法:
#   scripts/sync.sh push          # 本地 → a3-22:~/projects/vllm/tokenizer/
#   scripts/sync.sh pull-remote   # 把远端采集物拉回来（data/ figures/ 除外）
#   scripts/sync.sh status        # 看远端有什么
#
# 约定:
#   - 远端路径 a3-22:~/projects/vllm/tokenizer/（不碰别人目录）
#   - 不带 --delete，避免误删远端采集物
#   - 排除 .git / target / 大文件
set -euo pipefail

HOST="${TOKENIZER_HOST:-a3-22}"
REMOTE_DIR="${TOKENIZER_REMOTE_DIR:-projects/vllm/tokenizer}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXCLUDES=(
  --exclude=.git
  --exclude=target
  --exclude=__pycache__
  --exclude="*.perf.data"
  --exclude="*.so"
  --exclude="*.whl"
  --exclude=models
)

case "${1:-}" in
  push)
    ssh "$HOST" "mkdir -p ~/$REMOTE_DIR"
    rsync -az --info=stats2 "${EXCLUDES[@]}" "$ROOT/" "$HOST:$REMOTE_DIR/"
    echo "pushed -> $HOST:$REMOTE_DIR"
    ;;
  pull-remote)
    mkdir -p "$ROOT/data/remote"
    rsync -az --info=stats2 "${EXCLUDES[@]}" \
      "$HOST:$REMOTE_DIR/data/" "$ROOT/data/remote/"
    echo "pulled <- $HOST:$REMOTE_DIR/data/"
    ;;
  status)
    ssh "$HOST" "ls -la ~/$REMOTE_DIR 2>/dev/null | head -20; echo '--- disk ---'; df -h ~ | tail -1"
    ;;
  -h|--help|"") sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown: $1" >&2; exit 2 ;;
esac
