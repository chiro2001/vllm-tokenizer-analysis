#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# fetch_remote_bg.sh —— 后台反复重试拉取 a3-22 的历史 run 元数据（只读）
#
# 背景：a3-22 走 ProxyJump，链路不稳（多次 `Connection closed by UNKNOWN`）。
# 本脚本每轮重试 --list，成功后一次性 --fetch，全部只读。
#
# 用法:
#   nohup scripts/fetch_remote_bg.sh /tmp/c-remote > /tmp/c-fetch.log 2>&1 &
# ---------------------------------------------------------------------------
set -uo pipefail

usage() {
    cat <<'EOF'
用法: fetch_remote_bg.sh [选项] [目标目录]

  目标目录      拉取落盘位置（默认 /tmp/c-remote）
  -h, --help    显示本帮助

环境变量:
  MAX_ROUNDS    最多重试轮数（默认 12）
  SLEEP_S       每轮间隔秒数（默认 120）

行为: 在 a3-22 / a3-21 之间轮流重试 --list，成功后一次性 --fetch。
**全程只读远端**，不在远端写任何文件。
EOF
    exit 0
}

case "${1:-}" in
    -h|--help) usage ;;
esac

DEST="${1:-/tmp/c-remote}"
MAX_ROUNDS="${MAX_ROUNDS:-12}"
SLEEP_S="${SLEEP_S:-120}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[bg] 开始，目标 $DEST，最多 $MAX_ROUNDS 轮，每轮间隔 ${SLEEP_S}s"
for round in $(seq 1 "$MAX_ROUNDS"); do
    echo "[bg] === 第 $round 轮 $(date -Is)"
    for host in a3-22 a3-21; do
        echo "[bg] 尝试 $host"
        if timeout 180 env HOST="$host" "$HERE/sync_remote_runs.sh" --list \
                > "/tmp/c-remote-$host.txt" 2>/dev/null; then
            n=$(grep -c liteprof "/tmp/c-remote-$host.txt" || true)
            echo "[bg] $host 列表成功：$n 个 run"
            if [ "$n" -gt 0 ]; then
                if timeout 3600 env HOST="$host" "$HERE/sync_remote_runs.sh" --fetch "$DEST"; then
                    echo "[bg] 拉取完成 → $DEST"
                    exit 0
                fi
            fi
        else
            echo "[bg] $host 不可达（rc=$?）"
        fi
    done
    sleep "$SLEEP_S"
done
echo "[bg] 轮次用尽，未能拉取远端数据"
exit 1
