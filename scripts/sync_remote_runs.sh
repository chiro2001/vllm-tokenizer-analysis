#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync_remote_runs.sh —— 从 a3-22 **只读** 拉取历史 run 的元数据
#
# 只拉三样东西（每个 run 通常 < 2 MB，避免把 29 个 run 的 torch trace 拖下来）：
#   1. lite-profiler/lite.log       （或 run 根下的 lite.log）
#   2. requests/*.json / requests.tsv（用来判断负载形态）
#   3. validation-summary.json       （镜像 id / 端口 / 起止时间等口径）
#
# **绝不在远端写任何东西**：全程只用 ssh 的 ls/cat/tar 读操作。
#
# 用法:
#   scripts/sync_remote_runs.sh --list
#   scripts/sync_remote_runs.sh --fetch /tmp/c-remote
#   HOST=a3-21 scripts/sync_remote_runs.sh --list         # 换台机器
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${HOST:-a3-22}"
REMOTE_RUNS="${REMOTE_RUNS:-~/projects/vllm/HIST_PROJECT/runs}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no
          -o ServerAliveInterval=15 -o ServerAliveCountMax=8)

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

case "${1:-}" in
    --list)
        ssh "${SSH_OPTS[@]}" "$HOST" "ls -d $REMOTE_RUNS/liteprof_* 2>/dev/null" | sort
        ;;
    --fetch)
        DEST="${2:?用法: --fetch <目标目录>}"
        mkdir -p "$DEST"
        mapfile -t runs < <(ssh "${SSH_OPTS[@]}" "$HOST" \
            "ls -d $REMOTE_RUNS/liteprof_* 2>/dev/null" | sort)
        echo "远端 run 数: ${#runs[@]}" >&2
        for r in "${runs[@]}"; do
            name="$(basename "$r")"
            out="$DEST/$name"
            if [ -f "$out/lite-profiler/lite.log" ] || [ -f "$out/lite.log" ]; then
                echo "[skip] $name 已存在" >&2
                continue
            fi
            mkdir -p "$out/requests" "$out/lite-profiler"
            # 远端只做 tar 打包（读），本地解包 → 单向
            ssh "${SSH_OPTS[@]}" "$HOST" \
                "cd $r && tar cf - \
                   \$( [ -f lite-profiler/lite.log ] && echo lite-profiler/lite.log ) \
                   \$( [ -f lite.log ] && echo lite.log ) \
                   \$( [ -f validation-summary.json ] && echo validation-summary.json ) \
                   \$( ls requests/*.json requests/requests.tsv 2>/dev/null ) \
                   2>/dev/null" | tar xf - -C "$out" || {
                echo "[warn] $name 拉取失败" >&2; continue; }
            echo "[ok] $name ($(du -sh "$out" | cut -f1))" >&2
        done
        echo "完成 → $DEST" >&2
        ;;
    -h|--help|"") usage ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
esac
