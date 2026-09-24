#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 从官方 PyPI wheel 抽取 vllm-rs，做一次启动验证（不需要 Python / 不需要 NPU）。
#
# 做三件事：
#   1. 用 HTTP Range 抽取 wheel 里的 vllm/vllm-rs（只下十几 MB，不下全量 300 MB）
#   2. 枚举 CLI 面：--help / serve --help / frontend --help
#   3. 真启动一次：`vllm-rs serve <MODEL> --data-parallel-size-local 0`
#      —— 纯 Rust 路径，会真的加载 tokenizer.json、起 chat renderer、
#         在 ZMQ 上等 engine 握手。断言日志里出现 "loading tokenizer with"。
#
# 用法：
#   harness/rust-frontend/verify_vllm_rs.sh --help
#   harness/rust-frontend/verify_vllm_rs.sh                       # 默认 x86_64 + 本机 Qwen3-0.6B
#   harness/rust-frontend/verify_vllm_rs.sh --arch aarch64 --qemu  # 交叉架构（用 qemu-aarch64）
#   harness/rust-frontend/verify_vllm_rs.sh --model /path/to/model --serve-seconds 25
#
# 退出码：0 全部通过；非 0 表示某一步失败（会打印 FAIL 行）。
set -euo pipefail

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
  cat <<'EOF'

可选参数：
  --arch <x86_64|aarch64>   wheel 目标架构（默认：本机架构）
  --out <DIR>               抽取目录（默认 /tmp/d-nextgen-wheel[-<arch>]）
  --model <PATH>            用于启动验证的本地模型目录
  --serve-seconds <N>       启动验证观察秒数（默认 20）
  --qemu                    用 qemu-<arch> -L 运行交叉架构二进制
  --skip-serve              只做抽取 + --help，不启动
  --no-fetch                复用已抽取的二进制
  -h, --help                显示本帮助
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH=""
OUT=""
MODEL="${MODEL:-/home/chiro/models/Qwen3-0.6B}"
SERVE_SECONDS=20
USE_QEMU=0
SKIP_SERVE=0
NO_FETCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --serve-seconds) SERVE_SECONDS="$2"; shift 2 ;;
    --qemu) USE_QEMU=1; shift ;;
    --skip-serve) SKIP_SERVE=1; shift ;;
    --no-fetch) NO_FETCH=1; shift ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  case "$(uname -m)" in
    aarch64|arm64) ARCH=aarch64 ;;
    *) ARCH=x86_64 ;;
  esac
fi
if [[ -z "$OUT" ]]; then
  if [[ "$ARCH" == "$(uname -m)" || ( "$ARCH" == "aarch64" && "$(uname -m)" == "arm64" ) ]]; then
    OUT=/tmp/d-nextgen-wheel
  else
    OUT="/tmp/d-nextgen-wheel-$ARCH"
  fi
fi

BIN="$OUT/vllm-rs"
EVIDENCE_DIR="${EVIDENCE_DIR:-}"
fail=0

say() { printf '[verify_vllm_rs] %s\n' "$*"; }
fail_step() { printf '[verify_vllm_rs] FAIL: %s\n' "$*" >&2; fail=1; }

# ---- 1. 抽取 ----
if [[ "$NO_FETCH" == "0" || ! -x "$BIN" ]]; then
  say "从官方 wheel 抽取 vllm-rs → $OUT（HTTP Range，只取尾部）"
  python3 "$SCRIPT_DIR/fetch_vllm_rs.py" --arch "$ARCH" --out "$OUT" \
    --manifest "$OUT/manifest-$ARCH.json" > "$OUT/extract.json"
  say "抽取完成：$(python3 -c "import json;d=json.load(open('$OUT/extract.json'));print(f\"下载 {d['range_bytes_downloaded']/1e6:.1f} MB / wheel {d['wheel_size']/1e6:.1f} MB = {d['range_download_share']*100:.1f}%\")")"
else
  say "复用已有二进制 $BIN"
fi

[[ -x "$BIN" ]] || { fail_step "$BIN 不可执行"; exit 1; }
say "文件类型：$(file -b "$BIN")"

# ---- 2. 运行器 ----
run_bin() {
  if [[ "$USE_QEMU" == "1" ]]; then
    local sysroot
    sysroot=$(ls -d /usr/${ARCH}-linux-gnu 2>/dev/null | head -1 || true)
    [[ -n "$sysroot" ]] || { echo "找不到 ${ARCH} sysroot" >&2; return 127; }
    qemu-${ARCH} -L "$sysroot" "$BIN" "$@"
  else
    "$BIN" "$@"
  fi
}

# ---- 3. CLI 枚举 ----
say "枚举 CLI 面"
run_bin --help > /tmp/d-verify-help.txt 2>&1 || fail_step "vllm-rs --help 失败"
run_bin serve --help > /tmp/d-verify-serve-help.txt 2>&1 || fail_step "vllm-rs serve --help 失败"
run_bin frontend --help > /tmp/d-verify-frontend-help.txt 2>&1 || fail_step "vllm-rs frontend --help 失败"
for f in /tmp/d-verify-help.txt /tmp/d-verify-serve-help.txt /tmp/d-verify-frontend-help.txt; do
  say "  $(basename "$f"): $(wc -l < "$f") 行"
done
grep -q "frontend" /tmp/d-verify-help.txt || fail_step "顶层 help 里没有 frontend 子命令"
grep -q "serve" /tmp/d-verify-help.txt || fail_step "顶层 help 里没有 serve 子命令"

# ---- 4. 真启动（tokenizer 加载路径）----
if [[ "$SKIP_SERVE" == "0" ]]; then
  if [[ ! -d "$MODEL" ]]; then
    fail_step "模型目录不存在：$MODEL（用 --model 指定）"
  else
    PORT="${PORT:-8199}"
    HANDSHAKE_PORT="${HANDSHAKE_PORT:-29577}"
    LOG="/tmp/d-verify-serve.log"
    say "启动 frontend-only 模式（data-parallel-size-local 0，纯 CPU）"
    setsid timeout "$SERVE_SECONDS" "$BIN" serve "$MODEL" \
      --data-parallel-size 1 --data-parallel-size-local 0 \
      --handshake-port "$HANDSHAKE_PORT" --host 127.0.0.1 --port "$PORT" \
      --engine-ready-timeout-secs "$((SERVE_SECONDS - 2))" > "$LOG" 2>&1 &
    sleep $(( SERVE_SECONDS - 6 ))
    if grep -q "loading tokenizer with" "$LOG"; then
      say "  ✓ 加载了 tokenizer：$(grep -o 'loading tokenizer with [a-z-]*' "$LOG" | head -1)"
    else
      fail_step "日志里没有 'loading tokenizer with'，见 $LOG"
    fi
    grep -q "loaded chat backend" "$LOG" && say "  ✓ chat backend/renderer 就绪" \
      || fail_step "chat backend 未就绪"
    grep -q "waiting for engines to connect" "$LOG" \
      && say "  ✓ 已在 ZMQ 上等待 engine 握手（HTTP 端口要等握手完成才 bind）" \
      || fail_step "未观察到 engine 握手等待阶段"
    wait 2>/dev/null || true
    if [[ -n "$EVIDENCE_DIR" ]]; then
      mkdir -p "$EVIDENCE_DIR"
      # 仓库 .gitignore 屏蔽 *.log，证据改用 .log.txt 后缀才能入库
      cp "$LOG" "$EVIDENCE_DIR/verify-serve-$ARCH.log.txt"
      cp /tmp/d-verify-help.txt "$EVIDENCE_DIR/vllm-rs-help-$ARCH.txt"
      cp /tmp/d-verify-serve-help.txt "$EVIDENCE_DIR/vllm-rs-serve-help-$ARCH.txt"
    fi
  fi
fi

if [[ "$fail" == "0" ]]; then
  say "全部通过（arch=$ARCH）"
  exit 0
fi
exit 1
