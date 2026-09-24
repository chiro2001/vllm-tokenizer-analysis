#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 端到端验证 Rust 前端（tokenizer + chat 模板 + 增量 detokenizer），全 CPU、无 NPU。
#
# 为什么可以不要 Python engine：vLLM 上游自带 vllm-mock-engine
# （rust/src/mock-engine，引擎侧协议模拟器）。用官方 wheel 里的 vllm-rs 当前端、
# 自己编译的 mock engine 当引擎，走真实 ZMQ + msgpack 握手与数据面。
#
# 覆盖到的 tokenizer 相关环节：
#   - minijinja 渲染 HF chat 模板（--tokenizer-mode 默认 auto → hf renderer）
#   - fastokens 后端 load tokenizer.json + encode（/tokenize 与 chat prompt）
#   - DecodeStream 增量 detokenize（stream=true 的 SSE delta）
#
# 编译走 heavy_lock.sh + limit.sh（本机是共享开发机，12 核 / 29 GiB）。
#
# 用法：
#   harness/rust-frontend/e2e_mock_engine.sh --help
#   harness/rust-frontend/e2e_mock_engine.sh --model /home/chiro/models/Qwen3-0.6B
#   harness/rust-frontend/e2e_mock_engine.sh --skip-build
#   EVIDENCE_DIR=data/nextgen harness/rust-frontend/e2e_mock_engine.sh
#
# 退出码：0 全部断言通过；非 0 表示有断言失败（会打印 FAIL 行）。
set -euo pipefail

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
  cat <<'USAGE'

可选参数：
  --model <PATH>        本地模型目录（默认 /home/chiro/models/Qwen3-0.6B）
  --frontend <PATH>     vllm-rs 二进制（默认 /tmp/d-nextgen-wheel/vllm-rs）
  --build-dir <DIR>     mock engine 精简 workspace（默认 ~/.cache/d-nextgen-mock）
  --rust-src <DIR>      vLLM 源码里的 rust/ 目录（默认 HIST_PROJECT 只读副本）
  --main-repo <DIR>     主仓库（提供 scripts/limit.sh 与 heavy_lock.sh）
  --port <PORT>         HTTP 端口（默认 8199）
  --handshake-port <N>  ZMQ 握手端口（默认 29577）
  --vocab-size <N>      mock engine 采样上界（默认 32000）
  --skip-build          不编译，直接用 build-dir/target/release/vllm-mock-engine
  -h, --help            显示本帮助
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/home/chiro/models/Qwen3-0.6B}"
FRONTEND="${FRONTEND:-/tmp/d-nextgen-wheel/vllm-rs}"
BUILD_DIR="${BUILD_DIR:-$HOME/.cache/d-nextgen-mock}"
RUST_SRC="${RUST_SRC:-/home/chiro/projects/vllm/HIST_PROJECT/vllm/rust}"
MAIN_REPO="${MAIN_REPO:-/home/chiro/projects/vllm/tokenizer}"
PORT="${PORT:-8199}"
HANDSHAKE_PORT="${HANDSHAKE_PORT:-29577}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
SKIP_BUILD=0
EVIDENCE_DIR="${EVIDENCE_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) MODEL="$2"; shift 2 ;;
    --frontend) FRONTEND="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --rust-src) RUST_SRC="$2"; shift 2 ;;
    --main-repo) MAIN_REPO="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --handshake-port) HANDSHAKE_PORT="$2"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

LIMIT="$MAIN_REPO/scripts/limit.sh"
LOCK="$MAIN_REPO/scripts/heavy_lock.sh"
say() { printf '[e2e] %s\n' "$*"; }
fail=0
check() {
  if [[ "$2" == "1" ]]; then say "  ✓ $1"; else say "  ✗ FAIL: $1" >&2; fail=1; fi
}

[[ -x "$FRONTEND" ]] || { echo "找不到 vllm-rs：$FRONTEND（先跑 verify_vllm_rs.sh）" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "模型目录不存在：$MODEL" >&2; exit 2; }

# ---- 编译 mock engine（精简 workspace：只保留 engine-core-client + metrics + mock-engine）----
MOCK="$BUILD_DIR/target/release/vllm-mock-engine"
if [[ "$SKIP_BUILD" == "0" && ! -x "$MOCK" ]]; then
  say "准备精简 workspace → $BUILD_DIR"
  mkdir -p "$BUILD_DIR/src"
  for c in engine-core-client metrics mock-engine; do
    [[ -d "$BUILD_DIR/src/$c" ]] || cp -a "$RUST_SRC/src/$c" "$BUILD_DIR/src/"
  done
  if [[ ! -f "$BUILD_DIR/Cargo.toml" ]]; then
    python3 "$SCRIPT_DIR/trim_workspace.py" "$RUST_SRC/Cargo.toml" "$BUILD_DIR/Cargo.toml"
  fi
  say "编译 mock engine（heavy_lock + limit.sh，-j4）"
  ( cd "$BUILD_DIR"
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || true
    OWNER="D-nextgen-mock:$$" "$LOCK" "$LIMIT" cargo build --release -p vllm-mock-engine )
fi
[[ -x "$MOCK" ]] || { echo "mock engine 不可用：$MOCK" >&2; exit 2; }

# ---- 起前端 + mock engine ----
FE_LOG=/tmp/d-e2e-frontend.log
ME_LOG=/tmp/d-e2e-mockengine.log
say "启动 Rust 前端 $FRONTEND"
setsid timeout 90 "$FRONTEND" serve "$MODEL" \
  --data-parallel-size 1 --data-parallel-size-local 0 \
  --handshake-port "$HANDSHAKE_PORT" --host 127.0.0.1 --port "$PORT" \
  --engine-ready-timeout-secs 60 > "$FE_LOG" 2>&1 &
sleep 6
say "启动 mock engine"
setsid timeout 80 "$MOCK" --handshake-address "tcp://127.0.0.1:$HANDSHAKE_PORT" \
  --engine-count 1 --vocab-size "$VOCAB_SIZE" --seed 0 --log-requests > "$ME_LOG" 2>&1 &
sleep 9

grep -q 'loading tokenizer with' "$FE_LOG" && check "前端加载 tokenizer（$(grep -o 'loading tokenizer with [a-z-]*' "$FE_LOG" | head -1)）" 1 \
  || check "前端加载 tokenizer" 0
grep -q 'engines connected' "$FE_LOG" && check "ZMQ 握手完成（engines connected）" 1 || check "ZMQ 握手完成（engines connected）" 0
grep -q 'starting OpenAI server' "$FE_LOG" && check "HTTP server 已启动" 1 || check "HTTP server 已启动" 0

BASE="http://127.0.0.1:$PORT"

H=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$BASE/health" || echo 000)
check "/health 200" "$([[ "$H" == "200" ]] && echo 1 || echo 0)"
M=$(curl -s -m 8 -o /tmp/d-e2e-models.json -w '%{http_code}' "$BASE/v1/models" || echo 000)
check "/v1/models 200" "$([[ "$M" == "200" ]] && echo 1 || echo 0)"

TOK_CODE=$(curl -s -m 8 -o /tmp/d-e2e-tokenize.json -w '%{http_code}' -X POST "$BASE/tokenize" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"你好，Hello!\"}" || echo 000)
check "/tokenize 200（Rust 侧 encode 直接可用）" "$([[ "$TOK_CODE" == "200" ]] && echo 1 || echo 0)"

CHAT_CODE=$(curl -s -m 20 -o /tmp/d-e2e-chat.json -w '%{http_code}' -X POST "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，Hello!\"}],\"max_tokens\":5,\"temperature\":0}" || echo 000)
check "/v1/chat/completions 200" "$([[ "$CHAT_CODE" == "200" ]] && echo 1 || echo 0)"
grep -q '"prompt_tokens":[1-9]' /tmp/d-e2e-chat.json && check "chat prompt_tokens>0（模板渲染+encode 走了）" 1 \
  || check "chat prompt_tokens>0" 0

curl -s -m 20 -N -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":6,\"stream\":true}" \
  > /tmp/d-e2e-stream.txt 2>&1 || true
DELTAS=$(grep -c '"delta":{"content"' /tmp/d-e2e-stream.txt || true)
check "流式返回多个 content delta（增量 decode 生效，实测 $DELTAS 个）" \
  "$([[ "${DELTAS:-0}" -ge 2 ]] && echo 1 || echo 0)"

if [[ -n "$EVIDENCE_DIR" ]]; then
  mkdir -p "$EVIDENCE_DIR"
  # 仓库 .gitignore 屏蔽 *.log，证据改用 .log.txt 后缀才能入库
  cp "$FE_LOG" "$EVIDENCE_DIR/2026-09-24-native-e2e-frontend.log.txt"
  cp "$ME_LOG" "$EVIDENCE_DIR/2026-09-24-native-e2e-mockengine.log.txt"
  cp /tmp/d-e2e-tokenize.json "$EVIDENCE_DIR/e2e-tokenize-response.json"
  cp /tmp/d-e2e-chat.json "$EVIDENCE_DIR/e2e-chat-response.json"
  cp /tmp/d-e2e-stream.txt "$EVIDENCE_DIR/e2e-stream-sse.txt"
  say "证据已写入 $EVIDENCE_DIR"
fi

# 清理：只杀本次起的两条进程
pkill -f "timeout 90 $FRONTEND serve" 2>/dev/null || true
pkill -f "timeout 80 $MOCK" 2>/dev/null || true
sleep 1

if [[ "$fail" == "0" ]]; then say "全部断言通过"; exit 0; fi
say "有断言失败"; exit 1
