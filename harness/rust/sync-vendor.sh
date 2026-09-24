#!/usr/bin/env bash
# 校验/重新生成 vendored 的 vLLM `vllm-tokenizer` 快照。
#
# 为什么 vendor：上游 `rust/src/tokenizer` 是 `vllm` workspace 的一个成员，
# 直接把整个 vLLM rust workspace 拉进来会连带 server/engine-core 等几十个 crate。
# 这里的六个后端全在 `vllm-tokenizer` 一个包里，因此只复制它一个包 +
# 一份展开后的 Cargo.toml（把 workspace 继承字段写成字面量）。
#
# 用法:
#   harness/rust/sync-vendor.sh --check     # 校验快照与上游逐字节一致（CI/文档用）
#   harness/rust/sync-vendor.sh --update    # 从上游重新复制（需改 vLLM 源码时）
#
# 环境变量:
#   VLLM_SRC   vLLM 源码根（默认 /home/chiro/projects/vllm/HIST_PROJECT/vllm）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-/home/chiro/projects/vllm/HIST_PROJECT/vllm}"
UPSTREAM="$VLLM_SRC/rust/src/tokenizer"
VENDOR="$HERE/vendor/vllm-tokenizer"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

[[ $# -eq 1 ]] || { usage; exit 1; }

case "$1" in
  --check)
    [[ -d "$UPSTREAM" ]] || { echo "上游不存在: $UPSTREAM" >&2; exit 1; }
    fail=0
    # src/ 必须逐字节一致
    if ! diff -r "$UPSTREAM/src" "$VENDOR/src" > /tmp/vendor-src.diff 2>&1; then
      echo "!! vendor/vllm-tokenizer/src 与上游不一致:" >&2
      head -30 /tmp/vendor-src.diff >&2
      fail=1
    fi
    # benches/ 备份也必须一致
    if ! diff -r "$UPSTREAM/benches" "$VENDOR/upstream-benches" > /tmp/vendor-bench.diff 2>&1; then
      echo "!! vendor/vllm-tokenizer/upstream-benches 与上游不一致:" >&2
      head -30 /tmp/vendor-bench.diff >&2
      fail=1
    fi
    if [[ $fail -eq 0 ]]; then
      echo "OK: vllm-tokenizer 快照与 $UPSTREAM 一致（src/ 与 benches/）"
    fi
    exit $fail
    ;;
  --update)
    rm -rf "$VENDOR/src" "$VENDOR/upstream-benches"
    mkdir -p "$VENDOR"
    cp -r "$UPSTREAM/src" "$VENDOR/src"
    cp -r "$UPSTREAM/benches" "$VENDOR/upstream-benches"
    echo "已从 $UPSTREAM 更新（Cargo.toml 不覆盖：它是展开后的版本，需手工核对依赖版本）"
    ;;
  -h|--help|"") usage ;;
  *) echo "未知参数: $1" >&2; usage; exit 1 ;;
esac
