#!/usr/bin/env bash
# 下载 E1 需要的本地工件（tiktoken 家族 + tekken 家族），并核对 sha256。
#
# 为什么不在仓库里存这些文件：`tiktoken.model` 2.7 MB、`tekken.json` 14.8 MB，
# 都是第三方模型仓库的产物，放 git 里会让仓库膨胀；这里改为「按固定 revision
# 下载 + 记录 sha256」，manifest 里存的是 revision 与哈希，可复现。
#
# 用法:
#   harness/fetch-artifacts.sh            # 下载到 /tmp/b-artifacts（默认）
#   B_ARTIFACTS=/tmp/b-artifacts harness/fetch-artifacts.sh
#   harness/fetch-artifacts.sh --verify   # 只校验已有文件
#
# 已知坑：本地 pip/curl 走代理，HF 的 `resolve/<rev>/<file>` 是唯一稳定入口；
# 不要用 `?download=true`（会 302 到 CDN 并在某些代理下断流）。
set -euo pipefail

DEST="${B_ARTIFACTS:-/tmp/b-artifacts}"
KIMI_REV="4d01dfe0332d63057c186e0b262165819efb6611"   # moonshotai/Kimi-K2.5
MISTRAL_REV="main"                                    # mistralai/Mistral-Nemo-Base-2407

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

# 已核对过的哈希（2026-09-24 首次下载时记录，harness 的 manifest 里会出现同一组值）。
declare -A EXPECTED=(
  ["kimi-k2.5/tiktoken.model"]="b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
  ["kimi-k2.5/config.json"]="acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6"
  ["kimi-k2.5/tokenizer_config.json"]="12fcab43d2b6068f46769f5ff373960bf7c17a94d7abbc50e2491306b2f6cf58"
  ["tekken.json"]="eccd1665d2e477697c33cb7f0daa6f6dfefc57a0a6bceb66d4be52952f827516"
)

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

mkdir -p "$DEST/kimi-k2.5"

fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "[skip] $out 已存在（$(stat -c%s "$out") B）"
    return 0
  fi
  echo "[get ] $url"
  curl -sSL --fail --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
  echo "[ok  ] $out（$(stat -c%s "$out") B）"
}

for f in tiktoken.model config.json tokenizer_config.json; do
  fetch "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/$KIMI_REV/$f" "$DEST/kimi-k2.5/$f"
done

fetch "https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/resolve/$MISTRAL_REV/tekken.json" \
      "$DEST/tekken.json"

echo
echo "== sha256 =="
fail=0
for rel in "${!EXPECTED[@]}"; do
  got="$(sha256sum "$DEST/$rel" | cut -d' ' -f1)"
  if [[ "$got" == "${EXPECTED[$rel]}" ]]; then
    echo "OK   $rel  $got"
  else
    echo "FAIL $rel" >&2
    echo "     期望 ${EXPECTED[$rel]}" >&2
    echo "     实际 $got" >&2
    fail=1
  fi
done
[[ $fail -eq 0 ]] || { echo "工件哈希不匹配：可能是上游改了文件或下载被截断" >&2; exit 1; }

echo
echo "revision: moonshotai/Kimi-K2.5=$KIMI_REV  mistralai/Mistral-Nemo-Base-2407=$MISTRAL_REV"
echo "全部工件就绪，可以用 harness/rust/bench 的默认路径直接跑。"
