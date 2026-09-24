#!/usr/bin/env bash
# 跨装置桥梁校准：把 **Rust harness** 与 **Python harness** 的数字对齐。
#
# 为什么需要（`plan/COORDINATION.md` §5.4）：两个装置用不同的计时机制
# （Rust 侧自建 `Instant` 计时器 + 绑核；Python 侧 `time.perf_counter` +
# CPython 解释器开销），如果直接比，会把"装置差异"当成"实现差异"。
#
# 校准对象：**两个装置都存在的实现** —— HF `tokenizers` 与 `fastokens`。
# 判据：两装置对同一实现的相对差 ≤15%，否则禁止跨装置同表比较。
#
# 注意本脚本**串行**跑两次测量（Rust 一次、Python 一次），
# 中间不跑别的东西；两次都在同一个绑核窗口里，避免机器负载漂移
# 把"装置差异"污染成"负载差异"。
#
# 用法:
#   CORES=2 harness/bridge-calibration.sh                 # 直接跑（默认 40 轮）
#   ITERS=50 harness/bridge-calibration.sh                # 更多轮
#   harness/bridge-calibration.sh --help                  # 只打印用法
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
LOCK="${B_LOCK:-/home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh}"
LIM="${B_LIM:-/home/chiro/projects/vllm/tokenizer/scripts/limit.sh}"
BIN="${B_BIN:-$ROOT/harness/rust/target/release/b-backends}"
PY="${B_PYTHON:-$HOME/miniforge3/bin/python3}"
OUT="${B_OUT:-$ROOT/data/backends}"
ITERS="${ITERS:-40}"
CORPUS="${CORPUS:-mixed}"

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; }

# 无参数 = 直接跑（与其它 harness 脚本一致）；--help 只打印用法。
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  ""|--run) ;;
  *) echo "未知参数: $1" >&2; usage; exit 1 ;;
esac

[[ -x "$BIN" ]] || { echo "找不到 Rust harness: $BIN（先 cargo build --release --features gigatoken）" >&2; exit 1; }
command -v "$PY" >/dev/null || { echo "找不到 python: $PY" >&2; exit 1; }

mkdir -p "$OUT/tmp"
RS_JSON="$OUT/tmp/bridge-rust.jsonl"
PY_JSON="$OUT/tmp/bridge-python.json"

# ---- 1) Rust 侧：hf 与 fastokens_byte_level，单条 encode/decode ----------
echo "[1/3] Rust 装置（$(date --iso-8601=seconds)）"
"$LOCK" "$LIM" "$BIN" run \
    --backend hf,fastokens_byte_level \
    --corpus "$CORPUS" --lengths 128,1k --ops encode,decode \
    --max-iters "$ITERS" --budget-ms 200 \
    --prefix bridge-rust >&2

# ---- 2) Python 侧：同一批实现（注意 fastokens 在 Python 侧没有 ByteLevel 旁路）----
echo "[2/3] Python 装置（$(date --iso-8601=seconds)）"
"$LOCK" "$LIM" "$PY" "$HERE/python/bench_backends.py" \
    --impl hf,fastokens --corpus "$CORPUS" --lengths 128,1k \
    --iters "$ITERS" --json-out "$PY_JSON" >&2

# ---- 3) 汇总 ------------------------------------------------------------
echo "[3/3] 汇总"
"$PY" - "$OUT/bridge-rust.jsonl" "$PY_JSON" "$OUT/bridge-calibration.json" <<'PY'
import json, sys, time
from pathlib import Path

rust_path, py_path, out_path = map(Path, sys.argv[1:4])
rust_rows = [json.loads(l) for l in rust_path.read_text().splitlines() if l.strip()]
py_rows = json.loads(py_path.read_text())["rows"]

# Rust 侧：hf 与 fastokens_byte_level 的 encode/decode 中位数
def rust_get(backend, op, length):
    for r in rust_rows:
        if (r["backend"], r["op"], r["length"]) == (backend, op, length):
            return r["median_us"]
    return None

def py_get(impl, op, length):
    for r in py_rows:
        if (r.get("impl"), r.get("op"), str(r.get("length"))) == (impl, op, str(length)):
            return r.get("median_us")
    return None

lengths = ["128", "1k"]
py_len = {"128": 128, "1k": 1024}
items = []
for length in lengths:
    hf_r = rust_get("hf", "encode", length)
    hf_p = py_get("hf_tokenizers", "encode", py_len[length])
    ft_r = rust_get("fastokens_byte_level", "encode", length)
    ft_p = py_get("fastokens", "encode", py_len[length])
    if hf_r and hf_p:
        items.append(("hf encode", length, hf_r, hf_p))
    if ft_r and ft_p:
        items.append(("fastokens encode", length, ft_r, ft_p))

def rel(a, b):
    return abs(a - b) / max(a, b) * 1e6 / 1e6 if max(a, b) else float("nan")

report = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "purpose": "跨装置桥梁校准：Rust harness vs Python harness（plan/COORDINATION.md §5.4）",
    "threshold": 0.15,
    "note": (
        "两侧测的是**同一个实现**（HF tokenizers / fastokens），装置不同："
        "Rust 侧是 vllm-tokenizer crate 的自建计时器 + taskset 绑核；"
        "Python 侧是 CPython 解释器 + time.perf_counter。"
        "注意 Rust 的 fastokens_byte_level 是 vLLM 自研旁路，Python 侧无此旁路，"
        "所以这一行的差值同时含装置差与实现差。"
    ),
    "items": [],
}
for name, length, r, p in items:
    d = abs(r - p) / max(r, p)
    report["items"].append({
        "item": name,
        "length": length,
        "rust_us": round(r, 3),
        "python_us": round(p, 3),
        "rel_diff": round(d, 4),
        "within_threshold": d <= 0.15,
    })

ok = [i for i in report["items"] if i["within_threshold"]]
report["summary"] = {
    "total": len(report["items"]),
    "within_threshold": len(ok),
    "verdict": "允许跨装置比较" if len(ok) == len(report["items"]) and ok else "存在超阈值项：相关行禁止跨装置同表",
}
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"{'item':<20}{'len':>5}{'rust µs':>10}{'py µs':>10}{'rel':>9}  verdict")
for i in report["items"]:
    print(f"{i['item']:<20}{i['length']:>5}{i['rust_us']:>10.2f}{i['python_us']:>10.2f}"
          f"{i['rel_diff']*100:>8.1f}%  {'OK' if i['within_threshold'] else 'FAIL(>15%)'}")
print(f"\n{report['summary']['verdict']} → {out_path}")
PY
