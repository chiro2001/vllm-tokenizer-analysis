#!/usr/bin/env python3
"""把 perf 折叠栈（inferno-collapse-perf 的 folded 输出）算成可引用的帧表。

**为什么需要它**：SVG 火焰图为了塞进帧宽会**截断函数名**
（inferno-flamegraph 不给 `--fontsize` 之外的换行手段），图看一眼能看出形状，
但要**引用**"Rust 帧占多少"必须给出精确的函数名与权重。
本脚本输出 `data/cost/e3_flamegraph_frames.csv`，图 + 表一起才是证据。

权重口径：

* `inclusive`：该函数出现在栈里的样本总权重。**父子会重叠**，所以同层求和
  不等于总量；**不能**用它做"某类帧总共占多少"。
* `self`：该函数是**栈顶**的样本权重（真正花在它自己身上的时间）。
  各帧的 self 权重之和 = 总权重，所以**类别汇总一律用 self**。
* **单位是纳秒**：`perf -e cpu-clock` 的 period 以 ns 计，折叠文件里也是 ns。
  实测校验：`e3-1a` 总权重 4.6455e10 ns = **46.5 s**，采样窗口 70 s、进程
  单线程 ⇒ 量级吻合（窗口里有一段在做 import）。用 µs 解释会得到 46454 s，
  明显荒唐——这个坑记在这里，别再把权重当 µs。
权重单位与 `perf -e cpu-clock` 一致（µs）；总量应约等于采样窗口内该进程的 CPU 时间。

用法见 --help。
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import sys


def parse_folded(path: Path) -> tuple[Counter, Counter, float]:
    inclusive: Counter = Counter()
    self_w: Counter = Counter()
    total = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                stack, w_s = line.rsplit(" ", 1)
                w = int(float(w_s))
            except ValueError:
                continue
            total += w
            frames = stack.split(";")
            for fr in set(frames):
                inclusive[fr] += w
            self_w[frames[-1]] += w
    return inclusive, self_w, total


def classify(frame: str) -> str:
    # Rust 泛型帧会长成 `<core::iter::... as ...>`、`<alloc::vec::Vec<T> as ...>`，
    # 所以要先剥掉外层尖括号再判前缀（踩过：不剥的话它们全落到 "other"，
    # 把 Rust 自身的时间藏起来，看起来像"谁都说不清"）。
    probe = frame[1:].split(" as ")[0] if frame.startswith("<") else frame
    if probe.startswith("py::"):
        return "python-trampoline"
    if probe.startswith("tokenizers::"):
        return "rust-tokenizers"
    if re.match(
        r"^(core|alloc|hashbrown|aho_corasick|serde|serde_json|regex|regex_automata|onig|rayon|memchr|log|once_cell)::",
        probe,
    ):
        return "rust-stdlib-or-dep"
    if "Py" in frame or frame.startswith("_Py") or frame in ("python3",):
        return "cpython"
    if frame.startswith("[") or frame.startswith("lib"):
        return "shared-lib"
    if frame in ("[unknown]",) or frame.startswith("unknown"):
        return "unknown"
    if re.match(r"^(malloc|free|cfree|realloc|calloc|memcpy|memmove|memset)$", frame):
        return "libc-alloc-mem"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="folded 栈 → 精确帧表 CSV（补齐 SVG 截断的函数名）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--folded", action="append", required=True,
                    metavar="TAG=PATH", help="形如 e3-1a=/tmp/c-e3/e3-1a.folded")
    ap.add_argument("--out", default="data/cost/e3_flamegraph_frames.csv")
    ap.add_argument("--top", type=int, default=400, help="每个图导出前 N 个帧（按 inclusive）")
    args = ap.parse_args()

    rows = []
    for spec in args.folded:
        tag, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_file():
            print(f"跳过（不存在）: {p}", file=sys.stderr)
            continue
        incl, slf, total = parse_folded(p)
        if not total:
            continue
        for frame, w in incl.most_common(args.top):
            rows.append(
                {
                    "graph": tag,
                    "frame": frame,
                    "class": classify(frame),
                    "inclusive_weight_ns": w,
                    "inclusive_seconds": round(w / 1e9, 4),
                    "inclusive_pct": round(w / total * 100, 4),
                    "self_weight_ns": slf.get(frame, 0),
                    "self_seconds": round(slf.get(frame, 0) / 1e9, 4),
                    "self_pct": round(slf.get(frame, 0) / total * 100, 4),
                    "graph_total_weight_ns": total,
                    "graph_total_seconds": round(total / 1e9, 3),
                }
            )
        # 按类别汇总：**必须用 self**，否则父子重叠会把总和推到 100% 以上
        # （实测：用 inclusive 汇总时 cpython 一类就"占"184%）。
        by_class: Counter = Counter()
        for frame, w in slf.items():
            by_class[classify(frame)] += w
        for cls, w in by_class.most_common():
            rows.append(
                {
                    "graph": tag,
                    "frame": f"<CLASS TOTAL> {cls}",
                    "class": cls,
                    "inclusive_weight_ns": "",
                    "inclusive_seconds": "",
                    "inclusive_pct": "",
                    "self_weight_ns": w,
                    "self_seconds": round(w / 1e9, 4),
                    "self_pct": round(w / total * 100, 4),
                    "graph_total_weight_ns": total,
                    "graph_total_seconds": round(total / 1e9, 3),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph", "frame", "class",
        "inclusive_weight_ns", "inclusive_seconds", "inclusive_pct",
        "self_weight_ns", "self_seconds", "self_pct",
        "graph_total_weight_ns", "graph_total_seconds",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"写出 {out}（{len(rows)} 行）")
    for tag in sorted({r["graph"] for r in rows}):
        cls_rows = [r for r in rows if r["graph"] == tag and r["frame"].startswith("<CLASS")]
        print(f"  [{tag}]  self-time 占比（各帧 self 求和 = 100%）")
        for r in cls_rows:
            print(
                f"    {r['class']:<22} {r['self_pct']:>7.2f}%  "
                f"({r['self_seconds']:.1f} s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
