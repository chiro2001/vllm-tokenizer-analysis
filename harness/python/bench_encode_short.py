#!/usr/bin/env python3
"""E2.1 补充：**短 prompt 的 encode**——回答"固定成本有多大"。

## 为什么要加这个实验

`data/historical/historical_tokenizer_scopes.csv` 里 wave3/wave4 的稳态 run
（每个 41 次采样）用的是 **prompt_tokens = 10–11** 的极小请求，
encode 均值却是 **162–461 µs**（16 个 run）。而本机测 220 token 只要 283 µs。
两者放一起只能得出一个结论：**encode 有相当大的固定成本**，
线性外推在短 prompt 上不成立。

本脚本刻意把 ISL 从 8 一路加到 1024，把这条曲线的**拐点**量出来。

用法见 --help。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_renderer,
    make_manifest,
    measure,
    now_iso,
    summarize,
    write_json,
)
from corpora import make_text  # noqa: E402

DEFAULT_ISLS = [8, 10, 16, 32, 64, 128, 256, 512, 1024]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="短 prompt（8–1024 token）的 tokenizer: encode 成本，量固定成本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--corpus", default="mixed", choices=["en", "zh", "code", "mixed"])
    ap.add_argument("--isls", default=",".join(map(str, DEFAULT_ISLS)))
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="/workspace/data/cost/e2_encode_short.json")
    args = ap.parse_args()

    renderer, _cfg = build_renderer(args.model)
    tokenizer = renderer.get_tokenizer()
    from vllm.renderers.params import TokenizeParams

    params = TokenizeParams(max_total_tokens=40960, max_output_tokens=0)
    kwargs = params.get_encode_kwargs()

    isls = [int(x) for x in args.isls.split(",") if x]
    rows: list[dict[str, Any]] = []
    print(
        f"{'target':>7}{'actual':>8}{'chars':>8}{'mean_us':>10}"
        f"{'p50_us':>10}{'p99_us':>10}{'us/token':>10}"
    )
    for target in isls:
        text, actual = make_text(tokenizer, target, args.corpus)
        ns, _ = measure(lambda: tokenizer(text, **kwargs), args.repeats, args.warmup)
        st = summarize(ns)
        per_tok = st.mean_us / actual if actual else float("nan")
        rows.append(
            {
                "case": "encode_short",
                "scope": "tokenizer: encode",
                "target_tokens": target,
                "actual_tokens": actual,
                "n_chars": len(text),
                "stats_us": st.as_dict(),
                "us_per_token": round(per_tok, 3),
            }
        )
        print(
            f"{target:>7}{actual:>8}{len(text):>8}{st.mean_us:>10.1f}"
            f"{st.p50_us:>10.1f}{st.p99_us:>10.1f}{per_tok:>10.3f}",
            flush=True,
        )

    # 固定成本估计：用最小的两个点做差（比线性外推更稳，不假设线性）
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        dt = b["actual_tokens"] - a["actual_tokens"]
        marginal = (
            (b["stats_us"]["mean_us"] - a["stats_us"]["mean_us"]) / dt if dt else None
        )
    else:
        marginal = None

    write_json(
        args.out,
        {
            "manifest": make_manifest(
                experiment="E2.1 short-prompt encode",
                script=os.path.abspath(__file__),
                model=args.model,
                extra={
                    "args": vars(args),
                    "fixed_cost_note": (
                        "用小 token 数两点的差分估计边际成本；固定成本 = "
                        "mean(最小点) − 边际 × 该点 token 数"
                    ),
                    "marginal_us_per_token_from_two_smallest": (
                        round(marginal, 4) if marginal else None
                    ),
                    "finished_at": now_iso(),
                },
            ),
            "results": rows,
        },
    )
    print(f"\n写到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
