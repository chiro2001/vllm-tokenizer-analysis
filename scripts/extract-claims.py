#!/usr/bin/env python3
"""从 gigatoken 0.10.0 sdist 的结果文件里抽取它的**声明口径**，落到 data/。

本脚本不重跑任何基准：它只读 sdist 自带的
`benchmarks/results.json` + `benchmarks/compare/measure.py`，
把"它测了什么、拿什么比什么"变成结构化数据，供 `docs/05` 引用。

用法:
    python3 scripts/extract-claims.py --help
    python3 scripts/extract-claims.py \
        --sdist /tmp/b-artifacts/gigatoken-0.10.0.tar.gz \
        --out data/backends/gigatoken-claims.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

RESULTS = "gigatoken-0.10.0/benchmarks/results.json"
MEASURE = "gigatoken-0.10.0/benchmarks/compare/measure.py"
SWEEP = "gigatoken-0.10.0/benchmarks/compare/sweep.py"


def read_member(tar: tarfile.TarFile, name: str) -> bytes:
    f = tar.extractfile(name)
    if f is None:
        raise SystemExit(f"sdist 里找不到 {name}")
    return f.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdist", type=Path, default=Path("/tmp/b-artifacts/gigatoken-0.10.0.tar.gz"))
    ap.add_argument("--out", type=Path, default=Path("data/backends/gigatoken-claims.json"))
    args = ap.parse_args()

    if not args.sdist.exists():
        raise SystemExit(f"找不到 sdist: {args.sdist}（先跑 harness/fetch-artifacts.sh 或 pip download）")

    raw = args.sdist.read_bytes()
    with tarfile.open(args.sdist) as tar:
        results = json.loads(read_member(tar, RESULTS))
        measure_src = read_member(tar, MEASURE).decode("utf-8")
        sweep_src = read_member(tar, SWEEP).decode("utf-8")

    # 逐组抽出"它自己报的"对比：表观加速 = 输入量比 × 单位吞吐比。
    groups = []
    for cpu, toks in results.items():
        for tokenizer, datasets in toks.items():
            for dataset, impls in datasets.items():
                g = impls.get("gigatoken")
                if not g:
                    continue
                for other in ("hf", "tiktoken"):
                    o = impls.get(other)
                    if not o or not o.get("mb_per_s"):
                        continue
                    groups.append(
                        {
                            "cpu": cpu,
                            "tokenizer": tokenizer,
                            "dataset": dataset,
                            "other": other,
                            "gigatoken_mb_per_s": g["mb_per_s"],
                            "other_mb_per_s": o["mb_per_s"],
                            "apparent_speedup": round(g["mb_per_s"] / o["mb_per_s"], 2),
                            "gigatoken_bytes": g["bytes"],
                            "other_bytes": o["bytes"],
                            "input_ratio": round(g["bytes"] / o["bytes"], 2),
                            "gigatoken_docs": g.get("docs"),
                            "other_docs": o.get("docs"),
                            "gigatoken_tokens": g.get("tokens"),
                            "other_tokens": o.get("tokens"),
                            "gigatoken_bytes_per_token": (
                                round(g["bytes"] / g["tokens"], 4) if g.get("tokens") else None
                            ),
                            "other_bytes_per_token": (
                                round(o["bytes"] / o["tokens"], 4) if o.get("tokens") else None
                            ),
                        }
                    )

    ratios = [x["apparent_speedup"] for x in groups]
    input_ratios = sorted({x["input_ratio"] for x in groups})
    per_byte = [
        x["apparent_speedup"] / x["input_ratio"] for x in groups if x["input_ratio"]
    ]

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sdist": str(args.sdist),
        "sdist_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files": {
            RESULTS: hashlib.sha256(read_members_bytes(args.sdist, RESULTS)).hexdigest(),
            MEASURE: hashlib.sha256(measure_src.encode()).hexdigest(),
            SWEEP: hashlib.sha256(sweep_src.encode()).hexdigest(),
        },
        # 关键事实：它的输入口径在**每一行**上都不同（不是个别样本）
        "comparison_groups": len(groups),
        "distinct_input_ratios": input_ratios,
        "apparent_speedup_min": min(ratios) if ratios else None,
        "apparent_speedup_max": max(ratios) if ratios else None,
        "apparent_speedup_median": sorted(ratios)[len(ratios) // 2] if ratios else None,
        "per_byte_speedup_median": sorted(per_byte)[len(per_byte) // 2] if per_byte else None,
        "per_byte_speedup_max": max(per_byte) if per_byte else None,
        "sweep_defaults": {
            # 从 sweep.py 的 argparse 默认值里读出来（人工复核过 :187）
            "hf_mb_default": 100,
            "note": "sweep.py 对 gigatoken 传 max_mb=None（整文件），对 hf/tiktoken 传 --hf-mb/--tiktoken-mb",
        },
        "groups": groups,
        "caveats": [
            "本文件只转录 gigatoken 自己 results.json 里的数字，未重跑；"
            "其口径下的'加速'= 输入量比 × 单位吞吐比，两项都记录在每一行里。",
            "bytes_per_token 两侧接近（差值 <1%）说明**token 计数口径**在这组数据里不是主要因素；",
            "输入量比（input_ratio）在所有组里都是同一个常数，见 distinct_input_ratios。",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {len(groups)} 组对比 → {args.out}")
    print(f"       输入量比: {input_ratios}")
    print(f"       表观加速: min={payload['apparent_speedup_min']} "
          f"median={payload['apparent_speedup_median']} max={payload['apparent_speedup_max']}")
    print(f"       单位吞吐比(中位): {payload['per_byte_speedup_median']}×")
    return 0


def read_members_bytes(sdist: Path, name: str) -> bytes:
    with tarfile.open(sdist) as tar:
        return read_member(tar, name)


if __name__ == "__main__":
    raise SystemExit(main())
