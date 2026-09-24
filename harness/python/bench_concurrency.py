#!/usr/bin/env python3
"""E2.5：前端进程内的并发曲线（1→64 并发），找饱和点。

## 装置说明（重要）

真机上「前端」是一个 **asyncio 事件循环 + ThreadPoolExecutor**（`llm` worker 数
由 `renderer_num_workers` 决定，默认 1）。`BaseRenderer` 在 `__init__` 里用
`make_async(..., executor=self._executor)` 把 encode / render_messages / decode
三件事都 offload 到这个池里。所以本脚本**不启 HTTP server**，而是直接复现同一
套 offload 结构：asyncio 事件循环 + 同一个 renderer + 并发任务。

这样测的是**前端进程内部的排队与饱和**，不掺网络与 HTTP 解析（那些对三条
tokenizer scope 的口径无贡献）。

## 一个已知的隐藏劣化项

`vllm/tokenizers/hf.py:51-55`：tokenizer 池在 `renderer_num_workers + 1` 份
deepcopy 之外，**池空时不阻塞、而是现场 deepcopy 并让池无上限增长**。
所以并发超过 worker 数时，每次抢占都要付一次 deepcopy。本脚本用
`--worker-counts` 对比不同池大小下的曲线，把这个项量化出来。

用法见 --help。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_renderer,
    make_manifest,
    now_iso,
    read_launch_snapshot,
    write_json,
)
from corpora import TOOL_SPECS, make_chat_messages, make_text  # noqa: E402

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]


def loadavg() -> str:
    try:
        return open("/proc/loadavg").read().split()[0]
    except OSError:
        return "?"


async def drive(
    renderer,
    payloads: list[Any],
    kind: str,
    concurrency: int,
    n_requests: int,
    tok_params,
    chat_params,
    prewarm: int,
) -> dict[str, Any]:
    """以给定并发度打 n_requests 个请求，返回延迟分布与吞吐。

    ``prewarm`` 固定用 **worker 数**（不是被测并发度）：否则每个并发点的预热
    本身就会把 tokenizer 池撑到该并发度，测出来的曲线会把「池预热」当成
    「并发扩展性」。池的实际增长用 deepcopy 计数器单独量化。
    """
    # deepcopy 计数器：vllm/tokenizers/hf.py 的池空时现场 deepcopy tokenizer。
    # 直接包 copy.deepcopy，量化这一项，而不是靠推断。
    import vllm.tokenizers.hf as hf_mod

    real_deepcopy = hf_mod.copy.deepcopy
    counter = {"n": 0, "ns": 0}

    def counting_deepcopy(*a, **kw):
        t0 = time.perf_counter_ns()
        out = real_deepcopy(*a, **kw)
        counter["n"] += 1
        counter["ns"] += time.perf_counter_ns() - t0
        return out

    hf_mod.copy.deepcopy = counting_deepcopy
    latencies: list[int] = []
    completed = 0
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int):
        nonlocal completed
        payload = payloads[i % len(payloads)]
        async with sem:
            t0 = time.perf_counter_ns()
            if kind == "render":
                await renderer.render_chat_async([payload], chat_params, tok_params)
            elif kind == "cmpl":
                await renderer.render_cmpl_async([payload], tok_params)
            else:
                raise ValueError(kind)
            dt = time.perf_counter_ns() - t0
        async with lock:
            latencies.append(dt)
            completed += 1

    try:
        # 预热：固定轮数（= workers），让池与线程就位；不计时也不计数
        await asyncio.gather(*[one(i) for i in range(max(1, prewarm))])
        latencies.clear()
        counter["n"] = 0
        counter["ns"] = 0

        wall_t0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(n_requests)])
        wall = time.perf_counter() - wall_t0
    finally:
        hf_mod.copy.deepcopy = real_deepcopy

    us = sorted(v / 1000 for v in latencies)
    return {
        "concurrency": concurrency,
        "n_requests": len(latencies),
        "wall_s": round(wall, 4),
        "throughput_req_s": round(len(latencies) / wall, 2),
        "latency_mean_us": round(statistics.fmean(us), 1),
        "latency_p50_us": round(us[len(us) // 2], 1),
        "latency_p90_us": round(us[int(len(us) * 0.9)], 1),
        "latency_p99_us": round(us[min(int(len(us) * 0.99), len(us) - 1)], 1),
        "latency_max_us": round(us[-1], 1),
        "loadavg_after": loadavg(),
        "tokenizer_deepcopy_calls": counter["n"],
        "tokenizer_deepcopy_total_us": round(counter["ns"] / 1000, 1),
        "tokenizer_deepcopy_mean_us": (
            round(counter["ns"] / 1000 / counter["n"], 1) if counter["n"] else None
        ),
        "tokenizer_deepcopy_us_per_request": round(counter["ns"] / 1000 / len(latencies), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2.5 前端并发曲线（1→64），找饱和点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--out", default="/workspace/data/cost/e2_concurrency.json")
    ap.add_argument("--corpus", default="mixed")
    ap.add_argument("--kind", default="render", choices=["render", "cmpl"],
                    help="render=chat+tools（含 encode），cmpl=纯文本 completion")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCY)))
    ap.add_argument("--worker-counts", default="1,4,8",
                    help="renderer_num_workers 的取值（tokenizer 池 = workers+1 份 deepcopy）")
    ap.add_argument("--n-requests", type=int, default=256)
    ap.add_argument("--with-tools", action="store_true", default=True)
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrency.split(",") if x]
    worker_counts = [int(x) for x in args.worker_counts.split(",") if x]

    results: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for workers in worker_counts:
        renderer, cfg = build_renderer(args.model, renderer_num_workers=workers)
        from vllm.renderers.params import ChatParams, TokenizeParams

        tokenizer = renderer.get_tokenizer()
        tok_params = TokenizeParams(max_total_tokens=40960, max_output_tokens=256)
        filler, actual = make_text(tokenizer, args.isl, args.corpus)
        messages = make_chat_messages(filler, with_tools=args.with_tools)
        chat_params = ChatParams(
            chat_template=None,
            chat_template_content_format="auto",
            chat_template_kwargs=({"tools": TOOL_SPECS} if args.with_tools else {}),
        )
        payloads = [messages] if args.kind == "render" else [{"prompt": filler}]

        m = make_manifest(
            experiment="E2.5 frontend concurrency",
            script=os.path.abspath(__file__),
            model=args.model,
            extra={
                "args": vars(args),
                "renderer_num_workers": workers,
                "tokenizer_pool_size": workers + 1,
                "pool_growth_rule": (
                    "vllm/tokenizers/hf.py:51-55 —— 池空时不阻塞，现场 deepcopy "
                    "并把新副本放回池，池大小无上限"
                ),
                "kind": args.kind,
                "isl_target": args.isl,
                "isl_actual": actual,
                "n_payload_variants": len(payloads),
                "launch_snapshot": read_launch_snapshot().get("_raw"),
                "started_at": now_iso(),
            },
        )
        manifests.append(m)

        for conc in concs:
            stats = asyncio.run(
                drive(renderer, payloads, args.kind, conc, args.n_requests,
                      tok_params, chat_params, prewarm=workers)
            )
            stats["renderer_num_workers"] = workers
            stats["tokenizer_pool_size"] = workers + 1
            stats["kind"] = args.kind
            results.append(stats)
            print(
                f"[conc] workers={workers} pool={workers + 1:>2} c={conc:>3} "
                f"throughput={stats['throughput_req_s']:8.2f} req/s "
                f"p50={stats['latency_p50_us']:9.1f}us "
                f"p90={stats['latency_p90_us']:9.1f}us "
                f"p99={stats['latency_p99_us']:9.1f}us "
                f"deepcopy={stats['tokenizer_deepcopy_calls']:>3}"
                f"({stats['tokenizer_deepcopy_us_per_request']}us/req) "
                f"load={stats['loadavg_after']}",
                flush=True,
            )

        # 释放 renderer 的线程池，避免下一轮持有过多线程
        renderer._executor.shutdown(wait=True)

    # 饱和点判定：吞吐相对上一个并发点的提升 < 10% 视作已饱和
    saturation: dict[str, Any] = {}
    for workers in worker_counts:
        rows = [r for r in results if r["renderer_num_workers"] == workers]
        rows.sort(key=lambda r: r["concurrency"])
        sat = None
        for prev, cur in zip(rows, rows[1:]):
            gain = (
                (cur["throughput_req_s"] - prev["throughput_req_s"])
                / prev["throughput_req_s"]
                if prev["throughput_req_s"]
                else 0
            )
            if gain < 0.10:
                sat = {
                    "at_concurrency": cur["concurrency"],
                    "throughput_req_s": cur["throughput_req_s"],
                    "marginal_gain_vs_prev": round(gain, 4),
                    "prev_concurrency": prev["concurrency"],
                }
                break
        saturation[str(workers)] = sat

    write_json(
        args.out,
        {
            "manifests": manifests,
            "results": results,
            "saturation": saturation,
            "saturation_rule": (
                "按并发递增顺序，第一个「吞吐相对上一点提升 < 10%」的并发点；"
                "容器只有 2 核（cpuset 见 manifest），所以这是**前端结构**的饱和"
                "特征，不是 12 核整机的饱和点"
            ),
        },
    )
    print(f"\n写到 {args.out}（{len(results)} 条）")
    for k, v in saturation.items():
        print(f"  workers={k}: saturation={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
