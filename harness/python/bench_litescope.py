#!/usr/bin/env python3
"""E2 交叉校验：用 **真实 LiteProfiler 插桩** 采集同样的三个 scope。

与 `bench_paths.py` 的区别：

* `bench_paths.py` 在**自己的代码里**包 `time.perf_counter_ns()`，边界是我按
  patch 逐行对齐出来的；
* 本脚本设 `VLLM_LITE_PROFILER_LOG_PATH` 并启用覆盖层里**上游原版**
  `LiteScope`，让 vLLM 自己写 `scope|耗时us|开始us|tid|pid` 行。

两者一致 ⟹ `bench_paths.py` 的边界对齐成立、数字可用；
两者不一致 ⟹ 必须按 LiteScope 的口径修正。

用法：`scripts/c_cost_run.sh --liteprof harness/python/bench_litescope.py --help`
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_renderer,
    make_manifest,
    now_iso,
    write_json,
)
from corpora import TOOL_SPECS, make_chat_messages, make_text  # noqa: E402


def parse_lite_log(path: str) -> dict[str, list[float]]:
    """解析 scope|耗时us|开始us|tid|pid，只保留 scope 行（丢掉 phase 行）。"""
    out: dict[str, list[float]] = defaultdict(list)
    tids: dict[str, set[int]] = defaultdict(set)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("phase:"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            name, dur = parts[0], parts[1]
            try:
                out[name].append(float(dur))
            except ValueError:
                continue
            try:
                tids[name].add(int(parts[3]))
            except ValueError:
                pass
    out["__tids__"] = {k: sorted(v) for k, v in tids.items()}  # type: ignore[assignment]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="用真实 LiteScope 采集 tokenizer 三 scope（交叉校验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--log", default="/tmp/c-litescope/lite.log",
                    help="VLLM_LITE_PROFILER_LOG_PATH（必须在 import vllm 之前设）")
    ap.add_argument("--out", default="/workspace/data/cost/e2_litescope.json")
    ap.add_argument("--corpus", default="mixed")
    ap.add_argument("--isl", default="128,1024,8192")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--osl", default="256")
    ap.add_argument("--decode-offload", action="store_true",
                    help="用异步 offload 路径调 decode（_detokenize_prompt_async），"
                         "对应 API server 里的实际路径")
    args = ap.parse_args()

    if not os.environ.get("VLLM_LITE_PROFILER_LOG_PATH"):
        print(
            "错误：必须先设 VLLM_LITE_PROFILER_LOG_PATH（且要在 import vllm 之前），"
            "否则 record_function_or_nullcontext 会退化成 nullcontext。\n"
            "请用 scripts/c_cost_run.sh --liteprof 运行本脚本。",
            file=sys.stderr,
        )
        return 2

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # 关键：patch 后的 record_function_or_nullcontext 在**首次调用**时按 env
    # 选型并缓存，所以 env 必须已经生效（由 c_cost_run.sh --liteprof 保证）。
    # scope 工厂在 vllm.v1.utils；运行期开关在 vllm.utils.lite_profiler
    from vllm.utils.lite_profiler import (
        is_lite_profiler_enabled,
        set_lite_profiler_active,
    )
    from vllm.v1.utils import record_function_or_nullcontext

    if not is_lite_profiler_enabled():
        print("错误：LiteProfiler 未启用", file=sys.stderr)
        return 2

    renderer, _cfg = build_renderer(args.model)
    from vllm.renderers.params import ChatParams, TokenizeParams

    tokenizer = renderer.get_tokenizer()
    isls = [int(x) for x in args.isl.split(",") if x]

    ctx = record_function_or_nullcontext("c-cost: probe-warmup")
    print("context mgr:", type(ctx).__module__ + "." + type(ctx).__name__, flush=True)
    if "nullcontext" in type(ctx).__name__:
        print("警告：LiteScope 未启用（env 设置晚于 import）", file=sys.stderr)
        return 3

    tok_params = TokenizeParams(max_total_tokens=40960, max_output_tokens=256)
    chat_params = ChatParams(
        chat_template=None,
        chat_template_content_format="auto",
        chat_template_kwargs={"tools": TOOL_SPECS},
    )
    kwargs = tok_params.get_encode_kwargs()

    # 每个 case 一对 (label, in-scope fn, outer fn)：
    #   in_scope  —— 与 patch 里 `with record_function_or_nullcontext(...)` 的
    #                包裹范围 **完全等价** 的调用
    #   outer     —— 触发它的上层入口（scope 就在里面）
    # 两者在同一进程、同一轮里各测一次，差异就只可能来自边界本身。
    cases: list[tuple[str, str, Any]] = []
    for isl in isls:
        text, actual = make_text(tokenizer, isl, args.corpus)
        messages = make_chat_messages(text, with_tools=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        cases.append((f"encode@{isl}(actual={actual})", "encode", text))
        cases.append((f"render@{isl}", "render", messages))
        cases.append((f"decode@{len(ids)}", "decode", ids))

    def in_scope(kind: str, payload):
        if kind == "encode":
            return tokenizer(payload, **kwargs)
        if kind == "render":
            return renderer.render_messages(payload, chat_params)
        return renderer._decode(payload)

    def outer(kind: str, payload):
        if kind == "encode":
            return renderer.tokenize_prompt({"prompt": payload}, tok_params)
        if kind == "render":
            return renderer.render_chat([payload], chat_params, tok_params)
        return renderer._detokenize_prompt({"prompt_token_ids": list(payload)})

    # 预热：让线程池、缓存、JIT 全部就位（不计入日志）
    for _ in range(10):
        for _, kind, payload in cases:
            outer(kind, payload)

    set_lite_profiler_active(True)
    try:
        for _ in range(args.repeats):
            for _, kind, payload in cases:
                with record_function_or_nullcontext(f"c-cost: {kind}.outer"):
                    outer(kind, payload)
    finally:
        set_lite_profiler_active(False)

    # 同一进程内再做一次纯 perf_counter 直接计时（做为对照臂）
    from common import summarize, timer

    direct: dict[str, dict[str, Any]] = {}
    by_kind: dict[str, dict[str, list[int]]] = {}
    for kind in ("encode", "render", "decode"):
        by_kind[kind] = {"scope": [], "outer": []}
    for _ in range(args.repeats):
        for _, kind, payload in cases:
            with timer(by_kind[kind]["scope"]):
                in_scope(kind, payload)
            with timer(by_kind[kind]["outer"]):
                outer(kind, payload)
    for kind, buckets in by_kind.items():
        direct[kind] = {
            arm: summarize(v).as_dict() for arm, v in buckets.items() if v
        }

    parsed = parse_lite_log(str(log_path))
    tids = parsed.pop("__tids__", {})  # type: ignore[arg-type]
    summary: dict[str, Any] = {}
    for name, vals in sorted(parsed.items()):
        vals = sorted(vals)
        n = len(vals)
        summary[name] = {
            "n": n,
            "mean_us": round(sum(vals) / n, 3),
            "p50_us": round(vals[n // 2], 3),
            "min_us": round(vals[0], 3),
            "max_us": round(vals[-1], 3),
            "tids": tids.get(name, []),
        }

    # 前端进程一致性检查（§5.1）：三个 tokenizer scope 是否同一个 tid
    tok_tids = {
        name: info["tids"]
        for name, info in summary.items()
        if name.startswith("tokenizer: ")
    }
    distinct = {t for tids_ in tok_tids.values() for t in tids_}
    manifest = make_manifest(
        experiment="E2 cross-check (real LiteScope)",
        script=os.path.abspath(__file__),
        extra={
            "args": vars(args),
            "scope_line_raw_count": sum(v["n"] for v in summary.values()),
            "tokenizer_scope_tids": tok_tids,
            "tokenizer_scopes_single_frontend_thread": len(distinct) == 1,
            "log_path_in_container": str(log_path),
            "note": (
                "本脚本故意在 scope 外层再包一层 'c-cost: *.outer'，"
                "用来在同一份日志里同时读出 scope 内与 scope 外的成本"
            ),
            "finished_at": now_iso(),
        },
    )
    write_json(
        args.out,
        {"manifest": manifest, "litescope_scopes": summary, "direct_perf_counter": direct},
    )
    print(f"\n解析 {log_path} → {args.out}")
    for name, info in summary.items():
        print(
            f"  {name:<28} n={info['n']:>5} mean={info['mean_us']:>10.1f}us "
            f"p50={info['p50_us']:>10.1f}us tid={info['tids']}"
        )
    print("\n对照（同一进程、同样语料，perf_counter 直接计时）：")
    for kind, buckets in direct.items():
        s, o = buckets.get("scope"), buckets.get("outer")
        if s and o:
            print(
                f"  {kind:<8} scope_mean={s['mean_us']:>9.1f}us "
                f"outer_mean={o['mean_us']:>9.1f}us "
                f"(outer/scope={o['mean_us'] / s['mean_us']:.2f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
