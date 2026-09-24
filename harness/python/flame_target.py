#!/usr/bin/env python3
"""E3.1 火焰图的被测目标：在容器里连续跑前端 tokenizer 三条路径。

设计要点（为了拿到可读的火焰图）：

* **单线程**：`perf record -g` 对多线程采样会给出交错的栈，本脚本强制
  单线程（`renderer_num_workers=1` 且全部走同步路径），让栈是干净的。
* **三种负载轮流**：encode / render_messages(chat+tools) / detokenize:stream，
  这样一张图里能同时看到 Jinja、Python tokenizer 包装、Rust tokenizers 三类帧，
  并能在图上按函数名区分它们落在谁身上。
* **重复到指定时长**：火焰图需要足够样本（默认 ~2000 次迭代）。

用法见 --help。容器里跑法见 `scripts/run_flamegraph.sh`。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_renderer  # noqa: E402
from corpora import TOOL_SPECS, make_chat_messages, make_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E3.1 火焰图目标：轮流跑 encode / render_messages / 流式 detokenize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--iters", type=int, default=800, help="每类负载的迭代次数")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="按秒运行（覆盖 --iters）；给采样器一个稳定的时间窗")
    ap.add_argument("--isl", type=int, default=1024, help="encode/render 的输入 token 数")
    ap.add_argument("--osl", type=int, default=256, help="流式 detokenize 的生成 token 数")
    ap.add_argument("--tasks", default="encode,render,detokenize")
    ap.add_argument("--burn", type=int, default=0,
                    help="先空转这么多秒再开始（给 perf 时间就绪）")
    args = ap.parse_args()

    if args.burn:
        time.sleep(args.burn)

    renderer, _ = build_renderer(args.model, renderer_num_workers=1)
    tokenizer = renderer.get_tokenizer()

    from vllm.renderers.params import ChatParams, TokenizeParams
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    tok_params = TokenizeParams(max_total_tokens=40960, max_output_tokens=256)
    chat_params = ChatParams(
        chat_template=None,
        chat_template_content_format="auto",
        chat_template_kwargs={"tools": TOOL_SPECS},
    )
    text, actual = make_text(tokenizer, args.isl, "mixed")
    messages = make_chat_messages(text, with_tools=True)
    kwargs = tok_params.get_encode_kwargs()

    gen_ids: list[int] = []
    for name in ("en", "zh", "code", "mixed"):
        t, _ = make_text(tokenizer, args.osl * 2, name)
        e = tokenizer.encode(t, add_special_tokens=False)
        gen_ids.extend(e if isinstance(e, list) else e["input_ids"])
        if len(gen_ids) >= args.osl:
            break
    gen_ids = gen_ids[: args.osl]
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]

    sys.stderr.write(
        f"[flame_target] READY isl_target={args.isl} actual={actual} "
        f"osl={args.osl} iters={args.iters} duration={args.duration} "
        f"tasks={args.tasks}\n"
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    t_start = time.perf_counter()
    total = 0
    deadline = t_start + args.duration if args.duration > 0 else None
    i = -1
    while True:
        i += 1
        if deadline is None and i >= args.iters:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if "encode" in tasks:
            tokenizer(text, **kwargs)
            total += 1
        if "render" in tasks:
            renderer.render_messages(messages, chat_params)
            total += 1
        if "detokenize" in tasks:
            from vllm.sampling_params import SamplingParams
            from vllm.v1.engine import EngineCoreRequest

            req = EngineCoreRequest(
                request_id="flame",
                prompt_token_ids=list(prompt_ids),
                mm_features=None,
                sampling_params=SamplingParams(
                    max_tokens=args.osl, skip_special_tokens=True
                ),
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )
            det = FastIncrementalDetokenizer.from_new_request(tokenizer, req)
            for tid in gen_ids:
                det.update([tid], stop_terminated=False)
                det.get_next_output_text(finished=False, delta=True)
            total += 1
        if i % 100 == 0:
            # 注意：sys.stderr.write() **不接受 flush 关键字**（踩过：
            # TypeError: TextIOWrapper.write() takes no keyword arguments，
            # 而且正好在 READY 之后崩掉，刚好让 perf 采了个死进程）。
            print(
                f"  iter {i} elapsed={time.perf_counter() - t_start:.1f}s",
                file=sys.stderr,
                flush=True,
            )
    sys.stderr.write(
        f"[flame_target] 完成 {total} 次操作，用时 {time.perf_counter() - t_start:.1f}s\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
