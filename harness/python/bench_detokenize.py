#!/usr/bin/env python3
"""E2.3 的生成期一半：**流式 detokenize**（`tokenizer: decode` scope 不含它）。

历史 `lite.log` 里的 `tokenizer: decode` 只覆盖 prompt 反解
（`BaseRenderer._decode`，见 `bench_paths.py`）。服务里**每个生成步都跑**的
解码在 `vllm/v1/engine/detokenizer.py`，是一条**独立**的代码路径，本脚本专门测它。

两条路的判据（`vllm/v1/engine/detokenizer.py:24`、`:60`）::

    USE_FAST_DETOKENIZER = tokenizers.__version__ >= "0.22.0"
    if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
        FastIncrementalDetokenizer   # tokenizers.decoders.DecodeStream（Rust）
    else:
        SlowIncrementalDetokenizer   # Python 前缀偏移 + convert_tokens_to_string

**一个重要设计事实**：快路在构造时从 **模块属性** 查
`tokenizers.decoders.DecodeStream`（`detokenizer.py:180-182`），而不是在 import
期绑定名字。这样 fastokens 的 shim 无论在什么 import 顺序下替换该符号都能生效
——这是上游为第三方后端预留的接口。测 fastokens 臂时这一点必须写进文档。

## 三段成本（实测发现的关键结构，不是假设）

初次实测发现「per-token 成本随 OSL 变化」（32 token 时 9 µs/token，
1024 token 时 1.5 µs/token）。逐 index 剖析后定位到原因::

    idx0      fast 274 µs / slow 376 µs   ← 第 1 次 update() 把 prompt 的
                                            DecodeStream / 前缀状态**惰性**建好
    idx1..N   fast 1.27 µs / slow 1.80 µs ← 稳定后的每步成本

所以本脚本按三段分别测量，而不是把 init 摊到每 token：

| 段 | 含义 | 摊销 |
|---|---|---|
| `init` | `from_new_request()` | 每请求 1 次 |
| `first_step` | 第 1 次 `update([tid])` | 每请求 1 次（含 prompt 预热） |
| `steady_step` | 第 k 次 `update([tid])`（k ≥ steady_after） | **每 token 1 次 → 进 TPOT** |
| `emit_text` | 每次 `get_next_output_text(delta=True)` | **每 token 1 次 → 进 TPOT** |

用法见 --help。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_renderer,
    make_manifest,
    now_iso,
    summarize,
    write_json,
)
from corpora import make_text  # noqa: E402

DEFAULT_OSL = [32, 256, 1024]
DEFAULT_PSL = 1024
STEADY_AFTER = 32  # 前 32 步当作「首次/预热」不计入 steady 统计


def make_request(prompt_ids: list[int], max_tokens: int = 2048):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id="c-cost-detok",
        prompt_token_ids=list(prompt_ids),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def gen_token_ids(tokenizer, corpus: str, need: int) -> list[int]:
    """生成一段真实的 token 序列：混入 en/zh/code，覆盖 ASCII/多字节/空白三类。

    只用英文会让解码表现过于乐观（都是可空格切分的子词）；中文与代码会触发
    字节回退与 `U+FFFD` 前缀判定，是流式解码最贵的场景。
    """
    ids: list[int] = []
    for name in ("en", "zh", "code", "mixed"):
        text, _ = make_text(tokenizer, max(2048, need), name)
        enc = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(enc, dict):
            enc = enc["input_ids"]
        ids.extend(enc)
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise RuntimeError(f"生成语料不足: {len(ids)} < {need}")
    return ids[:need]


def bench_path(
    cls,
    label: str,
    tokenizer,
    prompt_ids: list[int],
    gen_ids: list[int],
    osls: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_osl = max(osls)

    # --- 每请求固定成本：init 与 first_step 分别单独测 ---------------
    def do_init():
        return cls.from_new_request(tokenizer, make_request(prompt_ids))

    def do_first_step():
        det = cls.from_new_request(tokenizer, make_request(prompt_ids))
        det.update([gen_ids[0]], stop_terminated=False)
        return det

    for _ in range(warmup):
        do_first_step()

    init_ns: list[int] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        do_init()
        init_ns.append(time.perf_counter_ns() - t0)

    first_ns: list[int] = []
    for _ in range(repeats):
        d = do_init()  # 计时窗口外：只量第一次 update
        t0 = time.perf_counter_ns()
        d.update([gen_ids[0]], stop_terminated=False)
        first_ns.append(time.perf_counter_ns() - t0)

    # --- 稳定期每步成本：逐 index 采样，取 mean/p50/p90/p99 --------
    runs = max(warmup, min(repeats, 25))
    upd_prof: list[list[int]] = [[] for _ in range(max_osl)]
    txt_prof: list[list[int]] = [[] for _ in range(max_osl)]
    for _ in range(runs):
        det = cls.from_new_request(tokenizer, make_request(prompt_ids))
        for i, tid in enumerate(gen_ids[:max_osl]):
            a = time.perf_counter_ns()
            det.update([tid], stop_terminated=False)
            b = time.perf_counter_ns()
            det.get_next_output_text(finished=False, delta=True)
            c = time.perf_counter_ns()
            upd_prof[i].append(b - a)
            txt_prof[i].append(c - b)

    # --- 整请求总耗时（直接测，作为三段之和的交叉校验） -------------
    for osl in osls:
        tok_ids = gen_ids[:osl]

        def do_full():
            det = cls.from_new_request(tokenizer, make_request(prompt_ids))
            for tid in tok_ids:
                det.update([tid], stop_terminated=False)
                det.get_next_output_text(finished=False, delta=True)
            return det.get_next_output_text(finished=True, delta=False)

        for _ in range(warmup):
            do_full()
        full_ns: list[int] = []
        text_out = ""
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            text_out = do_full()
            full_ns.append(time.perf_counter_ns() - t0)

        st_init = summarize(init_ns)
        st_first = summarize(first_ns)
        steady_upd = [v for i in range(STEADY_AFTER, osl) for v in upd_prof[i]]
        steady_txt = [v for i in range(STEADY_AFTER, osl) for v in txt_prof[i]]
        st_upd = summarize(steady_upd) if steady_upd else None
        st_txt = summarize(steady_txt) if steady_txt else None
        st_full = summarize(full_ns)

        predicted = (
            st_init.mean_us
            + st_first.mean_us
            + (osl - 1)
            * (
                (st_upd.mean_us if st_upd else 0)
                + (st_txt.mean_us if st_txt else 0)
            )
        )
        out.append(
            {
                "case": f"detokenize_stream_{label}",
                "scope": "streaming detokenize (NOT covered by LiteProfiler)",
                "path": label,
                "detokenizer_cls": cls.__name__,
                "prompt_tokens": len(prompt_ids),
                "osl_tokens": int(osl),
                "repeats": int(repeats),
                "steady_after_index": STEADY_AFTER,
                "init_us": st_init.as_dict(),
                "first_step_us": st_first.as_dict(),
                "steady_update_us_per_token": st_upd.as_dict() if st_upd else None,
                "steady_emit_text_us_per_token": st_txt.as_dict() if st_txt else None,
                "steady_total_us_per_token": (
                    round(st_upd.mean_us + st_txt.mean_us, 4)
                    if st_upd and st_txt
                    else None
                ),
                "steady_unavailable_reason": (
                    None
                    if st_upd
                    else f"OSL={osl} ≤ steady_after_index={STEADY_AFTER}，"
                    "无法在窗口内取到稳定期样本（首步开销未被 amortize）"
                ),
                "full_request_us": st_full.as_dict(),
                "full_definition": (
                    "from_new_request + OSL×(update + get_next_output_text(delta)) "
                    "+ 末次 get_next_output_text(finished=True)"
                ),
                "segment_sum_check": {
                    "measured_full_mean_us": st_full.mean_us,
                    "predicted_from_segments_us": round(predicted, 3),
                    "predicted_over_measured": round(predicted / st_full.mean_us, 4),
                    "prediction_definition": (
                        "init + first_step + (OSL-1)×(steady_update + steady_text)"
                    ),
                },
                "output_chars": len(text_out),
                "chars_per_token": round(len(text_out) / osl, 3),
            }
        )
        print(
            f"[detok.{label}] psl={len(prompt_ids)} osl={osl:>5} "
            f"init={st_init.mean_us:7.1f} first={st_first.mean_us:7.1f} "
            f"step={st_upd.mean_us if st_upd else float('nan'):5.3f} "
            f"emit={st_txt.mean_us if st_txt else float('nan'):5.3f} "
            f"full={st_full.mean_us:9.1f} "
            f"(pred={predicted:9.1f}, {predicted / st_full.mean_us * 100:5.1f}%) "
            f"chars/tok={len(text_out) / osl:.2f}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2.3 生成期流式 detokenize（Fast / Slow 两条路，按 OSL 分组）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--out", default="/workspace/data/cost/e2_detokenize.json")
    ap.add_argument("--corpus", default="mixed", choices=["en", "zh", "code", "mixed"])
    ap.add_argument("--osl", default=",".join(map(str, DEFAULT_OSL)))
    ap.add_argument("--psl", type=int, default=DEFAULT_PSL,
                    help="prompt token 数（影响 init 与 first_step）")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--paths", default="fast,slow",
                    help="fast=FastIncrementalDetokenizer / slow=SlowIncrementalDetokenizer")
    ap.add_argument("--out-tokens", default="/tmp/c-cost-gen-tokens.json",
                    help="把用到的 prompt/生成 token ids 落盘，便于复跑与审计")
    args = ap.parse_args()

    renderer, _cfg = build_renderer(args.model)
    tokenizer = renderer.get_tokenizer()

    from vllm.v1.engine.detokenizer import (
        USE_FAST_DETOKENIZER,
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    osls = sorted({int(x) for x in args.osl.split(",") if x})
    prompt_text, _ = make_text(tokenizer, args.psl, args.corpus)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    gen_ids = gen_token_ids(tokenizer, args.corpus, max(osls) + 16)
    write_json(
        args.out_tokens,
        {"prompt_token_ids": prompt_ids, "gen_token_ids": gen_ids, "corpus": args.corpus},
    )

    import tokenizers
    import transformers

    manifest = make_manifest(
        experiment="E2.3 streaming detokenize",
        script=os.path.abspath(__file__),
        model=args.model,
        extra={
            "args": vars(args),
            "tokenizers_version": tokenizers.__version__,
            "USE_FAST_DETOKENIZER": bool(USE_FAST_DETOKENIZER),
            "tokenizer_is_tokenizers_backend": isinstance(
                tokenizer, transformers.TokenizersBackend
            ),
            "path_selection_rule": (
                "USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend) "
                "-> FastIncrementalDetokenizer，否则 SlowIncrementalDetokenizer "
                "(vllm/v1/engine/detokenizer.py:60)"
            ),
            "fast_decodestream_lookup": (
                "detokenizer.py:180-182 在构造时从模块属性取 "
                "tokenizers.decoders.DecodeStream（不在 import 期绑定），"
                "因此 fastokens 的 shim 可生效"
            ),
            "prompt_tokens_actual": len(prompt_ids),
            "gen_tokens_written": args.out_tokens,
            "generated_token_mix": (
                "en+zh+code+mixed 拼接后截断，覆盖 ASCII / 多字节 CJK / 代码 / 空白"
            ),
            "finished_at": now_iso(),
        },
    )

    selected = [p.strip() for p in args.paths.split(",") if p.strip()]
    results: list[dict[str, Any]] = []
    for label, cls in (
        ("fast", FastIncrementalDetokenizer),
        ("slow", SlowIncrementalDetokenizer),
    ):
        if label not in selected:
            continue
        results += bench_path(
            cls, label, tokenizer, prompt_ids, gen_ids, osls, args.repeats, args.warmup
        )

    # 交叉校验：两条路对同一 token 序列必须产出同样文本（正确性前提）
    cross: dict[str, Any] = {}
    for osl in [min(osls), max(osls)]:
        texts = {}
        for label, cls in (
            ("fast", FastIncrementalDetokenizer),
            ("slow", SlowIncrementalDetokenizer),
        ):
            det = cls.from_new_request(tokenizer, make_request(prompt_ids))
            for tid in gen_ids[:osl]:
                det.update([tid], stop_terminated=False)
            texts[label] = det.get_next_output_text(finished=True, delta=False)
        identical = texts["fast"] == texts["slow"]
        entry: dict[str, Any] = {
            "identical": identical,
            "fast_len": len(texts["fast"]),
            "slow_len": len(texts["slow"]),
            "fast_head": texts["fast"][:120],
        }
        if not identical:
            a, b = texts["fast"], texts["slow"]
            i = next(
                (k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                min(len(a), len(b)),
            )
            entry.update(
                {
                    "first_diff_at": i,
                    "fast_at_diff": a[max(0, i - 40) : i + 40],
                    "slow_at_diff": b[max(0, i - 40) : i + 40],
                }
            )
        cross[str(osl)] = entry

    write_json(args.out, {"manifest": manifest, "results": results, "cross_check": cross})
    print(f"\n写到 {args.out}（{len(results)} 条）")
    for k, v in cross.items():
        print(
            f"  cross-check osl={k}: identical={v['identical']} "
            f"len fast/slow={v['fast_len']}/{v['slow_len']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
