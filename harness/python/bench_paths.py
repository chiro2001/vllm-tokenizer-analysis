#!/usr/bin/env python3
"""E2.1 / E2.2 / E2.3：三个 tokenizer scope 的成本（不启引擎、不需要 NPU）。

被测的**边界**与 LiteProfiler 插桩点逐条对齐（`liteprofiler-vllm.patch`
的 `vllm/renderers/base.py` 部分）：

| case | 边界（被测调用的闭合范围） | 对应插桩点 |
|---|---|---|
| `encode.scope`    | `tokenizer(text, **params.get_encode_kwargs())` | `_tokenize_prompt` 里的 `tokenizer: encode` |
| `encode.outer`    | `renderer.tokenize_prompt(prompt, params)` | 无（外层包装，用于算占比） |
| `render.scope`    | `renderer.render_messages(conv, chat_params)` | `render_chat` 列表推导里的 `tokenizer: render_messages` |
| `render.outer`    | `renderer.render_chat([conv], chat_params, tok_params)` | 无（含 chat 模板 + encode + process_for_engine） |
| `decode: prompt_reverse` | `renderer._decode(ids)` | `_decode` / `_detokenize_prompt` 里的 `tokenizer: decode` |

**生成期的流式解码不在这里**——见 `bench_detokenize.py`。

**为什么必须分开**：LiteProfiler 的 `tokenizer: decode` 只包住
`BaseRenderer._decode`，它把 prompt 的 token_ids 还原成文本，
**只在 `TokenizeParams.needs_detokenization=True` 时才被调用**
（服务里极少数请求才会走到）；而服务里**每个生成步都跑**的流式解码在
`vllm/v1/engine/detokenizer.py`，插桩完全没覆盖它。
把「prompt 反解」当成「生成期解码」是口径错误，两者在这里分开命名、分开测。

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
    measure,
    now_iso,
    summarize,
    write_json,
)
from corpora import TOOL_SPECS, make_chat_messages, make_text  # noqa: E402

DEFAULT_ISL = [128, 1024, 8192]
DEFAULT_OSL = [32, 256, 1024]


# --------------------------------------------------------------------------
# 单个实验
# --------------------------------------------------------------------------
def bench_encode(renderer, corpus: str, isls: list[int], repeats: int, warmup: int):
    from vllm.renderers.params import TokenizeParams

    tokenizer = renderer.get_tokenizer()
    params = TokenizeParams(max_total_tokens=40960, max_output_tokens=0)
    kwargs = params.get_encode_kwargs()
    out: list[dict[str, Any]] = []

    for isl in isls:
        text, actual = make_text(tokenizer, isl, corpus)
        # 预热 + 校准：确保 tokenizer 内部缓存（added tokens 等）就绪
        for _ in range(warmup):
            tokenizer(text, **kwargs)

        scope_ns, _ = measure(lambda: tokenizer(text, **kwargs), repeats, 0)
        outer_ns, _ = measure(
            lambda: renderer.tokenize_prompt({"prompt": text}, params), repeats, 0
        )
        out.append(
            {
                "case": "encode",
                "scope": "tokenizer: encode",
                "corpus": corpus,
                "target_isl_tokens": isl,
                "actual_tokens": actual,
                "n_chars": len(text),
                "scope_stats": summarize(scope_ns).as_dict(),
                "outer_stats": summarize(outer_ns).as_dict(),
                "outer_definition": "renderer.tokenize_prompt()",
            }
        )
        print(
            f"[encode] isl={isl:>5} actual={actual:>5} "
            f"scope_mean={summarize(scope_ns).mean_us:8.1f}us "
            f"outer_mean={summarize(outer_ns).mean_us:8.1f}us",
            flush=True,
        )
    return out


def bench_render_messages(
    renderer, corpus: str, isls: list[int], repeats: int, warmup: int
):
    from vllm.renderers.params import ChatParams, TokenizeParams

    tokenizer = renderer.get_tokenizer()
    tok_params = TokenizeParams(max_total_tokens=40960, max_output_tokens=256)
    out: list[dict[str, Any]] = []

    variants = [
        ("tools_1turn", dict(with_tools=True, n_turns=1, tool_call_turn=False)),
        ("no_tools_1turn", dict(with_tools=False, n_turns=1, tool_call_turn=False)),
        ("tools_2turn_toolresult", dict(with_tools=True, n_turns=2, tool_call_turn=True)),
    ]

    for variant, kw in variants:
        for isl in isls:
            filler, actual = make_text(tokenizer, isl, corpus)
            messages = make_chat_messages(filler, **kw)
            chat_params = ChatParams(
                chat_template=None,
                chat_template_content_format="auto",
                chat_template_kwargs=(
                    {"tools": TOOL_SPECS} if kw["with_tools"] else {}
                ),
            )
            renderer.render_messages(messages, chat_params)  # warmup
            for _ in range(warmup):
                renderer.render_messages(messages, chat_params)

            scope_ns, rendered = measure(
                lambda: renderer.render_messages(messages, chat_params),
                repeats,
                0,
            )
            outer_ns, _ = measure(
                lambda: renderer.render_chat([messages], chat_params, tok_params),
                repeats,
                0,
            )
            # 渲染出来的 prompt 实际有多少 token（模板自身的开销）
            _, dict_prompt = rendered
            try:
                tok_out = renderer.tokenize_prompt(dict_prompt, tok_params)
                rendered_tokens = len(tok_out["prompt_token_ids"])
            except Exception as exc:  # noqa: BLE001
                rendered_tokens = f"<{type(exc).__name__}: {exc}>"

            out.append(
                {
                    "case": "render_messages",
                    "scope": "tokenizer: render_messages",
                    "variant": variant,
                    "corpus": corpus,
                    "target_isl_tokens": isl,
                    "actual_filler_tokens": actual,
                    "rendered_prompt_tokens": rendered_tokens,
                    "n_messages": len(messages),
                    "n_tools": len(TOOL_SPECS) if kw["with_tools"] else 0,
                    "scope_stats": summarize(scope_ns).as_dict(),
                    "outer_stats": summarize(outer_ns).as_dict(),
                    "outer_definition": "renderer.render_chat([messages],...)",
                }
            )
            print(
                f"[render] {variant:<24} isl={isl:>5} "
                f"prompt_tokens={rendered_tokens!s:>6} "
                f"scope_mean={summarize(scope_ns).mean_us:8.1f}us "
                f"outer_mean={summarize(outer_ns).mean_us:8.1f}us",
                flush=True,
            )
    return out


def bench_render_decompose(
    renderer, corpus: str, isls: list[int], repeats: int, warmup: int
):
    """把 `render_messages` 的耗时拆成 **Jinja 模板** 与 **内部 encode** 两段。

    为什么要单独测：`render_messages` 的 scope 里其实**同时**做了 chat 模板渲染
    与 tokenization（`apply_chat_template` 默认 `tokenize=True`），所以
    `tokenizer: encode` 这个 scope 在 chat 请求上**根本不会被触发**。
    要回答「chat 模板本身花多少」，必须把两段分开量：

    * `tokenize=False` 的 `apply_chat_template` 只跑 Jinja 模板；
    * 再对同一条渲染出来的文本调 `tokenizer(...)`，就是模板内那一次的编码成本。

    两段之和与 `render_messages` 的 scope 时间对照，用来验证拆分是否自洽。
    """
    from vllm.renderers.params import ChatParams, TokenizeParams
    from vllm.renderers.hf import safe_apply_chat_template

    tokenizer = renderer.get_tokenizer()
    model_config = renderer.model_config
    tok_params = TokenizeParams(max_total_tokens=40960, max_output_tokens=256)
    kwargs = tok_params.get_encode_kwargs()
    out: list[dict[str, Any]] = []

    for variant, kw in (
        ("tools_1turn", dict(with_tools=True, n_turns=1, tool_call_turn=False)),
        ("no_tools_1turn", dict(with_tools=False, n_turns=1, tool_call_turn=False)),
        ("tools_2turn_toolresult", dict(with_tools=True, n_turns=2, tool_call_turn=True)),
    ):
        for isl in isls:
            filler, _ = make_text(tokenizer, isl, corpus)
            messages = make_chat_messages(filler, **kw)
            chat_kwargs = {"tools": TOOL_SPECS} if kw["with_tools"] else {}
            # 先拿到「模板渲染出的纯文本」，供 encode 臂复用
            rendered_text = safe_apply_chat_template(
                model_config, tokenizer, messages, tokenize=False, **chat_kwargs
            )
            if not isinstance(rendered_text, str):
                out.append(
                    {
                        "case": "render_decompose",
                        "variant": variant,
                        "target_isl_tokens": isl,
                        "skipped": f"apply_chat_template(tokenize=False) 返回 "
                        f"{type(rendered_text).__name__}，不是 str",
                    }
                )
                continue

            chat_params = ChatParams(
                chat_template=None,
                chat_template_content_format="auto",
                chat_template_kwargs=chat_kwargs,
            )
            for _ in range(warmup):
                safe_apply_chat_template(
                    model_config, tokenizer, messages, tokenize=False, **chat_kwargs
                )
                tokenizer(rendered_text, add_special_tokens=False)
                renderer.render_messages(messages, chat_params)
            # 再各自清一次 cache 影响（tokenizer 无 cache，这里只是保持一致性）
            if warmup:
                pass

            jinja_ns, _ = measure(
                lambda: safe_apply_chat_template(
                    model_config, tokenizer, messages, tokenize=False, **chat_kwargs
                ),
                repeats,
                0,
            )
            enc_ns, _ = measure(
                lambda: tokenizer(rendered_text, add_special_tokens=False), repeats, 0
            )
            total_ns, _ = measure(
                lambda: renderer.render_messages(messages, chat_params), repeats, 0
            )
            sj, se, st = summarize(jinja_ns), summarize(enc_ns), summarize(total_ns)
            ids = tokenizer(rendered_text, add_special_tokens=False)
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            out.append(
                {
                    "case": "render_decompose",
                    "variant": variant,
                    "target_isl_tokens": isl,
                    "rendered_prompt_tokens": len(ids),
                    "rendered_chars": len(rendered_text),
                    "jinja_tokenize_false_us": sj.as_dict(),
                    "encode_rendered_text_us": se.as_dict(),
                    "render_messages_scope_us": st.as_dict(),
                    "sum_vs_scope": {
                        "jinja_plus_encode_us": round(sj.mean_us + se.mean_us, 3),
                        "scope_us": st.mean_us,
                        "ratio": round((sj.mean_us + se.mean_us) / st.mean_us, 4),
                    },
                    "note": (
                        "encode 臂用 add_special_tokens=False；模板内部那次编码"
                        "用的是 chat 模板自己的 add_special_tokens 设定，"
                        "所以 sum/scope 不会精确等于 1"
                    ),
                }
            )
            print(
                f"[decompose] {variant:<24} isl={isl:>5} tokens={len(ids):>6} "
                f"jinja={sj.mean_us:8.1f}us encode={se.mean_us:8.1f}us "
                f"sum={sj.mean_us + se.mean_us:8.1f}us scope={st.mean_us:8.1f}us "
                f"ratio={(sj.mean_us + se.mean_us) / st.mean_us:.3f}",
                flush=True,
            )
    return out


def _make_request(prompt_ids: list[int], *, skip_special_tokens: bool = True):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id="c-cost-bench",
        prompt_token_ids=list(prompt_ids),
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=1024,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=True,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def bench_decode(renderer, corpus: str, osls: list[int], repeats: int, warmup: int):
    """E2.3 的 prompt 侧一半：`tokenizer: decode` = prompt 反向解码。

    生成期流式解码见 `bench_detokenize.py`（不同代码位置、不同成本量级）。
    """
    tokenizer = renderer.get_tokenizer()
    out: list[dict[str, Any]] = []

    # --- (a) scope: prompt 反向解码 -----------------------------------
    for target in (128, 1024, 8192):
        text, actual = make_text(tokenizer, target, corpus)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        renderer._decode(ids)  # warmup
        for _ in range(warmup):
            renderer._decode(ids)
        scope_ns, _ = measure(lambda: renderer._decode(ids), repeats, 0)
        outer_ns, _ = measure(
            lambda: renderer._detokenize_prompt({"prompt_token_ids": list(ids)}),
            repeats,
            0,
        )
        out.append(
            {
                "case": "decode_prompt_reverse",
                "scope": "tokenizer: decode (prompt reverse only)",
                "corpus": corpus,
                "n_tokens": len(ids),
                "scope_stats": summarize(scope_ns).as_dict(),
                "scope_per_token_us": summarize(scope_ns).mean_us / len(ids),
                "outer_stats": summarize(outer_ns).as_dict(),
                "outer_definition": "renderer._detokenize_prompt()",
            }
        )
        print(
            f"[decode.prompt] tokens={len(ids):>5} "
            f"scope_mean={summarize(scope_ns).mean_us:9.1f}us "
            f"({summarize(scope_ns).mean_us / len(ids):6.2f}us/token)",
            flush=True,
        )

    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2.1/E2.2/E2.3 tokenizer 三 scope 成本测量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=os.environ.get("COST_MODEL", "/models/Qwen3-0.6B"))
    ap.add_argument("--out", default="/workspace/data/cost/e2_paths.json")
    ap.add_argument("--corpus", default="mixed", choices=["en", "zh", "code", "mixed"])
    ap.add_argument("--isl", default=",".join(map(str, DEFAULT_ISL)),
                    help="encode/render 的输入长度分组（token 数，逗号分隔）")
    ap.add_argument("--osl", default=",".join(map(str, DEFAULT_OSL)),
                    help="流式 decode 的输出长度分组（token 数，逗号分隔）")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--tasks", default="encode,render,decode",
                    help="逗号分隔：encode,render,decompose,decode")
    ap.add_argument("--renderer-num-workers", type=int, default=None,
                    help="覆盖 model_config.renderer_num_workers（默认 1）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    isls = [int(x) for x in args.isl.split(",") if x]
    osls = [int(x) for x in args.osl.split(",") if x]
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    manifest = make_manifest(
        experiment="E2.1+E2.2+E2.3",
        script=os.path.abspath(__file__),
        extra={
            "args": vars(args),
            "started_at": now_iso(),
            "model_revision": None,
            "tokenizer_scope_boundaries": {
                "encode": "BaseRenderer._tokenize_prompt -> tokenizer(text, **kwargs)",
                "render_messages": "BaseRenderer.render_chat -> render_messages(conversation, chat_params)",
                "decode": "BaseRenderer._decode -> tokenizer.decode(...)",
                "streaming_detokenize": "vllm.v1.engine.detokenizer.*IncrementalDetokenizer (未插桩)",
            },
        },
    )

    t_start = time.perf_counter()
    renderer, vllm_config = build_renderer(
        args.model, renderer_num_workers=args.renderer_num_workers
    )
    manifest["renderer_init_s"] = round(time.perf_counter() - t_start, 3)
    manifest["renderer_cls"] = f"{type(renderer).__module__}.{type(renderer).__name__}"
    tok = renderer.get_tokenizer()
    manifest["tokenizer_cls"] = f"{type(tok).__module__}.{type(tok).__name__}"
    manifest["tokenizer_class_name"] = type(tok).__name__
    manifest["tokenizer_is_tokenizers_backend"] = type(tok).__name__ == "CachedQwen2Tokenizer"
    manifest["renderer_num_workers"] = vllm_config.model_config.renderer_num_workers
    manifest["vocab_size"] = getattr(tok, "vocab_size", None)
    from common import model_revision

    manifest["model_revision"] = model_revision(args.model)

    results: dict[str, Any] = {
        "manifest": manifest,
        "results": [],
        "notes": [],
    }

    if "encode" in tasks:
        results["results"] += bench_encode(renderer, args.corpus, isls, args.repeats, args.warmup)
    if "render" in tasks:
        results["results"] += bench_render_messages(
            renderer, args.corpus, isls, args.repeats, args.warmup
        )
    if "decompose" in tasks:
        results["results"] += bench_render_decompose(
            renderer, args.corpus, isls, args.repeats, args.warmup
        )
    if "decode" in tasks:
        results["results"] += bench_decode(renderer, args.corpus, osls, args.repeats, args.warmup)

    manifest["finished_at"] = now_iso()
    manifest["elapsed_s"] = round(time.perf_counter() - t_start, 3)
    write_json(args.out, results)
    print(f"\n写到 {args.out}（{len(results['results'])} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
