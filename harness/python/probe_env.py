#!/usr/bin/env python3
"""环境探针：确认容器内可以直接构造 vLLM renderer（不启引擎、不需要 NPU）。

用法：
    python3 probe_env.py --model /models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/models/Qwen3-0.6B")
    ap.add_argument("--tokenizer-mode", default="hf")
    args = ap.parse_args()

    report: dict[str, object] = {
        "python": sys.version,
        "cwd": os.getcwd(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_USE_RUST_FRONTEND",
                "VLLM_USE_FASTOKENS",
                "TORCH_DEVICE_BACKEND_AUTOLOAD",
                "LD_LIBRARY_PATH",
                "VLLM_LITE_PROFILER_LOG_PATH",
            )
        },
        "affinity": sorted(os.sched_getaffinity(0)),
    }

    import transformers
    import tokenizers

    report["transformers"] = transformers.__version__
    report["tokenizers"] = tokenizers.__version__

    t0 = time.perf_counter()
    from vllm.config import ModelConfig, VllmConfig

    report["import_vllm_config_ms"] = round((time.perf_counter() - t0) * 1e3, 3)

    mc = ModelConfig(
        model=args.model,
        tokenizer=args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
    )
    vllm_config = VllmConfig(model_config=mc)

    t1 = time.perf_counter()
    from vllm.renderers import renderer_from_config

    renderer = renderer_from_config(vllm_config)
    report["renderer_init_ms"] = round((time.perf_counter() - t1) * 1e3, 3)
    report["renderer_cls"] = f"{type(renderer).__module__}.{type(renderer).__name__}"

    tokenizer = renderer.get_tokenizer()
    report["tokenizer_cls"] = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    report["tokenizer_class_name"] = type(tokenizer).__name__
    report["vocab_size"] = getattr(tokenizer, "vocab_size", None)
    report["chat_template_present"] = bool(getattr(tokenizer, "chat_template", None))

    # 一次最小 encode：只走 renderer 的对外接口
    from vllm.renderers.params import TokenizeParams

    params = TokenizeParams()
    out = renderer.tokenize_prompt({"prompt": "你好，vLLM。"}, params)
    report["encode_smoke_ids"] = len(out["prompt_token_ids"])

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
