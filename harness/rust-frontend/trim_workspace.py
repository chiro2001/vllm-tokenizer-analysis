#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""从 vLLM 的 rust/Cargo.toml 裁剪出一个只含 mock-engine 依赖链的精简 workspace。

为什么要裁：完整 workspace 有 13 个成员（含 axum / tonic / minijinja / pyo3 等），
只为跑 `vllm-mock-engine` 而解析整个依赖图既慢又没必要。精简后只保留
`vllm-engine-core-client` + `vllm-metrics` + `vllm-mock-engine`。

用法：
    trim_workspace.py <源 Cargo.toml> <目标 Cargo.toml>
    trim_workspace.py --help
"""

from __future__ import annotations

import argparse
import pathlib
import sys

KEEP = ("src/engine-core-client", "src/metrics", "src/mock-engine")


def trim(text: str) -> str:
    start = text.index("members = [")
    end = text.index("]", start) + 1
    members = "members = [\n" + "".join(f'    "{m}",\n' for m in KEEP) + "]"
    return text[:start] + members + text[end:]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("source", nargs="?", help="vLLM rust/Cargo.toml 路径")
    p.add_argument("target", nargs="?", help="输出路径")
    args = p.parse_args()
    if not args.source or not args.target:
        p.print_help()
        return 2
    src = pathlib.Path(args.source).read_text()
    pathlib.Path(args.target).write_text(trim(src))
    print(f"trimmed workspace written to {args.target} (members={list(KEEP)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
