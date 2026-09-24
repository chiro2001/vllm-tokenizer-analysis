#!/usr/bin/env python3
"""线 B / 当代 Python 侧三实现对比（E1 的 Python 装置）。

测三个实现（见 `plan/EXECUTION.md` §2.2）：

1. **HF `TokenizersBackend`（默认）** —— `tokenizers` 0.22 的 `Tokenizer`，
   也就是 vLLM 0.26.0 的默认 Python 后端。
2. **fastokens**（`VLLM_USE_FASTOKENS=1` 走的那个包）—— 直接测 `fastokens` 包本身。
3. **gigatoken**（无 vLLM 集成）—— `gigatoken.Tokenizer(...)`。

装置定位（这是**桥梁校准**的两端之一，另一端是 Rust harness）：

- 本脚本在**裸库**上测（本地 miniforge / 容器内都行），不 import vLLM；
- 候选后端的 `batched()` / `encode_batch()` 走各自的多线程路径，但默认
  `--threads 1` 与 Rust 侧单条路径对齐；`RAYON_NUM_THREADS` / `TOKENIZERS_PARALLELISM`
  决定线程数，必须由 `scripts/limit.sh` 设。

口径（与 `harness/rust/bench/src/stats.rs` 一致，便于跨装置比较）：

- 单条路径：调用线程直接跑，`iters` 次取**中位数**；
- 报告 µs/op、MB/s、tokens/s；
- warmup 2 轮不计时。

用法：
    python3 harness/python/bench_backends.py --help
    CORES=2 scripts/limit.sh python3 harness/python/bench_backends.py \
        --impl hf,fastokens,gigatoken --lengths 128,1k --corpus mixed --json-out ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 语料：与 Rust harness 同源（同样的段落池、同样的截断规则）。
#
# 为什么在 Python 侧再实现一遍而不调用 Rust：本脚本要在**没有 cargo 的环境**
# （容器、aarch64 机器）里也能跑。语料生成的规则固定成：段落循环拼接 →
# 用参考后端的 token 数二分截断 → 误差 ≤1%。两侧的 sha256 会写进各自的
# manifest，跑完可以核对"两个装置用的是不是同一份输入"。
# ---------------------------------------------------------------------------

ZH = [
    "服务在一次批量推理任务中于凌晨两点出现了首包延迟突增，运维同学先看的是前端进程的分词耗时，"
    "因为这一段代码在请求进入调度器之前就要跑完，它不属于引擎核心，也不在任何一张卡上。",
    "我们把同一条提示词分别送进三个不同的分词实现，统计出来的 token 个数居然不完全一样，"
    "差异集中在中文标点、连续空格以及那些看起来像特殊标记的字符串上，这直接影响计费口径。",
    "在流式返回的场景里，解码器必须逐 token 吐出可读文本，遇到被切开的 UTF-8 多字节字符时"
    "要先把字节留在缓冲区里，等到下一个 token 补齐后再一起输出，否则前端会看到乱码方块。",
    "压测报告显示，当提示词长度从一百二十八增长到八千时，单条编码的耗时并不是线性增长，"
    "预分词阶段的正则匹配占比上升得更快，而合并阶段的哈希查找反而相对稳定。",
    "评审时有同事提出直接换成用其他语言写的前端，理由是启动更快、内存更省；"
    "我们要求他给出同条件的数据：同样的输入、同样的线程数、同样的硬件，并且逐 token 对齐编码结果。",
    "最终结论写在文档第三小节：分词器不是瓶颈，但它是可以预测的固定成本，"
    "在首包延迟里占比通常不到百分之五，除非你的请求里塞满了稀有字符和超长空白。",
]

EN = [
    "The tokenizer front end runs in the API server process, before any request reaches the "
    "scheduler, so its latency shows up directly in time to first token and never touches a GPU.",
    "We compared three implementations on the same prompt and the token counts disagreed by a "
    "fraction of a percent; the differences clustered around punctuation, leading whitespace, "
    "and strings that merely resemble registered special tokens.",
    "Streaming decode has to emit readable text one token at a time. Whenever a multi-byte "
    "character is split across two tokens, the decoder must hold the partial bytes in a buffer "
    "until the next token completes the sequence.",
    "Under load the encoding cost grows sub-linearly with prompt length: the pre-tokenization "
    "regex dominates the short prompts while the merge loop stays roughly constant per byte.",
    "A colleague suggested replacing the front end with a different language runtime because it "
    "starts faster and uses less memory, so we asked for numbers measured under identical "
    "conditions: same input, same thread count, same machine, identical token ids.",
    "The conclusion is deliberately narrow. The tokenizer is not the bottleneck, but it is a "
    "predictable fixed cost that rarely exceeds five percent of time to first token unless the "
    "prompt is full of rare characters or very long runs of whitespace.",
]

CODE = [
    "fn decode_stream(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Result<String> {\n    "
    "let mut out = String::new();\n    for id in ids {\n        "
    "let piece = tokenizer.id_to_token(*id).unwrap_or_default();\n        "
    "out.push_str(&piece);\n    }\n    Ok(out)\n}\n",
    "def render_messages(messages, tools=None, template=None):\n    "
    "# NOTE: this runs in the front-end process, before the engine core\n    "
    "rendered = template.render(messages=messages, tools=tools or [])\n    "
    "return rendered if isinstance(rendered, str) else rendered[0]\n",
    '{"model":"qwen3-0.6b","max_tokens":512,"temperature":0.7,"stream":true,'
    '"stop":["<|im_end|>"],"messages":[{"role":"user","content":"总结这段日志"}]}\n',
    "SELECT tokenizer_id, count(*) AS requests, avg(elapsed_us) AS avg_us\n  "
    "FROM frontend_metrics\n WHERE scope = 'tokenizer: encode'\n   "
    "GROUP BY tokenizer_id\n ORDER BY avg_us DESC;\n",
    "const MAX_CHARS_PER_TOKEN: usize = 128;\n\n"
    "pub fn assert_token_budget(text: &str, max_tokens: usize) -> bool {\n    "
    "text.chars().count() <= max_tokens * MAX_CHARS_PER_TOKEN\n}\n",
    "if __name__ == \"__main__\":\n    "
    "parser = argparse.ArgumentParser(description=\"backend matrix harness\")\n    "
    "parser.add_argument(\"--threads\", type=int, default=1)\n    "
    "parser.add_argument(\"--json-out\", type=Path)\n    args = parser.parse_args()\n",
]

MIXED = [
    "<|im_start|>system\n你是 Qwen3 的助手，回答尽量简短。"
    "<|im_end|>\n<|im_start|>user\n请用中英混合总结以下需求："
    "The service should stop cleanly at EOS, avoid leaking the next template turn."
    "\n<|im_end|>\n<|im_start|>assistant\n",
    "输入：4 个并发请求，20480 个 prompt tokens，生成 256 个 token；"
    "输出：首包延迟 180 ms，其中分词占 4.2 ms（front-end process, tok/s=4900）。"
    "Verify the numbers before quoting them.\n",
    "工具调用示例：<|tool_calls_section_begin|>{\"name\":\"summarize\","
    "\"arguments\":{\"style\":\"brief\",\"lang\":\"zh\"}}<|tool_calls_section_end|>"
    "请把这段 JSON 原样保留，不要改写字段名。\n",
    "日志片段 2026-09-24T02:14:07Z WARN frontend tokenizer: encode 8192-token prompt took 21.6 ms; "
    "结论是这一条请求的分词耗时约为首包延迟的 3%，属于可接受范围（阈值 5%）。\n",
    "压力测试结论（待复核）：throughput 12.4k tok/s at 8 threads, memory +38 MB RSS, "
    "但 riptoken 与 tiktoken-rs 的输出逐 id 一致，这一点在本机与容器内都验证过。\n",
    "最后的建议：在线路径保持单线程（1 thread per request），批量离线路径再开 rayon；"
    "keep bytes-per-second claims in the offline bucket and never mix them with TTFT.\n",
]

PARAGRAPHS = {"zh": ZH, "en": EN, "code": CODE, "mixed": MIXED}


@dataclass
class Measurement:
    median_us: float
    min_us: float
    max_us: float
    p90_us: float
    iters: int
    input_bytes: int
    input_tokens: int
    samples_us: list[float] = field(default_factory=list, repr=False)

    @property
    def mb_per_s(self) -> float:
        return (self.input_bytes / (self.median_us / 1e6)) / 1e6 if self.median_us else float("nan")

    @property
    def tokens_per_s(self) -> float:
        return self.input_tokens / (self.median_us / 1e6) if self.median_us else float("nan")

    def to_row(self, **extra) -> dict:
        row = {
            "median_us": round(self.median_us, 3),
            "min_us": round(self.min_us, 3),
            "max_us": round(self.max_us, 3),
            "p90_us": round(self.p90_us, 3),
            "iters": self.iters,
            "bytes": self.input_bytes,
            "tokens": self.input_tokens,
            "mb_per_s": round(self.mb_per_s, 3),
            "tokens_per_s": round(self.tokens_per_s, 1),
        }
        row.update(extra)
        return row


def measure(fn, input_bytes: int, input_tokens: int, iters: int, warmup: int = 2) -> Measurement:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    median = (
        samples_sorted[n // 2]
        if n % 2
        else (samples_sorted[n // 2 - 1] + samples_sorted[n // 2]) / 2
    )
    p90 = samples_sorted[min(n - 1, max(0, int(n * 0.9) - (1 if n % 10 == 0 and n > 1 else 0)))]
    return Measurement(
        median_us=median,
        min_us=samples_sorted[0],
        max_us=samples_sorted[-1],
        p90_us=p90,
        iters=n,
        input_bytes=input_bytes,
        input_tokens=input_tokens,
        samples_us=samples_sorted,
    )


# ---------------------------------------------------------------------------
# 三个实现各自的适配器
# ---------------------------------------------------------------------------


class Impl:
    """适配层：统一 encode/decode/decode_stream（有就暴露）。"""

    name: str
    family: str
    artifact: str
    version: str
    supports_stream: bool = False

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def load(self) -> None:  # noqa: B027 - 可选实现
        pass

    @property
    def vocab_size(self) -> int:
        return -1

    def meta(self) -> dict:
        return {
            "impl": self.name,
            "family": self.family,
            "artifact": self.artifact,
            "version": self.version,
            "vocab_size": self.vocab_size,
            "supports_streaming_decode": self.supports_stream,
        }


class HfImpl(Impl):
    """`tokenizers` 0.22.2 的 `Tokenizer`（vLLM 默认 Python 后端）。"""

    name = "hf_tokenizers"
    family = "hf"

    def __init__(self, path: Path):
        import tokenizers

        self.artifact = str(path)
        self.version = getattr(tokenizers, "__version__", "?")
        self.tok = tokenizers.Tokenizer.from_file(str(path))
        self.supports_stream = True

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def decode_stream(self, prompt_ids: list[int], generated: list[int]) -> str:
        """复刻 vLLM 的 `FastIncrementalDetokenizer`（vllm/v1/engine/detokenizer.py:183）：

        `tokenizers.decoders.DecodeStream(ids=prompt_token_ids, skip_special_tokens=...)`
        随后对每个生成 token 调 `stream.step(tokenizer, id)`。
        提示：`tokenizers` 0.22 是**模块级** `DecodeStream`（不是
        `Tokenizer.decode_stream` 方法），这也是 vLLM 从模块上取它的原因
        （fastokens shim 会替换这个类）。
        """
        import tokenizers.decoders

        stream = tokenizers.decoders.DecodeStream(
            ids=prompt_ids, skip_special_tokens=False
        )
        out: list[str] = []
        for i in generated:
            chunk = stream.step(self.tok, i)
            if chunk:
                out.append(chunk)
        return "".join(out)


class FastokensImpl(Impl):
    """`fastokens` 包（`VLLM_USE_FASTOKENS=1` 时 vLLM 用的那个）。"""

    name = "fastokens"
    family = "hf"

    def __init__(self, path: Path):
        import fastokens
        import importlib.metadata as md

        self.artifact = str(path)
        try:
            self.version = md.version("fastokens")
        except Exception:  # noqa: BLE001
            self.version = getattr(fastokens, "__version__", "?")
        # 版本的 API 差异（vLLM 0.26.0 在 Rust 侧钉的是 fastokens 0.2.1，
        # 而 PyPI 只有 0.2.0 / 0.3.x，这一条要写进文档的"口径"里）：
        #   0.2.x: from_json_str(text)  / vocab_size() 是方法
        #   0.3.x: from_json_str(text)  / vocab_size 是属性
        text = path.read_text(encoding="utf-8")
        self.tok = fastokens.Tokenizer.from_json_str(text)
        # fastokens 的 Python 绑定没有暴露 DecodeStream 等价物；
        # 有状态流式解码在 vLLM 的 Rust 侧（incremental.rs）里做。
        self.supports_stream = False

    @property
    def vocab_size(self) -> int:
        v = self.tok.vocab_size
        return v() if callable(v) else v

    def encode(self, text: str) -> list[int]:
        # 返回的是 `Encoding`（0.3.x 有 `.ids`），不是裸列表/数组。
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


class GigatokenImpl(Impl):
    """`gigatoken` 包（无 vLLM 集成，需自建适配）。"""

    name = "gigatoken"
    family = "gigatoken"

    def __init__(self, path: Path):
        import gigatoken

        self.artifact = str(path)
        self.version = getattr(gigatoken, "__version__", "?")
        self.tok = gigatoken.Tokenizer(str(path))
        self.supports_stream = False

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tok.encode(text)]

    def decode(self, ids: list[int]) -> str:
        raw = self.tok.decode(ids)
        # gigatoken 的 decode 返回 bytes（不是 str），这里显式解码：
        # 这就是"没有 skip_special_tokens、也没有替换字符策略"的直接体现。
        return raw.decode("utf-8", errors="replace")

    def decode_bytes_raw(self, ids: list[int]) -> bytes:
        return self.tok.decode(ids)

    def encode_batch(self, texts: list[str], parallel: bool) -> None:
        """批量入口（其内部自带 chunk 切分 + rayon）。"""
        self.tok.encode_batch(texts, parallel=parallel)


IMPLS = {"hf": HfImpl, "fastokens": FastokensImpl, "gigatoken": GigatokenImpl}


# ---------------------------------------------------------------------------
# 语料生成（与 Rust 侧同规则）
# ---------------------------------------------------------------------------


def build_corpus(kind: str, target_tokens: int, ref) -> tuple[str, list[int]]:
    paras = PARAGRAPHS[kind]
    text = ""
    idx = 0
    while len(ref.encode(text)) < target_tokens:
        text += paras[idx % len(paras)] + "\n"
        idx += 1
        if idx > 100_000:
            raise RuntimeError("语料拼接超过 100k 段仍未达到目标长度")

    lo, hi = 0, len(text)  # 字符数
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        n = len(ref.encode(text[:mid]))
        if n <= target_tokens:
            lo = mid
        else:
            hi = mid
    text = text[:lo].rstrip("\n ")
    ids = ref.encode(text)
    err = abs(len(ids) - target_tokens) / target_tokens
    if err > 0.01:
        raise RuntimeError(f"语料 {kind}-{target_tokens} 长度误差 {err:.3%} 超过 1%")
    return text, ids


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def host_info() -> dict:
    def cmd(*args: str) -> str:
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    loadavg = [0.0, 0.0, 0.0]
    try:
        loadavg = [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except Exception:  # noqa: BLE001
        pass
    mem_available_gb = float("nan")
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_gb = int(line.split()[1]) / 1024 / 1024
    except Exception:  # noqa: BLE001
        pass
    affinity = "unknown"
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                affinity = line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_affinity": affinity,
        "loadavg": loadavg,
        "mem_available_gb": round(mem_available_gb, 2),
        "nproc": os.cpu_count(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "RAYON_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
                "OMP_NUM_THREADS",
                "VLLM_USE_FASTOKENS",
                "CORES",
            )
        },
        "container_image": os.environ.get("B_CONTAINER_IMAGE"),
        "git_commit": cmd("git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--hf-dir",
        type=Path,
        default=Path(os.environ.get("B_HF_DIR", "/home/chiro/models/Qwen3-0.6B")),
        help="含 tokenizer.json 的目录（默认 %(default)s）",
    )
    ap.add_argument(
        "--impl",
        default="hf,fastokens,gigatoken",
        help="逗号分隔的实现：hf,fastokens,gigatoken（默认全部）",
    )
    ap.add_argument("--corpus", default="all", help="all 或 zh,en,code,mixed")
    ap.add_argument("--lengths", default="128,1k", help="all 或 128,1k,8k")
    ap.add_argument("--iters", type=int, default=20, help="每个测量点的计时轮数")
    ap.add_argument("--batch-docs", type=int, default=8, help="批量路径的文档数")
    ap.add_argument("--json-out", type=Path, default=None, help="结果 JSON 路径")
    ap.add_argument("--manifest-out", type=Path, default=None, help="manifest JSON 路径")
    ap.add_argument("--threads", type=int, default=0, help="提示值，写进 manifest（真实线程数由环境变量决定）")
    args = ap.parse_args()

    kinds = list(PARAGRAPHS) if args.corpus == "all" else args.corpus.split(",")
    aliases = {"128": 128, "1k": 1024, "1024": 1024, "8k": 8192, "8192": 8192}
    if args.lengths == "all":
        lengths = [128, 1024, 8192]
    else:
        lengths = []
        for token in args.lengths.split(","):
            token = token.strip()
            if not token:
                continue
            # `dict.get(s, default)` 会**先求值**默认参数，所以不能写
            # `aliases.get(s, int(s))`（"1k" 会在求值 int() 时直接抛）。
            lengths.append(aliases[token] if token in aliases else int(token))

    hf_json = args.hf_dir / "tokenizer.json"
    if not hf_json.exists():
        print(f"找不到 {hf_json}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    impl_meta: list[dict] = []
    missing: list[dict] = []

    # 参考后端固定用 HF tokenizers（与 Rust 侧同一实现），语料长度由它定义。
    reference = HfImpl(hf_json)
    corpora: dict[tuple[str, int], tuple[str, list[int]]] = {}
    for kind in kinds:
        for n in lengths:
            text, ids = build_corpus(kind, n, reference)
            corpora[(kind, n)] = (text, ids)
            print(
                f"[corpus] {kind:<6} {n:>5} tokens  {len(text.encode()):>7} B  "
                f"sha256={sha256_hex(text.encode())[:16]}",
                file=sys.stderr,
            )

    impls: list[Impl] = []
    for name in args.impl.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            impls.append(IMPLS[name](hf_json))
        except Exception as e:  # noqa: BLE001
            print(f"[impl] {name} 不可用: {type(e).__name__}: {e}", file=sys.stderr)
            missing.append({"impl": name, "reason": f"{type(e).__name__}: {e}"})

    for impl in impls:
        impl_meta.append(impl.meta())
        print(
            f"[impl] {impl.name:<16} v{impl.version:<10} vocab={impl.vocab_size}",
            file=sys.stderr,
        )

    for impl in impls:
        for (kind, n), (text, ref_ids) in corpora.items():
            data = text.encode()
            try:
                ids = impl.encode(text)
            except Exception as e:  # noqa: BLE001
                rows.append({"impl": impl.name, "corpus": kind, "length": n, "op": "encode",
                             "error": f"{type(e).__name__}: {e}"})
                continue

            m = measure(lambda: impl.encode(text), len(data), len(ids), args.iters)
            row = m.to_row(
                impl=impl.name, family=impl.family, corpus=kind, length=n, op="encode",
                mode="single", ids_sha256=sha256_hex(",".join(map(str, ids)).encode())[:16],
                ids_equal_reference=ids == ref_ids,
                ids_count=len(ids),
                ref_ids_count=len(ref_ids),
            )
            rows.append(row)
            print(
                f"[encode] {impl.name:<16} {kind:<6} {n:>5} {m.median_us:>10.2f} µs "
                f"({m.mb_per_s:>7.1f} MB/s) ids_eq_ref={ids == ref_ids}",
                file=sys.stderr,
            )

            try:
                decoded = impl.decode(ids)
            except Exception as e:  # noqa: BLE001
                rows.append({"impl": impl.name, "corpus": kind, "length": n, "op": "decode",
                             "error": f"{type(e).__name__}: {e}"})
                print(f"[decode] {impl.name} {kind}-{n} 失败: {e}", file=sys.stderr)
                continue
            m = measure(lambda: impl.decode(ids), len(decoded.encode()), len(ids), args.iters)
            rows.append(
                m.to_row(
                    impl=impl.name, family=impl.family, corpus=kind, length=n, op="decode",
                    mode="single", decoded_bytes=len(decoded.encode()),
                    roundtrip_ok=impl.encode(decoded) == ids,
                )
            )
            print(
                f"[decode] {impl.name:<16} {kind:<6} {n:>5} {m.median_us:>10.2f} µs "
                f"({m.mb_per_s:>7.1f} MB/s)",
                file=sys.stderr,
            )

            if impl.supports_stream and len(ids) > 16:
                prompt = ids[: len(ids) // 2]
                generated = ids[len(ids) // 2 :][:256]
                if generated:
                    m = measure(
                        lambda: impl.decode_stream(prompt, generated),  # type: ignore[attr-defined]
                        0,
                        len(generated),
                        args.iters,
                    )
                    rows.append(
                        m.to_row(
                            impl=impl.name, family=impl.family, corpus=kind, length=n,
                            op="stream_decode", mode="single", generated_tokens=len(generated),
                        )
                    )
                    print(
                        f"[stream] {impl.name:<16} {kind:<6} {n:>5} {m.median_us:>10.2f} µs "
                        f"({m.tokens_per_s:>9.0f} tok/s)",
                        file=sys.stderr,
                    )

    # 批量：gigatoken 有自己的批量入口；HF tokenizers 的 batched() 在 Python 侧
    # 走 Rust rayon 池；fastokens 的 Python 绑定没有批量 encode。
    for impl in impls:
        for (kind, n), (text, ids) in corpora.items():
            if n < 128:
                continue
            docs = split_docs(text, args.batch_docs)
            if isinstance(impl, GigatokenImpl):
                for parallel in (True, False):
                    m = measure(
                        lambda p=parallel: impl.encode_batch(docs, parallel=p),
                        len(text.encode()),
                        len(ids),
                        max(3, args.iters // 4),
                    )
                    rows.append(
                        m.to_row(
                            impl=impl.name, family=impl.family, corpus=kind, length=n,
                            op="encode_batch", mode="parallel" if parallel else "serial",
                            docs=len(docs),
                        )
                    )
                    print(
                        f"[batch ] {impl.name:<16} {kind:<6} {n:>5} "
                        f"{'parallel' if parallel else 'serial  '} {m.median_us:>10.2f} µs "
                        f"({m.mb_per_s:>7.1f} MB/s)",
                        file=sys.stderr,
                    )
            elif isinstance(impl, HfImpl):
                m = measure(
                    lambda: impl.tok.encode_batch(docs, add_special_tokens=False),
                    len(text.encode()),
                    len(ids),
                    max(3, args.iters // 4),
                )
                rows.append(
                    m.to_row(
                        impl=impl.name, family=impl.family, corpus=kind, length=n,
                        op="encode_batch", mode="batched", docs=len(docs),
                    )
                )
                print(
                    f"[batch ] {impl.name:<16} {kind:<6} {n:>5} batched    {m.median_us:>10.2f} µs "
                    f"({m.mb_per_s:>7.1f} MB/s)",
                    file=sys.stderr,
                )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host_info(),
        "impls": impl_meta,
        "missing": missing,
        "rows": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[done] {len(rows)} rows → {args.json_out}", file=sys.stderr)

    if args.manifest_out:
        manifest = {
            "line": "B-backends",
            "task": "E1 Python 侧三实现对比",
            "line_kind": "python-harness",
            "generated_at": payload["generated_at"],
            "upstream_commit": "568afb3a13806beb53bb2e6bd518269357b237c0",
            "container_image": os.environ.get("B_CONTAINER_IMAGE"),
            "host": payload["host"],
            "threads_requested": args.threads,
            "corpus_sha256": {
                f"{k}-{n}": sha256_hex(t.encode()) for (k, n), (t, _) in corpora.items()
            },
            "script_sha256": {"harness/python/bench_backends.py": sha256_hex(Path(__file__).read_bytes())},
            "artifacts": (
                {"path": str(args.json_out), "sha256": sha256_hex(args.json_out.read_bytes())}
                if args.json_out and args.json_out.exists()
                else {}
            ),
            "caveats": [
                "本装置是裸库（不 import vLLM）；与 Rust harness 的数字要靠 fastokens / HF tokenizers "
                "两个双端都存在的实现做桥梁校准后才能同表比较。",
                "gigatoken 的 decode 返回 bytes，本脚本用 errors='replace' 转 str，"
                "与 HF 的 decode 语义不完全等价。",
                "fastokens 的 Python 绑定没有暴露有状态流式解码接口，"
                "stream_decode 一行只对 HF tokenizers 有数；Rust 侧覆盖全部六个后端。",
            ],
        }
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[done] manifest → {args.manifest_out}", file=sys.stderr)

    return 0


def split_docs(text: str, n: int) -> list[str]:
    if n <= 1:
        return [text]
    step = len(text) // n
    return [text[i * step : (i + 1) * step] for i in range(n - 1)] + [text[(n - 1) * step :]]


if __name__ == "__main__":
    raise SystemExit(main())
