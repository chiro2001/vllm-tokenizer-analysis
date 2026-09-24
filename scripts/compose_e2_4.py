#!/usr/bin/env python3
"""E2.4：tokenizer 三段占 TTFT / TPOT 的比例。

## 分子与分母（口径纪律，引用数字前必读）

**分母来自真机**：`liteprof_wave*` / `liteprof_v*_uni_*` 这类历史 run 的
`lite.log` 里带绝对时间戳。engine core 的 `Step:Model` 是每次模型执行的墙钟区间
（µs 精度），据此可以还原：

    首步（prefill 步）的 Step:Model 时长  →  TTFT 的**模型侧下界**
    后续每步 Step:Model 时长的中位数      →  TPOT 的**模型侧下界**

这里的 denominators **只是模型执行段**，不含 scheduler / 前端 / 网络 / 排队的
时间，所以算出来的是**占比的上界**（分母越小，比例越大）。文档里必须写明。

**分子**：

* `tokenizer: encode` / `tokenizer: render_messages` —— 前端进程 scope，
  **目录不同**：encode 在前端进程 tid 上，engine core 的 Step:Model 在另一个
  tid 上。两者不能相减，只能各自相对自己的分母。因此本脚本把分子限定为
  **同机测得的 µs/req 常数**（C 线的本地容器测量），分母用**真机的**
  Step:Model 时长。跨装置拼接是**有意为之**，必须显式标注（§5.4）。
* `detokenize: stream` —— 本机 µs/token 常数 × 每步产生的 text 片段。

**不做的事**：不把前端 scope 时间从 Step:Model 里减掉（两者本来就不同进程，
在时序上部分重叠）。

用法见 --help。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_lite_log_events(path: Path) -> list[tuple[str, float, float, int, int]]:
    """返回 [(scope, dur_us, start_us, tid, pid)]，scope 行 + phase 行都保留。"""
    events: list[tuple[str, float, float, int, int]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 5:
                continue
            try:
                events.append(
                    (parts[0], float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]))
                )
            except ValueError:
                continue
    return events


def engine_denominators(path: Path) -> dict[str, Any]:
    """从 Step:Model 序列还原 TTFT / TPOT 的模型侧下界。"""
    events = parse_lite_log_events(path)
    step_model = [e for e in events if e[0] == "Step:Model"]
    if not step_model:
        return {"error": "无 Step:Model"}

    # Step:Model 在 engine core 线程（同一 tid）。按 tid 分组后取计数最多的那组。
    by_tid: dict[int, list[tuple]] = defaultdict(list)
    for e in step_model:
        by_tid[e[3]].append(e)
    tid, evs = max(by_tid.items(), key=lambda kv: len(kv[1]))
    evs.sort(key=lambda e: e[2])

    # 一次 HTTP 请求对应一个 prefill 步（Step:Model 很大）后面跟若干 decode 步。
    # 用请求日志（requests.tsv）里的时间窗切分更稳：这里先用「相邻步间隔 > 1ms
    # 或本步 > 10ms」判定 prefill 步。
    prefill_like = [e for e in evs if e[1] > 10_000]  # >10 ms
    decode_like = [e for e in evs if e[1] <= 10_000]

    def stat(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        s = sorted(xs)
        return {
            "n": len(s),
            "mean_us": round(statistics.fmean(s), 1),
            "p50_us": round(s[len(s) // 2], 1),
            "min_us": round(s[0], 1),
            "max_us": round(s[-1], 1),
        }

    return {
        "engine_core_tid": tid,
        "n_step_model": len(evs),
        "prefill_step_Step_Model": stat([e[1] for e in prefill_like]),
        "decode_step_Step_Model": stat([e[1] for e in decode_like]),
        "prefill_threshold_us": 10_000,
    }


def frontend_scope_stats(path: Path) -> dict[str, Any]:
    """前端进程的 HTTP 段与 tokenizer 段。

    **按 pid 分组，不按 tid**：实测（`liteprof_v1_torch_uni_*`）`tokenizer: encode`
    跑在 `tid=1169,pid=1`，而 `http: create_completion` 跑在 `tid=1,pid=1`
    ——**同一个进程、不同的线程**（encode 被 `make_async` 丢进了
    ThreadPoolExecutor 的 worker 线程）。engine core 是另一个进程（`pid=131`）。
    这就是 COORDINATION §5.1 说的「必须按 tid/pid 区分」的具体形态。
    """
    events = parse_lite_log_events(path)
    tok_pids = {e[4] for e in events if e[0].startswith("tokenizer: ")}
    front_tids = sorted({e[3] for e in events if e[4] in tok_pids})
    engine_pids = sorted({e[4] for e in events if e[4] not in tok_pids})
    out: dict[str, Any] = {
        "frontend_pids": sorted(tok_pids),
        "frontend_tids": front_tids,
        "non_frontend_pids": engine_pids,
        "evidence": (
            "tokenizer scope 出现在这些 pid 上；同一 pid 的其他线程（asyncio 事件循环 "
            "tid=1 等）也属于前端进程"
        ),
    }
    for name in (
        "http: create_completion",
        "http: create_chat_completion",
        "input: process_inputs",
        "tokenizer: encode",
        "tokenizer: decode",
        "tokenizer: render_messages",
    ):
        vals = [e[1] for e in events if e[0] == name and e[4] in tok_pids]
        if vals:
            s = sorted(vals)
            out[name] = {
                "n": len(s),
                "mean_us": round(statistics.fmean(s), 1),
                "p50_us": round(s[len(s) // 2], 1),
                "min_us": round(s[0], 1),
                "max_us": round(s[-1], 1),
                "tids": sorted({e[3] for e in events if e[0] == name and e[4] in tok_pids}),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E2.4 组装：tokenizer 三段 vs TTFT/TPOT（分子分母见模块 docstring）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--cost-json", action="append", default=[],
                    help="C 线产出的 JSON（可重复）：bench_paths / bench_detokenize")
    ap.add_argument("--anchor-run", action="append", default=[],
                    help="真机 run 目录（含 lite-profiler/lite.log），可重复")
    ap.add_argument("--out-json", default="/workspace/data/cost/e2_share.json")
    ap.add_argument("--out-csv", default="/workspace/data/cost/e2_share.csv")
    args = ap.parse_args()

    # --- 1. 收集本机常数 ---------------------------------------------
    numerators: dict[str, Any] = {}
    for path in args.cost_json:
        doc = json.loads(Path(path).read_text())
        for rec in doc.get("results", []):
            key = rec.get("case")
            if key == "encode":
                numerators[f"encode@{rec['target_isl_tokens']}"] = {
                    "us": rec["scope_stats"]["mean_us"],
                    "source": f"{path}:{key}",
                    "definition": rec["scope"],
                }
            elif key == "render_messages":
                k = f"render_messages@{rec['target_isl_tokens']}/{rec['variant']}"
                numerators[k] = {
                    "us": rec["scope_stats"]["mean_us"],
                    "source": f"{path}:{key}",
                    "definition": rec["scope"],
                    "rendered_prompt_tokens": rec.get("rendered_prompt_tokens"),
                }
            elif key == "decode_prompt_reverse":
                k = f"decode_prompt_reverse@{rec['n_tokens']}"
                numerators[k] = {
                    "us": rec["scope_stats"]["mean_us"],
                    "source": f"{path}:{key}",
                    "definition": rec["scope"],
                }
            elif key and key.startswith("detokenize_stream_"):
                k = f"{key}@osl{rec['osl_tokens']}"
                numerators[k] = {
                    "us_per_token": rec["steady_total_us_per_token"],
                    "first_step_us": rec["first_step_us"]["mean_us"],
                    "init_us": rec["init_us"]["mean_us"],
                    "full_request_us": rec["full_request_us"]["mean_us"],
                    "source": f"{path}:{key}",
                    "definition": rec["scope"],
                }

    # --- 2. 收集真机分母 ---------------------------------------------
    anchors: dict[str, Any] = {}
    for run in args.anchor_run:
        run_dir = Path(run)
        log = run_dir / "lite-profiler" / "lite.log"
        if not log.is_file():
            log = run_dir / "lite.log"
        if not log.is_file():
            continue
        anchors[run_dir.name] = {
            "lite_log": str(log),
            "engine": engine_denominators(log),
            "frontend": frontend_scope_stats(log),
        }

    # --- 3. 算比例 ----------------------------------------------------
    table: list[dict[str, Any]] = []

    def pct(num: float, den: float) -> float | None:
        return round(num / den * 100, 4) if den else None

    for run_id, a in anchors.items():
        eng = a["engine"]
        ttft = (eng.get("prefill_step_Step_Model") or {}).get("p50_us")
        tpot = (eng.get("decode_step_Step_Model") or {}).get("p50_us")
        front = a["frontend"]

        # 前端 HTTP 段（同 run、同装置）——用于校验「分子 vs 分母」的装置差异
        http = front.get("http: create_completion") or front.get(
            "http: create_chat_completion"
        )

        # 该 run 自己测到的 encode（同装置，可直接比）
        own_encode = front.get("tokenizer: encode")
        if own_encode and http:
            table.append(
                {
                    "anchor_run": run_id,
                    "numerator": "tokenizer: encode (同装置同 run)",
                    "numerator_us": own_encode["p50_us"],
                    "denominator": "http: create_completion (前端整请求)",
                    "denominator_us": http["p50_us"],
                    "share_pct": pct(own_encode["p50_us"], http["p50_us"]),
                    "comparable": "yes",
                    "note": "同进程同装置，唯一可直接相除的一行",
                }
            )
            table.append(
                {
                    "anchor_run": run_id,
                    "numerator": "tokenizer: encode (同装置同 run)",
                    "numerator_us": own_encode["p50_us"],
                    "denominator": "Step:Model prefill (引擎侧 TTFT 下界)",
                    "denominator_us": ttft,
                    "share_pct": pct(own_encode["p50_us"], ttft) if ttft else None,
                    "comparable": "yes",
                    "note": "跨进程但在同一台机器、同一次运行内",
                }
            )

        # C 线本机常数 vs 真机分母（跨装置，仅作上界参考）
        for key in ("encode@128", "encode@1024", "encode@8192"):
            num = numerators.get(key)
            if num and ttft:
                table.append(
                    {
                        "anchor_run": run_id,
                        "numerator": f"{key} (本机容器, ISL={key.split('@')[1]})",
                        "numerator_us": num["us"],
                        "denominator": "Step:Model prefill (真机, 引擎侧 TTFT 下界)",
                        "denominator_us": ttft,
                        "share_pct": pct(num["us"], ttft),
                        "comparable": "cross-device (上界参考)",
                        "note": "分子 x86 CPU 容器，分母 a3-22 NPU；不得当作同装置结论",
                    }
                )

        for label in ("detokenize_stream_fast", "detokenize_stream_slow"):
            for osl in (32, 256, 1024):
                num = numerators.get(f"{label}@osl{osl}")
                if num and tpot:
                    table.append(
                        {
                            "anchor_run": run_id,
                            "numerator": f"{label} 稳定期/step (PSL≈1k, OSL={osl})",
                            "numerator_us": num["us_per_token"],
                            "denominator": "Step:Model decode (真机, 引擎侧 TPOT 下界)",
                            "denominator_us": tpot,
                            "share_pct": pct(num["us_per_token"], tpot),
                            "comparable": "cross-device (上界参考)",
                            "note": (
                                "分子只含「update + emit_text」的稳定期每步成本；"
                                "不含首次 step（其 prompt 预热成本按请求摊销）"
                            ),
                        }
                    )

    payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "definitions": {
            "TTFT_denominator": (
                "lite.log 里 engine core 线程的 Step:Model（>10ms 的步）p50 —— "
                "只含模型执行，不含 scheduler / 前端 / 网络 / 队列 ⇒ 占比是**上界**"
            ),
            "TPOT_denominator": (
                "lite.log 里 engine core 线程的 Step:Model（≤10ms 的步）p50 —— 同上"
            ),
            "encode_numerator": (
                "API server 前端进程 vllm/renderers/base.py::_tokenize_prompt 里 "
                "tokenizer(text, **encode_kwargs) 的包裹区间"
            ),
            "render_messages_numerator": (
                "前端进程 vllm/renderers/base.py::render_chat 里 render_messages(conv, params) "
                "的包裹区间；**不包含** encode（chat 模板内部已 tokenize，见 docs/02）"
            ),
            "decode_numerator_historical": (
                "前端进程 vllm/renderers/base.py::_decode —— 只做 prompt token ids 反解，"
                "与生成期解码**不是同一段代码**"
            ),
            "detokenize_stream_numerator": (
                "vllm/v1/engine/detokenizer.py 的 Fast/SlowIncrementalDetokenizer，"
                "每个生成步 update()+get_next_output_text()；LiteProfiler **没有**覆盖它"
            ),
            "cross_device_warning": (
                "标 cross-device 的行，分子来自本机 x86 容器、分母来自 a3-22 NPU run，"
                "仅作量级上界参考；同装置结论只看 comparable=yes 的行"
            ),
        },
        "numerators": numerators,
        "anchors": anchors,
        "table": table,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "anchor_run", "numerator", "numerator_us", "denominator",
        "denominator_us", "share_pct", "comparable", "note",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in table:
            w.writerow(row)

    print(f"JSON → {out_json}")
    print(f"CSV  → {out_csv}（{len(table)} 行）\n")
    for row in table:
        print(
            f"  {row['anchor_run'][:38]:<38} {row['numerator'][:44]:<44} "
            f"num={row['numerator_us']:>9.2f} den={row['denominator_us']:>10.1f} "
            f"share={row['share_pct']:>7}% [{row['comparable']}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
