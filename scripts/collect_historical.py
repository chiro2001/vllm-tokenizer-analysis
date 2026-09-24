#!/usr/bin/env python3
"""E2.6：回收历史 LiteProfiler 日志里的 `tokenizer:` scope 计数。

**只读**：绝不修改任何历史 run 目录。

口径（对应 plan/COORDINATION.md §5）：

1. 一份 `lite.log` 里同时混着 **API server 前端进程** 与 **engine core**
   的 scope。`tokenizer: encode/decode/render_messages` 只可能出现在前端进程，
   必须按 `tid`（第 4 列）区分；本脚本按 tid 分组后，只把**含 tokenizer scope
   的 tid** 当作前端进程证据，并把该 run 的全部前端 scope 一起统计。
2. `count=0` 只说明**这次负载没走到那条分支**，不说明它不耗时。
   所以每条 run 都要记录「该 run 发的是什么请求」——脚本从
   `requests/req_*.json` 里推断负载形态（chat / completion-text /
   completion-token_ids），据此解释某一列为 0 的原因。
3. 原始 lite.log 是**整个进程生命周期**的采样；本脚本不做时间窗裁剪，
   只做 count/avg/min/max/mean 的直方统计，并在输出里带上 run 的
   起止时间戳。

用法见 --help。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

TOKENIZER_SCOPES = ("tokenizer: encode", "tokenizer: decode", "tokenizer: render_messages")


def parse_lite_log(path: Path) -> tuple[dict[str, list[int]], dict[int, dict[str, list[int]]]]:
    """返回 (全部 scope 的 tid 集合, {scope: {tid: [dur_us]}})。

    lite.log 行格式：`<scope>|<dur_us>|<start_us>|<tid>|<pid>`
    """
    all_tids: set[int] = set()
    per: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("phase:"):
                continue
            parts = line.split("|")
            if len(parts) < 5:
                bad += 1
                continue
            name, dur_s, _start_s, tid_s, _pid_s = parts[:5]
            try:
                dur = int(float(dur_s))
                tid = int(tid_s)
            except ValueError:
                bad += 1
                continue
            all_tids.add(tid)
            per[name][tid].append(dur)
    per["__malformed__"]["-1"] = [bad]
    return all_tids, per


def classify_requests(run_dir: Path) -> dict[str, object]:
    """推断负载形态。

    两类 run 目录的存法不同，都要能吃：

    * `liteprof_v*_*`：`requests/req_NN.json` 是**请求体**（含 `prompt`）；
    * `liteprof_wave*`：`requests/measuredN.json` 是**响应体**
      （含 `choices` / `usage` / `object`）。响应里 `object=text_completion`
      同样能判定端点，`usage.prompt_tokens` 更能直接给出该 run 的 prompt 规模
      ——这比请求体还准，所以单独统计。
    """
    info: dict[str, object] = {
        "n_requests": 0,
        "endpoint": None,
        "has_messages": False,
        "has_text_prompt": False,
        "has_token_ids_prompt": False,
        "has_tools": False,
        "sample_prompt_keys": [],
        "max_tokens_values": [],
        "ignore_eos_true": 0,
        "n_response_files": 0,
        "response_objects": [],
        "prompt_tokens_from_usage": [],
        "completion_tokens_from_usage": [],
        "evidence_kind": None,
    }
    reqs = sorted(run_dir.glob("requests/req_*.json"))
    if not reqs:
        reqs = sorted(run_dir.glob("requests/*.json")) or sorted(run_dir.glob("req_*.json"))
    info["n_requests"] = len(reqs)
    max_toks: set[int] = set()
    objs: set[str] = set()
    ptoks: list[int] = []
    ctoks: list[int] = []
    for p in reqs:
        try:
            obj = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(obj, dict):
            continue
        info["sample_prompt_keys"] = sorted(obj.keys())
        if "choices" in obj or "usage" in obj:
            info["n_response_files"] += 1
            if isinstance(obj.get("object"), str):
                objs.add(obj["object"])
            usage = obj.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int):
                    ptoks.append(usage["prompt_tokens"])
                if isinstance(usage.get("completion_tokens"), int):
                    ctoks.append(usage["completion_tokens"])
            if info["evidence_kind"] is None or info["evidence_kind"] == "request":
                info["evidence_kind"] = "response"
            continue
        if info["evidence_kind"] is None:
            info["evidence_kind"] = "request"
        if "messages" in obj:
            info["has_messages"] = True
            if obj.get("tools"):
                info["has_tools"] = True
        if isinstance(obj.get("prompt"), str):
            info["has_text_prompt"] = True
        if isinstance(obj.get("prompt"), list):
            info["has_token_ids_prompt"] = True
        if "prompt_token_ids" in obj:
            info["has_token_ids_prompt"] = True
        mt = obj.get("max_tokens")
        if isinstance(mt, int):
            max_toks.add(mt)
        if obj.get("ignore_eos") is True:
            info["ignore_eos_true"] += 1
    info["max_tokens_values"] = sorted(max_toks)
    info["response_objects"] = sorted(objs)
    info["prompt_tokens_from_usage"] = sorted(set(ptoks))
    info["completion_tokens_from_usage"] = sorted(set(ctoks))
    # 端点判定：请求体优先，没有就用响应体的 object 字段
    if info["has_messages"] or "chat.completion" in objs:
        info["endpoint"] = "/v1/chat/completions"
    elif info["has_text_prompt"] or info["has_token_ids_prompt"] or "text_completion" in objs:
        info["endpoint"] = "/v1/completions"
    elif any("chat" in o for o in objs):
        info["endpoint"] = "/v1/chat/completions (推断自 response.object)"
    elif any("completion" in o for o in objs):
        info["endpoint"] = "/v1/completions (推断自 response.object)"
    return info


def summarize_scopes(per: dict[str, dict[int, list[int]]], tids: list[int]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in TOKENIZER_SCOPES:
        vals: list[int] = []
        for t in tids:
            vals.extend(per.get(name, {}).get(t, []))
        out[name] = {
            "count": len(vals),
            "avg_us": round(fmean(vals), 3) if vals else None,
            "min_us": min(vals) if vals else None,
            "max_us": max(vals) if vals else None,
        }
    return out


def process_run(run_dir: Path, *, source: str, host: str | None) -> dict[str, object]:
    log = run_dir / "lite-profiler" / "lite.log"
    if not log.is_file():
        log = run_dir / "lite.log"
    rec: dict[str, object] = {
        "source": source,
        "host": host,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "lite_log": str(log) if log.is_file() else None,
        "lite_log_bytes": log.stat().st_size if log.is_file() else None,
        "run_mtime": datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
        if run_dir.exists()
        else None,
    }
    rec.update(classify_requests(run_dir))
    if not log.is_file():
        rec["error"] = "lite.log 不存在"
        return rec

    all_tids, per = parse_lite_log(log)
    malformed = per.get("__malformed__", {}).get("-1", [0])[0]
    # 前端进程 = 出现过任一 tokenizer scope 的 tid
    front_tids = sorted(
        {t for name in TOKENIZER_SCOPES for t in per.get(name, {})}
    )
    rec["all_tids_in_log"] = sorted(all_tids)
    rec["frontend_tids"] = front_tids
    rec["n_tids"] = len(all_tids)
    rec["malformed_lines"] = malformed
    rec["tokenizer_scopes"] = summarize_scopes(per, front_tids)
    rec["scope_names_present"] = sorted(
        k for k in per if not k.startswith("__") and per[k]
    )
    # 前端进程整体承载的 HTTP 层耗时（用于 E2.4 占比的交叉参考）
    for name in ("http: create_completion", "http: create_chat_completion", "input: process_inputs"):
        vals = [v for t in front_tids for v in per.get(name, {}).get(t, [])]
        if vals:
            rec[f"frontend_{name}_count"] = len(vals)
            rec[f"frontend_{name}_avg_us"] = round(fmean(vals), 3)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="回收历史 LiteProfiler 日志中的 tokenizer scope（只读）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--run", action="append", default=[], metavar="DIR[#source]",
                    help="run 目录，可重复；可加 #source 后缀标注来源，"
                         "例如 '/path/run#local'")
    ap.add_argument("--glob", action="append", default=[], metavar="PATTERN[#source]",
                    help="glob 模式，可重复")
    ap.add_argument("--out-csv", default="/workspace/data/historical/historical_tokenizer_scopes.csv")
    ap.add_argument("--out-json", default="/workspace/data/historical/historical_runs.json")
    args = ap.parse_args()

    targets: list[tuple[Path, str]] = []
    for spec in args.run:
        path, _, src = spec.partition("#")
        targets.append((Path(path).expanduser(), src or "unknown"))
    for spec in args.glob:
        pattern, _, src = spec.partition("#")
        for p in sorted(
            Path(pattern).expanduser().parent.glob(Path(pattern).expanduser().name)
        ):
            targets.append((p, src or "unknown"))

    if not targets:
        print("没有指定 --run/--glob", file=sys.stderr)
        return 2

    records = []
    for path, src in targets:
        if not path.is_dir():
            print(f"跳过（不是目录）: {path}", file=sys.stderr)
            continue
        rec = process_run(path, source=src, host=None)
        if "tokenizer_scopes" not in rec:
            # 例如远端还在拉、目录只有一半；不要因为一个残缺目录炸掉整批
            print(f"跳过（无 tokenizer_scopes，可能未拉全）: {rec['run_id']}", file=sys.stderr)
            continue
        records.append(rec)
        ts = rec["tokenizer_scopes"]  # type: ignore[index]
        print(
            f"[{src}] {rec['run_id']:<52} reqs={rec.get('n_requests'):>3} "
            f"tids={rec.get('n_tids'):>3} "
            f"encode={ts['tokenizer: encode']['count']:>4}"
            f"({ts['tokenizer: encode']['avg_us']}) "
            f"decode={ts['tokenizer: decode']['count']:>4} "
            f"render={ts['tokenizer: render_messages']['count']:>4} "
            f"endpoint={rec.get('endpoint')}"
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),
                "scope_definitions": {
                    "tokenizer: encode": "vllm/renderers/base.py:_tokenize_prompt -> tokenizer(text)",
                    "tokenizer: decode": "vllm/renderers/base.py:_decode -> tokenizer.decode(ids)",
                    "tokenizer: render_messages": "vllm/renderers/base.py:render_chat -> render_messages(conv)",
                },
                "runs": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source", "host", "run_id", "endpoint", "n_requests",
        "evidence_kind", "has_messages", "has_text_prompt", "has_token_ids_prompt",
        "has_tools", "max_tokens_values", "ignore_eos_true",
        "response_objects", "prompt_tokens_from_usage", "completion_tokens_from_usage",
        "n_tids", "frontend_tids",
        "encode_count", "encode_avg_us", "encode_min_us", "encode_max_us",
        "decode_count", "decode_avg_us", "decode_min_us", "decode_max_us",
        "render_count", "render_avg_us", "render_min_us", "render_max_us",
        "http_scope", "http_count", "http_avg_us",
        "lite_log_bytes", "run_dir",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            ts = rec["tokenizer_scopes"]  # type: ignore[index]
            http_name = None
            for cand in ("http: create_completion", "http: create_chat_completion"):
                if f"frontend_{cand}_count" in rec:
                    http_name = cand
                    break
            w.writerow(
                {
                    "source": rec["source"],
                    "host": rec.get("host"),
                    "run_id": rec["run_id"],
                    "endpoint": rec.get("endpoint"),
                    "n_requests": rec.get("n_requests"),
                    "evidence_kind": rec.get("evidence_kind"),
                    "has_messages": rec.get("has_messages"),
                    "has_text_prompt": rec.get("has_text_prompt"),
                    "has_token_ids_prompt": rec.get("has_token_ids_prompt"),
                    "has_tools": rec.get("has_tools"),
                    "max_tokens_values": ";".join(map(str, rec.get("max_tokens_values") or [])),
                    "ignore_eos_true": rec.get("ignore_eos_true"),
                    "response_objects": ";".join(rec.get("response_objects") or []),
                    "prompt_tokens_from_usage": ";".join(
                        map(str, rec.get("prompt_tokens_from_usage") or [])
                    ),
                    "completion_tokens_from_usage": ";".join(
                        map(str, rec.get("completion_tokens_from_usage") or [])
                    ),
                    "n_tids": rec.get("n_tids"),
                    "frontend_tids": ";".join(map(str, rec.get("frontend_tids") or [])),
                    "encode_count": ts["tokenizer: encode"]["count"],
                    "encode_avg_us": ts["tokenizer: encode"]["avg_us"],
                    "encode_min_us": ts["tokenizer: encode"]["min_us"],
                    "encode_max_us": ts["tokenizer: encode"]["max_us"],
                    "decode_count": ts["tokenizer: decode"]["count"],
                    "decode_avg_us": ts["tokenizer: decode"]["avg_us"],
                    "decode_min_us": ts["tokenizer: decode"]["min_us"],
                    "decode_max_us": ts["tokenizer: decode"]["max_us"],
                    "render_count": ts["tokenizer: render_messages"]["count"],
                    "render_avg_us": ts["tokenizer: render_messages"]["avg_us"],
                    "render_min_us": ts["tokenizer: render_messages"]["min_us"],
                    "render_max_us": ts["tokenizer: render_messages"]["max_us"],
                    "http_scope": http_name,
                    "http_count": rec.get(f"frontend_{http_name}_count") if http_name else None,
                    "http_avg_us": rec.get(f"frontend_{http_name}_avg_us") if http_name else None,
                    "lite_log_bytes": rec.get("lite_log_bytes"),
                    "run_dir": rec.get("run_dir"),
                }
            )
    print(f"\nCSV  → {out_csv}（{len(records)} run）")
    print(f"JSON → {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
