#!/usr/bin/env python3
"""E2 的图：从 data/cost/*.json 生成 figures/*.svg（+ PNG）。

只画**实测点**，不插值、不外推；每条曲线的定义写在轴标签与图例里，
避免读者误读口径。

用法见 --help。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文标签在多数 Linux 上缺字体，图内一律用英文 + 单位，
# 文档正文负责中文解释（避免生成一堆 tofu 方块）。
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
    }
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


# ------------------------------------------------------------------ 图 1
def fig_scope_cost(cost_dir: Path, out: Path) -> str:
    paths = load(cost_dir / "e2_paths.json")
    detok = load(cost_dir / "e2_detokenize.json")
    short_path = cost_dir / "e2_encode_short.json"
    short = load(short_path) if short_path.is_file() else None

    series: dict[str, tuple[list[float], list[float]]] = {}

    for rec in paths["results"]:
        if rec["case"] == "encode":
            xs, ys = series.setdefault("encode (tokenizer: encode)", ([], []))
            xs.append(rec["actual_tokens"])
            ys.append(rec["scope_stats"]["mean_us"])
        elif rec["case"] == "render_messages" and rec["variant"] == "tools_1turn":
            xs, ys = series.setdefault("render_messages (chat + tools)", ([], []))
            xs.append(rec["rendered_prompt_tokens"])
            ys.append(rec["scope_stats"]["mean_us"])
        elif rec["case"] == "render_messages" and rec["variant"] == "no_tools_1turn":
            xs, ys = series.setdefault("render_messages (chat, no tools)", ([], []))
            xs.append(rec["rendered_prompt_tokens"])
            ys.append(rec["scope_stats"]["mean_us"])
        elif rec["case"] == "decode_prompt_reverse":
            xs, ys = series.setdefault("decode: prompt_reverse (tokenizer: decode)", ([], []))
            xs.append(rec["n_tokens"])
            ys.append(rec["scope_stats"]["mean_us"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers = ["o", "s", "^", "D"]
    for (label, (xs, ys)), mk in zip(sorted(series.items()), markers):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order], marker=mk, label=label, lw=1.6)

    # 流式 detokenize 是「每 token 成本」，换算成整请求总时间后同轴可比。
    # 三个 OSL 点一次性画，才能只占一个图例项（否则后面两个点没有标签）。
    det_pts = [
        (rec["prompt_tokens"] + rec["osl_tokens"], rec["full_request_us"]["mean_us"])
        for rec in detok["results"]
        if rec["path"] == "fast" and rec["osl_tokens"] in (32, 256, 1024)
    ]
    if det_pts:
        det_pts.sort()
        ax.plot(
            [p[0] for p in det_pts],
            [p[1] for p in det_pts],
            marker="*",
            ms=12,
            ls="none",
            color="#b83280",
            label="detokenize: stream (fast, full request incl. 231us 1st step)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # 短 prompt 曲线单独画：它揭示 ~45 µs 的固定成本，是"线性只在 ISL≳64 成立"
    # 的证据；不画进去会让读者以为曲线一路线性。
    if short:
        pts = sorted(
            (r["actual_tokens"], r["stats_us"]["mean_us"]) for r in short["results"]
        )
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker="v",
            ls="--",
            lw=1.3,
            color="#718096",
            label="encode, short-prompt sweep (fixed cost 25-45us)",
        )
        ax.axhspan(25, 45, color="#718096", alpha=0.15)
        ax.annotate(
            "fixed cost 25-45us",
            xy=(4.5, 45),
            xytext=(5.2, 15),
            fontsize=7,
            color="#4a5568",
        )
    ax.set_xlabel("prompt / prompt+output tokens (measured, not targeted)")
    ax.set_ylabel("mean cost per request (us)")
    ax.set_title(
        "E2 tokenizer scope cost vs input size\n"
        "vLLM 0.26.0 / Qwen3-0.6B / x86 CPU container (2 cores, cpuset 4-5)"
    )
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------------ 图 2
def fig_concurrency(cost_dir: Path, out: Path) -> str:
    data = load(cost_dir / "e2_concurrency.json")
    rows = data["results"]
    workers = sorted({r["renderer_num_workers"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for w in workers:
        rs = sorted([r for r in rows if r["renderer_num_workers"] == w], key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in rs]
        axes[0].plot(xs, [r["throughput_req_s"] for r in rs], marker="o",
                     label=f"renderer_num_workers={w} (pool={w + 1})", lw=1.6)
        axes[1].plot(xs, [r["latency_p50_us"] / 1000 for r in rs], marker="s",
                     label=f"workers={w}", lw=1.6)

    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("client concurrency")
    axes[0].set_ylabel("throughput (req/s)")
    axes[0].set_title("frontend throughput vs concurrency")
    axes[0].legend(fontsize=8)

    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("client concurrency")
    axes[1].set_ylabel("p50 latency (ms)")
    axes[1].set_title("p50 latency vs concurrency")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "E2.5 vLLM frontend concurrency (asyncio + ThreadPoolExecutor) — "
        "chat+tools, ISL=1024, 4 cores, container 6 CPU",
        fontsize=9,
    )
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------------ 图 3
def fig_share(cost_dir: Path, out: Path) -> str:
    share = load(cost_dir / "e2_share.json")
    rows = [r for r in share["table"] if r["share_pct"] is not None]
    # 只画「同装置可比」与「跨装置上界」两类里最有代表性的
    keep_labels = []
    for r in rows:
        n = r["numerator"]
        if r["comparable"] == "yes" and "同装置同 run" in n:
            if "http:" in r["denominator"]:
                keep_labels.append(r)
        elif n.startswith("encode@8192") and "cross" in r["comparable"]:
            keep_labels.append(r)
        elif n.startswith("encode@128") and "cross" in r["comparable"]:
            keep_labels.append(r)
        # 分子串是 "detokenize_stream_fast 稳定期/step (PSL≈1k, OSL=1024)"，
        # 注意是 "OSL=1024" 不是 "osl1024"（踩过：写错大小写就一个都不匹配，
        # 图上会静默少一行）。
        elif n.startswith("detokenize_stream_fast") and "OSL=1024" in n and "cross" in r["comparable"]:
            keep_labels.append(r)

    if not keep_labels:
        raise SystemExit("e2_share.json 里没有可画的占比行")

    # 图中一律用 ASCII：中文字体在多数环境缺失，会渲染成方块
    # （实测 matplotlib 警告 "Glyph 22120 missing from font"）。
    def ascii_label(row: dict) -> str:
        n = row["numerator"]
        n = n.replace("本机容器", "local-container").replace("同装置同 run", "same-run")
        n = n.replace("稳定期/step", "steady/step").replace("(PSL≈1k, ", "(PSL~1k, ")
        d = row["denominator"]
        d = d.replace("真机", "real-NPU").replace("引擎侧 TTFT 下界", "engine-side TTFT lower bound")
        d = d.replace("引擎侧 TPOT 下界", "engine-side TPOT lower bound")
        d = d.replace("前端整请求", "whole frontend request")
        return f"{row['anchor_run'][:24]}...\n{n}\nvs {d}"

    labels = [ascii_label(r) for r in keep_labels]
    vals = [r["share_pct"] for r in keep_labels]
    colors = ["#2b6cb0" if r["comparable"] == "yes" else "#c05621" for r in keep_labels]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ypos = range(len(vals))
    ax.barh(list(ypos), vals, color=colors, height=0.6)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("share of denominator (%, log scale)")
    ax.set_title(
        "E2.4 tokenizer share of TTFT/TPOT\n"
        "blue = same device & same run (comparable) | orange = cross-device upper bound"
    )
    for y, v in zip(ypos, vals):
        ax.text(v * 1.06, y, f"{v:.3f}%", va="center", fontsize=7.5)
    ax.invert_yaxis()
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ------------------------------------------------------------------ 图 4
def fig_detok_segments(cost_dir: Path, out: Path) -> str:
    detok = load(cost_dir / "e2_detokenize.json")
    osls = sorted({r["osl_tokens"] for r in detok["results"]})
    paths = ["fast", "slow"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, p in zip(axes, paths):
        rs = sorted([r for r in detok["results"] if r["path"] == p], key=lambda r: r["osl_tokens"])
        xs = [r["osl_tokens"] for r in rs]
        init = [r["init_us"]["mean_us"] for r in rs]
        first = [r["first_step_us"]["mean_us"] for r in rs]
        # steady 段只在 OSL>steady_after 时存在
        steady = []
        for r in rs:
            s = r.get("steady_total_us_per_token")
            steady.append(s * (r["osl_tokens"] - 1) if s else 0.0)
        interior_missing = [
            (r["osl_tokens"] - 1) == 0 for r in rs
        ]
        ax.bar(range(len(xs)), init, label="init (per request)")
        ax.bar(range(len(xs)), first, bottom=init, label="first step (prompt prefill)")
        ax.bar(range(len(xs)), steady, bottom=[a + b for a, b in zip(init, first)],
               label="steady steps x (OSL-1)")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
            [f"{o}{'*' if m else ''}" for o, m in zip(xs, interior_missing)], fontsize=8
        )
        ax.set_xlabel("OSL (generated tokens); * = steady segment not measurable")
        ax.set_title(f"detokenize stream: {p}")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("per-request cost decomposition (us)")
    fig.suptitle(
        "E2.3b streaming detokenize cost structure — FastIncrementalDetokenizer vs "
        "SlowIncrementalDetokenizer (PSL~990, mixed corpus)",
        fontsize=9,
    )
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 data/cost/*.json 生成 figures/*.svg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--cost-dir", default="data/cost")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--also-png", action="store_true", default=True)
    args = ap.parse_args()

    cost_dir = Path(args.cost_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    made = []
    made.append(fig_scope_cost(cost_dir, fig_dir / "fig-02-scope-cost-vs-isl.svg"))
    made.append(fig_concurrency(cost_dir, fig_dir / "fig-02-frontend-concurrency.svg"))
    made.append(fig_detok_segments(cost_dir, fig_dir / "fig-02-detokenize-segments.svg"))
    made.append(fig_share(cost_dir, fig_dir / "fig-02-share-of-ttft-tpot.svg"))

    for m in made:
        print("写出", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
