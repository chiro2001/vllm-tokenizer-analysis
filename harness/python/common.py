"""C 线 harness 公共设施：renderer 构造、计时统计、manifest。

设计约束（对应 plan/COORDINATION.md §5）：

* **不启引擎、不碰 NPU**：只用 `VllmConfig` + `renderer_from_config()` 构造
  API server 前端进程里那一套 renderer/tokenizer 对象。
* **边界与 LiteProfiler 一致**：所有被测调用点与
  `liteprofiler-vllm.patch` 里 `vllm/renderers/base.py` 的 `with
  record_function_or_nullcontext(...)` 包裹范围逐行对应，见 `bench_paths.py`
  里每个 case 的 `scope` 字段。
* 计时用 `time.perf_counter_ns()`，与 `LiteScope` 用的是同一个时钟。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
@dataclass
class Stats:
    n: int
    mean_us: float
    p50_us: float
    p90_us: float
    p99_us: float
    min_us: float
    max_us: float
    stdev_us: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "mean_us": round(self.mean_us, 3),
            "p50_us": round(self.p50_us, 3),
            "p90_us": round(self.p90_us, 3),
            "p99_us": round(self.p99_us, 3),
            "min_us": round(self.min_us, 3),
            "max_us": round(self.max_us, 3),
            "stdev_us": round(self.stdev_us, 3),
        }


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def summarize(ns_samples: list[int]) -> Stats:
    us = sorted(x / 1000.0 for x in ns_samples)
    return Stats(
        n=len(us),
        mean_us=statistics.fmean(us),
        p50_us=_percentile(us, 0.50),
        p90_us=_percentile(us, 0.90),
        p99_us=_percentile(us, 0.99),
        min_us=us[0],
        max_us=us[-1],
        stdev_us=statistics.stdev(us) if len(us) > 1 else 0.0,
    )


class timer:
    """with 语句计时，把纳秒差值 append 到 samples。"""

    __slots__ = ("samples", "_t0")

    def __init__(self, samples: list[int]):
        self.samples = samples

    def __enter__(self):
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.samples.append(time.perf_counter_ns() - self._t0)
        return False


def measure(fn, n: int, warmup: int) -> tuple[list[int], Any]:
    """跑 warmup 次再采 n 次，返回 (ns 样本, 最后一次返回值)。"""
    out = None
    for _ in range(warmup):
        out = fn()
    samples: list[int] = []
    for _ in range(n):
        with timer(samples):
            out = fn()
    return samples, out


# --------------------------------------------------------------------------
# 环境 / manifest
# --------------------------------------------------------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path: str | os.PathLike) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except Exception:
        return None


def image_id(env_var: str = "COST_IMAGE_ID") -> str | None:
    return os.environ.get(env_var)


def model_revision(model_dir: str) -> dict[str, Any]:
    """模型 revision：优先读 HF 的 refs/main，其次记录关键文件 sha256。"""
    info: dict[str, Any] = {"path": model_dir}
    ref = Path(model_dir) / "refs" / "main"
    if ref.is_file():
        info["hf_revision"] = ref.read_text().strip()
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        p = Path(model_dir) / name
        if p.is_file():
            info[f"{name}_sha256"] = sha256_file(p)
            info[f"{name}_bytes"] = p.stat().st_size
    return info


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cores_available_to_process": len(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            k: os.environ.get(k)
            for k in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    for mod in ("transformers", "tokenizers", "torch", "vllm"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as exc:  # pragma: no cover - 环境相关
            info[f"{mod}_version"] = f"<import failed: {type(exc).__name__}>"
    return info


def read_launch_snapshot() -> dict[str, Any]:
    """读取宿主侧 launch 快照 `/workspace/system-snapshot.txt`（key=value 行）。"""
    out: dict[str, Any] = {}
    try:
        text = Path("/workspace/system-snapshot.txt").read_text()
    except OSError:
        return out
    out["_raw"] = text
    for line in text.splitlines():
        if "=" in line and not line.startswith(" "):
            k, _, v = line.partition("=")
            if k and " " not in k:
                out[k.strip()] = v.strip()
    return out


def system_load_snapshot() -> dict[str, Any]:
    """测量期间的系统负载（§9.3 要求写进 manifest）。

    `scripts/c_cost_run.sh` 在**启动容器之前**把宿主机的 loadavg / meminfo
    快照写到 `/workspace/system-snapshot.txt` 并只读挂载进来；容器内的
    /proc/loadavg 是宿主机的，所以两个来源都读。
    """
    snap: dict[str, Any] = {}
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
        snap["loadavg_1_5_15"] = [float(parts[0]), float(parts[1]), float(parts[2])]
        snap["running_procs"] = parts[3]
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key = line.split(":", 1)[0]
                if key in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
                    kb = int(line.split()[1])
                    snap[f"{key}_GiB"] = round(kb / 1048576, 2)
    except OSError:
        pass
    snap["host_snapshot_at_launch"] = read_launch_snapshot().get("_raw")
    return snap


def make_manifest(
    *,
    experiment: str,
    script: str,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    snap = read_launch_snapshot()
    model_path = model or os.environ.get("COST_MODEL", "/models/Qwen3-0.6B")
    manifest: dict[str, Any] = {
        "experiment": experiment,
        "timestamp": now_iso(),
        "script": script,
        "script_sha256": sha256_file(script) if Path(script).is_file() else None,
        # worktree 的 .git 在宿主机上，容器里看不到；由 c_cost_run.sh 采集后
        # 经 system-snapshot.txt 传进来（见 read_launch_snapshot）。
        "repo_commit": snap.get("worktree_head") or git_rev(root),
        "repo_branch": snap.get("worktree_branch"),
        "image": os.environ.get("COST_IMAGE")
        or snap.get("image")
        or "local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922",
        "image_id": image_id() or snap.get("image_id"),
        "model": model_path,
        # 模型 revision 在**每个**实验里都要有（验收判据要求），所以在 common
        # 里统一采集，而不是让每个 bench 脚本各写一遍（踩过：只有 bench_paths
        # 写了，其余 4 份 JSON 的 model_revision 是空的）。
        "model_revision": model_revision(model_path),
        "vllm_src": None,
        "vllm_src_commit": None,
        "liteprof_overlay": os.environ.get("COST_OVERLAY", ""),
        "vllm_lite_profiler_log_path": os.environ.get("VLLM_LITE_PROFILER_LOG_PATH"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "host": host_info(),
        "system_load": system_load_snapshot(),
    }
    try:
        import vllm

        manifest["vllm_src"] = str(Path(vllm.__file__).resolve().parent)
        manifest["vllm_version"] = getattr(vllm, "__version__", None)
        manifest["vllm_src_commit"] = git_rev(
            Path(vllm.__file__).resolve().parents[1]
        ) or snap.get("vllm_src_commit")
    except Exception as exc:
        manifest["vllm_src"] = f"<import failed: {type(exc).__name__}: {exc}>"
        manifest["vllm_src_commit"] = snap.get("vllm_src_commit")
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


# --------------------------------------------------------------------------
# renderer 构造
# --------------------------------------------------------------------------
def build_renderer(
    model: str,
    *,
    tokenizer_mode: str = "hf",
    renderer_num_workers: int | None = None,
    max_model_len: int | None = None,
    enable_prompt_embeds: bool = False,
):
    """构造 API server 前端进程使用的那套 renderer（不启引擎）。"""
    from vllm.config import ModelConfig, VllmConfig
    from vllm.renderers import renderer_from_config

    kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": model,
        "tokenizer_mode": tokenizer_mode,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "seed": 0,
    }
    if renderer_num_workers is not None:
        kwargs["renderer_num_workers"] = renderer_num_workers
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if enable_prompt_embeds:
        kwargs["enable_prompt_embeds"] = True

    model_config = ModelConfig(**kwargs)
    vllm_config = VllmConfig(model_config=model_config)
    renderer = renderer_from_config(vllm_config)
    return renderer, vllm_config
