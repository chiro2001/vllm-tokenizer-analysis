// SPDX-License-Identifier: Apache-2.0
//! 实验 manifest：每条实验产物旁边都要落一份。
//!
//! 至少包含（见 `plan/COORDINATION.md` §5.5）：
//! commit、镜像 id、模型 revision、脚本 sha256、时间戳、绑核、线程数。
//! 本 harness 不跑容器，因此「镜像 id」字段写成 `null`，另行记录本地工具链
//! 与 CPU 型号；Python 侧的 manifest 会带上镜像 id。

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use serde::Serialize;

use crate::corpus::sha256_hex;

#[derive(Clone, Debug, Serialize)]
pub struct Artifact {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

impl Artifact {
    pub fn from_path(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        Ok(Self {
            path: path.display().to_string(),
            bytes: data.len() as u64,
            sha256: sha256_hex(&data),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct HostInfo {
    pub hostname: String,
    pub cpu_model: String,
    pub nproc: usize,
    /// `/proc/self/status` 的 `Cpus_allowed_list`，即实际绑核。
    pub cpu_affinity: String,
    /// 采集瞬间的 loadavg 三元组（用户要求：写进 manifest）。
    pub loadavg: [f64; 3],
    /// `/proc/meminfo` 的 MemAvailable，GiB。
    pub mem_available_gb: f64,
    pub rustc: String,
    pub cargo: String,
    /// `RAYON_NUM_THREADS`（由 `scripts/limit.sh` 设置）。
    pub env_rayon_threads: Option<String>,
    pub env_cores: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub line: String,
    pub task: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    /// 本 worktree 的 git commit（线 B 的提交）。
    pub script_commit: String,
    /// 被测对象所在的 vLLM 源码 commit（0.26.0）。
    pub upstream_commit: String,
    /// 容器镜像：Rust 侧不跑容器，固定为 null。
    pub container_image: Option<String>,
    pub host: HostInfo,
    /// 线程数：请求值 vs 实际生效值（受 `RAYON_NUM_THREADS` 夹逼）。
    pub threads_requested: usize,
    pub threads_effective: usize,
    pub profile: String,
    pub features: Vec<String>,
    /// 语料指纹：`语料id -> sha256`。
    pub corpus_sha256: Vec<(String, String)>,
    pub artifacts: Vec<Artifact>,
    /// 脚本本身与 harness 关键文件的 sha256。
    pub script_sha256: Vec<Artifact>,
    /// 已知的口径限制（不要只给数字，也要给前提）。
    pub caveats: Vec<String>,
}

impl Manifest {
    pub fn new(
        task: &str,
        upstream_commit: &str,
        threads_requested: usize,
        threads_effective: usize,
        profile: &str,
        features: Vec<String>,
    ) -> Self {
        Self {
            line: "B-backends".into(),
            task: task.into(),
            started_at: now_rfc3339(),
            finished_at: None,
            script_commit: git_commit(),
            upstream_commit: upstream_commit.into(),
            container_image: None,
            host: host_info(),
            threads_requested,
            threads_effective,
            profile: profile.into(),
            features,
            corpus_sha256: Vec::new(),
            artifacts: Vec::new(),
            script_sha256: Vec::new(),
            caveats: Vec::new(),
        }
    }

    pub fn add_artifact(&mut self, path: &Path) -> Result<()> {
        self.artifacts.push(Artifact::from_path(path)?);
        Ok(())
    }

    pub fn add_script(&mut self, path: &Path) -> Result<()> {
        self.script_sha256.push(Artifact::from_path(path)?);
        Ok(())
    }

    pub fn add_caveat(&mut self, c: impl Into<String>) {
        self.caveats.push(c.into());
    }

    pub fn finish(&mut self) {
        self.finished_at = Some(now_rfc3339());
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

pub fn now_rfc3339() -> String {
    // 不引 chrono：直接用 `date` 的 ISO 输出，跨进程口径与 shell 完全一致。
    Command::new("date")
        .arg("--iso-8601=seconds")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

pub fn git_commit() -> String {
    let cwd = harness_root();
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&cwd)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn harness_root() -> PathBuf {
    // bench/src/manifest.rs -> harness/rust -> harness -> worktree 根
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .nth(3)
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest_dir)
}

fn host_info() -> HostInfo {
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let (loadavg, mem_available_gb) = read_pressure();
    HostInfo {
        hostname: std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".into()),
        cpu_model,
        nproc: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        cpu_affinity: cpu_affinity(),
        loadavg,
        mem_available_gb,
        rustc: tool_version("rustc"),
        cargo: tool_version("cargo"),
        env_rayon_threads: std::env::var("RAYON_NUM_THREADS").ok(),
        env_cores: std::env::var("CORES").ok(),
    }
}

/// 每次测量前后各采一次，用于判断"这次数是不是在机器被压着的时候测的"。
pub fn read_pressure() -> ([f64; 3], f64) {
    let loadavg = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| {
            let mut it = s.split_whitespace();
            let mut out = [0.0f64; 3];
            for slot in out.iter_mut() {
                *slot = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            }
            out
        })
        .unwrap_or([0.0; 3]);
    let mem_available_gb = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|kb| kb.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1024.0 / 1024.0)
        .unwrap_or(f64::NAN);
    (loadavg, mem_available_gb)
}

fn cpu_affinity() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Cpus_allowed_list:"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn tool_version(tool: &str) -> String {
    Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
