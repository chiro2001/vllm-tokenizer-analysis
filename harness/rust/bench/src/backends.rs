// SPDX-License-Identifier: Apache-2.0
//! 把六个 vLLM Rust 后端（+ 可选 gigatoken）收敛到一条统一接口。
//!
//! 六个后端的来源见 `docs/03-backend-matrix.md` §1：
//!
//! | # | 后端 | 构造函数 | 家族 |
//! |---|---|---|---|
//! | 1 | Hf | `HuggingFaceTokenizer::new_hf` | HF |
//! | 2 | Fastokens | `HuggingFaceTokenizer::new_fastokens` | HF |
//! | 3 | FastokensByteLevel | `new_fastokens` 内部自动判定（无独立入口） | HF |
//! | 4 | TiktokenRs | `TiktokenTokenizer::new_tiktoken_rs` | tiktoken |
//! | 5 | Riptoken | `TiktokenTokenizer::new_riptoken` | tiktoken |
//! | 6 | Tekken | `TekkenTokenizer::new` | tekken |

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use vllm_tokenizer::{
    HuggingFaceTokenizer, TekkenTokenizer, TiktokenTokenizer,
    Tokenizer as VllmTokenizer,
};

/// 后端家族。同家族内必须逐 id 一致。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    Hf,
    Tiktoken,
    Tekken,
    Gigatoken,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Hf => "hf",
            Family::Tiktoken => "tiktoken",
            Family::Tekken => "tekken",
            Family::Gigatoken => "gigatoken",
        }
    }
}

impl std::fmt::Display for Family {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 参与评估的后端全集（顺序 = 报告主表顺序）。
pub const ALL_BACKENDS: [BackendId; 7] = [
    BackendId::Hf,
    BackendId::Fastokens,
    BackendId::FastokensByteLevel,
    BackendId::TiktokenRs,
    BackendId::Riptoken,
    BackendId::Tekken,
    BackendId::Gigatoken,
];

/// 六个 vLLM 后端（不含 gigatoken）。
pub const VLLM_BACKENDS: [BackendId; 6] = [
    BackendId::Hf,
    BackendId::Fastokens,
    BackendId::FastokensByteLevel,
    BackendId::TiktokenRs,
    BackendId::Riptoken,
    BackendId::Tekken,
];

/// 家族列表（用于正确性断言分组）。
pub const FAMILIES: [Family; 4] =
    [Family::Hf, Family::Tiktoken, Family::Tekken, Family::Gigatoken];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendId {
    Hf,
    Fastokens,
    FastokensByteLevel,
    TiktokenRs,
    Riptoken,
    Tekken,
    Gigatoken,
}

impl BackendId {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendId::Hf => "hf",
            BackendId::Fastokens => "fastokens",
            BackendId::FastokensByteLevel => "fastokens_byte_level",
            BackendId::TiktokenRs => "tiktoken_rs",
            BackendId::Riptoken => "riptoken",
            BackendId::Tekken => "tekken",
            BackendId::Gigatoken => "gigatoken",
        }
    }

    pub fn family(self) -> Family {
        match self {
            BackendId::Hf | BackendId::Fastokens | BackendId::FastokensByteLevel => Family::Hf,
            BackendId::TiktokenRs | BackendId::Riptoken => Family::Tiktoken,
            BackendId::Tekken => Family::Tekken,
            BackendId::Gigatoken => Family::Gigatoken,
        }
    }

    /// 构造这个后端需要哪个工件（写进 manifest / 报告）。
    pub fn artifact(self) -> &'static str {
        match self {
            BackendId::Hf | BackendId::Fastokens | BackendId::FastokensByteLevel => {
                "tokenizer.json"
            }
            BackendId::TiktokenRs | BackendId::Riptoken => "tiktoken.model",
            BackendId::Tekken => "tekken.json",
            BackendId::Gigatoken => "tokenizer.json",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        ALL_BACKENDS.into_iter().find(|b| b.as_str() == s)
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// 本地工件路径集合。所有路径都由命令行给出，**不联网**。
#[derive(Clone, Debug)]
pub struct Paths {
    /// 含 `tokenizer.json` / `tokenizer_config.json` 的目录（Qwen3-0.6B）。
    pub hf_dir: PathBuf,
    /// 含 `tiktoken.model` / `config.json` / `tokenizer_config.json` 的目录。
    pub tiktoken_dir: PathBuf,
    /// Mistral `tekken.json`。
    pub tekken_path: PathBuf,
}

impl Paths {
    pub fn hf_tokenizer_json(&self) -> PathBuf {
        self.hf_dir.join("tokenizer.json")
    }

    pub fn tiktoken_model(&self) -> PathBuf {
        self.tiktoken_dir.join("tiktoken.model")
    }
}

/// `FastokensByteLevel` 的静态判定：复刻 `vllm-tokenizer` 的
/// `is_byte_level_only()`（见 `vendor/vllm-tokenizer/src/hf.rs`）。
///
/// 上游在 `new_fastokens()` 里根据 fastokens 解码出来的 `Decoder` 树自动选择
/// `Backend::FastokensByteLevel`，没有独立构造函数，因此外部无法直接指定；
/// 这里按同一规则从 `tokenizer.json` 的 `decoder` 字段复算，用于**确认**
/// 第三个后端确实被走到了（若判定为 false，主表的 2/3 两行数字必然相同）。
pub fn detect_byte_level_branch(tokenizer_json: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(tokenizer_json)
        .with_context(|| format!("读取 {} 失败", tokenizer_json.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("解析 {} 失败", tokenizer_json.display()))?;
    Ok(value
        .get("decoder")
        .map(count_byte_level_leaves)
        .unwrap_or(0)
        == 1)
}

/// 统计解码头里 `ByteLevel` 叶子的个数；`Sequence` 递归求和，
/// 其余类型（含 fastokens 用来表示 `Fuse` 的空 `Sequence`）记 0。
fn count_byte_level_leaves(node: &serde_json::Value) -> usize {
    match node.get("type").and_then(|t| t.as_str()) {
        Some("ByteLevel") => 1,
        Some("Sequence") => node
            .get("decoders")
            .and_then(|d| d.as_array())
            .map(|steps| steps.iter().map(count_byte_level_leaves).sum())
            .unwrap_or(0),
        _ => 0,
    }
}

/// 已加载的后端实例。
pub struct Loaded {
    pub id: BackendId,
    pub detail: LoadDetail,
    /// 构造（含文件解析）耗时，µs。
    pub load_us: f64,
    inner: Inner,
}

/// 后端加载过程中值得记录的细节（写进 manifest）。
#[derive(Clone, Debug, Serialize)]
pub struct LoadDetail {
    pub artifact: String,
    pub vocab_size: usize,
    /// `FastokensByteLevel` 是否被判定生效（其余后端为 None）。
    pub byte_level_branch: Option<bool>,
    /// 额外说明（例如 tekken 的序号空间、gigatoken 的加载器）。
    pub note: String,
}

enum Inner {
    Hf(HuggingFaceTokenizer),
    Tiktoken(TiktokenTokenizer),
    Tekken(TekkenTokenizer),
    #[cfg(feature = "gigatoken")]
    Gigatoken(GigatokenShim),
}

/// 把 `vllm_tokenizer::TokenizerError` 收敛到 `anyhow`（harness 只关心信息文本）。
fn to_anyhow(e: vllm_tokenizer::TokenizerError) -> anyhow::Error {
    anyhow!("{e}")
}

macro_rules! dispatch {
    ($self:expr, $t:ident => $body:expr) => {
        match &$self.inner {
            Inner::Hf($t) => $body,
            Inner::Tiktoken($t) => $body,
            Inner::Tekken($t) => $body,
            #[cfg(feature = "gigatoken")]
            Inner::Gigatoken($t) => $body,
        }
    };
}

impl Loaded {
    pub fn id(&self) -> BackendId {
        self.id
    }

    /// panic 安全层。
    ///
    /// 六个后端里只有 fastokens 是"自己的预处理实现"，而它在**长中文/长英文输入**
    /// 上会越界 panic（`fastokens-0.2.1/src/pre_tokenizers/split.rs:419`，
    /// `n_chunks` 被 `.max(2)` 抬到 2、但正则表只有 1 条 → `pcre2[1]` 越界）。
    /// 这是上游 bug，不是本次实验的配置问题；harness 的职责是**如实记录**它，
    /// 而不是让整轮矩阵被一个输入带走。所有对外入口因此都套 `guard_panic`。
    fn guard<T>(
        id: BackendId,
        op: &'static str,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        crate::stats::guard_panic(op, std::panic::AssertUnwindSafe(f))
            .map_err(|e| anyhow!("{e}"))?
            .map_err(|e| anyhow!("{} {op} 失败: {e:#}", id.as_str()))
    }

    /// 单条 encode（`add_special_tokens = false`，与在线路径一致：
    /// vLLM 的 chat 模板自己插入控制符，tokenizer 不再加）。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Self::guard(self.id, "encode", || {
            dispatch!(self, t => t.encode(text, false).map_err(to_anyhow))
        })
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Self::guard(self.id, "decode", || {
            dispatch!(self, t => t.decode(ids, false).map_err(to_anyhow))
        })
    }

    /// 逐 token 的增量解码：`min_bytes_to_buffer` 扫一遍。
    pub fn decode_incremental(
        &self,
        prompt_ids: &[u32],
        generated: &[u32],
        min_bytes_to_buffer: usize,
    ) -> Result<(String, Vec<usize>)> {
        Self::guard(self.id, "stream_decode", || {
            let mut stream = dispatch!(
                self,
                t => t.create_decode_stream(prompt_ids, false, min_bytes_to_buffer)
            );
            let mut emitted: Vec<usize> = Vec::with_capacity(generated.len());
            for &id in generated {
                let n = stream.push_token(id).map_err(to_anyhow)?;
                emitted.push(n);
                // 真实服务路径会在需要时取走文本；这里同样把 chunk 取出来，
                // 否则测的就是一条被优化掉的路径。
                if n > 0 {
                    std::hint::black_box(stream.next_chunk());
                }
            }
            std::hint::black_box(stream.flush(None).map_err(to_anyhow)?);
            Ok((stream.output().to_string(), emitted))
        })
    }

    pub fn vocab_size(&self) -> usize {
        dispatch!(self, t => t.vocab_size())
    }

    /// 流式（增量）解码的来源：
    /// - `Native`：后端自带的实现；
    /// - `Adapter`：后端**没有**流式 API，靠 vLLM 的 `DecodeStream` 适配层补上
    ///   （每个 token 重解一次当前全序列，O(n²)）；
    /// - `None`：不可用。
    ///
    /// 六个 vLLM 后端用的都是 `lib.rs::create_decode_stream` 的默认实现
    /// （同一个 `DecodeStream`），所以在本 harness 里它们都是 `Adapter` 形态，
    /// 但"后端自己的 decode 有多快"仍会直接决定流式解码的总耗时。
    pub fn streaming_kind(&self) -> StreamingKind {
        match self.id {
            #[cfg(feature = "gigatoken")]
            BackendId::Gigatoken => StreamingKind::Adapter,
            _ => StreamingKind::Native,
        }
    }

    pub fn has_streaming_decode(&self) -> bool {
        self.streaming_kind() != StreamingKind::None
    }

    /// 批量 encode（并行路径，线程数由 [`install_thread_pool`] 决定）。
    pub fn encode_batch(&self, docs: &[&str]) -> Result<Vec<Vec<u32>>> {
        Self::guard(self.id, "encode_batch", || {
            #[cfg(feature = "gigatoken")]
            {
                if let Inner::Gigatoken(shim) = &self.inner {
                    // gigatoken 的批量路径自带 rayon 切块 + WorkerPool（`src/batch.rs`），
                    // 与逐文档 par_iter 不是同一套；这一点正是审计对象之一。
                    let raw: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
                    return shim.encode_docs_parallel(&raw);
                }
            }
            use rayon::prelude::*;
            docs.par_iter()
                .map(|doc| dispatch!(self, t => t.encode(doc, false).map_err(to_anyhow)))
                .collect::<Result<Vec<_>>>()
        })
    }

    /// 批量 encode 的**串行**形态（在线/单线程对照臂）。
    pub fn encode_batch_serial(&self, docs: &[&str]) -> Result<Vec<Vec<u32>>> {
        Self::guard(self.id, "encode_batch_serial", || {
            #[cfg(feature = "gigatoken")]
            {
                if let Inner::Gigatoken(shim) = &self.inner {
                    let raw: Vec<&[u8]> = docs.iter().map(|d| d.as_bytes()).collect();
                    let (ids, lens) = shim.encode_docs_serial(&raw);
                    return Ok(gigatoken_shim::ragged_to_rows(&ids, &lens));
                }
            }
            docs.iter()
                .map(|doc| dispatch!(self, t => t.encode(doc, false).map_err(to_anyhow)))
                .collect()
        })
    }
}

/// 加载某个后端。缺失工件时返回 `Err`，由调用方记成「未测 + 原因」。
pub fn load(id: BackendId, paths: &Paths) -> Result<Loaded> {
    let t0 = Instant::now();
    let (inner, detail) = match id {
        BackendId::Hf => {
            let p = paths.hf_tokenizer_json();
            let tok = HuggingFaceTokenizer::new_hf(&p)
                .map_err(|e| anyhow!("新 HF 后端加载失败: {e}"))?;
            check_artifact(&p)?;
            (
                Inner::Hf(tok),
                LoadDetail {
                    artifact: p.display().to_string(),
                    vocab_size: 0,
                    byte_level_branch: None,
                    note: "HuggingFace tokenizers 0.22（与 Python 侧同一实现）".into(),
                },
            )
        }
        BackendId::Fastokens | BackendId::FastokensByteLevel => {
            let p = paths.hf_tokenizer_json();
            check_artifact(&p)?;
            let byte_level = detect_byte_level_branch(&p)?;
            let tok = HuggingFaceTokenizer::new_fastokens(&p)
                .map_err(|e| anyhow!("fastokens 后端加载失败: {e}"))?;
            if id == BackendId::Fastokens && byte_level {
                bail!(
                    "fastokens 后端在 {} 上被判为 ByteLevel-only，此时 `Fastokens` 与 \
                     `FastokensByteLevel` 是同一个对象（上游行为），请只保留一行",
                    p.display()
                );
            }
            if id == BackendId::FastokensByteLevel && !byte_level {
                bail!(
                    "{} 的解码头不是 ByteLevel-only，FastokensByteLevel 旁路不会被走到",
                    p.display()
                );
            }
            (
                Inner::Hf(tok),
                LoadDetail {
                    artifact: p.display().to_string(),
                    vocab_size: 0,
                    byte_level_branch: Some(byte_level),
                    note: if byte_level {
                        "fastokens 解码为 ByteLevel-only → 自动走 FastokensByteLevel 旁路"
                            .into()
                    } else {
                        "fastokens 解码非 ByteLevel-only → 走通用 Fastokens 路径".into()
                    },
                },
            )
        }
        BackendId::TiktokenRs | BackendId::Riptoken => {
            let p = paths.tiktoken_model();
            check_artifact(&p)?;
            let tok = match id {
                BackendId::TiktokenRs => TiktokenTokenizer::new_tiktoken_rs(&p)
                    .map_err(|e| anyhow!("tiktoken-rs 后端加载失败: {e}"))?,
                _ => TiktokenTokenizer::new_riptoken(&p)
                    .map_err(|e| anyhow!("riptoken 后端加载失败: {e}"))?,
            };
            (
                Inner::Tiktoken(tok),
                LoadDetail {
                    artifact: p.display().to_string(),
                    vocab_size: 0,
                    byte_level_branch: None,
                    note: match id {
                        BackendId::TiktokenRs => "tiktoken-rs 0.9.1 后端".into(),
                        _ => "riptoken 0.3.0 后端".into(),
                    },
                },
            )
        }
        BackendId::Tekken => {
            let p = paths.tekken_path.clone();
            check_artifact(&p)?;
            let tok = TekkenTokenizer::new(&p)
                .map_err(|e| anyhow!("tekken 后端加载失败: {e}"))?;
            (
                Inner::Tekken(tok),
                LoadDetail {
                    artifact: p.display().to_string(),
                    vocab_size: 0,
                    byte_level_branch: None,
                    note: "tekken-rs 0.1.1；无第二个独立实现可做同家族 id 对齐，\
                           正确性只做 roundtrip + 不变量"
                        .into(),
                },
            )
        }
        #[cfg(feature = "gigatoken")]
        BackendId::Gigatoken => {
            let p = paths.hf_tokenizer_json();
            check_artifact(&p)?;
            let shim = GigatokenShim::load(&p)?;
            (
                Inner::Gigatoken(shim),
                LoadDetail {
                    artifact: p.display().to_string(),
                    vocab_size: 0,
                    byte_level_branch: None,
                    note: "gigatoken 0.10.0 sdist（Rust 源码 path 依赖，nightly 工具链）".into(),
                },
            )
        }
        #[cfg(not(feature = "gigatoken"))]
        BackendId::Gigatoken => bail!(
            "本二进制未开启 gigatoken feature；请用 \
             `cargo +nightly build --release --features gigatoken` 构建"
        ),
    };

    let load_us = t0.elapsed().as_secs_f64() * 1e6;
    let mut loaded = Loaded { id, detail, load_us, inner };
    let vs = loaded.vocab_size();
    loaded.detail.vocab_size = vs;
    Ok(loaded)
}

/// **旁路对照臂**：直接用 fastokens 自己的通用 `decode`（不经 vLLM 的
/// `FastokensByteLevel` 快路径）。
///
/// 存在的理由：3 号后端（`FastokensByteLevel`）没有独立构造入口，
/// 在 ByteLevel-only 的模型上它与 2 号（`Fastokens`）是**同一个对象**，
/// 因此"旁路值多少"无法从 2/3 的对比里读出来。这里绕过 vLLM wrapper，
/// 直接调 `fastokens::Tokenizer::decode`——也就是旁路生效时**没有走**的那条路，
/// 两者的差就是旁路的收益。
pub struct FastokensGeneric {
    inner: fastokens::Tokenizer,
}

impl FastokensGeneric {
    pub fn load(tokenizer_json: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(tokenizer_json)
            .with_context(|| format!("读取 {} 失败", tokenizer_json.display()))?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("解析 {} 失败", tokenizer_json.display()))?;
        let inner = fastokens::Tokenizer::from_json(value)
            .map_err(|e| anyhow!("fastokens 直接从 JSON 构造失败: {e}"))?;
        Ok(Self { inner })
    }

    /// fastokens 的通用解码：内部组装 `Vec<String>` 再 `join("")`
    /// （vLLM 的旁路正是为了省掉这一步）。
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!("fastokens 通用 decode 失败: {e}"))
    }
}

fn check_artifact(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("工件不存在: {}", path.display());
    }
    Ok(())
}

#[cfg(feature = "gigatoken")]
mod gigatoken_shim;
#[cfg(feature = "gigatoken")]
use gigatoken_shim::GigatokenShim;

/// 未开启 gigatoken feature 时的占位（`Engine` 里不会引用它）。
#[cfg(not(feature = "gigatoken"))]
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct GigatokenShim;

pub const GIGATOKEN_STREAMING_NOTE: &str =
    "gigatoken 无原生流式 decode API；此处的数是 vLLM `DecodeStream` 适配层（前缀差分，O(n²)）\
     架在其整段 decode 上的结果";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamingKind {
    Native,
    Adapter,
    /// 无流式解码能力（本 harness 目前没有这种后端，保留给未来）。
    #[allow(dead_code)]
    None,
}

impl StreamingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            StreamingKind::Native => "native",
            StreamingKind::Adapter => "adapter(vLLM DecodeStream)",
            StreamingKind::None => "none",
        }
    }
}

/// 线程数设置：
///
/// - `threads == 0`：**不建池**，让 rayon 按 `RAYON_NUM_THREADS`（由
///   `scripts/limit.sh` 设成绑核数）自行初始化；
/// - `threads >= 1`：显式建全局池。
///
/// 在线路径（单条请求）用 [`run_single_threaded`] 在**调用线程**上直接跑，
/// 完全不进 rayon，避免把"请求内并行"混进"请求间并行"。
pub fn install_thread_pool(threads: usize) -> Result<()> {
    if threads == 0 {
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| anyhow!("构建全局 rayon 线程池失败: {e}"))
}

/// 实际生效的 rayon 线程数（写进 manifest）。
pub fn effective_threads() -> usize {
    rayon::current_num_threads()
}

/// 在调用线程上直接执行（在线服务路径的形态）。
pub fn run_single_threaded<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// 简单互斥包装，供需要 `&mut self` 的实现使用。
pub struct Mutexed<T>(pub Mutex<T>);

impl<T> Mutexed<T> {
    pub fn new(v: T) -> Self {
        Self(Mutex::new(v))
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }
}
