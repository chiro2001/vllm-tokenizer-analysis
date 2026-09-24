// SPDX-License-Identifier: Apache-2.0
//! gigatoken 0.10.0 的 Rust 侧适配层。
//!
//! 用到的公开 API（crate 名 `gigatoken_rs`，来自 sdist 的 `[lib] name`）：
//!
//! | 用途 | 入口 | 备注 |
//! |---|---|---|
//! | 加载 | `load_tokenizer::hf::load_hf_bpe(path)` | ByteLevel BPE，无 byte_fallback |
//! | 单条 encode | `Tokenizer::encode_with_added_tokens_flat(bytes, out)` | 需要 `&mut self`（内部有 pretoken cache） |
//! | 并行批量 | `encode_docs_ragged(&WorkerPool, &Tokenizer, &[&[u8]])` | 自带 chunk 切分 + LPT + rayon |
//! | 串行批量 | `encode_docs_ragged_serial(...)` | 同一入口的单线程形态 |
//! | decode | `Tokenizer::decode(&[TokenId])` | **返回 bytes**，无特殊 token 过滤 |
//!
//! ## `TokenId` 不可命名的处理
//!
//! `gigatoken_rs::bpe` 与 `gigatoken_rs::token` 都是 `pub(crate)`，
//! 因此 `pub struct TokenId(pub u32)` 在 crate 外**无法命名**，
//! 而 `Tokenizer::decode` 的签名要求 `&[TokenId]`（见 `decode_bytes` 里的 SAFETY 注释）。
//! 这是**对外 API 不完整**的直接证据（写进 `docs/05-gigatoken-audit.md` 的缺口一节），
//! 不是我们的实现选择。正确性由与 Python 侧 `Tokenizer.decode()` 的逐字节对比兜底。

//! ## 一个 API 缺口（影响本 harness 的"串行"对照臂）
//!
//! `src/batch.rs:791` 有 `pub fn encode_docs_ragged_serial`，但 `batch` 模块是
//! `pub(crate)`（`src/lib.rs:3`），且 crate 根只 `pub use` 了
//! `WorkerPool / encode_docs_ragged / sp_encode_docs_ragged`。
//! 因此**外部 Rust 调用者拿不到官方的串行批量入口**（Python 侧通过
//! `parallel=False` 可以走到它对应用户路径）。
//! 本 harness 的串行臂因此自己实现：逐文档调 `encode_with_added_tokens_flat`
//! （与 `encode_docs_ragged_serial` 内部循环同一入口；由于本 harness 的文档
//! 单个体量远低于其 chunk 阈值，切片策略不引入差异）。

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use gigatoken_rs::{Tokenizer as GtTokenizer, WorkerPool, encode_docs_ragged};
use vllm_tokenizer::Tokenizer as VllmTokenizer;

pub struct GigatokenShim {
    /// 单条/解码路径用的原型 tokenizer。
    /// 单条 encode 需要 `&mut`（内部 pretoken cache）， поэтому用 `Mutex` 包一层。
    proto: Mutex<GtTokenizer>,
    /// 批量路径用的 worker 池：rayon 每个线程一个 fork，cache 常驻。
    workers: WorkerPool,
}

impl GigatokenShim {
    pub fn load(path: &Path) -> Result<Self> {
        // 注意：gigatoken 用 `eyre::Report` 作错误类型，而 `eyre::Report` 不实现
        // `std::error::Error`，所以 `anyhow::Context` 套不上——只能手工转字符串。
        let proto = gigatoken_rs::load_tokenizer::hf::load_hf_bpe(path)
            .map_err(|e| anyhow!("gigatoken 加载 {} 失败: {e}", path.display()))?;
        Ok(Self { proto: Mutex::new(proto), workers: WorkerPool::new() })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GtTokenizer> {
        self.proto.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 单文档 encode：与 `benchmarks/compare/measure.py` 的 gigatoken 侧同一入口。
    pub fn encode_single(&self, text: &str) -> Vec<u32> {
        let mut tok = self.lock();
        let mut out: Vec<u32> = Vec::new();
        tok.encode_with_added_tokens_flat(text.as_bytes(), &mut out);
        out
    }

    /// 串行批量（单线程，完全不碰 rayon 池）。
    ///
    /// 官方入口 `encode_docs_ragged_serial` 在 crate 外不可见（见模块头注释），
    /// 这里复刻它的核心循环：一个 fork 出来的 worker 顺序吃所有文档。
    pub fn encode_docs_serial(&self, docs: &[&[u8]]) -> (Vec<u32>, Vec<i64>) {
        let mut tok = self.lock();
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let mut ids: Vec<u32> = Vec::with_capacity(total / 4 + 16);
        let mut lens: Vec<i64> = Vec::with_capacity(docs.len());
        for doc in docs {
            let before = ids.len();
            tok.encode_with_added_tokens_flat(doc, &mut ids);
            lens.push((ids.len() - before) as i64);
        }
        (ids, lens)
    }

    /// 并行批量（rayon + 内部 chunk 切分 + LPT 负载均衡）。
    pub fn encode_docs_parallel(&self, docs: &[&[u8]]) -> Result<Vec<Vec<u32>>> {
        let guard = self.lock();
        let (ids, lens) = encode_docs_ragged(&self.workers, &guard, docs);
        Ok(ragged_to_rows(&ids, &lens))
    }

    /// 整段 decode：**返回 bytes**，不做特殊 token 过滤，也没有流式接口。
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let tok = self.lock();
        // SAFETY: TokenId 是 #[repr(transparent)] 的 u32（src/token.rs:6），
        // 且参数类型由 `decode` 的签名推断；u32 到 TokenId 的位模式一致。
        let toks = unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const _, ids.len()) };
        tok.decode(toks).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.lock().vocab_size()
    }

}

/// 把 gigatoken 适配成 vLLM 的 `Tokenizer` trait。
///
/// 这样做的两个理由：
///
/// 1. 六个 vLLM 后端全部实现这个 trait，适配后它们在同一条接口上比较；
/// 2. 更重要的：trait 的 `create_decode_stream` **有默认实现**
///    （`vendor/vllm-tokenizer/src/lib.rs`），它把 `DecodeStream`
///    （前缀差分算法，`incremental.rs`）架在任何 `Tokenizer` 之上。
///    也就是说：gigatoken 官方**没有**流式 decode API，但给它套上 vLLM 的
///    DecodeStream 就能跑——代价是每个 token 重解一次当前全序列（O(n²)）。
///    这正好把"缺 API"这件事**量化**成一组可以写在文档里的数。
///
/// 注意：vLLM 的 `DecodeStream` 依赖 `decode` 对不完整 UTF-8 返回替换字符
/// （而不是报错），所以这里的 `decode` 必须是 lossy 的。
impl VllmTokenizer for GigatokenShim {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        Ok(self.encode_single(text))
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> vllm_tokenizer::Result<String> {
        // lossy：与 vLLM Hafnium/other backends 一致，非法字节变 U+FFFD。
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // gigatoken 没有暴露反查表；用单 token 编码近似（与其它后端一致的做法）。
        let ids = self.encode_single(token);
        if ids.len() == 1 { Some(ids[0]) } else { None }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        let bytes = self.decode_bytes(&[id]);
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        GigatokenShim::vocab_size(self)
    }
}

pub fn ragged_to_rows(ids: &[u32], lens: &[i64]) -> Vec<Vec<u32>> {
    let mut out = Vec::with_capacity(lens.len());
    let mut start = 0usize;
    for &len in lens {
        let len = len.max(0) as usize;
        out.push(ids[start..start + len].to_vec());
        start += len;
    }
    out
}
