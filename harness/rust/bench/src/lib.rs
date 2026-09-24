// SPDX-License-Identifier: Apache-2.0
//! 线 B（多后端评估）的 Rust harness 主体。
//!
//! 结构：
//! - [`corpus`]：确定性语料生成（中文/英文/代码/混合 × 128/1k/8k tokens）
//! - [`backends`]：把 vLLM 0.26.0 的六个 Rust 后端 + 可选的 gigatoken 收敛成同一条接口
//! - [`stats`]：自实现的计时器（warmup + 多轮 median，单线程/多线程都可用）
//! - [`manifest`]：实验元信息（commit / 模型 revision / 脚本哈希 / 时间戳 / 绑核 / 线程数）

pub mod backends;
pub mod corpus;
pub mod manifest;
pub mod stats;

pub use backends::{FAMILIES, Loaded};
pub use corpus::{Corpus, CorpusKind, CorpusSpec, LengthTarget};
pub use manifest::Manifest;
pub use stats::{Measurement, Timer};
