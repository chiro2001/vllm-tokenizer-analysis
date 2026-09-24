// SPDX-License-Identifier: Apache-2.0
//! criterion 交叉验证装置。
//!
//! 主矩阵（`b-backends run`）用的是自建计时器（见 `src/stats.rs` 的口径说明），
//! 这个 bench 用 criterion 独立复测**少量关键组合**，两套装置的中位数若落在
//! 彼此 10% 以内，就说明主矩阵的计时没有系统性偏差。
//!
//! 上游 vLLM 的 `benches/{hf,tiktoken}.rs` 是这里的结构原型
//! （`vendor/vllm-tokenizer/upstream-benches/`），两处改造：
//! (a) `hf-hub` 联网 → 本地路径；(b) `SAMPLE_TEXT.repeat(32)` → 真实长度分布语料。

use std::path::PathBuf;

use b_backends::backends::{self, BackendId, Loaded, Paths};
use b_backends::corpus::{self, CorpusKind, CorpusSpec, LengthTarget};
use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

fn paths() -> Paths {
    Paths {
        hf_dir: PathBuf::from(
            std::env::var("B_HF_DIR").unwrap_or_else(|_| "/home/chiro/models/Qwen3-0.6B".into()),
        ),
        tiktoken_dir: PathBuf::from(
            std::env::var("B_TIKTOKEN_DIR").unwrap_or_else(|_| "/tmp/b-artifacts/kimi-k2.5".into()),
        ),
        tekken_path: PathBuf::from(
            std::env::var("B_TEKKEN").unwrap_or_else(|_| "/tmp/b-artifacts/tekken.json".into()),
        ),
    }
}

fn load_or_skip(id: BackendId) -> Option<Loaded> {
    match backends::load(id, &paths()) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("[criterion] 跳过 {}: {e:#}", id.as_str());
            None
        }
    }
}

/// criterion 组：1k 中英混合语料 × {hf, fastokens} 的 encode/decode。
/// 这正是上游 `benches/hf.rs` 的两条曲线，只是换成本地工件与真实语料。
fn bench_hf_family(c: &mut Criterion) {
    let reference = match corpus::load_reference(&paths().hf_tokenizer_json()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[criterion] 跳过 hf 家族: {e:#}");
            return;
        }
    };
    let corpus = corpus::generate(
        CorpusSpec::new(CorpusKind::Mixed, LengthTarget::Tok1k),
        &reference,
    )
    .expect("生成语料");
    let text = corpus.text.clone();
    let ids = corpus.token_ids.clone();

    let mut group = c.benchmark_group("criterion_encode_1k_mixed");
    group.throughput(Throughput::Bytes(text.len() as u64));
    for id in [BackendId::Hf, BackendId::Fastokens] {
        let Some(b) = load_or_skip(id) else { continue };
        // 与上游一样的正确性断言：同家族必须逐 id 一致。
        assert_eq!(
            b.encode(&text).expect("encode"),
            ids,
            "{} 的 token ids 与参考后端不一致",
            id.as_str()
        );
        group.bench_function(id.as_str(), |bench| {
            bench.iter(|| b.encode(black_box(text.as_str())).expect("encode"))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("criterion_decode_1k_mixed");
    group.throughput(Throughput::Elements(ids.len() as u64));
    for id in [BackendId::Hf, BackendId::Fastokens] {
        let Some(b) = load_or_skip(id) else { continue };
        group.bench_function(id.as_str(), |bench| {
            bench.iter(|| b.decode(black_box(ids.as_slice())).expect("decode"))
        });
    }
    group.finish();
}

/// tiktoken 家族（上游 `benches/tiktoken.rs` 的两条曲线）。
fn bench_tiktoken_family(c: &mut Criterion) {
    let reference = match corpus::load_reference(&paths().hf_tokenizer_json()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[criterion] 跳过 tiktoken 家族: {e:#}");
            return;
        }
    };
    let corpus = corpus::generate(
        CorpusSpec::new(CorpusKind::English, LengthTarget::Tok1k),
        &reference,
    )
    .expect("生成语料");
    let text = corpus.text.clone();

    let mut group = c.benchmark_group("criterion_encode_1k_en_tiktoken");
    group.throughput(Throughput::Bytes(text.len() as u64));
    let mut reference_ids: Option<Vec<u32>> = None;
    for id in [BackendId::TiktokenRs, BackendId::Riptoken] {
        let Some(b) = load_or_skip(id) else { continue };
        let got = b.encode(&text).expect("encode");
        match &reference_ids {
            None => reference_ids = Some(got),
            Some(want) => assert_eq!(&got, want, "{} 与 tiktoken-rs 的 ids 不一致", id.as_str()),
        }
        group.bench_function(id.as_str(), |bench| {
            bench.iter(|| b.encode(black_box(text.as_str())).expect("encode"))
        });
    }
    group.finish();
}

/// 流式 decode：`min_bytes_to_buffer` 三个取值（上游没有这块 bench）。
fn bench_incremental_decode(c: &mut Criterion) {
    let reference = match corpus::load_reference(&paths().hf_tokenizer_json()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[criterion] 跳过流式 decode: {e:#}");
            return;
        }
    };
    let corpus = corpus::generate(
        CorpusSpec::new(CorpusKind::Chinese, LengthTarget::Tok1k),
        &reference,
    )
    .expect("生成语料");
    let prompt = corpus.token_ids[..8].to_vec();
    let generated = corpus.token_ids[8..264].to_vec();

    let mut group = c.benchmark_group("criterion_stream_decode_256tok_zh");
    group.throughput(Throughput::Elements(generated.len() as u64));
    for id in [BackendId::Hf, BackendId::Fastokens] {
        for min_bytes in [1usize, 4, 8] {
            let Some(b) = load_or_skip(id) else { continue };
            group.bench_function(format!("{}/min_bytes={min_bytes}", id.as_str()), |bench| {
                bench.iter(|| {
                    b.decode_incremental(black_box(&prompt), black_box(&generated), min_bytes)
                        .expect("stream decode")
                })
            });
        }
    }
    group.finish();
}

/// 直接用上游 `tokenizers` crate 的 `DecodeStream`（Python 侧同一条路径的 Rust 形态），
/// 用于回答"vLLM 自研 DecodeStream 与上游实现差多少"。
fn bench_upstream_decode_stream(c: &mut Criterion) {
    use tokenizers::Tokenizer;
    let path = paths().hf_tokenizer_json();
    let Ok(tok) = Tokenizer::from_file(&path) else {
        eprintln!("[criterion] 跳过上游 DecodeStream: 加载 {} 失败", path.display());
        return;
    };
    let reference = match corpus::load_reference(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[criterion] 跳过上游 DecodeStream: {e:#}");
            return;
        }
    };
    let corpus = corpus::generate(
        CorpusSpec::new(CorpusKind::Chinese, LengthTarget::Tok1k),
        &reference,
    )
    .expect("生成语料");
    let prompt = corpus.token_ids[..8].to_vec();
    let generated = corpus.token_ids[8..264].to_vec();

    let mut group = c.benchmark_group("criterion_upstream_decodestream_256tok_zh");
    group.throughput(Throughput::Elements(generated.len() as u64));
    group.bench_function("tokenizers::DecodeStream", |bench| {
        bench.iter(|| {
            // tokenizers 0.22 的 DecodeStream 是**有状态**的：`step(id)` 返回该步
            // 新解出的文本（可能是 None）；语义与 vLLM 的 DecodeStream 类似但不含
            // 提示词前缀。这里只测生成段，保持与 vLLM 侧同尺度。
            let mut stream = tok.decode_stream(true);
            for &id in black_box(&prompt) {
                let _ = stream.step(id);
            }
            let mut out = String::new();
            for &id in black_box(&generated) {
                if let Ok(Some(chunk)) = stream.step(id) {
                    out.push_str(&chunk);
                }
            }
            out
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_hf_family,
    bench_tiktoken_family,
    bench_incremental_decode,
    bench_upstream_decode_stream
);
criterion_main!(benches);
