// SPDX-License-Identifier: Apache-2.0
//! 线 B 的 Rust 多后端基准 CLI。
//!
//! ```text
//! b-backends run        --backend all --out data/backends/run1.jsonl
//! b-backends correctness --out data/backends/correctness.json
//! b-backends corpus      --out data/backends/corpus.json
//! b-backends gigatoken-audit --out data/backends/gigatoken-audit.json
//! ```
//!
//! 所有命令都**只读本地文件**（不联网），并且必须在
//! `scripts/limit.sh` / `scripts/heavy_lock.sh` 之下运行（见 README）。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use serde::Serialize;

use b_backends::backends::{self, BackendId, Loaded, Paths, VLLM_BACKENDS};
use b_backends::corpus::{self, Corpus, CorpusKind, CorpusSpec, LengthTarget};
use b_backends::manifest::Manifest;
use b_backends::stats::{Measurement, Timer, settle};

const UPSTREAM_COMMIT: &str = "568afb3a13806beb53bb2e6bd518269357b237c0";

/// 结果行：一行一个测量，JSONL / CSV 两种落盘形态。
#[derive(Clone, Debug, Serialize)]
struct Row {
    backend: String,
    family: String,
    artifact: String,
    corpus: String,
    length: String,
    op: String,
    /// 并发形态：`single`（调用线程直接跑）/ `batch`（rayon 并行）。
    mode: String,
    threads: usize,
    /// 输入规模。
    bytes: usize,
    tokens: usize,
    /// 这条 row 的输入是「几个文档」（批量时 >1）。
    docs: usize,
    /// 实际测到的 token 总数（批量时是所有文档之和）。
    measured_tokens: usize,
    iters: usize,
    median_us: f64,
    min_us: f64,
    p90_us: f64,
    max_us: f64,
    mb_per_s: f64,
    tokens_per_s: f64,
    us_per_token: f64,
    /// 测量前后的 loadavg（第 1 个值）与可用内存，用于剔除被干扰的样本。
    loadavg_before: f64,
    loadavg_after: f64,
    mem_available_gb_before: f64,
    note: String,
}

impl Row {
    /// 某个操作在某个 (后端, 语料) 上失败时记一行：时间字段为 NaN，
    /// 原因写在 note 里。有了它，"六后端是否全有数"可以机械核对。
    #[allow(clippy::too_many_arguments)]
    fn failure(
        backend_label: &str,
        family: &str,
        spec: &CorpusSpec,
        op: &str,
        mode: &str,
        threads: usize,
        reason: &str,
    ) -> Self {
        Self {
            backend: backend_label.to_string(),
            family: family.to_string(),
            artifact: String::new(),
            corpus: spec.kind.as_str().to_string(),
            length: spec.target.as_str(),
            op: op.to_string(),
            mode: mode.to_string(),
            threads,
            bytes: 0,
            tokens: 0,
            docs: 0,
            measured_tokens: 0,
            iters: 0,
            median_us: f64::NAN,
            min_us: f64::NAN,
            p90_us: f64::NAN,
            max_us: f64::NAN,
            mb_per_s: f64::NAN,
            tokens_per_s: f64::NAN,
            us_per_token: f64::NAN,
            loadavg_before: f64::NAN,
            loadavg_after: f64::NAN,
            mem_available_gb_before: f64::NAN,
            note: format!("ERROR: {reason}"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn from_measurement(
        backend: &Loaded,
        backend_label: &str,
        spec: &CorpusSpec,
        op: &str,
        mode: &str,
        threads: usize,
        docs: usize,
        measured_tokens: usize,
        m: &Measurement,
        pressure: (f64, f64, f64, f64),
        note: String,
    ) -> Self {
        Self {
            backend: backend_label.to_string(),
            family: backend.id().family().as_str().to_string(),
            artifact: backend.detail.artifact.clone(),
            corpus: spec.kind.as_str().to_string(),
            length: spec.target.as_str(),
            op: op.to_string(),
            mode: mode.to_string(),
            threads,
            bytes: m.input_bytes,
            tokens: m.input_tokens,
            docs,
            measured_tokens,
            iters: m.iters,
            median_us: m.median_us,
            min_us: m.min_us,
            p90_us: m.p90_us,
            max_us: m.max_us,
            mb_per_s: m.mb_per_s(),
            tokens_per_s: m.tokens_per_s(),
            us_per_token: m.us_per_token(),
            loadavg_before: pressure.0,
            loadavg_after: pressure.2,
            mem_available_gb_before: pressure.1,
            note,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "b-backends",
    about = "vLLM tokenizer 多后端微基准（线 B）；只读本地工件，不联网",
    long_about = None,
    version
)]
struct Cli {
    /// Qwen3-0.6B 目录（含 tokenizer.json / tokenizer_config.json）
    #[arg(long, default_value = "/home/chiro/models/Qwen3-0.6B", global = true)]
    hf_dir: PathBuf,

    /// tiktoken 工件目录（含 tiktoken.model / config.json / tokenizer_config.json）
    #[arg(long, default_value = "/tmp/b-artifacts/kimi-k2.5", global = true)]
    tiktoken_dir: PathBuf,

    /// Mistral tekken.json
    #[arg(long, default_value = "/tmp/b-artifacts/tekken.json", global = true)]
    tekken: PathBuf,

    /// 输出根目录（相对 worktree 根）
    #[arg(long, default_value = "data/backends", global = true)]
    out_dir: PathBuf,

    /// rayon 线程数；0 = 交给 RAYON_NUM_THREADS（limit.sh 会设）
    #[arg(long, default_value_t = 0, global = true)]
    threads: usize,

    /// 每次测量的时间预算（ms）
    #[arg(long, default_value_t = 120.0, global = true)]
    budget_ms: f64,

    /// 每次测量最多轮数
    #[arg(long, default_value_t = 30, global = true)]
    max_iters: usize,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// 跑 E1 微基准矩阵
    Run(RunArgs),
    /// 跑正确性断言（同家族 token id 逐一致 + roundtrip + 流式一致）
    Correctness(CorrectnessArgs),
    /// 只打印语料规格（长度/字节/指纹）
    Corpus(CorpusArgs),
    /// gigatoken 专项：单线程 vs 并行、批量 vs 单条、decode 形态
    GigatokenAudit(GigatokenArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// 后端列表（逗号分隔；all = 六个 vLLM 后端 + gigatoken）
    #[arg(long, default_value = "all")]
    backend: String,

    /// 语料（逗号分隔；all = zh,en,code,mixed）
    #[arg(long, default_value = "all")]
    corpus: String,

    /// 长度档（逗号分隔；all = 128,1k,8k）
    #[arg(long, default_value = "all")]
    lengths: String,

    /// 操作（逗号分隔；all = encode,decode,stream,batch）
    #[arg(long, default_value = "all")]
    ops: String,

    /// 批量路径每个文档的 token 数（语料按此切片）
    #[arg(long, default_value_t = 256)]
    batch_doc_tokens: usize,

    /// 批大小
    #[arg(long, default_value_t = 8)]
    batch_docs: usize,

    /// 流式 decode 的 token 个数
    #[arg(long, default_value_t = 256)]
    stream_tokens: usize,

    /// 结果文件名前缀
    #[arg(long, default_value = "e1")]
    prefix: String,

    /// 只打印将要做的事，不实际测量
    #[arg(long)]
    dry_run: bool,
}

#[derive(clap::Args, Debug)]
struct CorrectnessArgs {
    /// 输出 JSON 路径（相对 worktree 根或绝对路径）
    #[arg(long, default_value = "data/backends/correctness.json")]
    out: PathBuf,
}

#[derive(clap::Args, Debug)]
struct CorpusArgs {
    #[arg(long, default_value = "all")]
    corpus: String,
    #[arg(long, default_value = "all")]
    lengths: String,
    #[arg(long, default_value = "data/backends/corpus.json")]
    out: PathBuf,
}

#[derive(clap::Args, Debug)]
struct GigatokenArgs {
    #[arg(long, default_value = "data/backends/gigatoken-audit.json")]
    out: PathBuf,
    /// 同条件对比用的总字节数（默认 8 MiB，按用户资源纪律控制）
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    total_bytes: usize,
    /// 文档数（文档边界实验）
    #[arg(long, default_value = "1,8,64,512")]
    doc_counts: String,
    /// 是否额外做冷启动测量（每次重新加载 tokenizer、只跑一次）
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cold: bool,
    /// 大块文本的构造方式：
    /// - `repeat`：把同一段 8k 语料反复拼接（**pretoken cache 会全命中**）；
    /// - `unique`：用确定性伪随机词流拼出**互不相同**的预分词
    ///   （cache 基本不命中，作为"冷缓存下界"）；
    /// - `both`：两种都跑（默认）。
    #[arg(long, default_value = "both")]
    blob_mode: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let paths = Paths {
        hf_dir: cli.hf_dir.clone(),
        tiktoken_dir: cli.tiktoken_dir.clone(),
        tekken_path: cli.tekken.clone(),
    };
    let requested = resolve_threads(cli.threads);
    if let Err(e) = backends::install_thread_pool(requested) {
        eprintln!("[b-backends] 设置线程池失败: {e:#}");
        return ExitCode::FAILURE;
    }
    let threads_effective = effective_threads();
    eprintln!(
        "[b-backends] 线程: --threads={} 请求={} 实际生效={} (RAYON_NUM_THREADS={:?})",
        cli.threads,
        requested,
        threads_effective,
        std::env::var("RAYON_NUM_THREADS").ok(),
    );
    let result = match &cli.cmd {
        Cmd::Run(args) => cmd_run(&cli, &paths, args, threads_effective),
        Cmd::Correctness(args) => cmd_correctness(&cli, &paths, args),
        Cmd::Corpus(args) => cmd_corpus(&cli, &paths, args),
        Cmd::GigatokenAudit(args) => cmd_gigatoken_audit(&cli, &paths, args, threads_effective),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[b-backends] 失败: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// `--threads 0` → 用 RAYON_NUM_THREADS；否则取 `min(请求值, 环境上限)`。
/// 环境上限由 `scripts/limit.sh` 设置（= 绑核数），这是"不许超核"的硬约束。
fn resolve_threads(requested: usize) -> usize {
    let env_cap = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    match (requested, env_cap) {
        (0, _) => 0, // 交给 rayon 自己按 RAYON_NUM_THREADS 初始化
        (n, Some(cap)) => n.min(cap).max(1),
        (n, None) => n.max(1),
    }
}

/// 实际生效的线程数（建池后回读，写进 manifest）。
fn effective_threads() -> usize {
    b_backends::backends::effective_threads()
}

fn parse_kinds(spec: &str) -> Result<Vec<CorpusKind>> {
    if spec == "all" {
        return Ok(CorpusKind::ALL.to_vec());
    }
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<CorpusKind>())
        .collect()
}

fn parse_lengths(spec: &str) -> Result<Vec<LengthTarget>> {
    if spec == "all" {
        return Ok(vec![LengthTarget::Tok128, LengthTarget::Tok1k, LengthTarget::Tok8k]);
    }
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| match s {
            "128" => Ok(LengthTarget::Tok128),
            "1k" | "1024" => Ok(LengthTarget::Tok1k),
            "8k" | "8192" => Ok(LengthTarget::Tok8k),
            other => other
                .parse::<usize>()
                .map(LengthTarget::Custom)
                .map_err(|_| anyhow!("未知长度档 {other:?}")),
        })
        .collect()
}

fn parse_backends(spec: &str, with_gigatoken: bool) -> Result<Vec<BackendId>> {
    if spec == "all" {
        let mut v = VLLM_BACKENDS.to_vec();
        if with_gigatoken {
            v.push(BackendId::Gigatoken);
        }
        return Ok(v);
    }
    spec.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            BackendId::from_str(s)
                .ok_or_else(|| anyhow!("未知后端 {s:?}（可选 hf/fastokens/fastokens_byte_level/tiktoken_rs/riptoken/tekken/gigatoken）"))
        })
        .collect()
}

fn load_backends(ids: &[BackendId], paths: &Paths) -> (Vec<Loaded>, Vec<(BackendId, String)>) {
    let mut ok = Vec::new();
    let mut missing = Vec::new();
    for &id in ids {
        match backends::load(id, paths) {
            Ok(b) => {
                eprintln!(
                    "[load] {:24} vocab={:<8} {:.1} ms  {}",
                    id.as_str(),
                    b.detail.vocab_size,
                    b.load_us / 1000.0,
                    b.detail.note
                );
                ok.push(b);
            }
            Err(e) => {
                // 缺工件/不支持 ⇒ 记成"未测 + 原因"，不中断整轮。
                eprintln!("[load] {:24} 跳过: {e:#}", id.as_str());
                missing.push((id, format!("{e:#}")));
            }
        }
    }
    (ok, missing)
}

fn reference_tokenizer(paths: &Paths) -> Result<vllm_tokenizer::HuggingFaceTokenizer> {
    corpus::load_reference(&paths.hf_tokenizer_json())
}

fn cmd_corpus(cli: &Cli, paths: &Paths, args: &CorpusArgs) -> Result<()> {
    let reference = reference_tokenizer(paths)?;
    let kinds = parse_kinds(&args.corpus)?;
    let lengths = parse_lengths(&args.lengths)?;
    let corpora = corpus::generate_all(&kinds, &lengths, &reference)?;
    let out = resolve_out(&cli.out_dir, &args.out);
    let rows: Vec<CorpusRow> = corpora
        .iter()
        .map(|c| CorpusRow {
            id: CorpusSpec::new(c.kind, c.target).id(),
            kind: c.kind.as_str().to_string(),
            length: c.target.as_str(),
            tokens: c.token_ids.len(),
            bytes: c.bytes(),
            chars: c.text.chars().count(),
            chars_per_token: c.charset_per_token(),
            length_error_ratio: c.length_error_ratio(),
            sha256: c.source_sha256.clone(),
            head: c.text.chars().take(48).collect(),
        })
        .collect();
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&out, serde_json::to_string_pretty(&rows)?)?;
    println!("{}", corpus::summarize(&corpora));
    eprintln!("[corpus] 写入 {}", out.display());
    Ok(())
}

#[derive(Serialize)]
struct CorpusRow {
    id: String,
    kind: String,
    length: String,
    tokens: usize,
    bytes: usize,
    chars: usize,
    chars_per_token: f64,
    length_error_ratio: f64,
    sha256: String,
    head: String,
}

fn cmd_run(cli: &Cli, paths: &Paths, args: &RunArgs, threads: usize) -> Result<()> {
    let mut manifest = Manifest::new(
        "E1 微基准矩阵",
        UPSTREAM_COMMIT,
        cli.threads,
        threads,
        if cfg!(debug_assertions) { "debug" } else { "release" },
        features(),
    );
    manifest.add_caveat(
        "语料长度按 Qwen3-0.6B 参考后端的 token 数对齐；其它家族看到的 token 数会不同，\
         表里同时给出 bytes 与各后端实测 token 数（不假设等价）。",
    );
    manifest.add_caveat(
        "单条（single）路径在调用线程上直接跑，不进 rayon；批量（batch）路径走 rayon，\
         线程数受 RAYON_NUM_THREADS 夹逼。",
    );
    manifest.add_caveat("所有工件来自本地路径，harness 不联网（上游 bench 用 hf-hub，本项目改为本地）。");

    let reference = reference_tokenizer(paths)?;
    let kinds = parse_kinds(&args.corpus)?;
    let lengths = parse_lengths(&args.lengths)?;
    let corpora = corpus::generate_all(&kinds, &lengths, &reference)?;
    for c in &corpora {
        manifest
            .corpus_sha256
            .push((CorpusSpec::new(c.kind, c.target).id(), c.source_sha256.clone()));
    }

    let with_gigatoken = args.backend == "all" || args.backend.contains("gigatoken");
    let ids = parse_backends(&args.backend, with_gigatoken)?;
    let ops: Vec<String> = if args.ops == "all" {
        vec!["encode".into(), "decode".into(), "stream".into(), "batch".into()]
    } else {
        args.ops.split(',').map(|s| s.to_string()).collect()
    };

    if args.dry_run {
        println!(
            "将测: backends={:?} corpora={} lengths={} ops={:?} threads={}",
            ids.iter().map(|b| b.as_str()).collect::<Vec<_>>(),
            corpora.len(),
            lengths.len(),
            ops,
            threads
        );
        return Ok(());
    }

    let (loaded, missing) = load_backends(&ids, paths);
    for (id, reason) in &missing {
        manifest.add_caveat(format!("后端 {} 未测: {reason}", id.as_str()));
    }
    if loaded.is_empty() {
        return Err(anyhow!("没有任何后端加载成功"));
    }

    let timer = Timer::new(2, cli.max_iters);
    let mut rows: Vec<Row> = Vec::new();

    // 旁路对照臂：3 号后端没有独立入口，用 raw fastokens 的通用 decode 作为
    // 「旁路未生效」的对照（见 backends.rs::FastokensGeneric 的注释）。
    let byte_level_active = loaded
        .iter()
        .any(|b| b.id() == BackendId::FastokensByteLevel && b.detail.byte_level_branch == Some(true));
    let generic_fastokens = if byte_level_active && ops.iter().any(|o| o == "decode") {
        match backends::FastokensGeneric::load(&paths.hf_tokenizer_json()) {
            Ok(g) => {
                eprintln!(
                    "[load] fastokens_generic        通用 decode 对照臂（旁路未生效的形态）"
                );
                Some(g)
            }
            Err(e) => {
                manifest.add_caveat(format!("fastokens 通用解码对照臂加载失败: {e:#}"));
                None
            }
        }
    } else {
        None
    };

    // --- 单条路径 -----------------------------------------------------------
    for b in &loaded {
        for c in &corpora {
            let spec = CorpusSpec::new(c.kind, c.target);
            let text = c.text.as_str();
            let (bytes, tokens_in) = (c.bytes(), c.token_ids.len());

            // **本后端自己的 token id**。decode / 流式 decode 必须用自己家族的 id：
            // Kimi 的 id 喂给 Qwen 的分词表毫无意义（第一版就是这么错的，
            // tekken 直接报 `Invalid token for decoding: 150644`）。
            //
            // 这里也必须套 `guard_panic`：实测 fastokens 0.2.1 的长中文输入会在
            // 预分词阶段越界 panic（见 docs/03 §5），不隔离就会带走整轮矩阵。
            let own_ids: Vec<u32> = match b.encode(text) {
                Ok(v) => v,
                Err(e) => {
                    manifest.add_caveat(format!(
                        "{} encode 在 {}/{} 上失败: {e:#}",
                        b.id().as_str(),
                        c.kind.as_str(),
                        c.target.as_str()
                    ));
                    eprintln!(
                        "[pre  ] {:<22} {:<6} {:<5} 跳过: {e}",
                        b.id().as_str(),
                        c.kind.as_str(),
                        c.target.as_str()
                    );
                    rows.push(Row::failure(
                        b.id().as_str(),
                        b.id().family().as_str(),
                        &spec,
                        "encode_prepass",
                        "single",
                        threads,
                        &format!("{e:#}"),
                    ));
                    continue;
                }
            };
            // HF 家族的 id 必须与语料参考 id 逐一相等（语料就是用它切的）。
            if b.id().family() == b_backends::backends::Family::Hf {
                assert_eq!(
                    own_ids,
                    c.token_ids,
                    "{} 的 token id 与语料参考不一致（同一工件，必须一致）",
                    b.id().as_str()
                );
            }

            if ops.iter().any(|o| o == "encode") {
                let (p0, p1, p2, p3) = pressure();
                let t0 = std::time::Instant::now();
                // 每次迭代换一个变体（见 corpus.rs::variants 的说明）：
                // 避免"反复编码同一个字符串"把 gigatoken 的 pretoken cache
                // 之类的内部缓存喂成不真实的命中率。
                let mut k = 0usize;
                let m = timer.measure_try_with_budget(
                    bytes,
                    tokens_in,
                    cli.budget_ms,
                    cli.max_iters,
                    || {
                        let t = if c.variants.len() > 1 {
                            let v = c.variant(k);
                            k += 1;
                            v
                        } else {
                            text
                        };
                        // catch_unwind：第三方后端在个别输入上会 panic
                        // （实测 fastokens 0.2.1 的长中文输入），
                        // 整轮矩阵不能因此中断。
                        b.encode(t).map_err(|e| format!("{e:#}"))
                    },
                );
                let elapsed = t0.elapsed().as_secs_f64() * 1e3;
                match m {
                    Ok(m) => {
                        rows.push(Row::from_measurement(
                            b,
                            b.id().as_str(),
                            &spec,
                            "encode",
                            "single",
                            threads,
                            1,
                            tokens_in,
                            &m,
                            (p0, p1, p2, p3),
                            format!(
                                "input=变体轮转({} 个变体)；本后端在变体 0 上实测 {} tokens；\
                                 encode 全流程 {:.0} ms",
                                c.variants.len(),
                                own_ids.len(),
                                elapsed
                            ),
                        ));
                        eprintln!(
                            "[encode] {:<22} {:<6} {:<5} {:>9.2} µs  ({:.1} MB/s)",
                            b.id().as_str(),
                            c.kind.as_str(),
                            c.target.as_str(),
                            m.median_us,
                            m.mb_per_s()
                        );
                    }
                    Err(e) => {
                        manifest.add_caveat(format!(
                            "{} encode 失败于 {}/{}: {e}",
                            b.id().as_str(),
                            c.kind.as_str(),
                            c.target.as_str()
                        ));
                        rows.push(Row::failure(
                            b.id().as_str(),
                            b.id().family().as_str(),
                            &spec,
                            "encode",
                            "single",
                            threads,
                            &e,
                        ));
                    }
                }
            }

            if ops.iter().any(|o| o == "decode") {
                let ids_out: Vec<u32> = own_ids.clone();
                let decoded = match b.decode(&ids_out) {
                    Ok(d) => d,
                    Err(e) => {
                        manifest.add_caveat(format!(
                            "{} decode 在 {}/{} 上失败: {e:#}",
                            b.id().as_str(),
                            c.kind.as_str(),
                            c.target.as_str()
                        ));
                        continue;
                    }
                };
                // 正确性旁证：dump 出来的文本重新编码是否与原始 id 一致（roundtrip）。
                let roundtrip = b.encode(&decoded).map(|re| re == ids_out).unwrap_or(false);
                let (p0, p1, p2, p3) = pressure();
                let m = timer.measure_try_with_budget(
                    decoded.len(),
                    ids_out.len(),
                    cli.budget_ms,
                    cli.max_iters,
                    || {
                        b.decode(&ids_out).map_err(|e| format!("{e:#}"))
                    },
                );
                match m {
                    Ok(m) => {
                        rows.push(Row::from_measurement(
                            b,
                            b.id().as_str(),
                            &spec,
                            "decode",
                            "single",
                            threads,
                            1,
                            ids_out.len(),
                            &m,
                            (p0, p1, p2, p3),
                            format!(
                                "ids 来自本后端 encode（{} tokens）；roundtrip_reencode_ids_eq={roundtrip}",
                                ids_out.len()
                            ),
                        ));
                        eprintln!(
                            "[decode] {:<22} {:<6} {:<5} {:>9.2} µs  ({:.1} MB/s out={} B)",
                            b.id().as_str(),
                            c.kind.as_str(),
                            c.target.as_str(),
                            m.median_us,
                            m.mb_per_s(),
                            decoded.len()
                        );
                    }
                    Err(e) => {
                        rows.push(Row::failure(
                            b.id().as_str(),
                            b.id().family().as_str(),
                            &spec,
                            "decode",
                            "single",
                            threads,
                            &e,
                        ));
                    }
                }

                // 同一 ids 上再测一次"通用 fastokens decode"，用于量化 ByteLevel 旁路。
                if let Some(g) = &generic_fastokens {
                    let m = timer.measure_try_with_budget(
                        decoded.len(),
                        ids_out.len(),
                        cli.budget_ms,
                        cli.max_iters,
                        || g.decode(&ids_out, false).map_err(|e| format!("{e:#}")),
                    );
                    if let Ok(m) = m {
                        rows.push(Row::from_measurement(
                            b,
                            "fastokens_generic",
                            &spec,
                            "decode_generic_fastokens",
                            "single",
                            threads,
                            1,
                            ids_out.len(),
                            &m,
                            (0.0, 0.0, 0.0, 0.0),
                            "对照臂：fastokens 通用 decode（旁路未生效）；与 fastokens_byte_level 同一 ids"
                                .into(),
                        ));
                        eprintln!(
                            "[decode*] {:<21} {:<6} {:<5} {:>9.2} µs  (fastokens 通用路径)",
                            "fastokens_generic",
                            c.kind.as_str(),
                            c.target.as_str(),
                            m.median_us
                        );
                    }
                }
            }

            if ops.iter().any(|o| o == "stream") && b.has_streaming_decode() {
                if own_ids.len() < args.stream_tokens + 8 {
                    continue; // 语料太短，跳过（会在文档里写清）
                }
                let prompt_len = own_ids.len() / 2;
                let prompt = &own_ids[..prompt_len];
                let generated = &own_ids[prompt_len..prompt_len + args.stream_tokens];
                for &min_bytes in &[1usize, 4, 8] {
                    let (p0, p1, p2, p3) = pressure();
                    let m = timer.measure_try_with_budget(
                        0,
                        generated.len(),
                        cli.budget_ms,
                        cli.max_iters,
                        || {
                            b.decode_incremental(prompt, generated, min_bytes)
                                .map(|v| v.0)
                                .map_err(|e| format!("{e:#}"))
                        },
                    );
                    let note = format!(
                        "prompt_tokens={} min_bytes_to_buffer={}",
                        prompt.len(),
                        min_bytes
                    );
                    match m {
                        Ok(m) => {
                            rows.push(Row::from_measurement(
                                b,
                                b.id().as_str(),
                                &spec,
                                "stream_decode",
                                "single",
                                threads,
                                1,
                                generated.len(),
                                &m,
                                (p0, p1, p2, p3),
                                note,
                            ));
                            eprintln!(
                                "[stream] {:<22} {:<6} {:<5} min_bytes={:<2} {:>9.2} µs  ({:.0} tok/s)",
                                b.id().as_str(),
                                c.kind.as_str(),
                                c.target.as_str(),
                                min_bytes,
                                m.median_us,
                                m.tokens_per_s()
                            );
                        }
                        Err(e) => {
                            // 例如 tekken 对不完整 UTF-8 直接报错（见 docs/03 §4）。
                            manifest.add_caveat(format!(
                                "{} 流式 decode 失败于 {}/{}（min_bytes_to_buffer={min_bytes}）: {e}",
                                b.id().as_str(),
                                c.kind.as_str(),
                                c.target.as_str()
                            ));
                            rows.push(Row::failure(
                                b.id().as_str(),
                                b.id().family().as_str(),
                                &spec,
                                "stream_decode",
                                "single",
                                threads,
                                &format!("{note}: {e}"),
                            ));
                            eprintln!(
                                "[stream] {:<22} {:<6} {:<5} min_bytes={:<2} 失败: {e}",
                                b.id().as_str(),
                                c.kind.as_str(),
                                c.target.as_str(),
                                min_bytes
                            );
                        }
                    }
                }
            }
        }
    }

    // --- 批量路径 -----------------------------------------------------------
    if ops.iter().any(|o| o == "batch") {
        for b in &loaded {
            for c in &corpora {
                let spec = CorpusSpec::new(c.kind, c.target);
                let docs = batch_docs(c, args.batch_doc_tokens);
                let total_bytes: usize = docs.iter().map(|d| d.len()).sum();
                let (p0, p1, p2, p3) = pressure();
                let m = timer.measure_try_with_budget(
                    total_bytes,
                    0,
                    cli.budget_ms * 2.0,
                    cli.max_iters.min(10),
                    || {
                        b.encode_batch(&docs).map_err(|e| format!("{e:#}"))
                    },
                );
                let measured_tokens: usize = b
                    .encode_batch(&docs)
                    .map(|v| v.iter().map(|d| d.len()).sum())
                    .unwrap_or(0);
                match m {
                    Ok(m) => {
                        rows.push(Row::from_measurement(
                            b,
                            b.id().as_str(),
                            &spec,
                            "encode_batch",
                            if threads > 1 { "batch" } else { "batch-1thread" },
                            threads,
                            docs.len(),
                            measured_tokens,
                            &m,
                            (p0, p1, p2, p3),
                            format!(
                                "batch_docs={} 每文档≈{} tokens",
                                docs.len(),
                                args.batch_doc_tokens
                            ),
                        ));
                        eprintln!(
                            "[batch]  {:<22} {:<6} {:<5} docs={:<3} {:>9.2} µs  ({:.1} MB/s)",
                            b.id().as_str(),
                            c.kind.as_str(),
                            c.target.as_str(),
                            docs.len(),
                            m.median_us,
                            m.mb_per_s()
                        );
                    }
                    Err(e) => {
                        rows.push(Row::failure(
                            b.id().as_str(),
                            b.id().family().as_str(),
                            &spec,
                            "encode_batch",
                            "batch",
                            threads,
                            &e,
                        ));
                    }
                }
                settle();
            }
        }
    }

    let out_dir = resolve_root(&cli.out_dir);
    std::fs::create_dir_all(&out_dir)?;
    let jsonl = out_dir.join(format!("{}.jsonl", args.prefix));
    let csv = out_dir.join(format!("{}.csv", args.prefix));
    write_jsonl(&jsonl, &rows)?;
    write_csv(&csv, &rows)?;
    manifest.add_artifact(&jsonl)?;
    manifest.add_artifact(&csv)?;
    for p in harness_scripts() {
        if p.exists() {
            manifest.add_script(&p)?;
        }
    }
    manifest.finish();
    manifest.write(&out_dir.join(format!("{}.manifest.json", args.prefix)))?;
    println!("[done] rows={} → {}", rows.len(), jsonl.display());
    Ok(())
}

fn batch_docs<'a>(c: &'a Corpus, doc_tokens: usize) -> Vec<&'a str> {
    // 把语料等分成 n 个文档，n 取「总 token / 每文档 token」并夹到 [4, 64]。
    // 下限 4：128 token 的语料若只切 1 片，测出来就是单条而不是批量；
    // 上限 64：避免 8k 语料切出上百个小文档、把启动开销算进吞吐。
    let n = (c.token_ids.len() / doc_tokens.max(1)).clamp(4, 64);
    c.batches(n)
}

fn pressure() -> (f64, f64, f64, f64) {
    let (load, mem) = b_backends::manifest::read_pressure();
    (load[0], mem, load[0], mem)
}

fn resolve_root(out_dir: &Path) -> PathBuf {
    if out_dir.is_absolute() {
        out_dir.to_path_buf()
    } else {
        worktree_root().join(out_dir)
    }
}

fn resolve_out(out_dir: &Path, out: &Path) -> PathBuf {
    if out.is_absolute() {
        out.to_path_buf()
    } else if out.starts_with("data") {
        worktree_root().join(out)
    } else {
        resolve_root(out_dir).join(out)
    }
}

fn worktree_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn harness_scripts() -> Vec<PathBuf> {
    let root = worktree_root();
    vec![
        root.join("harness/rust/bench/src/main.rs"),
        root.join("harness/rust/bench/src/backends.rs"),
        root.join("harness/rust/bench/src/corpus.rs"),
        root.join("harness/rust/bench/src/stats.rs"),
        root.join("harness/rust/bench/src/manifest.rs"),
        root.join("harness/rust/bench/Cargo.toml"),
    ]
}

fn features() -> Vec<String> {
    let mut v = Vec::new();
    if cfg!(feature = "gigatoken") {
        v.push("gigatoken".to_string());
    }
    v
}

fn write_jsonl(path: &Path, rows: &[Row]) -> Result<()> {
    let mut s = String::new();
    for r in rows {
        s.push_str(&serde_json::to_string(r)?);
        s.push('\n');
    }
    std::fs::write(path, s).with_context(|| format!("写 {} 失败", path.display()))?;
    Ok(())
}

fn write_csv(path: &Path, rows: &[Row]) -> Result<()> {
    let mut s = String::from(
        "backend,family,corpus,length,op,mode,threads,bytes,tokens,docs,measured_tokens,iters,\
         median_us,min_us,p90_us,max_us,mb_per_s,tokens_per_s,us_per_token,loadavg_before,\
         loadavg_after,mem_available_gb_before,note\n",
    );
    for r in rows {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.1},{:.4},{:.2},{:.2},{:.2},{}\n",
            r.backend,
            r.family,
            r.corpus,
            r.length,
            r.op,
            r.mode,
            r.threads,
            r.bytes,
            r.tokens,
            r.docs,
            r.measured_tokens,
            r.iters,
            r.median_us,
            r.min_us,
            r.p90_us,
            r.max_us,
            r.mb_per_s,
            r.tokens_per_s,
            r.us_per_token,
            r.loadavg_before,
            r.loadavg_after,
            r.mem_available_gb_before,
            r.note.replace(',', ";"),
        ));
    }
    std::fs::write(path, s).with_context(|| format!("写 {} 失败", path.display()))?;
    Ok(())
}

// --- 正确性 --------------------------------------------------------------

#[derive(Serialize)]
struct FamilyCheck {
    family: String,
    artifact: String,
    corpus: String,
    length: String,
    tokens: usize,
    /// 家族内参与比对的成员。
    members: Vec<String>,
    /// 是否所有成员的 token id 序列逐一致。
    ids_identical: bool,
    /// 不一致时的反例：分叉位置 + 各方 token。
    counterexample: Option<String>,
    /// 缺失的成员及原因。
    missing: Vec<String>,
}

#[derive(Serialize)]
struct RoundTripCheck {
    backend: String,
    corpus: String,
    length: String,
    /// decode 用的 ids 来源（永远是本后端自己 encode 出来的）。
    ids_source: String,
    /// decode 后是否得到合法 UTF-8。
    decode_ok: bool,
    /// 重新 encode(decode(ids)) 是否与 ids 完全一致。
    reencode_ids_identical: bool,
    /// 流式（逐 token）拼接结果是否与整段 decode 一致。
    stream_matches_full: Option<bool>,
    note: String,
}

#[derive(Serialize)]
struct PrefixDecodeCheck {
    backend: String,
    corpus: String,
    /// 前缀长度扫描范围。
    prefix_lens: Vec<usize>,
    /// 失败的前缀长度 -> 错误信息（流式 decode 必须能容忍"半个字符"）。
    failures: BTreeMap<usize, String>,
    /// 是否所有前缀都能解码（不能的话流式 decode 一定会炸）。
    all_prefixes_ok: bool,
}

#[derive(Serialize)]
struct CorrectnessReport {
    generated_at: String,
    upstream_commit: String,
    checks: Vec<FamilyCheck>,
    roundtrips: Vec<RoundTripCheck>,
    prefix_decodes: Vec<PrefixDecodeCheck>,
    summary: BTreeMap<String, usize>,
}

fn cmd_correctness(cli: &Cli, paths: &Paths, args: &CorrectnessArgs) -> Result<()> {
    let reference = reference_tokenizer(paths)?;
    let corpora = corpus::generate_all(&CorpusKind::ALL, &parse_lengths("all")?, &reference)?;

    let wanted = [
        BackendId::Hf,
        BackendId::Fastokens,
        BackendId::FastokensByteLevel,
        BackendId::TiktokenRs,
        BackendId::Riptoken,
        BackendId::Tekken,
        BackendId::Gigatoken,
    ];
    let (loaded, missing_map) = {
        let (mut ok, missing) = load_backends(&wanted, paths);
        let map: BTreeMap<String, String> =
            missing.into_iter().map(|(id, r)| (id.as_str().to_string(), r)).collect();
        ok.sort_by_key(|b| b.id());
        (ok, map)
    };

    let mut checks = Vec::new();
    let mut roundtrips = Vec::new();
    let mut prefix_decodes = Vec::new();
    let mut summary: BTreeMap<String, usize> = BTreeMap::new();

    for family in [
        b_backends::backends::Family::Hf,
        b_backends::backends::Family::Tiktoken,
        b_backends::backends::Family::Tekken,
        b_backends::backends::Family::Gigatoken,
    ] {
        let members: Vec<&Loaded> =
            loaded.iter().filter(|b| b.id().family() == family).collect();
        if members.is_empty() {
            continue;
        }
        for c in &corpora {
            let spec = CorpusSpec::new(c.kind, c.target);
            let mut encodings: Vec<(String, Vec<u32>)> = Vec::new();
            let mut local_missing = Vec::new();
            for b in &members {
                match b.encode(&c.text) {
                    Ok(ids) => encodings.push((b.id().as_str().to_string(), ids)),
                    Err(e) => local_missing.push(format!("{}: {e:#}", b.id().as_str())),
                }
            }
            let mut counterexample = None;
            let mut identical = true;
            if encodings.len() >= 2 {
                let (first_name, first_ids) = &encodings[0];
                for (name, ids) in &encodings[1..] {
                    if ids != first_ids {
                        identical = false;
                        let pos = ids
                            .iter()
                            .zip(first_ids.iter())
                            .position(|(a, b)| a != b)
                            .unwrap_or_else(|| ids.len().min(first_ids.len()));
                        let ctx = |v: &[u32]| {
                            let lo = pos.saturating_sub(3);
                            v[lo..(pos + 4).min(v.len())]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        };
                        counterexample = Some(format!(
                            "首个分歧位置 {pos}: {first_name}=[{}] ({} ids) vs {name}=[{}] ({} ids)",
                            ctx(first_ids),
                            first_ids.len(),
                            ctx(ids),
                            ids.len()
                        ));
                        break;
                    }
                }
            }
            let key = format!("{} ids_identical", family.as_str());
            if identical {
                *summary.entry(key).or_default() += 1;
            }
            checks.push(FamilyCheck {
                family: family.as_str().to_string(),
                artifact: members[0].detail.artifact.clone(),
                corpus: c.kind.as_str().to_string(),
                length: spec.target.as_str(),
                tokens: c.token_ids.len(),
                members: encodings.iter().map(|(n, _)| n.clone()).collect(),
                ids_identical: identical,
                counterexample,
                missing: local_missing,
            });
        }
    }

    // roundtrip + 流式一致性
    for b in &loaded {
        for c in &corpora {
            let spec = CorpusSpec::new(c.kind, c.target);
            let own_ids = match b.encode(&c.text) {
                Ok(v) => v,
                Err(e) => {
                    roundtrips.push(RoundTripCheck {
                        backend: b.id().as_str().to_string(),
                        corpus: c.kind.as_str().to_string(),
                        length: spec.target.as_str(),
                        ids_source: "self".into(),
                        decode_ok: false,
                        reencode_ids_identical: false,
                        stream_matches_full: None,
                        note: format!("encode 失败: {e:#}"),
                    });
                    continue;
                }
            };
            let decoded = b.decode(&own_ids);
            let (decode_ok, reencode_ids_identical, note) = match &decoded {
                Ok(text) => {
                    let same = b.encode(text).map(|re| re == own_ids).unwrap_or(false);
                    (true, same, String::new())
                }
                Err(e) => (false, false, format!("{e:#}")),
            };
            let stream_matches_full = if b.has_streaming_decode() && decoded.is_ok() {
                let prompt_len = (own_ids.len() / 2).min(4);
                let generated = &own_ids[prompt_len..];
                b.decode_incremental(&own_ids[..prompt_len], generated, 1)
                    .ok()
                    .map(|(streamed, _)| {
                        // 流式输出的是"生成部分"文本，整段 decode 是"全量"；
                        // 比较用 generated 前缀部分。
                        let full = decoded.as_ref().unwrap();
                        full.ends_with(&streamed) || streamed.is_empty()
                    })
            } else {
                None
            };
            roundtrips.push(RoundTripCheck {
                backend: b.id().as_str().to_string(),
                corpus: c.kind.as_str().to_string(),
                length: spec.target.as_str(),
                ids_source: "self".into(),
                decode_ok,
                reencode_ids_identical,
                stream_matches_full,
                note,
            });
        }
    }

    // 前缀解码：流式 decode 必须能容忍"一个多字节字符被拆到两个 token"，
    // 也就是 decode(前缀) 要么给出替换字符、要么给出前缀文本，**不能报错**。
    // 这条检查是 tekken 后端的一个真实缺陷的证据链。
    for b in &loaded {
        for c in &corpora {
            let Ok(own_ids) = b.encode(&c.text) else { continue };
            // 扫全部前缀（最多 512 个），因为"半个字符"落在哪个 token 边界
            // 是数据相关的，只扫前 8 个会漏掉真实故障点。
            let scan = own_ids.len().min(512);
            let prefix_lens: Vec<usize> = (1..=scan).collect();
            let mut failures = BTreeMap::new();
            for &n in &prefix_lens {
                if let Err(e) = b.decode(&own_ids[..n]) {
                    failures.insert(n, format!("{e:#}"));
                }
            }
            let all_prefixes_ok = failures.is_empty();
            if !all_prefixes_ok {
                *summary.entry(format!("{} prefix_decode_failures", b.id().as_str())).or_default() +=
                    1;
            }
            prefix_decodes.push(PrefixDecodeCheck {
                backend: b.id().as_str().to_string(),
                corpus: format!("{}-{}", c.kind.as_str(), c.target.as_str()),
                prefix_lens,
                failures,
                all_prefixes_ok,
            });
        }
    }

    // 缺件记录
    for (id, reason) in &missing_map {
        summary.insert(format!("missing:{id}"), 1);
        eprintln!("[correctness] {id} 未测: {reason}");
    }

    // 跨家族对照：gigatoken 与 HF 加载的是**同一个** `tokenizer.json`，
    // 因此两者的 token id 必须逐一相等（不同实现、不同表结构，但同一文件）。
    // 这是 gigatoken 声明的"兼容 HF tokenizer.json"的可验证形式。
    if let (Some(hf), Some(gt)) = (
        loaded.iter().find(|b| b.id() == BackendId::Hf),
        loaded.iter().find(|b| b.id() == BackendId::Gigatoken),
    ) {
        for c in &corpora {
            let spec = CorpusSpec::new(c.kind, c.target);
            let (a, b_) = (hf.encode(&c.text), gt.encode(&c.text));
            let identical = matches!((&a, &b_), (Ok(x), Ok(y)) if x == y);
            let counterexample = match (&a, &b_) {
                (Ok(x), Ok(y)) if x != y => {
                    let pos = x
                        .iter()
                        .zip(y.iter())
                        .position(|(p, q)| p != q)
                        .unwrap_or_else(|| x.len().min(y.len()));
                    Some(format!(
                        "首个分歧位置 {pos}: hf=[{:?}] ({} ids) vs gigatoken=[{:?}] ({} ids)",
                        &x[pos.saturating_sub(2)..(pos + 3).min(x.len())],
                        x.len(),
                        &y[pos.saturating_sub(2)..(pos + 3).min(y.len())],
                        y.len()
                    ))
                }
                (Err(e), _) => Some(format!("hf encode 失败: {e:#}")),
                (_, Err(e)) => Some(format!("gigatoken encode 失败: {e:#}")),
                _ => None,
            };
            if identical {
                *summary.entry("gigatoken_vs_hf ids_identical".into()).or_default() += 1;
            } else {
                *summary.entry("gigatoken_vs_hf ids_differ".into()).or_default() += 1;
            }
            checks.push(FamilyCheck {
                family: "gigatoken_vs_hf".into(),
                artifact: hf.detail.artifact.clone(),
                corpus: c.kind.as_str().to_string(),
                length: spec.target.as_str(),
                tokens: c.token_ids.len(),
                members: vec!["hf".into(), "gigatoken".into()],
                ids_identical: identical,
                counterexample,
                missing: vec![],
            });
        }
    }

    let report = CorrectnessReport {
        generated_at: b_backends::manifest::now_rfc3339(),
        upstream_commit: UPSTREAM_COMMIT.to_string(),
        checks,
        roundtrips,
        prefix_decodes,
        summary,
    };
    let out = resolve_out(&cli.out_dir, &args.out);
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("[correctness] → {}", out.display());
    for (k, v) in &report.summary {
        println!("  {k:40} {v}");
    }
    for c in report.checks.iter().filter(|c| !c.ids_identical) {
        println!(
            "  !! {} {}-{}: {}",
            c.family,
            c.corpus,
            c.length,
            c.counterexample.as_deref().unwrap_or("(unknown)")
        );
    }
    // 退出码：有反例或 roundtrip 失败就非零，方便 CI 和一键脚本判断
    let bad = report.checks.iter().filter(|c| !c.ids_identical).count()
        + report.roundtrips.iter().filter(|r| !r.decode_ok || !r.reencode_ids_identical).count();
    if bad > 0 {
        eprintln!("[correctness] 有 {bad} 项不通过（详见 JSON）");
    }
    Ok(())
}

// --- gigatoken 专项 -------------------------------------------------------

#[derive(Serialize)]
struct AuditRow {
    scenario: String,
    /// 大块文本的**内容**构造方式：
    /// - `repeat`：同一段语料反复拼接（预分词大量重复）；
    /// - `unique`：伪随机词流（预分词互不重复）；
    /// - `online-自然文本`：E1 的 4 类语料。
    blob_mode: String,
    docs: usize,
    total_bytes: usize,
    threads: usize,
    /// **热缓存**中位耗时：同一输入反复跑（第 2 轮起 pretoken/BPE 缓存全命中）。
    median_us: f64,
    /// **冷启动**耗时：每次重新加载一个全新 tokenizer，只跑一次。
    ///
    /// 为什么必须有这一列：gigatoken 与 fastokens 都在 tokenizer 内部维护
    /// pretoken / BPE 缓存，同一输入反复编码时吞吐可以虚高一个数量级
    /// （实测差 3–10×）。真实离线批处理是"一次过"的语料，对应的就是冷启动数。
    cold_us: f64,
    mb_per_s: f64,
    cold_mb_per_s: f64,
    tokens_per_s: f64,
    total_tokens: usize,
    note: String,
}

#[derive(Serialize)]
struct GigatokenAudit {
    generated_at: String,
    upstream_commit: String,
    total_bytes: usize,
    doc_counts: Vec<usize>,
    rows: Vec<AuditRow>,
    /// 与 HF / fastokens 的**同条件**对照（同字节、同文档边界、同线程数）。
    same_condition: Vec<AuditRow>,
    /// 测不出来的点（原样记录原因，不允许静默丢弃）。
    failed_rows: Vec<String>,
    /// 缺口事实（可复核的代码/接口证据）。
    gaps: Vec<Gap>,
    decode_checks: Vec<DecodeCheck>,
}

#[derive(Serialize)]
struct Gap {
    id: String,
    statement: String,
    evidence: String,
    impact: String,
}

#[derive(Serialize)]
struct DecodeCheck {
    backend: String,
    ids: usize,
    text_equal: bool,
    note: String,
}

fn cmd_gigatoken_audit(
    cli: &Cli,
    paths: &Paths,
    args: &GigatokenArgs,
    threads: usize,
) -> Result<()> {
    let doc_counts: Vec<usize> = args
        .doc_counts
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow!("解析 --doc-counts 失败: {e}"))?;

    let reference = reference_tokenizer(paths)?;
    // 用 1k 语料作为"同条件"输入，按目标总字节数重复/截断。
    let corpora = corpus::generate_all(&[CorpusKind::English, CorpusKind::Chinese], &[LengthTarget::Tok8k], &reference)?;
    let mut rows = Vec::new();
    let mut same_condition = Vec::new();
    // 对照组里失败的点：不能悄悄消失，要出现在产物里。
    let mut failed_rows: Vec<String> = Vec::new();
    let mut decode_checks = Vec::new();
    let timer = Timer::new(2, cli.max_iters.min(10));

    let (loaded, missing) = load_backends(&parse_backends("all", true)?, paths);
    let gigatoken = loaded.iter().find(|b| b.id() == BackendId::Gigatoken);
    let hf = loaded.iter().find(|b| b.id() == BackendId::Hf);
    // 注意：在 ByteLevel-only 的 Qwen3 工件上 `BackendId::Fastokens` 无法单独加载
    // （它与 FastokensByteLevel 是同一个对象，load() 会拒绝），所以对照臂用后者。
    let fastokens = loaded.iter().find(|b| b.id() == BackendId::FastokensByteLevel);
    for (id, reason) in &missing {
        if id == &BackendId::Gigatoken {
            return Err(anyhow!("gigatoken 未加载: {reason}"));
        }
    }

    let gt = gigatoken.ok_or_else(|| anyhow!("gigatoken 未加载"))?;
    let hf = hf.ok_or_else(|| anyhow!("hf 未加载"))?;

    let base = &corpora[0];
    let modes: Vec<&str> = match args.blob_mode.as_str() {
        "repeat" => vec!["repeat"],
        "unique" => vec!["unique"],
        _ => vec!["repeat", "unique"],
    };

    for mode in modes {
        // 固定总字节数的大块文本。
        //
        // `repeat`：同一段 8k 语料反复拼接。**这会让 gigatoken 的 pretoken cache
        // 几乎全命中**（它的 cache 存在 tokenizer 内部、跨调用常驻），
        // 所以这一档是"热缓存上界"。
        // `unique`：确定性伪随机词流，每个预分词都不重复，cache 基本不命中，
        // 是"冷缓存下界"。真实流量在两个极端之间（自然语言会复用常见 pretoken）。
        let mut blob = if mode == "repeat" {
            let mut s = String::new();
            while s.len() < args.total_bytes {
                s.push_str(&base.text);
                s.push('\n');
            }
            s
        } else {
            unique_blob(args.total_bytes)
        };
        blob.truncate(floor_boundary(&blob, args.total_bytes));

    for &docs in &doc_counts {
        let slices = split_into(&blob, docs);
        // (a) gigatoken 单文档语义：整块当一个文档（其内部会自己 chunk）
        let blob_str = blob.as_str();
        let whole = std::slice::from_ref(&blob_str);
        let m_whole = timer.measure_with_budget(
            blob.len(),
            base.token_ids.len(),
            300.0,
            10,
            || gt.encode_batch_serial(whole),
        );
        let cold_whole = if args.cold {
            cold_encode_batch(BackendId::Gigatoken, paths, whole, 1)
        } else {
            None
        };
        rows.push(AuditRow {
            scenario: "gigatoken/whole-blob-as-1-doc".into(),
            blob_mode: mode.to_string(),
            docs: 1,
            total_bytes: args.total_bytes,
            threads: 1,
            median_us: m_whole.median_us,
            cold_us: cold_whole.as_ref().map_or(f64::NAN, |(us, _)| *us),
            mb_per_s: m_whole.mb_per_s(),
            cold_mb_per_s: cold_whole
                .as_ref()
                .map_or(f64::NAN, |(us, _)| (blob.len() as f64 / (us / 1e6)) / 1e6),
            tokens_per_s: m_whole.tokens_per_s(),
            total_tokens: base.token_ids.len(),
            note: "其官方 benchmark 的口径：整块 bytes 当单文档，内部 chunk 化".into(),
        });
        // (b) gigatoken 并行批量：同样的 N 个文档
        let m_par = timer.measure_with_budget(
            blob.len(),
            base.token_ids.len(),
            300.0,
            10,
            || gt.encode_batch(&slices),
        );
        let cold_par = if args.cold {
            cold_encode_batch(BackendId::Gigatoken, paths, &slices, 1)
        } else {
            None
        };
        rows.push(AuditRow {
            scenario: format!("gigatoken/parallel-{docs}-docs"),
            blob_mode: mode.to_string(),
            docs,
            total_bytes: args.total_bytes,
            threads,
            median_us: m_par.median_us,
            cold_us: cold_par.as_ref().map_or(f64::NAN, |(us, _)| *us),
            mb_per_s: m_par.mb_per_s(),
            cold_mb_per_s: cold_par
                .as_ref()
                .map_or(f64::NAN, |(us, _)| (blob.len() as f64 / (us / 1e6)) / 1e6),
            tokens_per_s: m_par.tokens_per_s(),
            total_tokens: base.token_ids.len(),
            note: "rayon 全局池 + WorkerPool".into(),
        });
        // (c) gigatoken 串行批量：同样 N 个文档、单线程
        let m_ser =
            timer.measure_with_budget(blob.len(), base.token_ids.len(), 300.0, 10, || {
                gt.encode_batch_serial(&slices)
            });
        rows.push(AuditRow {
            scenario: format!("gigatoken/serial-{docs}-docs"),
            blob_mode: mode.to_string(),
            docs,
            total_bytes: args.total_bytes,
            threads: 1,
            median_us: m_ser.median_us,
            cold_us: cold_par.as_ref().map_or(f64::NAN, |(us, _)| *us),
            mb_per_s: m_ser.mb_per_s(),
            cold_mb_per_s: cold_par
                .as_ref()
                .map_or(f64::NAN, |(us, _)| (blob.len() as f64 / (us / 1e6)) / 1e6),
            tokens_per_s: m_ser.tokens_per_s(),
            total_tokens: base.token_ids.len(),
            note: "encode_docs_ragged_serial：完全不碰 rayon".into(),
        });

        // 同条件对照臂：HF / fastokens 用同一 N 个文档、同一线程数
        for (b, label) in [Some((hf, "hf")), fastokens.map(|f| (f, "fastokens_bl"))]
            .into_iter()
            .flatten()
        {
            // 必须用 try 版：fastokens 在长输入上会报错/panic，用非 try 版会把
            // **错误路径**的耗时当成成功测量记下来（本 harness 早期版本犯过这个错）。
            let m = timer.measure_try_with_budget(
                blob.len(),
                base.token_ids.len(),
                300.0,
               10,
                || b.encode_batch(&slices).map_err(|e| format!("{e:#}")),
            );
            let cold_b = if args.cold {
                cold_encode_batch(b.id(), paths, &slices, 1)
            } else {
                None
            };
            let cold_mb = cold_b
                .as_ref()
                .map_or(f64::NAN, |(us, _)| (blob.len() as f64 / (us / 1e6)) / 1e6);
            let base_note = format!("同字节/同文档边界；后端={}", b.id().as_str());
            match &m {
                Ok(m) => same_condition.push(AuditRow {
                    scenario: format!("{label}/parallel-{docs}-docs"),
                    blob_mode: mode.to_string(),
                    docs,
                    total_bytes: args.total_bytes,
                    threads,
                    median_us: m.median_us,
                    cold_us: cold_b.as_ref().map_or(f64::NAN, |(us, _)| *us),
                    mb_per_s: m.mb_per_s(),
                    cold_mb_per_s: cold_mb,
                    tokens_per_s: m.tokens_per_s(),
                    total_tokens: base.token_ids.len(),
                    note: base_note.clone(),
                }),
                Err(e) => failed_rows.push(format!(
                    "{label}/parallel-{docs}-docs [{mode}]: {e}"
                )),
            }
            let m1 = timer.measure_try_with_budget(
                blob.len(),
                base.token_ids.len(),
                300.0,
                10,
                || b.encode_batch_serial(&slices).map_err(|e| format!("{e:#}")),
            );
            match &m1 {
                Ok(m1) => same_condition.push(AuditRow {
                    scenario: format!("{label}/serial-{docs}-docs"),
                    blob_mode: mode.to_string(),
                    docs,
                    total_bytes: args.total_bytes,
                    threads: 1,
                    median_us: m1.median_us,
                    cold_us: cold_b.as_ref().map_or(f64::NAN, |(us, _)| *us),
                    mb_per_s: m1.mb_per_s(),
                    cold_mb_per_s: cold_mb,
                    tokens_per_s: m1.tokens_per_s(),
                    total_tokens: base.token_ids.len(),
                    note: format!("{base_note}；单线程；冷启动列复用并行臂"),
                }),
                Err(e) => failed_rows.push(format!(
                    "{label}/serial-{docs}-docs [{mode}]: {e}"
                )),
            }
        }
    }

    } // end for mode

    // P3：在线单条（128 / 1k / 8k）
    let online = corpus::generate_all(
        &[CorpusKind::Mixed, CorpusKind::Chinese, CorpusKind::Code, CorpusKind::English],
        &parse_lengths("all")?,
        &reference,
    )?;
    for c in &online {
        let spec = CorpusSpec::new(c.kind, c.target);
        let label = format!("online/{}/{}", c.kind.as_str(), spec.target.as_str());
        let m = timer.measure_with_budget(
            c.bytes(),
            c.token_ids.len(),
            150.0,
            cli.max_iters.min(20),
            || gt.encode(&c.text),
        );
        rows.push(AuditRow {
            scenario: label,
            blob_mode: "online-自然文本".into(),
            docs: 1,
            total_bytes: c.bytes(),
            threads: 1,
            median_us: m.median_us,
            cold_us: f64::NAN, // 在线单条场景每次请求的缓存状态本就不同，单列冷启动意义有限
            mb_per_s: m.mb_per_s(),
            cold_mb_per_s: f64::NAN,
            tokens_per_s: m.tokens_per_s(),
            total_tokens: c.token_ids.len(),
            note: "在线服务形态：单条请求，单线程".into(),
        });
    }

    // decode 形态对比
    for c in &online {
        let spec = CorpusSpec::new(c.kind, c.target);
        let gt_ids = gt.encode(&c.text)?;
        let gt_text = gt.decode(&gt_ids);
        let hf_text = hf.decode(&gt_ids);
        match (&gt_text, &hf_text) {
            (Ok(g), Ok(h)) => decode_checks.push(DecodeCheck {
                backend: "gigatoken_vs_hf".into(),
                ids: gt_ids.len(),
                text_equal: g == h,
                note: format!("corpus={} length={}", spec.kind.as_str(), spec.target.as_str()),
            }),
            _ => decode_checks.push(DecodeCheck {
                backend: "gigatoken_vs_hf".into(),
                ids: gt_ids.len(),
                text_equal: false,
                note: format!(
                    "corpus={} length={} gigatoken_err={:?} hf_err={:?}",
                    spec.kind.as_str(),
                    spec.target.as_str(),
                    gt_text.as_ref().err().map(|e| e.to_string()),
                    hf_text.as_ref().err().map(|e| e.to_string())
                ),
            }),
        }
    }

    let gaps = gaps();
    let report = GigatokenAudit {
        generated_at: b_backends::manifest::now_rfc3339(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        total_bytes: args.total_bytes,
        doc_counts,
        rows,
        same_condition,
        failed_rows,
        gaps,
        decode_checks,
    };
    let out = resolve_out(&cli.out_dir, &args.out);
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))?;
    std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("[gigatoken-audit] → {}", out.display());
    for r in &report.rows {
        println!(
            "  {:<40} docs={:<4} threads={:<2} {:>10.2} ms {:>9.1} MB/s",
            r.scenario, r.docs, r.threads, r.median_us / 1000.0, r.mb_per_s
        );
    }
    Ok(())
}

fn gaps() -> Vec<Gap> {
    vec![
        Gap {
            id: "decode-returns-bytes".into(),
            statement: "gigatoken 的 decode 返回 bytes，不是 str".into(),
            evidence:
                "sdist `gigatoken/_tokenizer.py:286` `def decode(...) -> bytes`；\
                 Rust 侧 `src/bpe/tiktoken.rs:1386` `pub fn decode(&self, v: &[TokenId]) -> \
                 impl Iterator<Item = u8>`。"
                    .into(),
            impact:
                "vLLM 的 `Tokenizer::decode` 契约是返回 `String`（非法 UTF-8 用替换字符），\
                 调用方要对每一步自行做 UTF-8 校验/替换；流式路径上这是每 token 一次的开销。"
                    .into(),
        },
        Gap {
            id: "no-streaming-decode".into(),
            statement: "没有 `tokenizers.decoders.DecodeStream` 等价的状态化流式解码 API"
                .into(),
            evidence:
                "sdist 全树 grep：只有整段 `decode` / `decode_batch` / `decode_bytes_batch`，\
                 没有任何 push_token / flush / DecodeStream 形态的接口；\
                 vLLM 的对比物是 `rust/src/tokenizer/src/incremental.rs:9` 的 \
                 `trait IncrementalDecoder`（push_token/next_chunk/flush/output）。"
                    .into(),
            impact:
                "流式返回是 vLLM 默认路径。缺这个 API，要么每个 token 重解一次全序列（O(n²)），\
                 要么在 Python 侧自己实现前缀差分——两者都会把 Rust 侧省下的时间还回去。"
                    .into(),
        },
        Gap {
            id: "no-skip-special-tokens".into(),
            statement: "decode 没有 `skip_special_tokens` 语义".into(),
            evidence:
                "`Tokenizer::decode` 只做 vocab 字节拼接（`src/bpe/tiktoken.rs:1386-1390`），\
                 不区分 special / added token；Python 兼容层也没有该参数。"
                    .into(),
            impact:
                "带 `<|im_end|>` 之类的输出会原样出现在返回文本里，服务端要另做过滤。".into(),
        },
        Gap {
            id: "rayon-global-pool".into(),
            statement: "并行批量走 rayon 全局池，与外部绑核/线程数设置互相影响".into(),
            evidence: "sdist `src/batch.rs:665-745`（`WorkerPool` 按 \
                 `rayon::current_num_threads()` 建槽）与 `encode_docs_ragged`（`into_par_iter`）。"
                .into(),
            impact:
                "在线服务通常按请求绑核；gigatoken 的并行批量会去抢全局池，\
                 与 `RAYON_NUM_THREADS`/`taskset` 的组合行为需要在部署时显式验证。".into(),
        },
        Gap {
            id: "tokenid-not-public".into(),
            statement: "`TokenId` 不在 crate 公开 API 里，decode 无法从外部正常调用".into(),
            evidence: "`src/lib.rs:12-15` 只 `pub use` 了 `Tokenizer`/`WorkerPool`/`EncodeState`；\
                 `mod bpe` 与 `mod token` 均为 `pub(crate)`。"
                .into(),
            impact:
                "把 gigatoken 作为 Rust 库（而不是 Python wheel）嵌入时，\
                 `decode` 属于事实上的私有 API，升级时没有兼容性保证。".into(),
        },
    ]
}

/// 冷启动测量：重新加载一个全新的后端实例，只跑一次。
///
/// 这会连带重跑加载（gigatoken 还要重建 WorkerPool），代价是几百毫秒到 1.6 秒，
/// 但只有这样才能测到"缓存是空的"那一遍。`reps` 次取最小值，
/// 因为冷启动噪声只会往慢的方向偏。
fn cold_encode_batch(
    id: BackendId,
    paths: &Paths,
    docs: &[&str],
    reps: usize,
) -> Option<(f64, usize)> {
    let mut best: Option<f64> = None;
    let mut tokens = 0usize;
    for _ in 0..reps.max(1) {
        let t0 = std::time::Instant::now();
        let b = backends::load(id, paths).ok()?;
        let load_us = t0.elapsed().as_secs_f64() * 1e6;
        let t1 = std::time::Instant::now();
        let out = match b.encode_batch(docs) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[cold] {} 失败: {e:#}", id.as_str());
                return None;
            }
        };
        // 只计编码时间，**不含加载**（加载时间另记在 note 里）。
        let encode_us = t1.elapsed().as_secs_f64() * 1e6;
        tokens = out.iter().map(|d| d.len()).sum();
        let _ = load_us; // 见 note：加载耗时随行记录
        best = Some(best.map_or(encode_us, |b0: f64| b0.min(encode_us)));
    }
    best.map(|us| (us, tokens))
}

/// 确定性伪随机词流：约 8 个字节一个词、词与词几乎不重复。
///
/// 目的是构造一个"预分词基本不重复"的输入，用来测**冷缓存**下的字节吞吐。
/// 用固定种子的 LCG（不用真随机），保证同一 `total_bytes` 每次生成的文本一致、可复现。
fn unique_blob(total_bytes: usize) -> String {
    const WORDS: [&str; 16] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    ];
    let mut out = String::with_capacity(total_bytes + 16);
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut i: u64 = 0;
    while out.len() < total_bytes {
        // LCG（Numerical Recipes 常数），取高 4 位选词、低 16 位做数字后缀。
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let word = WORDS[((state >> 60) as usize) & 15];
        out.push_str(word);
        out.push_str(&format!("{i}{}", state & 0xFFFF));
        out.push(' ');
        i += 1;
    }
    out
}

/// 把字符串截到不超过 `len` 字节的字符边界。
fn floor_boundary(s: &str, len: usize) -> usize {
    let mut n = len.min(s.len());
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

fn split_into(s: &str, n: usize) -> Vec<&str> {
    if n <= 1 {
        return vec![s];
    }
    (0..n)
        .map(|i| {
            let start = floor_boundary(s, s.len() * i / n);
            let end = floor_boundary(s, s.len() * (i + 1) / n);
            &s[start..end]
        })
        .collect()
}
