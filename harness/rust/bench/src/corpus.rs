// SPDX-License-Identifier: Apache-2.0
//! 确定性语料生成。
//!
//! 上游 bench 的做法是 `SAMPLE_TEXT.repeat(32)`（同一段模板文本反复拼接），
//! 既没有长度控制，也只有一种语料。这里改成：
//!
//! 1. 四种语料（中文 / 英文 / 代码 / 中英混合），每种由 6 个**不同**段落组成；
//! 2. 目标长度按**参考后端的 token 数**对齐（128 / 1024 / 8192），
//!    通过「循环拼接段落 → 字符级二分截断」逼近，误差 ≤1%；
//! 3. 生成结果与参考 token id 一起返回，全部后端共用同一份文本，
//!    保证跨后端比较是「同条件」。
//!
//! 语料的 sha256 会写进 manifest，任何一次改动都能被发现。

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

/// 语料种类。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CorpusKind {
    /// 纯中文（含标点、数字、专有名词）。
    Chinese,
    /// 纯英文（散文 + 技术描述）。
    English,
    /// 代码（Rust/Python/JSON 混合，含注释）。
    Code,
    /// 中英混合 + 少量结构标记（接近 chat 模板渲染后的形态）。
    Mixed,
}

impl CorpusKind {
    pub const ALL: [CorpusKind; 4] = [
        CorpusKind::Chinese,
        CorpusKind::English,
        CorpusKind::Code,
        CorpusKind::Mixed,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            CorpusKind::Chinese => "zh",
            CorpusKind::English => "en",
            CorpusKind::Code => "code",
            CorpusKind::Mixed => "mixed",
        }
    }

    /// 该语料的段落池（6 段，长度量级相近但内容不同）。
    fn paragraphs(self) -> &'static [&'static str] {
        match self {
            CorpusKind::Chinese => ZH_PARAGRAPHS,
            CorpusKind::English => EN_PARAGRAPHS,
            CorpusKind::Code => CODE_PARAGRAPHS,
            CorpusKind::Mixed => MIXED_PARAGRAPHS,
        }
    }
}

impl std::fmt::Display for CorpusKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for CorpusKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "zh" | "chinese" | "cn" => Ok(CorpusKind::Chinese),
            "en" | "english" => Ok(CorpusKind::English),
            "code" | "rs" => Ok(CorpusKind::Code),
            "mixed" | "mix" => Ok(CorpusKind::Mixed),
            other => Err(anyhow!(
                "未知语料种类 {other:?}（可选 zh/en/code/mixed）"
            )),
        }
    }
}

/// 目标长度档位（单位为**参考后端的 token 数**，不是字节数）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LengthTarget {
    Tok128,
    Tok1k,
    Tok8k,
    /// 由命令行显式给出的任意 token 数。
    Custom(usize),
}

impl LengthTarget {
    pub fn tokens(self) -> usize {
        match self {
            LengthTarget::Tok128 => 128,
            LengthTarget::Tok1k => 1024,
            LengthTarget::Tok8k => 8192,
            LengthTarget::Custom(n) => n,
        }
    }

    pub fn as_str(self) -> String {
        match self {
            LengthTarget::Tok128 => "128".to_string(),
            LengthTarget::Tok1k => "1k".to_string(),
            LengthTarget::Tok8k => "8k".to_string(),
            LengthTarget::Custom(n) => format!("{n}"),
        }
    }
}

/// 一份生成好的语料。
#[derive(Clone, Debug)]
pub struct Corpus {
    pub kind: CorpusKind,
    pub target: LengthTarget,
    pub text: String,
    /// 同一段落池的**变体**：段落顺序不同、token 数相近。
    ///
    /// 为什么必须有这个：gigatoken 在 tokenizer 内部维护 pretoken cache
    /// （`src/bpe/pretoken_cache.rs`），对**同一个字符串**反复编码会全命中缓存，
    /// 测出来的字节吞吐会显著虚高。用同一段落池的不同排列循环喂进去，
    /// 既保留"自然语言反复复用常见 pretoken"的现实（缓存仍会命中一部分），
    /// 又不会退化成"把同一份输入缓存着复读"。
    pub variants: Vec<String>,
    /// 参考后端（HF `tokenizers`）对该文本的编码结果。
    pub token_ids: Vec<u32>,
    pub source_sha256: String,
}

impl Corpus {
    pub fn bytes(&self) -> usize {
        self.text.len()
    }

    pub fn charset_per_token(&self) -> f64 {
        self.text.chars().count() as f64 / self.token_ids.len().max(1) as f64
    }

    /// 变体 0 是主文本；第 i 次调用返回第 i 个变体（循环）。
    pub fn variant(&self, i: usize) -> &str {
        if self.variants.is_empty() {
            return self.text.as_str();
        }
        &self.variants[i % self.variants.len()]
    }

    /// 变体集合的 token 数范围（写进 manifest，说明变体长度是可控的）。
    pub fn variant_token_range(&self, reference: &HuggingFaceTokenizer) -> (usize, usize) {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for v in &self.variants {
            if let Ok(ids) = reference.encode(v.as_str(), false) {
                lo = lo.min(ids.len());
                hi = hi.max(ids.len());
            }
        }
        if lo == usize::MAX { (0, 0) } else { (lo, hi) }
    }

    /// 与目标 token 数的相对误差。
    pub fn length_error_ratio(&self) -> f64 {
        let target = self.target.tokens() as f64;
        (self.token_ids.len() as f64 - target).abs() / target
    }

    /// 同一份语料按**预先算好的 token 切片**构成的小批量。
    pub fn batches(&self, docs: usize) -> Vec<&str> {
        // 每个文档取语料的等分切片，避免每个文档都是同一份文本
        // （同一文本会被 CPU 分支预测/缓存吃掉，批量结果虚高）。
        let total = self.text.len();
        let n = docs.max(1);
        (0..n)
            .map(|i| {
                let start = total * i / n;
                let end = total * (i + 1) / n;
                crate::corpus::floor_char_boundary(&self.text, start, end)
            })
            .collect()
    }
}

/// 把 `[start, end)` 收缩到合法的 UTF-8 字符边界并返回切片。
pub fn floor_char_boundary(text: &str, start: usize, end: usize) -> &str {
    let mut s = start.min(text.len());
    let mut e = end.min(text.len());
    while s < text.len() && !text.is_char_boundary(s) {
        s += 1;
    }
    while e > s && !text.is_char_boundary(e) {
        e -= 1;
    }
    &text[s..e]
}

/// 语料规格：种类 + 目标长度。
#[derive(Clone, Copy, Debug)]
pub struct CorpusSpec {
    pub kind: CorpusKind,
    pub target: LengthTarget,
}

impl CorpusSpec {
    pub fn new(kind: CorpusKind, target: LengthTarget) -> Self {
        Self { kind, target }
    }

    pub fn id(&self) -> String {
        format!("{}-{}", self.kind.as_str(), self.target.as_str())
    }
}

/// 按段落池的偏移 `shift` 拼出文本，并截断到目标 token 数。
///
/// 返回 `(text, ids)`。**不涉及变体**：变体由 `generate()` 用不同 shift 调它拼出来，
/// 这样不会递归。
fn build_text(
    kind: CorpusKind,
    target: usize,
    shift: usize,
    reference: &HuggingFaceTokenizer,
) -> Result<(String, Vec<u32>)> {
    let paragraphs = kind.paragraphs();
    if paragraphs.is_empty() {
        return Err(anyhow!("语料段落池为空: {kind}"));
    }

    // 第 1 步：按段落循环拼接，直到参考 tokenizer 认为已超过目标。
    let mut text = String::with_capacity(target * 4);
    let mut idx = shift;
    let mut ids = reference
        .encode(text.as_str(), false)
        .context("参考后端编码空串失败")?;
    let mut guard = 0usize;
    while ids.len() < target {
        text.push_str(paragraphs[idx % paragraphs.len()]);
        text.push('\n');
        idx += 1;
        ids = reference.encode(text.as_str(), false)?;
        guard += 1;
        if guard > 100_000 {
            return Err(anyhow!("语料拼接超过 100k 段仍未达到目标长度"));
        }
    }

    // 第 2 步：二分找到「token 数 ≤ target」的最长字符前缀。
    let char_len = text.chars().count();
    let mut lo = 0usize;
    let mut hi = char_len;
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        let byte_end = char_to_byte(&text, mid);
        let n = reference.encode(&text[..byte_end], false)?.len();
        if n <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    text.truncate(char_to_byte(&text, lo));
    // 去掉可能被截断在中间的空行，避免最后一个 token 是半个词。
    while text.ends_with('\n') || text.ends_with(' ') {
        text.pop();
    }
    let ids = reference.encode(text.as_str(), false)?;
    Ok((text, ids))
}

/// 生成一份语料（含 1 个主文本 + 最多 3 个同段落池的排列变体）。
pub fn generate(
    spec: CorpusSpec,
    reference: &HuggingFaceTokenizer,
) -> Result<Corpus> {
    let target = spec.target.tokens();
    let (text, token_ids) = build_text(spec.kind, target, 0, reference)?;

    let mut variants = vec![text.clone()];
    for variant_shift in 1..=3 {
        let (variant_text, variant_ids) = build_text(spec.kind, target, variant_shift, reference)?;
        let ratio = variant_ids.len() as f64 / token_ids.len().max(1) as f64;
        if (0.8..=1.25).contains(&ratio) {
            variants.push(variant_text);
        }
    }

    let corpus = Corpus {
        kind: spec.kind,
        target: spec.target,
        source_sha256: sha256_hex(text.as_bytes()),
        text,
        variants,
        token_ids,
    };

    if corpus.length_error_ratio() > 0.01 {
        return Err(anyhow!(
            "语料 {} 长度误差 {:.3}% 超过 1% 阈值（实际 {} tokens，目标 {}）",
            spec.id(),
            corpus.length_error_ratio() * 100.0,
            corpus.token_ids.len(),
            target
        ));
    }
    Ok(corpus)
}

/// 生成全部「语料 × 长度」组合。
pub fn generate_all(
    kinds: &[CorpusKind],
    targets: &[LengthTarget],
    reference: &HuggingFaceTokenizer,
) -> Result<Vec<Corpus>> {
    let mut out = Vec::with_capacity(kinds.len() * targets.len());
    for &kind in kinds {
        for &target in targets {
            out.push(generate(CorpusSpec::new(kind, target), reference)?);
        }
    }
    Ok(out)
}

/// 把 token 边界附近的富文本对齐输出成一行摘要（写报告用）。
pub fn summarize(corpora: &[Corpus]) -> String {
    let mut s = String::new();
    for c in corpora {
        s.push_str(&format!(
            "{:<6} {:>5} tokens  {:>7} bytes  {:.2} chars/token  sha256={}\n",
            c.kind.as_str(),
            c.token_ids.len(),
            c.bytes(),
            c.charset_per_token(),
            &c.source_sha256[..16],
        ));
    }
    s
}

/// 读取本地 `tokenizer.json` 作为参考后端。
pub fn load_reference(path: &Path) -> Result<HuggingFaceTokenizer> {
    HuggingFaceTokenizer::new_hf(path)
        .map_err(|e| anyhow!("加载参考后端 {} 失败: {e}", path.display()))
}

fn char_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// 纯 Rust 实现的 sha256（避免引额外依赖）。
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// 语料池：每种 6 段，内容互不相同（写入报告时可核对）。
// ---------------------------------------------------------------------------

const ZH_PARAGRAPHS: &[&str] = &[
    "服务在一次批量推理任务中于凌晨两点出现了首包延迟突增，运维同学先看的是前端进程的分词耗时，\
     因为这一段代码在请求进入调度器之前就要跑完，它不属于引擎核心，也不在任何一张卡上。",
    "我们把同一条提示词分别送进三个不同的分词实现，统计出来的 token 个数居然不完全一样，\
     差异集中在中文标点、连续空格以及那些看起来像特殊标记的字符串上，这直接影响计费口径。",
    "在流式返回的场景里，解码器必须逐 token 吐出可读文本，遇到被切开的 UTF-8 多字节字符时\
     要先把字节留在缓冲区里，等到下一个 token 补齐后再一起输出，否则前端会看到乱码方块。",
    "压测报告显示，当提示词长度从一百二十八增长到八千时，单条编码的耗时并不是线性增长，\
     预分词阶段的正则匹配占比上升得更快，而合并阶段的哈希查找反而相对稳定。",
    "评审时有同事提出直接换成用其他语言写的前端，理由是启动更快、内存更省；\
     我们要求他给出同条件的数据：同样的输入、同样的线程数、同样的硬件，并且逐 token 对齐编码结果。",
    "最终结论写在文档第三小节：分词器不是瓶颈，但它是可以预测的固定成本，\
     在首包延迟里占比通常不到百分之五，除非你的请求里塞满了稀有字符和超长空白。",
];

const EN_PARAGRAPHS: &[&str] = &[
    "The tokenizer front end runs in the API server process, before any request reaches the \
     scheduler, so its latency shows up directly in time to first token and never touches a GPU.",
    "We compared three implementations on the same prompt and the token counts disagreed by a \
     fraction of a percent; the differences clustered around punctuation, leading whitespace, \
     and strings that merely resemble registered special tokens.",
    "Streaming decode has to emit readable text one token at a time. Whenever a multi-byte \
     character is split across two tokens, the decoder must hold the partial bytes in a buffer \
     until the next token completes the sequence.",
    "Under load the encoding cost grows sub-linearly with prompt length: the pre-tokenization \
     regex dominates the short prompts while the merge loop stays roughly constant per byte.",
    "A colleague suggested replacing the front end with a different language runtime because it \
     starts faster and uses less memory, so we asked for numbers measured under identical \
     conditions: same input, same thread count, same machine, identical token ids.",
    "The conclusion is deliberately narrow. The tokenizer is not the bottleneck, but it is a \
     predictable fixed cost that rarely exceeds five percent of time to first token unless the \
     prompt is full of rare characters or very long runs of whitespace.",
];

const CODE_PARAGRAPHS: &[&str] = &[
    "fn decode_stream(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Result<String> {\n    \
     let mut out = String::new();\n    for id in ids {\n        \
     let piece = tokenizer.id_to_token(*id).unwrap_or_default();\n        \
     out.push_str(&piece);\n    }\n    Ok(out)\n}\n",
    "def render_messages(messages, tools=None, template=None):\n    \
     # NOTE: this runs in the front-end process, before the engine core\n    \
     rendered = template.render(messages=messages, tools=tools or [])\n    \
     return rendered if isinstance(rendered, str) else rendered[0]\n",
    "{\"model\":\"qwen3-0.6b\",\"max_tokens\":512,\"temperature\":0.7,\"stream\":true,\
     \"stop\":[\"<|im_end|>\"],\"messages\":[{\"role\":\"user\",\"content\":\"总结这段日志\"}]}\n",
    "SELECT tokenizer_id, count(*) AS requests, avg(elapsed_us) AS avg_us\n  \
     FROM frontend_metrics\n WHERE scope = 'tokenizer: encode'\n   \
     GROUP BY tokenizer_id\n ORDER BY avg_us DESC;\n",
    "const MAX_CHARS_PER_TOKEN: usize = 128;\n\n\
     pub fn assert_token_budget(text: &str, max_tokens: usize) -> bool {\n    \
     text.chars().count() <= max_tokens * MAX_CHARS_PER_TOKEN\n}\n",
    "if __name__ == \"__main__\":\n    \
     parser = argparse.ArgumentParser(description=\"backend matrix harness\")\n    \
     parser.add_argument(\"--threads\", type=int, default=1)\n    \
     parser.add_argument(\"--json-out\", type=Path)\n    args = parser.parse_args()\n",
];

const MIXED_PARAGRAPHS: &[&str] = &[
    "<|im_start|>system\n你是 Qwen3 的助手，回答尽量简短。\
     <|im_end|>\n<|im_start|>user\n请用中英混合总结以下需求：\
     The service should stop cleanly at EOS, avoid leaking the next template turn.\
     \n<|im_end|>\n<|im_start|>assistant\n",
    "输入：4 个并发请求，20480 个 prompt tokens，生成 256 个 token；\
     输出：首包延迟 180 ms，其中分词占 4.2 ms（front-end process, tok/s=4900）。\
     Verify the numbers before quoting them.\n",
    "工具调用示例：<|tool_calls_section_begin|>{\"name\":\"summarize\",\
     \"arguments\":{\"style\":\"brief\",\"lang\":\"zh\"}}<|tool_calls_section_end|>\
     请把这段 JSON 原样保留，不要改写字段名。\n",
    "日志片段 2026-09-24T02:14:07Z WARN frontend tokenizer: encode 8192-token prompt took 21.6 ms; \
     结论是这一条请求的分词耗时约为首包延迟的 3%，属于可接受范围（阈值 5%）。\n",
    "压力测试结论（待复核）：throughput 12.4k tok/s at 8 threads, memory +38 MB RSS, \
     但 riptoken 与 tiktoken-rs 的输出逐 id 一致，这一点在本机与容器内都验证过。\n",
    "最后的建议：在线路径保持单线程（1 thread per request），批量离线路径再开 rayon；\
     keep bytes-per-second claims in the offline bucket and never mix them with TTFT.\n",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn floor_char_boundary_is_utf8_safe() {
        let s = "中文abc";
        let slice = floor_char_boundary(s, 1, 4);
        assert!(s.contains(slice));
        assert!(!slice.is_empty());
    }
}
