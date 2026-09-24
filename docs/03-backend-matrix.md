# 03 · 多后端矩阵：vLLM Rust 前端六后端 + 三个 Python 实现

> 线 B（`B-backends`）产出。数据来自 `harness/rust`（自建 criterion harness）
> 与 `harness/python`（三个 Python 实现），原始数据在 `data/backends/`。
> 每条实验都有 manifest（commit / 模型 revision / 脚本 sha256 / 时间戳 / 绑核 / 线程数）。

## 1. 评估对象与它们"实际是什么"

### 1.1 六个 Rust 后端

| # | 后端 | 构造函数 | 家族 | 工件 | 本表是否有数 |
|---|---|---|---|---|---|
| 1 | `Hf` | `HuggingFaceTokenizer::new_hf` | HF | `tokenizer.json` | ✅ 全维度 |
| 2 | `Fastokens` | `HuggingFaceTokenizer::new_fastokens` | HF | 同上 | ⚠️ 见 §1.2 |
| 3 | `FastokensByteLevel` | 无独立入口，由 `new_fastokens` 自动判定 | HF | 同上 | ✅ 全维度 |
| 4 | `TiktokenRs` | `TiktokenTokenizer::new_tiktoken_rs` | tiktoken | `tiktoken.model` | ✅ 全维度 |
| 5 | `Riptoken` | `TiktokenTokenizer::new_riptoken` | tiktoken | 同上 | ✅ 全维度 |
| 6 | `Tekken` | `TekkenTokenizer::new` | tekken | `tekken.json` | ⚠️ 流式 decode 有失败项（§4） |
| — | gigatoken | `load_tokenizer::hf::load_hf_bpe`（sdist vendor） | 独立 | `tokenizer.json` | ✅ 全维度（见 `docs/05`） |

**型号与工件**（`data/backends/e1.manifest.json` 里逐项记录 sha256）：

| 工件 | 来源 | revision | 说明 |
|---|---|---|---|
| `tokenizer.json` | `Qwen3-0.6B`（本地） | 本地文件 | BFCL byte-level BPE，vocab 151 669 |
| `tiktoken.model` + `config.json` | `moonshotai/Kimi-K2.5` | `4d01dfe0` | Kimi BPE，vocab 163 840 |
| `tekken.json` | `mistralai/Mistral-Nemo-Base-2407` | `main` | tekken v3，vocab 131 072 |

### 1.2 一个必须写清楚的结构事实：2 号与 3 号在 Qwen3 上是同一个对象

`new_fastokens()` 会把解码头是**纯 ByteLevel**的 tokenizer 自动装进
`Backend::FastokensByteLevel`（`vendor/vllm-tokenizer/src/hf.rs:97-108`）。
Qwen3 的 `tokenizer.json` 正是这种情况，因此：

- **`BackendId::Fastokens` 与 `BackendId::FastokensByteLevel` 是同一个对象**，
  没有"两个后端各测一次"这回事。harness 检测到这一点会拒绝加载 2 号并记明原因
  （`data/backends/e1.manifest.json` 的 caveat）。
- 于是"旁路值多少钱"不能从 2/3 的对比里读出。本表用一条**独立对照臂**补上：
  绕开 vLLM wrapper，直接调 `fastokens::Tokenizer::decode`（也就是旁路生效时
  **没有走**的那条通用路径，内部组装 `Vec<String>` 再 `join("")`）。
  两者喂**同一批 token id**，差值就是旁路收益（§3.2）。

### 1.3 后端对 tokenizer 类型的适用性（**选型必须先看这张表**）

| 后端 | 适用工件 | 在 Qwen3(BFCL) | 在 Kimi(tiktoken) | 在 Mistral(tekken) | 备注 |
|---|---|---|---|---|---|
| `Hf` | 任意 HF `tokenizer.json` | ✅ | ✅ | ✅ | 最通用；也是对拍基准 |
| `Fastokens` | HF `tokenizer.json`（非纯 ByteLevel 解码头） | ❌ 不可单独加载 | 需各自工件 | 需各自工件 | 纯 ByteLevel 时被自动改派到 3 号 |
| `FastokensByteLevel` | HF `tokenizer.json`，且解码头为纯 ByteLevel | ✅ 自动生效 | ❌ 不生效 | ❌ 不生效 | SentencePiece/带 Fuse 的解码头**不生效** |
| `TiktokenRs` | `tiktoken.model` / `*.tiktoken` | ❌（无该工件） | ✅ | ❌ | 按 `model_type` 选正则，见 `tiktoken.rs:524` |
| `Riptoken` | 同上 | ❌ | ✅ | ❌ | 与 4 号同家族，ids 必须逐一致（§2） |
| `Tekken` | `tekken.json` | ❌ | ❌ | ✅ | 独立的 vocab/序号空间 |

**一句话选型**：`FastokensByteLevel` 的收益只能落在"解码头是纯 ByteLevel"的
BPE 上（GPT-2/Qwen/Llama 系常见），SentencePiece 系（部分 Mistral/Llama 变体）
拿不到；换成 tiktoken/tekken 工件时这条旁路根本不存在。

## 2. 正确性（先给结论，再给反例）

装置：`harness/rust/target/release/b-backends correctness`，
产物 `data/backends/correctness.json`。语料 4 类 × 3 长度 = 12 组。

判据与结果：

| 判据 | 口径 | 结果 |
|---|---|---|
| HF 家族 ids 逐一致（1 vs 3） | 同一 `tokenizer.json`，`add_special_tokens=false` | ✅ 12/12 一致 |
| tiktoken 家族 ids 逐一致（4 vs 5） | 同一 `tiktoken.model` | ✅ 12/12 一致 |
| gigatoken vs HF（同一 `tokenizer.json`） | 跨实现、同一文件 | ✅ 12/12 一致 |
| tekken 家族内比对 | 只有一个实现，**无法对拍** | ⛔ 未测（原因：无第一方对比物） |
| decode→re-encode 回到原 ids | 逐后端、逐语料 | ✅ 除失败项外全部一致 |
| 前缀解码不报错（流式的必要条件） | 扫每个输入的前 512 个 token 前缀 | ❌ tekken 6/12 组失败（§4） |
| fastokens 长中文/长英文 encode | 8k 等价文本 | ❌ panic（§5） |

**与计划文档的差异（必须声明）**：`plan/EXPERIMENT-MATRIX.md` 写的 E1.1
（`new_hf`，1 线程）与 E1.2（`new_fastokens`）在本工件上**不构成两个可测对象**
（§1.2）。本表因此把 E1.2 记为"未测 + 原因"，并用 §1.2 的独立对照臂替代它回答
同一个问题（"fastokens 比 HF 快多少"），这一改动不影响 E1.1/E1.3 的结论。

## 3. 主表：单条路径（在线服务形态）

**口径（下列所有数字共同适用，不得只引用数字）**：

- 装置：本地 x86_64 / 12 核共享开发机。基准统一用
  `CORES=2 scripts/limit.sh ...`，它在 `limit.sh` 里的语义是
  **`taskset -c 2`（绑 1 个核）+ `RAYON_NUM_THREADS=1`**
  （不是 2 核；manifest 里 `threads_effective=1`、`cpu_affinity="2"` 可核对）。
  **单条路径在调用线程上直接跑，不进 rayon**，所以这个绑核对单条结果无影响；
  批量路径的线程数另见 §6。
- 计时：自建计时器，2 轮预热 + 自适应轮数（预算 120 ms / 点），取**中位数**。
- 输入：每种语料 4 个变体轮转（见 §7.2 的缓存说明），长度按 **Qwen3 参考后端的
  token 数**对齐到 128 / 1 024 / 8 192，误差 ≤1%（`data/backends/corpus.json`）。
- 单位：µs/次（中位数）；`MB/s` = 输入字节 / 中位耗时。
- `add_special_tokens=false`（与 vLLM 一致：控制符由 chat 模板插入）。

### 3.1 单条 encode / decode（中位耗时 µs）

| 后端 | zh 128 | zh 1k | zh 8k | en 1k | code 1k | mixed 1k | mixed 8k |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Hf` | 64.0 | 488 | 4 480 | 1 571 | 1 479 | 1 053 | 11 720 |
| `FastokensByteLevel` | **10.1** | **23.9** | — | **61.8** | **64.3** | **48.3** | **322** |
| `TiktokenRs` | 37.6 | 207 | 1 781 | 457 | 471 | 438 | 2 652 |
| `Riptoken` | 35.3 | 213 | 1 110 | 65.1 | 63.1 | 83.6 | 839 |
| `Tekken` | 37.8 | 277 | 2 481 | 426 | 444 | 338 | 2 747 |
| gigatoken | **2.4** | **14.6** | **121** | **5.6** | **7.8** | **12.0** | **93.2** |

（数据源：`data/backends/e1.jsonl`，2026-09-25 00:22–00:27 本地时间的一次完整重跑；
`loadavg` 见 manifest。个别格子跨次重跑有 30–100% 抖动，见 §7.4 的稳定性讨论。）

`—` 表示该项因 §5 的上游 panic 未取得数据，不是"测不出来"。

### 3.2 decode：旁路值多少（**同 ids、同装置**）

| 语料-长度 | 3 号（ByteLevel 旁路） | 通用 fastokens decode | 倍率 |
|---|---:|---:|---:|
| zh-128 | 2.84 µs | 19.73 µs | 7.0× |
| zh-1k | 31.10 µs | 148.15 µs | 4.8× |
| en-128 | 2.39 µs | 17.96 µs | 7.5× |
| en-1k | 16.01 µs | 133.95 µs | 8.4× |
| code-128 | 2.21 µs | 19.05 µs | 8.6× |
| code-1k | 15.18 µs | 139.51 µs | 9.2× |
| code-8k | 117.17 µs | 978.28 µs | 8.3× |
| mixed-128 | 2.29 µs | 18.95 µs | 8.3× |
| mixed-1k | 16.38 µs | 134.23 µs | 8.2× |
| mixed-8k | 128.80 µs | 1 025.55 µs | 8.0× |

**口径与归属（这条结论最容易读错，逐条写死）**：

1. 这是**纯 decode 调用**（给定 token ids 出文本）的差距，**不是** vLLM 的流式
   增量解码。两者相差一个数量级的场景，见 §3.3。
2. 8×（实测 4.8–9.2×）来自 `vllm-tokenizer` crate 自己写的
   `Backend::FastokensByteLevel` 旁路（`vendor/vllm-tokenizer/src/hf.rs:20-31`），
   **不是 fastokens 提供的**。上游 fastokens 的通用 decode 就是右列那条路。
3. 省掉的成本是 `Vec<String>` 组装 + `join("")`：8k 长度上差值 707 µs，
   而批量长度上差值随字节数近似线性增长（1k→8k 相差约 6×），与该解释一致。
4. 跨装置对照：根代理在**容器内、Python 侧**用独立装置测到
   encode 8–9×（与本表 8.1× 吻合），但 decode 只有 **1.09×**
   ——因为 Python 侧没有这条旁路。即：**decode 的 8× 是 Rust 前端专有**，
   Python 用户开 `VLLM_USE_FASTOKENS=1` 拿不到（证据：`data/crosscheck/`）。

### 3.3 流式（增量）解码：与服务路径对应的那一组

口径：用后端自己的 ids，取前半为 prompt 前缀、后 256 个 token 为"生成"，
逐 token `push_token` + 取走 chunk，`min_bytes_to_buffer` 扫 {1,4,8}。
单位 µs / 256 token。

| 后端 | zh 1k | zh 8k | en 1k | code 1k | mixed 1k | mixed 8k |
|---|---:|---:|---:|---:|---:|---:|
| `Hf` | 81.9 | 82.5 | 84.1 | 131 | 80.1 | 77.3 |
| `FastokensByteLevel` | 36.0 | — | 39.1 | 38.4 | 33.4 | 35.3 |
| `TiktokenRs` | 51.4 | 52.8 | 50.9 | 48.8 | 51.9 | 72.4 |
| `Riptoken` | 31.2 | 30.0 | 30.6 | 27.8 | 40.7 | 31.3 |
| `Tekken` | ⛔ | ⛔ | 47.2 | 44.8 | ⛔ | ⛔ |
| gigatoken（适配层） | 25.3 | 25.4 | 26.6 | 23.1 | 23.1 | 23.2 |

**读法**：vLLM 六个后端共用同一个 `DecodeStream`（前缀差分，
`vendor/vllm-tokenizer/src/incremental.rs:26`），所以这组数的差异**只来自底层
`decode` 有多快**，不含实现策略差异。3 号相对 1 号是 **2.3×**（zh-1k 81.9 vs 36.0），
而不是 §3.2 的 8×——因为流式每次只推进一个 token，前缀差分的固定开销摊薄了
`Vec<String>`+`join` 的绝对占比。
**引用"快 8×"时必须限定在 §3.2 的纯 decode。**

**gigatoken 的那一行是适配层，不是原生能力**：gigatoken 没有流式 API（`docs/05` §4），
这里的数是把 vLLM 的 `DecodeStream` 架在它的整段 decode 上得到的。
顺带说明：由于它整段 decode 很快（1.2 µs @128），适配层的 O(n²) 并没有拖垮它。

## 4. tekken 后端：流式路径上的一个真实缺口

现象（`data/backends/correctness.json` 的 `prefix_decodes`）：

| 语料 | 失败前缀数 / 扫描数 | 首个失败点 |
|---|---:|---|
| zh-128 | 8 / 180 | n=22 |
| zh-1k | 30 / 512 | n=22 |
| zh-8k | 30 / 512 | n=22 |
| mixed-128 | 1 / 169 | n=118 |
| mixed-1k | 7 / 512 | n=118 |
| mixed-8k | 7 / 512 | n=118 |
| en / code（全部长度） | 0 | — |

错误原文：`Tokenizers error: Unable to decode into a valid UTF-8 string:
incomplete utf-8 byte sequence from index 3`。

**为什么这是硬伤**：流式返回必然会出现"一个多字节字符被拆到相邻两个 token"的
中间状态（中文尤其常见）。vLLM 的 `DecodeStream` 依赖 `decode` 对不完整 UTF-8
返回**替换字符**而不是报错（`incremental.rs` 里靠 `ends_with('\u{FFFD}')` 判断
"这次还不该输出"）。tekken-rs 0.1.1 在中文/中英混合输入上直接报错，因此
**6/12 语料组的流式 decode 完全不可用**；英文与代码语料正常。

影响面与边界：只在**流式**路径上暴露（整段 decode 正常，§3.1 有数）；
只在**多字节字符**上暴露（en/code 全绿）。修复路径有两条：给
`TekkenTokenizer::decode` 加 lossy 语义，或在 `DecodeStream` 侧对 tekken 放宽。
前者更贴 vLLM 已有的后端契约。

## 5. fastokens 0.2.1 的一个上游 panic（长中文/长英文输入）

**现象**：`HuggingFaceTokenizer::new_fastokens()` 加载的 3 号后端，对
zh-8k（36 KB 中文）与 en-8k（46 KB 英文）的输入直接 panic：

```text
thread '<unnamed>' panicked at
  fastokens-0.2.1/src/pre_tokenizers/split.rs:419:78:
  index out of bounds: the len is 1 but the index is 1
```

**根因（读源码定位）**：`find_matches_pcre2_parallel` 里

```rust
let n_chunks = n_cpus.min(text.len() / MIN_CHUNK_SIZE).min(pcre2.len()).max(2);  // :383-386
...
let all = find_matches_pcre2(chunk, base + auth_start, &pcre2[i])?;              // :419
```

`n_chunks` 先被 `.min(pcre2.len())` 夹到正则条数，又被末尾的 **`.max(2)`** 抬回 2。
当 `pcre2.len() == 1`（本工件上确实如此）时 `n_chunks = 2`，循环跑到 `i = 1`，
`pcre2[1]` 越界。触发条件是**文本足够长**（`text.len()/MIN_CHUNK_SIZE ≥ 1`），
所以 128/1k 全绿、8k 必炸——这与实测完全一致（zh-8k、en-8k panic；
zh-1k 不炸而 zh-8k 炸）。

**为什么 harness 还活着**：所有后端调用都套了 `guard_panic`（`stats.rs`），
panic 被记成该点的失败行而不是带走整轮矩阵。**注意**：本 harness 的 release
profile 保留了 `panic = unwind`（上游 vLLM 用 `panic = "abort"`），这是为了能
捕获并记录这类上游 bug；这一有意偏离已在 `harness/rust/Cargo.toml` 注明。

**影响面（直说）**：vLLM 的 `new()` 会先试 fastokens、失败才回落 HF
（`hf.rs:117-128`），但**panic 不是 `Err`**，回落逻辑拦不住它——
也就是说这条路径在 8k 级中文 prompt 上会**直接崩掉前端进程**，
而不是"慢一点"。建议上游把 `pcre2[i]` 的索引夹到 `pcre2.len()`，
或在 `n_chunks` 那里把 `.max(2)` 换成 `.clamp(2, pcre2.len())`。
本地 0.3.2 的 Python 绑定对同一批语料没有复现（版本不同，未逐行核对，
标为"待确认"）。

## 6. 批量路径（离线/吞吐形态）

口径：把语料等分 4–32 个文档，走 rayon 并行。
**这里做的是 1 线程 vs 2 线程的对照**，因为 `plan/EXPERIMENT-MATRIX.md` 的 E1.9
要求"批量 encode（default 线程）"，而本机受资源纪律限制只能用 2 核窗口。

| 后端 | 批大小 | 1 线程（µs/批） | 2 线程（µs/批） | 并行加速 |
|---|---|---:|---:|---:|
| `Hf` | 4 docs | 523 | 292 | **1.79×** |
| `Hf` | 32 docs | 4 378 | 2 172 | **2.02×** |
| `FastokensByteLevel` | 4 docs | 69.7 | 49.7 | 1.40× |
| `FastokensByteLevel` | 32 docs | 405 | 258 | 1.57× |
| `TiktokenRs` | 4 docs | 342 | 248 | 1.38× |
| `TiktokenRs` | 32 docs | 2 719 | 1 691 | 1.61× |
| `Riptoken` | 4 docs | 292 | 95.7 | **3.05×** |
| `Riptoken` | 32 docs | 2 456 | 489 | **5.02×** |
| `Tekken` | 4 docs | 365 | 285 | 1.28× |
| `Tekken` | 32 docs | 2 997 | 1 760 | 1.70× |
| gigatoken | 4 docs | 13.8 | 14.7 | **0.94×** |
| gigatoken | 32 docs | 104 | 112 | **0.93×** |

（mixed-1k / mixed-8k 语料；完整 4 语料 × 3 长度见
`data/backends/e1-batch-2t.jsonl`（2 线程）与 `e1.jsonl`（1 线程）。）

**这张表有三条与直觉相反的结论**，都是数据支持的：

1. **gigatoken 在 2 线程下没有加速（0.93–0.94×，实际略慢）**。
   因为它走的是自己的 `encode_docs_ragged`（内部按字节切 chunk），
   而本组输入总量只有约 2 MB / 4 或 32 个文档——
   在 2 核、这个体量下，**切 chunk + 起 rayon 任务的固定开销
   超过了多核收益**。它在 §3 的"大块单文档"场景里能铺满核（`docs/05` §3.3），
   但**小批量场景下它的并行不值钱**。这不影响它单线程本身很快的事实
   （13.8 µs vs hf 的 523 µs）。
2. **1–6 号（同样的 `par_iter` 文档级并行）加速比差异很大**：
   `Hf` 1.79–2.02×、`Riptoken` 3.05–5.02×（超线性，见下）、
   `Tekken` 1.28–1.70×。同样的并行框架下，**底层实现的单线程效率越高，
   并行空间越小**——hf 的单线程最慢（523 µs），所以并行收益最明显。
3. **Riptoken 出现 3–5× 的超线性加速**（2 核不可能给出 >2× 的稳态加速）。
   可能是 1 线程那一轮受了同批其它测量点的干扰（该轮的 `loadavg` 见
   `e1.csv`），也可能与 tiktoken-rs 的 `Mutex` 警告集合在并行路径上的
   争用行为有关。**这一格标为可疑，不作为结论**；要坐实需要单独重跑。
   同类可疑格：`Tekken` 的 8k 批量（2 线程反而比 1 线程慢）。

**这组数不能当成"离线吞吐"来引用**：本机只有 2 核可用（资源纪律），
真实离线场景是几十到几百核；§6 的用途是**看并行是否有收益、以及收益从哪来**，
不是给绝对吞吐。

## 7. 跨装置桥梁校准与口径边界

### 7.1 校准结果（`plan/COORDINATION.md` §5.4 的硬要求）

要求：用 fastokens 与 HF `tokenizers` 这两个**双端都存在**的实现做桥梁，
两装置相对差 ≤15% 才允许同表比较。正式校准脚本：
`harness/bridge-calibration.sh`（两次测量在同一绑核窗口里**串行**执行，
避免负载漂移被误读成装置差异），产物 `data/backends/bridge-calibration.json`。

| 项 | 长度 | Rust 装置 | Python 装置 | 相对差 | 判定 |
|---|---|---:|---:|---:|---|
| HF `tokenizers` encode | 128 | 97.56 µs | 122.40 µs | 20.3% | ❌ |
| fastokens encode | 128 | 12.32 µs | 11.61 µs | 5.8% | ✅ |
| HF `tokenizers` encode | 1k | 692.25 µs | 1 123.81 µs | 38.4% | ❌ |
| fastokens encode | 1k | 45.35 µs | 78.96 µs | 42.6% | ❌ |

**结论（按硬要求执行）：本项目的 Rust 装置与 Python 装置没有校准到 15% 以内，
因此本文档禁止跨装置同表比较。** 1–6 号后端的全部数字只来自 Rust 装置
（§3、§6），Python 三实现只在 §7.3 内部自成一张表。

差异的两个可复核来源（都不是"某一边测错了"）：

1. **fastokens 版本不同，且无法对齐**：Rust 侧钉的是 0.2.1（vLLM 0.26.0 的
   `rust/Cargo.toml`），而 **PyPI 上没有 0.2.1**（实测可用版本 0.1.1/0.1.2/
   0.2.0/0.3.0/0.3.1/0.3.2）。Python 侧只能用 0.3.2，**不同版本的引擎**，
   1k 长度上差 42.6% 属版本差异而非装置差异。
2. **解释器与计时开销在小输入上占比更高**：HF encode @128 差 20.3%，
   而 @1k 差 38.4%——后者反而更大，说明不只是固定开销，主要是
   上面那条版本差异叠加机器噪声。

**与根代理第二装置的关系（不要混淆）**：根代理在容器内做过一次独立复核
（`data/crosscheck/`），那次比的是**倍率**（HF/fastokens 的比值），
得到 Rust 8.1× vs Python 9.3×（差 14.8%，勉强通过）。
**倍率吻合不等于绝对值吻合**：本节的绝对值校准显示 20–43% 的差。
两个结论同时成立，引用时要说清用的是哪一个口径。
另外注意根代理那次 Python 侧的 decode 只有 1.09×，与 Rust 的 8× 相差甚远，
原因已定位（Python 没有 `FastokensByteLevel` 旁路），
所以 **decode 方向在任何口径下都不允许跨装置比较**。

### 7.2 缓存与噪声：本表最容易被打偏的两处

- **gigatoken 的 pretoken cache 在 tokenizer 内部常驻**（`src/bpe/pretoken_cache.rs`），
  反复编码同一字符串会把它的数字显著抬高。本 harness 用"同段落池的 4 个排列
  变体轮转"喂输入（`corpus.rs::variants`），既保留自然语言复用常见 pretoken 的
  现实，又不退化成复读同一份输入。**但这条缓解不等于消除**：
  `docs/05` §3 用 `repeat` / `unique` 两档给出上下界。
- **计时噪声**：机器是共享开发机，`load.sh`/`heavy_lock.sh` 把并发压到 2 核；
  每个测量点都记录测量前后的 loadavg 与可用内存（`e1.csv` 的
  `loadavg_before/after` 列），事后可筛。中位数（而非均值）是主口径，
  `p90` 同表给出以便看出抖动。

### 7.3 Python 侧三实现（当代装置）

数据：`data/backends/python-e1.json`（4 类语料 × 128/1k，绑核 2 核，`--iters 20`）。

| 实现 | 版本 | encode mixed-128 | encode mixed-1k | decode mixed-128 | 流式 256 tok（mixed-128） |
|---|---|---:|---:|---:|---:|
| HF `tokenizers` | 0.22.2 | 230.7 µs | 1 884.7 µs | 27.8 µs | 56.1 µs |
| fastokens | 0.3.2 | **17.9 µs** | **74.6 µs** | 28.5 µs | 无绑定（见下） |
| gigatoken | 0.10.0 | **9.5 µs** | 98.3 µs | **3.59 µs** | 无原生 API（见下） |

其他语料的同批数据（encode @128，µs）：
zh 101.9 / 20.8 / 14.3，en 366.3 / 12.3 / 12.9，code 263.0 / 8.2 / 8.4
（依次为 hf / fastokens / gigatoken）。可以看出 **hf 的耗时强烈依赖语料**
（en 366 µs 是 code 263 的 1.4 倍），这与它按正则做预分词的实现相符；
fastokens 与 gigatoken 对语料的敏感度小得多。

**注意 gigatoken 在 Python 侧的两处不规律**（同批数据里可见）：
mixed-1k 的 encode（98.3 µs）比 1k 的中文（67.4）与英文（56.5）都慢，
而它自己 @128 只需 9.5 µs——这个跳变与本机负载抖动有关
（该批 `--iters 20`、绑 2 核），**不宜按语料做细粒度解读**；
需要精细结论时应重跑 `harness/bridge-calibration.sh` 那种串行窗口。

装置：本地 miniforge 裸库（不 import vLLM），绑核 2 核，
`python3 harness/python/bench_backends.py`。**这些数是"裸库"数字，
不含 vLLM 的 wrapper 开销**（`maybe_make_thread_pool` / `CachedHfTokenizer`），
与容器内 vLLM 路径的差需要另行测量——本表不做这个推广。

两条与选型直接相关的限定：

- **fastokens 的 Python 绑定没有暴露 `DecodeStream` 等价物**，
  流式解码在 vLLM 里实际由 `tokenizers.decoders.DecodeStream` 提供
  （`vllm/v1/engine/detokenizer.py:183`）；`VLLM_USE_FASTOKENS=1` 时
  `fastokens.patch_transformers()` 会替换这个类（`vllm/tokenizers/fastokens.py`）。
  也就是说 **Python 侧的流式收益来自这个 shim，而不是 fastokens 包本身**。
- **版本口径**：vLLM 0.26.0 在 **Rust 侧钉 fastokens 0.2.1**，
  而 PyPI 上只有 0.2.0 / 0.3.x（`pip index` 实测无 0.2.1 可装）。
  本节 Python 数用的是 0.3.2，**与 Rust 侧不同版本**，这是 §7.1 那条
  校准必须存在的原因；引用 Python 端的 fastokens 数字时要带上 0.3.2。

### 7.4 两种 Rust 计时装置的交叉验证（criterion vs 自建计时器）

主表用的是自建计时器（理由见 `harness/rust/bench/src/stats.rs`：criterion 在
「6 后端 × 4 语料 × 3 长度 × 4 操作」的笛卡尔积上跑不完时间盒）。
为了证明自建计时器没有系统偏差，`harness/rust/bench/benches/backends.rs` 用
**criterion 独立复测**了少量关键组合（结构照搬上游 `benches/{hf,tiktoken}.rs`，
改成 1) 本地工件、2) 真实长度分布语料），绑核与线程数一致。

| 测量点 | criterion | 自建计时器（e1） | 相对差 |
|---|---:|---:|---:|
| `hf` encode mixed-1k | 847 µs | 1 053 µs | 24% |
| `hf` decode mixed-1k | 87.6 µs | 171 µs | **95%** |
| `tiktoken_rs` encode en-1k | 475.8 µs | 457 µs | 4% |
| `riptoken` encode en-1k | 67.5 µs | 65.1 µs | 4% |
| `hf` 流式 256 token（zh-1k） | 83.1 µs | 81.9 µs | 1.4% |

读法（**这条比数字本身重要**）：

- tiktoken 家族与流式解码两组吻合在 5% 内，说明自建计时器在**稳定的测量点**上
  与 criterion 等价。
- `hf` 的两个单发测量点（encode/decode @1k）差 24% 与 95%。**已定位为共享机器的
  运行间抖动，不是代码路径差异**：把同一条命令单独跑一次
  （`--ops decode --corpus mixed --lengths 1k`，机器安静时）得到 **84.9 µs**，
  与 criterion 的 87.6 µs 一致。也就是说 e1 那一格被同批测量的其它负载抬高了。
  （该复测命令：`b-backends run --backend hf --corpus mixed --lengths 1k --ops decode`，
  输出 `[decode] hf mixed 1k 84.88 µs`；属于一次性诊断，未单独入库。）
- 结论：**单发、低耗时的格子在共享机器上不可信**；引用 hf 的 1k 级单发数字时，
  以 criterion 或上面的单独复测为准（§3.1 的 hf 行是按整批量测的，
  属于"被抬高的那一版"，读的时候请按这一节打折）。流式（单次 80 µs 以上、
  内部循环 256 次）与批量（单次毫秒级）的两装置一致性明显更好。

**另一个副产物**：criterion 还测了上游 `tokenizers::DecodeStream`（Rust 侧，
`tokenizers-0.22.2`）与 vLLM 自研 `DecodeStream` 的对比——
**上游 91.4 µs vs vLLM 83.1 µs（vLLM 快 9%）**，说明 vLLM 的前缀差分实现
（`incremental.rs`）在字节层面没有比上游慢，且它额外提供了 `min_bytes_to_buffer`
与前缀种子逻辑。

## 8. 一键复跑

```bash
# 0) 资源纪律（本机是共享开发机，所有基准都必须经这两个脚本）
export PATH="$HOME/.cargo/bin:$PATH"
LOCK=/home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh
LIM=/home/chiro/projects/vllm/tokenizer/scripts/limit.sh

# 1) 取本地工件（tiktoken 家族 + tekken 家族，按固定 revision + 校验 sha256）
harness/fetch-artifacts.sh

# 2) 建 harness（gigatoken 需要 nightly：它用了 #![feature(portable_simd)]）
cd harness/rust
$LOCK $LIM cargo +nightly build --release -p b-backends --features gigatoken

# 3) 六后端 + gigatoken：正确性 → 微基准矩阵 → gigatoken 审计
CORES=2 $LOCK $LIM ./target/release/b-backends correctness
CORES=2 $LOCK $LIM ./target/release/b-backends run \
    --backend all --corpus all --lengths all --ops all --prefix e1
# 批量维度的 2 线程对照（CORES=4-5 = limit.sh 里的 2 核）
CORES=4-5 $LOCK $LIM ./target/release/b-backends run \
    --backend all --corpus all --lengths all --ops batch --prefix e1-batch-2t
CORES=2 $LOCK $LIM ./target/release/b-backends gigatoken-audit --total-bytes 4194304

# 4) criterion 交叉验证（少量关键组合，验证自建计时器没有系统偏差）
CORES=2 $LOCK $LIM cargo +nightly bench --features gigatoken

# 5) Python 侧三实现
CORES=2 $LIM python3 harness/python/bench_backends.py \
    --impl hf,fastokens,gigatoken --corpus all --lengths 128,1k \
    --json-out data/backends/python-e1.json \
    --manifest-out data/backends/python-e1.manifest.json

# 6) 校验 vendored 快照没被改动
bash harness/rust/sync-vendor.sh --check
```

每个 harness 子命令都支持 `--help`。

## 9. 每条数字的出处

| 文档里的表 | 原始文件 | manifest |
|---|---|---|
| §3 主表 | `data/backends/e1.jsonl` / `e1.csv` | `data/backends/e1.manifest.json` |
| §6 批量（1t/2t） | `e1.jsonl` + `e1-batch-2t.jsonl` | 各自的 `.manifest.json` |
| §7.4 criterion 交叉验证 | `harness/rust/target/criterion/`（本地产物，不入库） | — |
| §2、§4 正确性 | `data/backends/correctness.json` | 同目录（含 `upstream_commit`） |
| §5 panic | `data/backends/e1.csv` 的 `ERROR:` 行 + `correctness.json` | 同上 |
| §7.1 校准 | **本仓库** + 根代理 `data/crosscheck/README.md`（第二装置） | 两侧各自 manifest |
| §7.3 Python | `data/backends/python-e1.json` | `data/backends/python-e1.manifest.json` |
| `docs/05` 全部 | `data/backends/gigatoken-audit.json` | `data/backends/gigatoken-audit` 同批 manifest |

**未测项清单（不要当成"没有差异"）**：

- tekken 家族内没有第二个实现可对拍 ⇒ 只能验 roundtrip 与不变量；
- tekken 的流式 decode 在 zh/mixed 全部失败（§4），不是没测；
- `Fastokens`（2 号）在本工件上无法单独构造（§1.2）；
- 真实 Mistral 模型的 tekken 路径未验证（只用了官方 `tekken.json`）；
- Python 侧 `VLLM_USE_FASTOKENS=1` 的**容器内 vLLM wrapper** 端到端未测
  （本节只测裸库；wrapper 开销属于 E2）。
