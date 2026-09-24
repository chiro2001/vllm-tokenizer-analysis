# 05 · gigatoken 审计：它的"快 1000×"到底在说什么

> 线 B（`B-backends`）产出。原始数据：
> - 它自己声明的口径：`data/backends/gigatoken-claims.json`（由 `scripts/extract-claims.py` 从 sdist 转录）
> - 本机复测：`data/backends/gigatoken-audit.json`
> - E1 里的 gigatoken 行：`data/backends/e1.jsonl`
>
> **前提声明**：本文档**没有**去搬 10 GB 级语料复现它的端到端数字。
> P1 是"读它的脚本与结果、把口径讲清楚"；P2/P3 是本机同条件复测（MB 级）。

## 1. 它是谁，为什么值得单独审

gigatoken 是一个独立的 Rust/Python 分词库（PyPI `gigatoken==0.10.0`，
crate 源码随 sdist 发布）。它的卖点是**离线语料吞吐**：一句话概括是
"在 288 核 EPYC 上把 11.9 GB 文本按 19 GB/s 分完词"。

把它和 vLLM 放在一起看的时候，**必须先把"它测的场景"和"vLLM 服务的场景"
分开**——这是本文档的全部意义。三条硬事实：

1. `crates.io` 上**没有** `gigatoken` crate，它只以 Python wheel +
   sdist 形态发布（sdist 里带完整 Cargo workspace，可作 path 依赖 vendor）。
2. 它的公开 API 面向**批量文件/文档吞吐**：`encode_batch` / `encode_files`，
   内部按字节切 chunk、用 rayon 并行、LPT 负载均衡（`src/batch.rs`）。
3. 它**没有流式 decode**，`decode` 返回 `bytes`（§4）。

## 2. P1 · 复现声明：其口径下的数 + 前提

### 2.1 它的输入口径（读源码得到，不是推测）

`benchmarks/compare/measure.py` 的 docstring 自己写得很清楚（原文）：

> "hf and tiktoken only parallelize across documents, so their input is split
> on `--separator` (before timing); gigatoken is handed the raw un-split bytes
> object as a **single document** — both its BPE and SentencePiece backends
> chunk oversized documents internally"

代码层面（`measure.py:170-184`）：

```python
if args.max_mb is not None:          # ← 只有 hf/tiktoken 会走到这里
    data = data[:budget]
if args.library == "gigatoken":
    inputs = [data]                  # ← 整块当 1 个文档
else:
    inputs = text.split(args.separator)   # ← 按 <|endoftext|> 切成几万个文档
```

`sweep.py:187, 206-208` 决定 `max_mb`：`--hf-mb` 默认 **100 MB**；
gigatoken 一律 `max_mb=None`（整文件）。

于是同一个"对比"里，两侧的输入是：

| | gigatoken | hf |
|---|---:|---:|
| 输入字节 | 11 920 511 059（11.9 GB） | 100 000 000（100 MB） |
| 文档数 | **1** | 20 342 |
| 并行来源 | 单个 11.9 GB 文档 → 内部切 chunk + rayon（288 核） | 20 342 个文档 → 文档级并行 |

**这两侧比的不是同一个东西**：左侧是"一个巨大的单文档被内部并行切分"，
右侧是"两万个小文档并行"。前者能在 288 核上均匀铺开，后者受文档数上限
（以及每文档的固定开销）约束。

### 2.2 它的"加速比"由两部分乘起来（这一节是本文档的核心）

对 `results.json` 里**全部 72 组对比**做同样的分解
（`data/backends/gigatoken-claims.json`）：

```
表观加速比  =  输入量比  ×  单位吞吐比
            = (11.9 GB / 100 MB) × (gigatoken MB/s / hf MB/s)
               = 119.21 × (…)
```

实测分布（全部 72 组）：

| 项 | 值 |
|---|---|
| 输入量比（**每一组都相同**） | 119.21×（另有 6 组是 11.92×，即 1 GB 档） |
| 表观加速比 中位 / 最大 | 110× / 1 299× |
| **剥掉输入量比后的单位吞吐比** 中位 | **2.39×** |

也就是说：**"1000×"里，119× 来自"喂了 119 倍的输入"，剩下约 2.4× 才是
单位字节上的真实差距**（在它选定的并行条件下）。

以任务里点名的 `AMD EPYC 9565` / `Qwen/Qwen2-1.5B-Instruct` / `owt_train.txt`
那一行为例（逐字段转录，未改口径）：

| 字段 | gigatoken | hf | 说明 |
|---|---:|---:|---|
| bytes | 11 920 511 059 | 100 000 000 | **输入量差 119.21×** |
| docs | 1 | 20 342 | 文档边界完全不同 |
| time_s | 0.6234 | 3.6124 | 各自的计时区间 |
| mb_per_s | 19 121.88 | 27.68 | 表观 **690.8×** |
| tokens | 2 611 765 417 | 21 862 732 | 各自口径下的 token 数 |
| bytes/token | 4.5642 | 4.5740 | 差 **0.21%** |

**token 计数这一项要特别说明**：任务背景里提到"切分口径不同导致 token 计数
不一致"。逐组核对 72 组后，`bytes/token` 的相对差是
**中位 0.28% / 最大 0.75%**——这是个**很小**的差异，量级上解释不了 690×。
它的来源是 `measure.py` 自己承认的那一句："gigatoken also encodes the
separator strings themselves (as special tokens), so its token count differs
from the others by roughly one token per document"——即 gigatoken 把
`<|endoftext|>` 分隔符本身也编码了，而 hf 是**在分隔符上切开**所以没编它。
每文档 1 个 token 的差异摊到 20 342 个文档上，就是千分之几。

> 结论修正：**"token 计数不一致"是真实存在的，但它是 0.3% 量级的偏差，
> 不是 690× 的原因。** 690× 的原因 99.8% 来自输入量差与并行粒度差。
> 写报告时不要把这两件事混成一条。

### 2.3 P1 的本地小规模忠实复现

不搬 10 GB，但可以**按它的口径**在本机复现"比值趋势"：把 4 MB 大块
**当成单文档**交给 gigatoken（其内部切 chunk），同样 4 MB 按文档切给 hf，
同线程数。这就是 `data/backends/gigatoken-audit.json` 的
`same_condition` 组里 `docs=1` 的那些行。结果见 §3.3。

## 3. P2 · 同条件对比：把"并行策略"与"算法速度"分开

装置：`b-backends gigatoken-audit`，绑核 2 核（`taskset -c 4-5`），
总输入固定 **4 MiB**，文档数扫 {1, 8, 64, 512}，线程数分别取 1（串行）
与 2（并行）。产物：`data/backends/gigatoken-audit.json`。

### 3.1 两条输入构造（这是读懂下面所有数字的钥匙）

gigatoken 在 tokenizer 内部维护 **pretoken cache**（`src/bpe/pretoken_cache.rs`，
还有 `token_arena` 分段缓存）。同一个输入反复编码时，第 2 轮起几乎全是缓存命中。
本审计因此用两种输入，给出**上下界**：

| 档 | 构造方式 | 含义 |
|---|---|---|
| `repeat` | 同一段 8 k 语料反复拼到 4 MiB | 预分词高度重复 = **缓存命中上界** |
| `unique` | 确定性伪随机词流（词后带递增数字）拼到 4 MiB | 预分词互不重复 = **缓存冷下界** |

并且每一行都给两个耗时：

| 列 | 口径 |
|---|---|
| `median_us` | **热缓存**：同一 tokenizer 实例反复跑同一输入，取中位数 |
| `cold_us` | **冷启动**：每次**重新加载**一个全新 tokenizer、只跑一次（取 1 次） |

为什么必须分开：真实离线语料是"一次过"的，对应的就是 `cold_us`；
而 `median_us` 反映的是"同一批数据被反复处理"（例如多 epoch 训练、
或对同一批请求做多轮实验）。

### 3.2 缓存的影响有多大（同条件，只是输入构造不同）

| 后端 / 场景 | 输入 | 热缓存 MB/s | 冷启动 MB/s | 热/冷 |
|---|---|---:|---:|---:|
| gigatoken（整块当 1 文档） | `repeat` | 923.3 | 93.1 | **9.9×** |
| gigatoken（并行 512 文档） | `repeat` | 824.1 | 93.8 | 8.8× |
| gigatoken（整块当 1 文档） | `unique` | 285.6 | 73.5 | **3.9×** |
| gigatoken（并行 512 文档） | `unique` | 243.4 | 75.1 | 3.2× |
| hf（并行 512 文档） | `repeat` | 7.4 | 7.5 | 1.0× |
| hf（并行 512 文档） | `unique` | 3.3 | 3.1 | 1.1× |

读法：

- **gigatoken 的缓存效应是 3.2–9.9×**，而 hf 侧基本没有（1.0× 上下）。
  这意味着"同一条 11.9 GB 单文档跑一遍"这种口径下，**gigatoken 自己的
  数字里已经包含了一部分缓存收益**——自然语料的预分词会大量重复，缓存会命中很多。
  这是它真实设计收益的一部分，**不是作弊**，但引用时必须说明
  "这个数包含预分词缓存的贡献"，而且**离线一遍过（unique）与反复处理
  （repeat）要分开报**。
- 顺带一个反直觉的观察：`unique` 档的热缓存数（243–286 MB/s）
  只有 `repeat` 档（824–923）的 1/3。也就是**它的"快"里有相当大一块来自
  输入本身的重复度**，而不是纯算力。§3.3 的倍率因此**统一用冷启动列**。

### 3.3 并行策略 vs 算法速度（分离）

同 4 MiB、同线程数（2 核）、同文档边界下，把四种组合摆在一起
（统一取**冷启动列**，避开 §3.2 的缓存效应；单位 MB/s）：

| 文档边界 | gigatoken | hf | 倍率 | gigatoken | hf | 倍率 |
|---|---:|---:|---:|---:|---:|---:|
| | \(repeat\) ——— | | | \(unique\) ——— | | |
| 4 MiB 当 **1 个文档**（它的官方口径） | 93.1 | 3.1 | **30.0×** | 73.5 | 1.8 | **39.8×** |
| 切成 **8 个文档** | 92.2 | 5.4 | 17.0× | 77.4 | 2.3 | 33.4× |
| 切成 **64 个文档** | 95.9 | 7.6 | 12.6× | 75.9 | 3.0 | 25.5× |
| 切成 **512 个文档** | 93.8 | 7.5 | 12.5× | 75.1 | 3.1 | 24.4× |

完整表见 `gigatoken-audit.json` 的 `same_condition` 与 `rows`。

可以下的四条结论：

1. **在"1 个文档"这个它的官方口径上，hf 完全吃不到并行**（3.1 MB/s ≈
   单线程速度），而 gigatoken 能把自己内部切成 chunk 铺满核。
   这正是它宣称的优势，也是**它的设计目标**（离线大文件吞吐）。
2. 一旦把输入切成 64–512 个文档，**hf 侧也上来了**（3.1 → 7.5 MB/s，2.4×，
   因为文档级并行生效），而 gigatoken 基本持平（93 → 94）。
   即：**"输入切分粒度"这一项对两侧的影响都可以量化**，
   在 `repeat` 档下是 2.4×（hf）对 1.0×（gigatoken）。
3. 即便在最有利的同条件组合下，**单位字节差距是 12.5–30×**
   （`repeat`）/ 24–40×（`unique`），而不是 690×，也不是 2.4×
   （§2.2 那个 2.4× 是它在 288 核 EPYC 上、hf 只吃 100 MB 文档级并行时的结果）。
   **三个数字口径不同，引用时必须写明是哪一个。**
4. **fastokens 在这组同条件对照里大面积缺席**：4 MiB 的长输入会触发
   `03` §5 那个上游 panic，只有切成 512 个小文档时才测得出来
   （`repeat` 冷启动 80.8 MB/s、`unique` 28.8 MB/s，约为 hf 的 10×，
   但仍只有 gigatoken 的 1/1.2–1/2.6）。失败的 12 个点原样记在
   `gigatoken-audit.json` 的 `failed_rows` 里，没有静默丢弃。

> 口径提醒：本机 12 核、绑 2 核，与它的 288 核 EPYC **不可比**。
> 我们复现的是"比值趋势与口径结构"，不是它的绝对值。

### 3.4 并行到底有没有用（线程列的含义）

本审计的线程列只有 1 与 2 两种（受资源纪律约束：本项目所有基准绑 2 核，
`RAYON_NUM_THREADS=2`）。同条件对照里两侧都在同一线程预算下跑，
所以 §3.3 的倍率**已经把并行预算对齐**。

**未测**：4/8/16/288 核下的曲线。这在 12 核共享机上做不了，
也不能用 2 核的结果外推（gigatoken 的 chunk + LPT 调度在核数增长时
是否仍然线性，取决于它的 chunk 切分是否够细——`chunk_target_bytes`
按总字节数决定，核数越多单核 chunk 越小，固定开销占比上升，
这个拐点必须在真机上扫）。这是本次审计最大的未测项。

## 4. P3 · vLLM 相关域：在线单条才是结论落点

### 4.1 在线单条（128 / 1k / 8k 等价文本，单线程）

装置同上，输入是 E1 的四类语料（中文/英文/代码/混合），单条请求、单线程。

| 语料-长度 | gigatoken 中位耗时 | 吞吐 |
|---|---:|---:|
| mixed-128 | 2.4 µs | 209 MB/s |
| mixed-1k | 12.0 µs | 304 MB/s |
| mixed-8k | 93.2 µs | 318 MB/s |
| zh-128 | 2.4 µs | 240 MB/s |
| zh-1k | 14.6 µs | 316 MB/s |
| zh-8k | 121 µs | 296 MB/s |
| code-128 | 2.2 µs | 507 MB/s |
| code-1k | 7.8 µs | 552 MB/s |
| code-8k | 58.3 µs | 570 MB/s |
| en-128 | 1.0 µs | 764 MB/s |
| en-1k | 5.6 µs | 1 018 MB/s |
| en-8k | 43.9 µs | 1 043 MB/s |

（数据源：`data/backends/e1.jsonl`，1 核 / 单线程 × 自然文本变体轮转。）

对照同装置、同语料的其他后端（`data/backends/e1.csv`）：

| 后端 | mixed-1k encode | zh-8k encode | 相对 gigatoken |
|---|---:|---:|---|
| gigatoken | 12.0 µs | 121 µs | 1.0× |
| fastokens（ByteLevel 旁路） | 48.3 µs | panic（`03` §5） | 慢 4.0× |
| riptoken | 83.6 µs | 1 110 µs | 慢 7–9× |
| hf | 1 053 µs | 4 480 µs | 慢 37–88× |

**这是"vLLM 该关心"的那一组数**。读法：

- 在线单条路径上，gigatoken **确实是最快的**（比 vLLM 现役最快的
  fastokens 旁路还快约 4 倍，比默认 hf 快数十倍）——
  但注意它的 8k 数字 ~121 µs 仍然要**乘进 TTFT**，
  而 §4.2 的缺口会让这个优势在流式路径上大幅缩水。
- 长度 128→8k（64×）时，gigatoken 的耗时 2.4→121 µs（50×），
  **近似线性**。这说明它的在线路径没有明显的固定开销，
  也没有随长度恶化的现象。
- 这些数字**包含 pretoken cache 的部分命中**（输入是自然文本变体轮转，
  见 `03` §7.2）。冷启动下会下降（§3.2）。

### 4.2 缺口量化（**这是选型的决定性部分**）

#### 缺口 1：`decode` 返回 `bytes`，没有 `skip_special_tokens`

代码证据：

- Python：`gigatoken/_tokenizer.py:286` → `def decode(self, tokens) -> bytes`
- Rust：`src/bpe/tiktoken.rs:1386` → `pub fn decode(&self, v: &[TokenId]) -> impl Iterator<Item = u8>`

vLLM 的契约是 `Tokenizer::decode(&self, ids, skip_special_tokens) -> Result<String>`
（`vendor/vllm-tokenizer/src/lib.rs:25`），且**要求对不完整 UTF-8 返回替换字符**
（`DecodeStream` 靠 `ends_with('\u{FFFD}')` 判断"这次还不该输出"）。

影响：调用方每一步都要自己做 UTF-8 校验/替换；`<|im_end|>` 之类的特殊 token
会原样出现在返回文本里（它的 decode 不区分 special）。适配成本是**每个 token
一次**的额外工作，在流式路径上直接落到 TPOT 上。

#### 缺口 2：没有流式 decode API（**vLLM 依赖它**）

vLLM 的流式返回走 `tokenizers.decoders.DecodeStream`（带状态、逐 token、
`step(tokenizer, id)`），vLLM 自己还有一份等价的 Rust 实现
（`vendor/vllm-tokenizer/src/incremental.rs:9` 的 `trait IncrementalDecoder`：
`push_token` / `next_chunk` / `flush` / `output`）。

gigatoken 全树没有这类接口：只有整段 `decode` / `decode_batch` /
`decode_bytes_batch`（grep `gigatoken/` 与 `src/` 全树）。

**量化**：本 harness 给它套上 vLLM 的 `DecodeStream`（前缀差分，O(n²)：
每推一个 token 就重解一次当前全序列），跑 256 个生成 token：

| 后端（流式 256 token，mixed-1k） | 中位耗时 |
|---|---:|
| gigatoken + vLLM DecodeStream 适配层 | 23.1 µs |
| fastokens（ByteLevel 旁路） | 33.4 µs |
| riptoken | 40.7 µs |
| hf | 80.1 µs |

也就是说：**适配层补上了功能，代价可以接受**（因为它整段 decode 只有 1.6 µs
@128，O(n²) 的常数被摊薄了）。但这是**我们的适配**，
不是 gigatoken 的能力；上游若发生行为变化，这层适配需要重验。

#### 缺口 3：rayon 全局线程池与绑核/线程数的相互影响

证据（读源码）：

- `src/batch.rs:665-745` 的 `WorkerPool` 按 `rayon::current_num_threads()`
  建槽，**每线程一个 fork**，fork 常驻（cache warm）。
- `encode_docs_ragged` 用 `into_par_iter()`（`src/batch.rs:760-789`）走**全局** rayon 池。
- `chunk_target_bytes` / LPT 决定切块大小，**不接受调用方指定线程数**。

影响（对部署）：

1. 在线服务通常把前端进程绑核（`taskset` / cgroup），
   gigatoken 的并行批量会去**抢全局 rayon 池**，实际并发由
   `RAYON_NUM_THREADS` 与绑核的交集决定；本项目所有基准都用
   `scripts/limit.sh` 设了 `RAYON_NUM_THREADS`，因此测出的并行度是 2。
2. `WorkerPool` 的 fork **常驻内存**（每线程一份 cache），
   在"一个进程处理很多小请求"的服务形态下，这是常驻 RSS 而不是峰值 RSS。
   本次没有测它的内存占用（**未测**，见 §6）。
3. 池在首次使用时惰性建立；**先跑小批量再跑大批量**与**反过来**，
   测出的并行效率会不同。本审计所有行都在独立进程里、先小后大，
   顺序固定（见 `gigatoken-audit.json` 的 `doc_counts` 顺序）。

#### 缺口 4：Rust 侧 `TokenId` 不在公开 API 里（库集成成本）

`src/lib.rs:12-15` 只 `pub use` 了 `Tokenizer` / `WorkerPool` / `EncodeState`；
`mod bpe`、`mod token` 都是 `pub(crate)`。而 `Tokenizer::decode` 的签名要求
`&[TokenId]`，外部调用者**无法命名这个类型**。本 harness 的绕法是
`#[repr(transparent)]` + `slice::from_raw_parts`（见
`harness/rust/bench/src/backends/gigatoken_shim.rs` 的 SAFETY 注释）。
同类情况还有 `encode_docs_ragged_serial`（有 `pub fn` 但模块是 `pub(crate)`），
本 harness 只能复刻它的循环。

影响：把 gigatoken 作为 **Rust 库**（而不是 Python wheel）嵌入时，
decode 属于事实上的私有 API，升级没有兼容性保证。vLLM 的
`vllm-tokenizer` 是 Rust crate，这一点直接影响集成方案。

## 5. 三档结论（一句话各一条）

| 档 | 结论 |
|---|---|
| **P1（复现其声明）** | 它的"~1000×" = **119× 输入量差 × 约 2.4× 单位吞吐差**（72 组对比的中位；最大 1 299×），前提是 gigatoken 拿整块 11.9 GB 单文档、hf 只拿按分隔符切碎的 100 MB；其脚本与结果都明确写了这个口径。 |
| **P2（同条件）** | 同 4 MiB、同文档边界、同线程数下，单位字节差距收敛到 **12.5–30×**（`repeat` 冷启动列）/ **24–40×**（`unique` 冷启动列），远小于 P1 的 690×；缓存效应本身是 3.2–9.9×；把输入切碎能让 hf 侧多拿 2.4× 而 gigatoken 不变，**并行策略差异与算法速度差异是可分离的两个量**。 |
| **P3（vLLM 相关域）** | 在线单条 128/1k/8k 上 gigatoken 是最快的（mixed-1k 12.0 µs，比 fastokens 旁路快 4.0×、比默认 hf 快 88×），**但**它缺 vLLM 依赖的流式 decode 与 `skip_special_tokens` 语义，需要外挂适配层（vLLM `DecodeStream` 实测 23.9 µs/256 token），且 Rust 侧集成面对非公开 API。**结论落点：它的收益主要在离线批量，在线收益存在但要连同缺口一起算账。** |

## 6. 明确未测 / 待确认

| 项 | 状态 | 原因 |
|---|---|---|
| 11.9 GB 级端到端复现 | **未做** | 按 `plan/EXECUTION.md` §4 的取舍；本机 29 GiB 内存 + 资源纪律不允许 |
| 288 核并行曲线 | **未做** | 本机 12 核，项目规定所有基准绑 2 核 |
| 常驻内存 / RSS | **未测** | `limit.sh` 限制 8 GiB 地址空间；需要在放开限制的机器上另测 |
| `unique` 档的 512 文档冷启动 | 冷启动列有数（57.6 MB/s） | — |
| fastokens 0.3.2（Python）对同一批长输入的 panic | **待确认** | `03` §5 的 panic 在 Rust 1.0.2（fastokens 0.2.1）复现；0.3.2 未逐行核对 |
| gigatoken 的 SentencePiece 后端 | **未测** | 本审计只覆盖 BPE（`load_hf_bpe`）；其 `sp_encode_docs_ragged` 是另一条路径 |
| gigatoken 与 HF 的 ids 一致性 | ✅ 12/12 一致 | `data/backends/correctness.json` 的 `gigatoken_vs_hf` |
| gigatoken 的 decode 文本与 HF 是否一致 | ✅ 6/6 一致（见 audit 的 `decode_checks`） | 同一 `tokenizer.json`、同一 id 序列 |

## 7. 一键复跑

```bash
export PATH="$HOME/.cargo/bin:$PATH"
LOCK=/home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh
LIM=/home/chiro/projects/vllm/tokenizer/scripts/limit.sh

# (a) 转录它自己的声明口径（只读 sdist，不重跑大语料）
python3 scripts/extract-claims.py \
    --sdist /tmp/b-artifacts/gigatoken-0.10.0.tar.gz \
    --out data/backends/gigatoken-claims.json

# (b) 本机同条件复测（P2 + P3），默认 4 MiB、文档数 1/8/64/512、两种输入
CORES=2 $LOCK $LIM ./harness/rust/target/release/b-backends \
    gigatoken-audit --total-bytes 4194304 \
    --out data/backends/gigatoken-audit.json

# (c) Python 侧对照（gigatoken 不需要 vLLM，本地装即可）
CORES=2 $LIM python3 harness/python/bench_backends.py \
    --impl gigatoken,hf,fastokens --corpus all --lengths 128,1k,8k
```

两个 harness 命令都支持 `--help`。
