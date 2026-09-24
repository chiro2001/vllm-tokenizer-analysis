# 06 · 结论、选型建议与待验证项

> 根代理收口。综合 `01`–`05` 五篇的四条线产出。
> **引用任何数字前先读对应文档的口径节**——本项目有多处"同一现象在不同口径下
> 数字相差数倍"的情况，本文会把它们逐条点明。

---

## 0. 一句话总结

**vLLM 的 tokenizer 正在换代：从"Python 进程里的 HF `tokenizers`"，走向
"Rust 前端的统一多后端抽象"。** 这次调研量化了现在的价（成本占比）、
各家的成色（六后端 + 三实现）、以及下一代的路（`vllm-rs`）。
在此之上有三个可以直接行动的结论：**fastokens 有一个会崩进程的上游 bug**、
**chat 模板不是瓶颈（Jinja 是常数级）**、**gigatoken 的战场在离线批量而不在在线服务**。

---

## 1. 现状：tokenizer 链路长什么样

### 1.1 边界（`01` §1）

链路只有**三段真实工作量，且全在 API server 前端进程**：

```
HTTP body(messages/prompt 文本)
  → chat 模板渲染（Jinja）        ← chat 请求；内部直接产出 token ids
  → 编码 encode                   ← text 请求；chat 请求已含在上面
  → ZMQ/msgpack 下发 list[int]    ─┐
                                   │  engine core（调度/forward/sample）
  ← 回来的 list[int]              ─┘
  → 流式增量解码 + 停止串判定      ← 前端进程，每个生成步都跑
```

**engine core 不在这条链路上**，唯一例外是 structured output 会为语法编译
**另加载一份** tokenizer（`vllm/v1/structured_output/__init__.py:79`）。

### 1.2 三个容易被搞错的事实（都经独立验证）

| # | 事实 | 为什么重要 |
|---|---|---|
| 1 | **`tokenizer_pool_size` 在 0.26.0 不存在**（全仓 0 命中） | 真实参数是 `renderer_num_workers`，池大小 = `workers + 1` 份 deepcopy；**池空时不阻塞**，而是现场 deepcopy 并让池**无上限增长**（`vllm/tokenizers/hf.py:51-55`）——这是链路里唯一"随并发劣化"的隐藏项 |
| 2 | **chat 请求永远不触发 `tokenizer: encode` scope** | chat 的编码在 `apply_chat_template(tokenize=True)` 内部完成，被 `render_messages` 包住。两个 scope 是**互补负载**；拿"encode 计数为 0"推断"encode 不耗时"是错的 |
| 3 | **`tokenizer: decode` 不是生成期解码** | 它只是 prompt 反解（每请求 ≤1 次）。真正每步都跑的流式解码在 `vllm/v1/engine/detokenizer.py`，**LiteProfiler 从未插桩它** |

---

## 2. 成本：按输入规模分层，8k 是分水岭（`02`）

Qwen3-0.6B / 单请求 / x86 2 核 / 无批量：

| 环节 | @220 tok | @1k | @8k | 边际成本 |
|---|---:|---:|---:|---:|
| `tokenizer: encode`（text 路径） | 286 µs | 1 100 µs | 10 537 µs | **≈1.47 µs/token**（R²=0.99975） |
| `render_messages`（chat + tools，**含模板内 encode**） | 1 015 µs | 2 075 µs | 12 231 µs | 同上 |
| ├ 其中 **Jinja 渲染** | 63 µs | 69 µs | 63 µs | **常数级** |
| └ 其中模板内 encode | 603 µs | 1 403 µs | 10 009 µs | 随 ISL 线性 |
| `decode: prompt_reverse`（历史口径，非生成期） | — | — | — | 0.17–0.19 µs/token |
| `detokenize: stream`（生成期增量解码） | — | — | — | **1.39–1.59 µs/token**（另有首步 216–292 µs） |

**占 TTFT 的比例**（跨装置上界口径）：1k prompt ≈ **0.3–1.2%**、
8k prompt ≈ **5.3–10.4%**。⇒ **8k 以下 tokenizer 不是瓶颈；8k 以上开始进入
TTFT 的两位数百分比区间。** 同装置可比的那一格更小：
`tokenizer: encode ÷ http:` = **0.09%**、`÷ Step:Model` prefill = **0.26%**
（分母只含模型执行，所以跨装置那组是**上界**）。

### 2.1 三条可直接行动的结论

1. **优化 chat 请求的前端成本，矛头应指向编码速度，不是 Jinja。**
   Jinja 在 128→8k 全程都是 47–147 µs 的同一个量级，**换模板引擎收益有限**
   （`02` §3.1 实测）。
2. **encode 在 64 token 以下有 25–45 µs 的固定成本**，线性只在 ≳64 token
   之后成立（`02` §2.1）。短 prompt 场景下线性外推会给出负数。
3. **历史"163–461 µs"的数字必须带口径引用**：那些稳态样本（52/52 全部回收）
   的 prompt 只有 **10–11 token**（从响应体 `usage` 读出），不是"几百 token"。
   但同量级 prompt 在本机只需 **36–47 µs**，**相差 3.4–12.7 倍且原因未定**
   （候选：机器差异 / 插桩开销 / 池冷启动；C 线明确**不判定**，需同装置复核）。
   ⇒ 引用时要同时说明"10–11 token + 原因未定"，不要当成典型成本，也不要
   擅自归因给 LiteProfiler。

### 2.2 插桩开销：本文的测量装置自证 <3%

C 线在**同一进程、同一轮**里把 LiteScope 读数与自己的 `perf_counter` 对齐：
encode 差 **1.5%**、render **2.3%**、decode **2.9%**（`02` §9）。
⇒ 本项目的成本数字**不是插桩撑起来的**。
（注意这**不能**反推历史数字也干净——历史的 LiteProfiler 采样器另当别论，
见 `02` §7.1。）

### 2.3 火焰图：热点不在 BPE merge（`02` §8）

四张 perf 火焰图（`figures/e3-1{a,b,c,d}-python-frontend.svg`）的 self-time 分类：

| 类别 | encode | render(chat+tools) | detokenize |
|---|---:|---:|---:|
| `rust-tokenizers`（`tokenizers::*`） | 15.9% | 5.8% | 13.6% |
| `rust-stdlib-or-dep`（core/alloc/hashbrown/regex/serde） | **48.1%** | 12.6% | 26.1% |
| CPython | 9.5% | 14.9% | **34.3%** |
| libc 符号 + malloc/free | 6.8% | **54.9%** | 10.0% |

**三条可引用的结论**：

1. **encode 由 Rust 主导（两类合计 64%），但最大头不是 BPE merge**
   ——`BPE::tokenize` 的 inclusive 只有 ~1%。真正花时间的是**词表/缓存查找与
   正则预分词**：`hash_one` 8.5%、`RawTable::reserve_rehash` 5.4%、
   `HashMap::insert` 4.9%、正则匹配 5.1%、`Cache::get` 3.2%。
   ⇒ **这一层还有优化空间**（更便宜的哈希 / 预分配 / 关缓存换内存）。
2. **chat 渲染里真正贵的是内存分配**（libc+malloc/free ≈55%，Rust 仅 18%）：
   Jinja 拼串 + serde_json 序列化 tools 是堆分配大户。
   与"Jinja 是常数级"互相印证——它是**固定量级**的分配成本，不随 ISL 增长。
3. **detokenize 里 Python 侧占比最高（34%）**，Rust 侧集中在
   `id_to_token`(4.2%) + `is_special_token`(2.6%)，即每步的"id→串 + 特判"。

---

## 3. 多后端成色（`03` / `05`）

### 3.1 主表摘录（单条在线路径，中位耗时 µs；Rust 装置）

| 后端 | zh 128 | zh 1k | zh 8k | mixed 1k | 适用工件 |
|---|---:|---:|---:|---:|---|
| `Hf`（基线） | 72.6 | 568.3 | 5 357 | 1 021 | 任意 HF `tokenizer.json` |
| `FastokensByteLevel` | **10.4** | **24.3** | — ⚠️ | **50.5** | 解码头为**纯 ByteLevel** 的 BPE |
| `TiktokenRs` | 37.5 | 206.7 | 1 668 | 307.6 | `tiktoken.model` |
| `Riptoken` | 22.8 | 136.2 | 1 072 | 82.7 | 同 tiktoken |
| `Tekken` | 35.6 | 271.8 | 2 205 | 559.6 | `tekken.json` |
| gigatoken（vendor） | **2.45** | **14.8** | **117.9** | **11.9** | `tokenizer.json`（BPE） |

`— ⚠️` = 上游 fastokens 0.2.1 在 8k 输入上 panic（§4.1）。

### 3.2 "快 8×"必须限定在纯 decode

| 口径 | 差距 | 说明 |
|---|---:|---|
| **纯 decode**（给 ids 出文本） | **4.8–9.2×** | 来自 `Backend::FastokensByteLevel` 旁路 |
| **流式增量解码**（服务实际用的） | **2.3×** | 前缀差分的固定开销摊薄了 assemble 占比 |

**归属**：这条旁路是 **`vllm-tokenizer` crate 自己写的，不是 fastokens 提供的**
（上游 fastokens 的通用 decode 就是慢的那条）。根代理在容器内用独立装置复核：
encode 的 8–13× 两端都有，**decode 的 8× 只在 Rust 侧**（Python 侧只有 1.09×）。
⇒ **Python 用户开 `VLLM_USE_FASTOKENS=1` 拿不到 decode 的收益。**

### 3.3 gigatoken：把"快 1000×"拆开看（`05`）

| 档 | 结论 |
|---|---|
| **P1 它声明的** | "~1000×" = **119× 输入量差 × 约 2.4× 单位吞吐差**。其 benchmark 对 HF 只喂 **100 MB 且切成 20 342 个文档**，对 gigatoken 喂**整个 11.9 GB 当单文档**（内部 chunk + rayon） |
| **P2 同条件** | 同 4 MiB / 同文档边界 / 同线程下收敛到 **12.5–30×**（冷启动），其中缓存自贡献 **3.2–9.9×** |
| **P3 vLLM 相关域** | 在线单条它**最快**（mixed-1k 12.0 µs，比 fastokens 旁路快 4.0×、比默认 hf 快 88×），**但缺 vLLM 依赖的流式 decode 与 `skip_special_tokens`**，需外挂适配层 |

**落点：它的收益主要在离线批量。** 在线收益存在但要把缺口一起算账
（无流式 API、rayon 全局池与绑核冲突、Rust 侧 API 非公开）。

### 3.4 一个反直觉的并行结论（`03` §6）

2 线程小批量下：**gigatoken 没有加速（0.93×，略慢）**，
因为它内部切 chunk + 起 rayon 任务的固定开销超过了 2 核收益；
而 `Hf` 有 1.79–2.02×。⇒ **底层单线程越快，并行空间越小**，
且"多快"与"多核下多快"是两个问题。

---

## 4. 必须向上游反馈的三个问题

这三条是本次调研最有实操价值的产出，**都不是"慢"而是"会坏"**。

### 4.1 fastokens 0.2.1 的越界 panic —— 会崩掉前端进程

```text
thread '<unnamed>' panicked at fastokens-0.2.1/src/pre_tokenizers/split.rs:419:78:
  index out of bounds: the len is 1 but the index is 1
```

**根因**（读源码定位，已由根代理独立复核）：

```rust
let n_chunks = n_cpus.min(text.len() / MIN_CHUNK_SIZE).min(pcre2.len()).max(2);  // :383-386
...
let all = find_matches_pcre2(chunk, base + auth_start, &pcre2[i])?;              // :419
```

`n_chunks` 先被 `.min(pcre2.len())` 夹到 1，又被末尾的 **`.max(2)`** 抬回 2，
于是 `pcre2[1]` 越界。触发条件是文本足够长（zh-8k 的 36 KB、en-8k 的 46 KB 必炸）。

**为什么这比"某后端用不了"严重**：vLLM 的 `new()` 会先试 fastokens、失败回落 HF
（`hf.rs:117-128`），**但 panic 不是 `Err`，回落逻辑拦不住**——
8k 级中文 prompt 会**直接崩掉前端进程**。建议上游把 `.max(2)` 改成
`.clamp(2, pcre2.len())`。

> ⚠️ 补充：vLLM 的 Rust 侧 `panic = "abort"`，本 harness 有意用 `unwind`
> 才捕获到它（已在 `harness/rust/Cargo.toml` 注明）。生产环境只会看到进程消失。

### 4.2 tekken-rs 0.1.1 的中文流式解码不可用

流式返回必然出现"多字节字符被拆到相邻两个 token"的中间态。
vLLM 的 `DecodeStream` 依赖 `decode` 对不完整 UTF-8 **返回替换字符**，
而 tekken-rs 直接报错（`Unable to decode into a valid UTF-8 string`）。
**6/12 语料组（zh 与 mixed）的流式 decode 完全不可用**，en/code 正常。
修复路径：给 `TekkenTokenizer::decode` 加 lossy 语义（更贴 vLLM 现有后端契约）。

### 4.3 `Fastokens`（2 号）与 `FastokensByteLevel`（3 号）在 Qwen3 上是同一个对象

`new_fastokens()` 会把解码头为纯 ByteLevel 的 tokenizer 自动装进
`Backend::FastokensByteLevel`（`hf.rs:97-108`），因此**不存在"两个后端各测一次"**。
`plan/experiment-matrix.md` 的 E1.1/E1.2 在本工件上不构成两个可测对象——
这条差异已在 `03` §1.2 声明，并用独立对照臂回答同一个问题。

---

## 5. 下一代：Rust 前端（`04`）

**它不是"更快的 tokenizer"，是北向服务层整体换语言。**

| | Python 前端（当代） | Rust 前端（`vllm-rs`） |
|---|---|---|
| HTTP 层 | FastAPI / uvicorn | axum |
| chat 模板 | Jinja2 | minijinja（+ pycompat） |
| tokenizer | HF `tokenizers`；fastokens 要 `VLLM_USE_FASTOKENS=1` | 六后端抽象，**fastokens 优先** |
| 流式解码 | `detokenizer_utils.py` | `incremental.rs::DecodeStream` |
| 与引擎通信 | ZMQ + msgpack | **同样**（南北向边界没动） |
| Python 侧剩下 | 全部 | **只剩 engine core** |

**关键点**：Python 前端里要用户手动开的 `VLLM_USE_FASTOKENS=1`，
在 Rust 前端里是**默认路径**（先试 fastokens、失败回落 HF），
且是逐模型自动降级而非全局开关。

### 5.1 成熟度：实验性，但差距可枚举

`vllm-rs serve --help` 显式列出 **50 个未实现参数 + 7 个接受但无效（Noop）**，
对比 33 个已支持参数（`rust/src/cmd/src/cli/unsupported.rs`，根代理复核：
57 个字段 = 50 + 7）。

**比"未实现"更危险的是 7 个 `Noop` 参数**——传了不报错也不生效，
静默不一致。其中与 tokenizer 相关的：`--enable-tokenizer-info-endpoint`。
另有一批传了就**直接拒绝启动**：`--tokenizer` / `--tokenizer-revision` /
`--skip-tokenizer-init` / `--hf-overrides` / `--trust-request-chat-template`。

### 5.2 实测到哪一步（`04` §2–§3）

- **官方 PyPI wheel 自带 `vllm-rs` 二进制**（x86_64 42.79 MB / aarch64 39.60 MB，
  ELF64 PIE），**零编译可抽取**（HTTP Range 只下 ~6% 的 wheel 字节）。
- **启动验证成功**：日志出现 `loading tokenizer with fastokens` +
  `loaded chat backend ... renderer=hf`。
- **全链路 E2E 成功**（配上游 `vllm-mock-engine`，9 条断言全过）：
  `/health`、`/v1/models`、`/tokenize`、`/v1/chat/completions`、流式 6 个 delta。
- **正确性交叉核对**：`你好，Hello!` 在 Rust `/tokenize` 与 Python `tokenizers`
  都得到 `[108386, 3837, 9707, 0]`，逐 id 相同。

**未做**：Python 前端的同负载对照 ⇒ **"Rust 前端省了多少"没有数**，
只有链路证明与微基准（后者见 `03`）。

---

## 6. 口径纪律（引用数字前必读）

1. **禁止跨装置同表比较**。桥梁校准（fastokens + HF `tokenizers` 双端各测一次）
   **未通过 15% 判据**（1/4 通过，HF 方向差 20–38%），原因已定位：
   Rust 侧钉 fastokens **0.2.1**、PyPI 上**没有 0.2.1**（Python 只能用 0.3.2），
   属版本差异而非装置差异。⇒ `03` 的 1–6 号后端只用 Rust 装置的数，
   Python 三实现自成一张表（`03` §7.3）。
2. **"倍率吻合" ≠ "绝对值吻合"**。根代理的独立复核比的是倍率
   （Rust 8.1× vs Python 9.3×，差 14.8% 勉强通过），而绝对值校准差 20–43%。
   两个结论同时成立，引用时说清用的是哪个口径。
3. **decode 方向任何口径下都不允许跨装置比较**（Python 侧没有 ByteLevel 旁路）。
4. **gigatoken 的倍数必须带前提**：输入量、文档边界、并行粒度三者都不同。
5. **单发、低耗时的格子在共享机器上不可信**：`03` §7.4 里 hf 的 1k 单发格
   被同批负载抬高 24–95%，单独复测才回到 84.9 µs。

---

## 7. 选型建议

### 7.1 如果你在用 Python 前端（当代，绝大多数人）

| 场景 | 建议 |
|---|---|
| 默认 | 保持现状。8k 以下 tokenizer 不是瓶颈（占 TTFT <2%） |
| 长 prompt（≥8k）、高 QPS、前端 CPU 紧张 | 开 `VLLM_USE_FASTOKENS=1`：encode 拿 8–13×，**但先看 §4.1 的 panic 风险**——若 prompt 会到 8k 且含中文，先确认该 bug 是否已修 |
| chat 密集 | 不要为了 Jinja 做优化（它是常数级）；优化编码 |
| 并发高且出现延迟劣化 | 检查 `renderer_num_workers`。实测默认值 1 时**在 2–4 并发就饱和**（吞吐 ~750 req/s），提到 4 约 **3.2×**（4 核容器）、再提到 8 **反而更低**（核已用尽，多出的线程只增加竞争）——`02` §6。**注意这不是免费旋钮**：池按 `N+1` 预建，启动期约 0.35–0.65 s/份（8 worker 时渲染器构造要 **9.5 s**），且超额借用时的现场 deepcopy 平均 **3.6 s** |

### 7.2 如果你在评估下一代（Rust 前端）

- **已经在可用区间**：**单模型 + 标准目录 + 模板在 `tokenizer_config.json` +
  不需要 LoRA/多模态**的标准 HF chat 服务——本次 E2E 恰好落在这个交集里并全程通过。
- **值得跟进**：南北向边界没动、二进制零编译可得、tokenizer 默认走 fastokens、
  六后端统一抽象、`DecodeStream` 是自研且比上游快 9%。
- **暂时不能上生产**：**怪配置还早**——一旦用到 `--tokenizer` /
  `--skip-tokenizer-init` / `--hf-overrides` / 请求级 chat 模板，**直接拒绝启动**；
  另有 7 个 `Noop` 参数静默不生效。功能差距清单见 `04` §4。
- **建议的第一步**：用 `harness/rust-frontend/verify_vllm_rs.sh` 在自己的模型上
  跑一次抽取 + 启动，确认启动参数落不落在"未实现"名单里。

> **跨线交叉推论（本节由根代理合成，注意等级）**：把 tokenizer 搬进 Rust 的
> **收益上限**可以由 C 线的占比数据框定——tokenizer 在 8k 以下只占 TTFT **<2%**，
> 8k 才到 5.3–10.4%。因此**"换代"本身不该以"tokenizer 更快"为主要理由**；
> 它的真实价值更可能在**换掉整层 Python 前端**（GIL、跨进程对象、线程池、
> 进程内编排）与**统一多后端**上。D 线原文把这条净收益标为【推断】，
> 本项目**没有**做同负载端到端对照，所以这里只是**上界框定**，不是实测结论。

### 7.3 如果你在评估 gigatoken

- **离线批量 / 语料预处理**：值得评估，同条件下 12.5–30× 于 HF。
- **在线服务**：单条确实最快，但要先补两个缺口——**流式 decode API**、
  **`skip_special_tokens` 语义**，还要处理 rayon 全局池与绑核的冲突。
- **别引用"1000×"**：那个数字的 119/120 来自输入量差异（§3.3）。

---

## 8. 待验证项（不要当成"没有差异"）

| # | 项 | 为什么重要 |
|---|---|---|
| 1 | fastokens **0.3.2** 是否仍有 §4.1 的 panic | 0.2.1 已确认；0.3.2 未逐行核对，而它才是当前 PyPI 最新 |
| 2 | Rust 前端 vs Python 前端的**同负载端到端对照** | "换代值多少"目前只有链路证明 |
| 3 | aarch64（Kunpeng 920B）上的原生验证 | `vllm-rs` 只在 x86 原生跑过，aarch64 走的是 qemu |
| 4 | 大核数并行曲线 | 本机受资源纪律限制只能用 2 核窗口，**不能外推到几十核** |
| 5 | gigatoken 常驻内存 / RSS | `WorkerPool` 每线程一个常驻 fork，未测 |
| 6 | 真机（a3-22）chat 负载锚点 | `render_messages` 的真机数没有，只有 x86 |
| 7 | 三个 scope 的**插桩开销** | 历史数字都带 LiteProfiler，未做关/开对照 |
| 8 | E3.2（Rust `vllm-tokenizer` 内部火焰图） | 只有 Python 前端火焰图（`figures/e3-1*.svg`） |
| 9 | 非 Qwen3 系 tokenizer 的三条成本 | 本项目成本数据全基于 Qwen3-0.6B |

---

## 9. 复现入口

```bash
# === 资源纪律（本机是共享开发机，所有重活必过）===
scripts/limit.sh <命令...>              # 绑 4 核 + 8 GiB + 4 并行编译
scripts/heavy_lock.sh <命令...>         # 全项目唯一重活锁
scripts/heavy_lock.sh --status          # 看谁持有

# === 成本（线 C）===
scripts/docker_run.sh -c '...'          # 无卡容器跑 vLLM 的 Python tokenizer 路径
scripts/c_cost_run.sh <task>            # E2 各任务；见 docs/02 §8

# === 后端矩阵（线 B）===
export PATH="$HOME/.cargo/bin:$PATH"
cd harness/rust
CORES=2 $LOCK $LIM cargo +nightly build --release -p b-backends --features gigatoken
CORES=2 $LOCK $LIM ./target/release/b-backends correctness
CORES=2 $LOCK $LIM ./target/release/b-backends run --backend all --corpus all --lengths all --ops all --prefix e1

# === 下一代（线 D）===
harness/rust-frontend/verify_vllm_rs.sh --no-fetch     # 抽取 + 启动验证
harness/rust-frontend/e2e_mock_engine.sh               # 全链路 E2E

# === 发布（净化 + private push）===
scripts/export_publish.sh --no-push    # 先干跑复核
scripts/export_publish.sh               # 净化 + 建 private repo + push
```

各 harness 子命令均支持 `--help`；每篇文档末尾都有该篇的一键复跑命令。
