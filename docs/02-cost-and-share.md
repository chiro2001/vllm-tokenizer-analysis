# 02 · tokenizer 的成本与占比

> C 线（`c_cost`）产出。数据：`data/cost/*.json`、`data/historical/*.csv`；
> 图：`figures/fig-02-*.svg`；复跑：`scripts/c_cost_run.sh`、`harness/python/*.py`。
> 引用本文任何数字前，请先读 §1 口径。

## 0. 一句话结论

在 Qwen3-0.6B / 单请求 / 无批量的条件下，tokenizer 链路**不是瓶颈但要分层看**：

- `encode`：**0.29 ms @220 token**、**1.1 ms @1k**、**10.5 ms @8k**
  （128–8k 严格线性，≈1.47 µs/token；**64 token 以下有 25–45 µs 固定成本**）
- `render_messages`（chat + tools，含模板内 encode）：**1.0–23.1 ms**；
  其中 Jinja 只有 **50–216 µs 且不随 ISL 变**
- `decode: prompt_reverse`（历史口径）：**0.17–0.19 µs/token**
- `detokenize: stream`（生成期）：**1.39–1.59 µs/token**，另加首步 216–292 µs
- 占 TTFT（引擎侧下界）：短 prompt **<1%**，**8k prompt ≈5.3–10.4%**（跨装置上界）

**8k 是分水岭**：再往上，tokenizer 开始进入 TTFT 的两位数百分比区间。

**必须同时记住的三条**：

1. `tokenizer: encode` 在 **chat 请求上永远不会被触发**——chat 的编码发生在
   `apply_chat_template(tokenize=True)` 内部，被 `render_messages` 这个 scope 包住。
   拿历史数据里 "encode 计数为 0" 去推断 encode 不耗时是错的（历史里 chat 请求本来就少）。
2. 生成期解码 **不在** `tokenizer: decode` scope 里。后者只是 prompt 反解。
3. 本机是 x86 CPU 容器；历史数据是 a3-22 / a3-21 Ascend NPU 上的 arm64 前端。
   **跨装置的绝对时间不可直接比**，本文只做量级与占比的上界参考。

## 1. 口径（引用数字前必读）

### 1.1 进程归属

`tokenizer:` 三个 scope 都在 **API server 前端进程**，不在 engine core。
实测证据（`liteprof_v1_torch_uni_a321_chip15_20260914T2142Z`）：

| scope | tid | pid |
|---|---|---|
| `tokenizer: encode` | **1169** | **1** |
| `http: create_completion` | 1 | 1 |
| `output: process_outputs` | 1 | 1 |
| `Step:Model` / `phase: *`（engine core） | 131 | **131** |

`tokenizer: encode` 与 `http:` **同进程不同线程**——因为 `BaseRenderer.__init__`
用 `make_async(..., executor=self._executor)` 把 tokenize 丢进了
ThreadPoolExecutor，跑在 worker 线程上（`vllm/renderers/base.py:97-101`）。
所以**按 tid 分组是对的，按 "出现 tokenizer scope 的 pid" 归组也是对的**，
但两者不能混用：同一个 pid 下面既有 asyncio 事件循环线程（tid=1），
也有 tokenizer worker 线程（tid=1169）。

### 1.2 三个 scope 的边界与「谁会触发谁」

| scope | 代码位置 | 闭合范围 | 何时触发 |
|---|---|---|---|
| `tokenizer: encode` | `renderers/base.py::_tokenize_prompt` | `tokenizer(prompt_text, **encode_kwargs)` | **仅**文本 prompt（`/v1/completions` 且给 `prompt` 字符串） |
| `tokenizer: render_messages` | `renderers/base.py::render_chat(_async)` | `render_messages(conversation, chat_params)` | **仅** chat 请求（`/v1/chat/completions`） |
| `tokenizer: decode` | `renderers/base.py::_decode` | `tokenizer.decode(prompt_token_ids)` | 仅 `TokenizeParams.needs_detokenization=True` 的请求 |

**关键（实测确认，不是推断）**：`render_messages` 的 scope 体里**包含编码**。
`chat_template_kwargs` 默认不给 `tokenize`，`safe_apply_chat_template` 于是按
HuggingFace 默认走 `tokenize=True`，Jinja 渲染完直接返回 ids。
我用 `BaseRenderer._tokenize_prompt` 打桩计数验证过：

- `render_chat(messages)` → `_tokenize_prompt` 调用次数 = **0**；
- `render_cmpl([{"prompt": text}])` → **1**；
- `render_cmpl([{"prompt_token_ids": ids}])` → **0**。

所以：**`encode` 与 `render_messages` 两个 scope 只对互补的负载形态生效，
不存在"两段可以直接相加"的请求。** 历史日志里两者之一必然为 0，属正常。

### 1.3 生成期解码是另一套代码（口径修正）

`tokenizer: decode`（scope） = **prompt token ids 反解成文本**，每个请求最多一次。
服务里**每个生成步都跑**的解码在 `vllm/v1/engine/detokenizer.py`，
LiteProfiler **完全没有插桩它**。本文把两者分开命名：

| 名称 | 位置 | 摊销 |
|---|---|---|
| `decode: prompt_reverse` | `renderers/base.py::_decode`（有 scope） | 每请求 ≤1 次 |
| `detokenize: stream` | `v1/engine/detokenizer.py`（**无 scope**） | 每 token 1 次 |

路径判据（`detokenizer.py:24`、`:60`）：
`tokenizers >= 0.22.0` **且** tokenizer 是 `TokenizersBackend` → `FastIncrementalDetokenizer`
（Rust `DecodeStream`）；否则 `SlowIncrementalDetokenizer`（Python 前缀偏移）。
本装置 `tokenizers 0.22.2` + `isinstance(tok, TokenizersBackend) == True`
⇒ **服务默认走快路**。

### 1.4 装置与负载形态

| 项 | 值 |
|---|---|
| 本机实验 | x86_64 AMD Eng Sample（12 核），容器 2 核（cpuset 4-5）、6 CPU 上限、8 GiB |
| 镜像 | `local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922` |
| vLLM | 0.26.0（容器内 `/vllm-workspace/vllm`，commit `568afb3`） |
| 模型 | `/home/chiro/Qwen3-0.6B`（`/models/Qwen3-0.6B`），HF `tokenizer.json` sha256 见 manifest |
| tokenizer | `TokenizerPoolCachedQwen2Tokenizer`（`renderer_num_workers=1` ⇒ 池 2 份 deepcopy） |
| 语料 | `mixed`：英文 + 中文 + 代码单元交替；ISL/OSL 都用**真实 tokenizer 编码后计数**，不用字符数估 |
| 历史锚点 | a3-21/a3-22，`Qwen3.5-2B`，`liteprof_v1/v2_*` run |

### 1.5 分子与分母（E2.4 专用）

- **分子**：`bench_paths.py` / `bench_detokenize.py` 在**本机容器**里测的 scope 内耗时。
- **分母**：历史 run 的 `lite.log` 里 **engine core 线程的 `Step:Model`** 墙钟
  （>10 ms 的步 = prefill 步 ⇒ TTFT 下界；≤10 ms 的步 = decode 步 ⇒ TPOT 下界）。
- `Step:Model` **只含模型执行**，不含 scheduler / 前端 / 网络 / 排队 ⇒ 分母偏小 ⇒
  **算出来的占比是上界**。
- 分子来自 x86 CPU、分母来自 Ascend NPU，标 `cross-device`；只有
  「同一 run 内 `tokenizer: encode` ÷ `http:` 或 `÷ Step:Model`」才是同装置可比。

## 2. E2.1 `tokenizer: encode`

100 次采样、10 次预热，`max_total_tokens=40960`（`truncation=True`）：

| 目标 ISL | 实测 token | scope mean | p50 | p99 | 外层 `tokenize_prompt()` mean |
|---|---|---|---|---|---|
| 128 | 220 | **286.1 µs** | 277.9 | 383.7 | 290.5 µs |
| 1k | 990 | **1100.0 µs** | 1083.5 | 1370.7 | 1073.0 µs |
| 8k | 8140 | **10536.8 µs** | 10404.7 | 12089.9 | 12940.8 µs |

对三点做最小二乘拟合：`cost_us ≈ 1.4688 × tokens − 136.3`，**R² = 0.99975**。
也就是说 128–8k 区间内 encode 是**严格线性**的，边际成本 **≈1.47 µs/token**
（本机 2 核、单线程、无批量）。

外层 `renderer.tokenize_prompt()` 只比 scope 多 2–10%：长度校验、截断检查、
`TokensPrompt` 构造都是小数。

> **run-to-run 波动**：同一配置在不同次运行之间波动明显（本节三个点在本线
> 各次运行里分别是 283/1211/11830、262/987/10897、286/1100/10537 µs，
> 极差约 **9–13%**）。引用单次绝对值时请同时看 `manifest` 里的 load average；
> 本文所有表格都取自**同一份已提交的 JSON**（`data/cost/e2_paths.json`）。

### 2.1 短 prompt：存在约 45 µs 的固定成本，线性只在 ISL ≳ 64 之后成立

把 ISL 压到 4–64 再测一遍（`bench_encode_short.py`，100 次采样）：

| 实测 token | 4 | 8 | 10 | 16 | 32 | 64 | 220 | 440 | 990 |
|---|---|---|---|---|---|---|---|---|---|
| mean (µs) | **26.5** | 36.7 | 36.2 | 41.9 | 60.0 | 79.5 | 280.4 | 512.3 | 1095.2 |
| µs/token | 6.63 | 4.58 | 3.62 | 2.62 | 1.88 | 1.24 | 1.28 | 1.16 | 1.11 |

**读法**：4–16 token 区间成本几乎是**常数 ≈26–42 µs**（token 数翻两番只涨 58%，
远低于线性应给的 4×），这就是 encode 的固定开销；边际成本在 64 token 之后
收敛到 **≈1.1–1.3 µs/token**。
所以 §2 那条"严格线性、R²=0.99975"的拟合只覆盖 128–8k；
**在 64 token 以下线性外推会严重低估**（它给出负数）。

> **固定成本的数值范围**：同一条曲线在本线两次运行里分别是
> **43.5 µs**（4 token，机器较静）与 **26.5 µs**（4 token，机器有并发负载），
> 所以本文只给 **"约 25–45 µs"** 这个区间，不给单一数字。
> 这个绝对量与 §7.1 的历史值对比有关，引用时请注意。

> 这个实验是**后补的，起因是历史数据对不上**：见 §7.1——历史 wave3/wave4 的
> 稳态 run 用 10–11 token 的 prompt，却报 162–461 µs。本机的固定成本只有 25–45 µs，
> 说明那个差距不能全归因于固定成本，还需要同装置复核。

**与历史对照**：上个项目在 a3-21 NPU 上测到 407–476 µs（`real-b1`，10 次），
本次同 run 的 `Qwen3.5-2B` 数据里是 452–613 µs。本机 220 token 时 283 µs、
990 token 时 1211 µs——**两者 prompt 长度不同，不能直接比**；量级（数百 µs 到
1 ms）一致，这已经足够说明"历史 163–613 µs 是几百 token 量级下的数"。

> ⚠️ 上面这句话在 §7.1 被**修正**了：历史稳态样本其实是 **10–11 token**，
> 不是"几百 token"。请以 §7.1 为准。

## 3. E2.2 `tokenizer: render_messages`（历史零实测项）

chat + tools（两个 function 定义），100 次采样。三种形态：

| 形态 | 目标 ISL | 渲染出的 prompt token | scope mean | p50 | 外层 `render_chat()` mean |
|---|---|---|---|---|---|
| `tools_1turn` | 128 | 519 | **1015.3 µs** | 962.6 | 982.0 µs |
| `tools_1turn` | 1k | 1289 | **2075.4 µs** | 2021.5 | 2027.5 µs |
| `tools_1turn` | 8k | 8439 | **12230.8 µs** | 11737.8 | 13229.7 µs |
| `no_tools_1turn` | 128 | 253 | **496.2 µs** | 449.1 | 553.5 µs |
| `no_tools_1turn` | 1k | 1023 | **1477.4 µs** | 1416.9 | 1285.7 µs |
| `no_tools_1turn` | 8k | 8173 | **9952.0 µs** | 9449.9 | 9667.1 µs |
| `tools_2turn_toolresult` | 128 | 812 | **1258.5 µs** | 1213.2 | 1222.7 µs |
| `tools_2turn_toolresult` | 1k | 2352 | **2875.3 µs** | 2834.6 | 3172.9 µs |
| `tools_2turn_toolresult` | 8k | 16652 | **23086.6 µs** | 21949.3 | 23074.1 µs |

读法：**`render_messages` 的开销主要由"渲染出多少 token"决定**，因为里面就是
一次 encode。tools 定义本身（约 700 字符 JSON）在 128 目标 ISL 上把成本从
496 µs 抬到 1015 µs，其中大部分是**定义本身被渲染进 prompt** 后多出来的
266 token（253 → 519），模板机制自身的增量只是 §3.1 里的几十 µs。

### 3.1 拆开 Jinja 与模板内 encode（`--tasks decompose`）

`render_messages` 的 scope 里同时有 Jinja 和编码，只报一个总数无法回答
"chat 模板贵不贵"。把两段分开量：`apply_chat_template(tokenize=False)` 只跑
Jinja，再对渲染出来的同一段文本单独 `tokenizer(...)` 就是模板内那一次编码。

| 形态 | ISL | 渲染 token | **Jinja（tokenize=False）** | encode(rendered) | 两段之和 | scope | 和/scope |
|---|---|---|---|---|---|---|---|
| `tools_1turn` | 128 | 519 | **63.2 µs** | 603.2 µs | 666.5 | 804.4 | 0.83 |
| `tools_1turn` | 1k | 1289 | **69.1 µs** | 1403.0 µs | 1472.2 | 1560.9 | 0.94 |
| `tools_1turn` | 8k | 8439 | **62.8 µs** | 10009.3 µs | 10072.1 | 10358.3 | 0.97 |
| `no_tools_1turn` | 128 | 253 | **50.7 µs** | 286.9 µs | 337.5 | 411.2 | 0.82 |
| `no_tools_1turn` | 1k | 1023 | **49.7 µs** | 1033.3 µs | 1083.1 | 1202.6 | 0.90 |
| `no_tools_1turn` | 8k | 8173 | **54.7 µs** | 9557.2 µs | 9611.9 | 11291.7 | 0.85 |
| `tools_2turn_toolresult` | 128 | 812 | **139.0 µs** | 1096.8 µs | 1235.8 | 2118.1 | 0.58 |
| `tools_2turn_toolresult` | 1k | 2352 | **215.7 µs** | 3049.2 µs | 3264.9 | 2958.9 | 1.10 |
| `tools_2turn_toolresult` | 8k | 16652 | **147.5 µs** | 23991.7 µs | 24139.1 | 25571.6 | 0.94 |

**结论（据上表，实测不是推断）**：

1. **Jinja 渲染是常数级**：单轮 50–69 µs、含 tool 结果的两轮 139–216 µs，
   **从 128 token 到 8k token 几乎不变**（变的是文本长度，模板只做一次
   消息拼接 + JSON 序列化 tools）。
2. 随 ISL 线性增长的那部分**全部来自模板内部的 encode**。
3. 两段之和占 scope 的 **58–110%**（多数 83–97%）。这个区间**跨过了 1**，
   说明拆分是**量级正确但不是严格分解**：未计入的
   `parse_chat_messages`、content-format 判定属于正向偏差，而模板内那次编码
   与我的 `add_special_tokens=False` 臂的参数差异属于负向偏差。
   引用时请只用"**Jinja 是常数级、增长全部来自 encode**"这个定性结论，
   不要引用具体的百分比。

**实践含义**：优化 chat 请求的前端成本，矛头应指向 **tokenizer 的编码速度**
（Rust 后端、fastokens 一类），**不是 Jinja**。Jinja 的百 µs 在所有 ISL 上
都是同一个量级，换模板引擎收益有限。

## 4. E2.3 解码的两半

### 4.1 `decode: prompt_reverse`（历史 `tokenizer: decode` 口径）

| token 数 | scope mean | p50 | 每 token |
|---|---|---|---|
| 220 | **42.6 µs** | 41.5 | 0.194 µs |
| 990 | **164.2 µs** | 161.7 | 0.166 µs |
| 8140 | **1497.9 µs** | 1396.6 | 0.184 µs |

线性、每 token 约 **0.17–0.20 µs**，比 encode 便宜约 **8 倍**。
注意这个 scope 在真实服务里**极少触发**（只有 `needs_detokenization` 的请求），
历史日志里计数为 0 是负载形态决定的，不是"不耗时"。

### 4.2 `detokenize: stream`（生成期，**无插桩**）

PSL≈990，混合语料（英/中/代码），生成 token 覆盖多字节 CJK。
逐 index 剖析后发现成本是**三段结构**，不是均匀的每 token 成本：

| 段 | 含义 | fast | slow |
|---|---|---|---|
| `init` | `from_new_request()` | **16.6–17.1 µs** | 16.2–16.7 µs |
| `first_step` | 第 1 次 `update()`（惰性建 prompt 前缀状态） | **215.6–257.5 µs** | 219.9–291.9 µs |
| `steady` update | 稳定期每次 `update()` | **1.135–1.205 µs** | 1.206–1.290 µs |
| `steady` emit | 每次 `get_next_output_text(delta=True)` | **0.260–0.304 µs** | 0.293–0.302 µs |
| **稳定期合计** | **进 TPOT 的部分** | **1.394–1.509 µs/token** | **1.504–1.590 µs/token** |

整请求实测（三段之和的交叉校验）：

| OSL | fast full | slow full |
|---|---|---|
| 32 | 262.1–467.2 µs | 262.1–270.3 µs |
| 256 | 532.4–586.6 µs | 521.8–599.0 µs |
| 1024 | 1517.2–1540.5 µs | 1575.2–1635.1 µs |

> **区间表示跨运行**：上表每个格子给了本线两次运行的极值，不是误差棒。
> OSL=32 的 fast 臂出现 262 → 467 µs 的跳动，是因为该臂的「整请求」里
> 首步成本（215–257 µs）占比极高，而首步对机器负载很敏感。
>
> 交叉校验的偏差在 OSL=256/1024 上是 **+3% ~ +20%**（预测值偏高），
> 在 OSL=32 上是 **−41% ~ +15%**。原因：三段是**分别测量**的（每次都是
> 一个新 detokenizer 实例），而整请求臂把它们串在同一个实例上。
> **引用 per-token 成本时请用三段表，引用整请求成本时用 full 表，
> 不要互相换算。**

**这两条路的实测差在 1% 以内**（本线两次运行分别 7% 与 0.4%），即在本装置上
**「快路」并没有真的更快**（因为本机 HF tokenizer 的快路走 `DecodeStream`，
慢路走 Python 前缀偏移，两者在 OSL ≤ 1k 上打平）。设计上它们差别很大
（Rust vs Python 循环），但**在本装置 + 本语料上量级相同**——不要据此说
"Fast 永远没有收益"，只能说"**在 OSL ≤ 1k 的 mix 语料上收益未显现**"；
更长的 OSL、更多字节回退（emoji/罕见 CJK）下两者的差距可能才显现，本文未测。

**正确性交叉校验**：同一 token 序列，两条路输出的文本**完全一致**
（OSL=32：169 字符两边相同；OSL=1024：5465 字符两边相同）。

> 一个易踩的坑：`FastIncrementalDetokenizer` 的第 1 次 `update()` 要付
> ~231–263 µs（把整个 prompt 灌进 `DecodeStream` 做 native prefill）。
> 这不进 `from_new_request` 的账，所以**只测构造函数会严重低估**，
> 只测"整请求 ÷ OSL"又会在小 OSL 上严重高估（32 token 时 9.7 µs/token）。

## 5. E2.4 三段占 TTFT / TPOT

占比表：`data/cost/e2_share.csv`，图 `figures/fig-02-share-of-ttft-tpot.svg`。
**分母是引擎侧下界，所以占比是上界**（§1.5）。三个真机锚点：

| anchor run | `tokenizer: encode` p50 | `http:` p50 | `Step:Model` prefill p50 | decode 步 p50 |
|---|---|---|---|---|
| `liteprof_v1_torch_uni_...T2142Z` | 524.9 µs | 583.2 ms | 199.2 ms | 422.4 µs |
| `liteprof_v1_torch_uni_wide_...T2158Z` | 484.3 µs | 603.2 ms | 209.1 ms | 425.3 µs |
| `liteprof_v2_only_uni_wide_...T2148Z` | 391.6 µs | 305.7 ms | 101.0 ms | 240.4 µs |

同装置可比的两行（**这才是能直接引用的**）：

| 分子 | 分母 | 占比 |
|---|---|---|
| `tokenizer: encode` 524.9 µs | `http: create_completion` 583.2 ms | **0.090%** |
| `tokenizer: encode` 524.9 µs | `Step:Model` prefill 199.2 ms | **0.264%** |
| `tokenizer: encode` 391.6 µs | `http:` 305.7 ms | 0.128% |
| `tokenizer: encode` 391.6 µs | `Step:Model` prefill 101.0 ms | 0.388% |

跨装置上界参考（本机分子 ÷ 真机分母，标 cross-device）：

| 分子（本机） | 分母 prefill 199.2 ms | 分母 prefill 101.0 ms |
|---|---|---|
| `encode` @220 token 286 µs | **0.14%** | 0.28% |
| `encode` @990 token 1100 µs | **0.55%** | 1.09% |
| `encode` @8140 token 10537 µs | **5.29%** | **10.43%** |
| `detokenize: stream` fast 1.51 µs/token | **0.357%**（占 decode 步 422 µs） | **0.628%**（占 240 µs） |
| `detokenize: stream` slow 1.51 µs/token | 0.358% | 0.630% |

**读法**：

1. 几百 token 的短 prompt 上，tokenizer 占 TTFT **不到 1%**——历史数据里
   那些"163–613 µs"的 encode 确实是小项，这个结论站得住。
2. **8k prompt 上占比跳到 5.9–11.7%**（上界口径）。这是"encode 随 ISL 线性增长、
   而 prefill 近似二次"两者相对关系的转折区，是本文最值得记住的定量结论。
3. 流式解码占 TPOT **≈0.3–0.6%**（上界）。TPOT 的分母是模型一整个 decode 步
   （420 µs / 240 µs），1.4 µs 的 detokenize 在其中很小。
   但注意 OSL 越大，detokenize 是**每 token 恒定**而模型步也恒定 ⇒ 两者比值
   不随 OSL 变化，占比稳定在千分之几。

## 6. E2.5 前端并发曲线

装置：复现前端结构（asyncio 事件循环 + ThreadPoolExecutor），不启 HTTP server；
chat+tools、ISL=1024、每点 128 请求、4 核容器。
图：`figures/fig-02-frontend-concurrency.svg`。

| `renderer_num_workers` | tokenizer 池 | 饱和并发点 | 饱和吞吐 | 峰值 p50 延迟 |
|---|---|---|---|---|
| 1（默认） | 2 | **2–4** | ~740–770 req/s | 1.4 ms @c=1 → 80–107 ms @c=64 |
| 4 | 5 | **16–32** | ~2200–2470 req/s | 1.4–1.7 ms @c=1 → 23 ms @c=64 |
| 8 | 9 | 16–32（受核数限制） | ~1970–2030 req/s | 1.7–2.1 ms @c=1 → 26 ms @c=64 |

三条实测结论：

1. **默认 `renderer_num_workers=1` 时，前端在 4 并发附近就饱和**，
   之后吞吐持平甚至下降（c=64 时吞吐掉到 562 req/s），而延迟线性上升
   （c=64 时 p50 107 ms）。
2. 提到 4 后吞吐约 **3.2×**（4 核容器）；再提到 8 **反而更低**
   （~2000 vs ~2400 req/s）——4 核已经被用完，多出的线程只增加 context switch
   与 tokenizer 池竞争。
3. **deepcopy 增长路径在这条路径上不会触发**：`renderer_num_workers=N` 时
   线程池 N 个线程、tokenizer 池 N+1 份，并发借用数最多 N < N+1，
   所以"池空 ⇒ 现场 deepcopy 且池无上限增长"（`vllm/tokenizers/hf.py:51-55`）
   在默认 async 路径上**测到 0 次**（21 个配置点全部 0）。

### 6.1 但那个路径确实存在，而且很贵（受控实验）

用一个**故意超额**的借用实验（8 线程 vs `renderer_num_workers=2` 的 3 份池）
强制触发：**5 次现场 deepcopy，平均每次 3.64 s**；同参数下 3 线程（池内够用）
触发 **0 次**。

同一条成本也出现在**启动期**：`HfRenderer.__init__` 会按
`renderer_num_workers + 1` 预建池，用 deepcopy 计数包抄实测：

| `renderer_num_workers` | 池 | 构造 renderer 用时 | deepcopy 次数 | deepcopy 总时长 | 平均 |
|---|---|---|---|---|---|
| 1 | 2 | 2.75 s | 7 | 2.45 s | 350 ms |
| 2 | 3 | 3.48 s | 8 | 3.07 s | 384 ms |
| 4 | 5 | 6.61 s | 10 | 5.94 s | 594 ms |
| 8 | 9 | 9.54 s | 14 | 9.17 s | 655 ms |

（deepcopy 次数多于池容量，说明除池以外还有别的调用方在拷贝 tokenizer。）

**实践含义**：调大 `renderer_num_workers` 换吞吐，代价是**启动期线性变慢**
（8 worker 约 9.5 s），以及每次超额借用的 0.35–0.65 s 卡顿。
这是一条"能换吞吐但要付启动时间"的旋钮，不是免费的。

## 7. E2.6 历史数据回收

`data/historical/historical_tokenizer_scopes.csv`（按 run 一行）+ `historical_runs.json`。
生成脚本 `scripts/collect_historical.py`（**只读**，全程不写任何历史目录）。

**覆盖情况：52 个 run 全部回收成功**（远端 48 + 本地 4）。

| source | run 数 | 说明 |
|---|---|---|
| `remote-a322` | **48** | a3-22 的 `liteprof_wave*`（wave1/3/4/5/6 × chip × v1/v2） |
| `local-a321` | 3 | `liteprof_v1/v2_*_uni_*`（每个 8 次 encode） |
| `local-preparing-input-phase` | 1 | 上个项目的 `real-b1`（10 次） |

`encode` 均值范围 **162.3 – 2382.0 µs**；其中 **16 个 run 是 41 次采样的稳态样本**
（wave3 的 v2 与 wave4 的 v1），范围 **162.3 – 461.0 µs**，中位数 275.6 µs。
这与任务书里给的"5 个 count=41 的稳态样本，avg 163–461 µs"**完全吻合**，
且本次多找到了 11 个同类样本。

**`decode` / `render_messages` 在 52 个 run 里全部为 0**，且这不是"不耗时"的证据：
这些 run 发的全是 `/v1/completions` + 文本 `prompt`（响应体 `object=text_completion`），
按 §1.2 的触发条件，chat 分支与 `needs_detokenization` 分支根本没有进入。

### 7.1 一个必须写下来的发现：稳态样本的 prompt 只有 10–11 token

wave3/wave4 的响应体里带 `usage`，从中可以直接读到该 run 的 prompt 规模：

```
prompt_tokens_from_usage = 10;11     （16 个稳态 run 全部如此）
completion_tokens_from_usage = 64    （max_tokens=64，ignore_eos）
```

也就是说，**"163–461 µs"这个历史数字对应的是 10–11 token 的 prompt**，
不是"数百 token"。这个口径修正很重要，因为它把该数字的性质从
"长 prompt 的成本"变成了"极小 prompt 下的成本"，进而和本机的短 prompt 曲线对撞：

| 装置 | prompt token | encode mean |
|---|---|---|
| 本机 x86 容器（本次，两次运行） | 10 | **36.2 / 47.4 µs** |
| a3-22 NPU 前端（历史稳态 16 run） | 10–11 | **162.3 – 461.0 µs** |

**同一量级的 prompt，相差 3.4–12.7 倍。** 这不能用固定成本解释（本机固定成本
只有 25–45 µs，历史值远高于此）。可能的解释有三类，**本文不判定哪一个**，
需要同装置复核才能定：

1. **机器差异**：aarch64 前端 vs x86，主频、NUMA、当时负载（历史 run 是
   wave 并发采集，同机可能在跑别的负载）；
2. **插桩开销**：历史 run 开着 LiteProfiler，本机 §2 的主数据**没有**插桩
   （只有 `bench_litescope.py` 开了）；LiteScope 每次 scope 要
   `open/append/close` 一次文件，在小 prompt 上固定开销的占比最大；
3. **首次调用/池冷启动**：wave1 类 run 每个只有 1 次 encode（均值 253–2382 µs），
   明显包含冷启动；稳态 run 复用同一进程，但若并发下每次借用都要 deepcopy
   （§6.1），10 token 的请求也会被摊上毫秒级成本。

**结论**：引用"163–461 µs"时必须同时说明**它是 10–11 token prompt 下的数、
且开着 LiteProfiler**；不要把它当成"几百 token 的典型成本"，也不要直接和
本机 x86 的数字相减。这条修正正是"跨装置不可比"（§5.4）的一个实例。

### 7.2 拉取过程（供复跑参考）

2026-09-24 23:38 起 a3-22 走 ProxyJump 不稳（反复 `Connection closed by UNKNOWN`），
`scripts/fetch_remote_bg.sh` 后台重试到 00:2x 链路恢复，一次性拉完 48 个 run
（每个约 56 KB：`lite-profiler/lite.log` + 响应体，**不拉 torch trace**）。
全程只读。远端 raw run 未进 git（在 `/tmp/c-remote`）；进包的是汇总 CSV/JSON。

## 8. E3 火焰图

### 8.1 装置（与 E2 不同，必须说明）

- 镜像里**没有 perf、没有 flamegraph**（实测 `which perf` 为空）。采样改在
  **宿主机**做（perf 7.1.6 + `inferno-collapse-perf` + `inferno-flamegraph`），
  被采样的 python 进程仍在同一镜像、同一容器里跑同一套代码路径。
- 容器进程在宿主机上是 **root**（uid 0），`perf_event_paranoid=2` 下普通用户
  不能给它开事件 ⇒ 必须 `sudo -n perf`。
- 采样参数：`-e cpu-clock -F 99 --call-graph dwarf,16384`，容器 2 核（cpuset 4-5）。
- **权重单位是纳秒**：`e3-1a` 总权重 4.6455e10 ns = 46.5 s，窗口 70 s、单线程，
  量级吻合。
- Python 3.12 的帧在堆上，perf 看不到 `py::` 帧（本机 perf **没有 JIT 接口选项**），
  所以图上是 C/Rust 帧；Python 层的归属由 §2–§4 的微基准给出。

### 8.2 四张图

| 图 | 被测路径 | 备注 |
|---|---|---|
| `figures/e3-1a-python-frontend.svg` | encode + render + detokenize 混合 | 70 s 窗口 |
| `figures/e3-1b-python-frontend.svg` | **仅 encode** | ISL=1024 |
| `figures/e3-1c-python-frontend.svg` | **仅 render_messages（chat+tools）** | ISL=1024 |
| `figures/e3-1d-python-frontend.svg` | **仅 detokenize:stream** | OSL=256 |

SVG 里帧宽太窄时 inferno 会**截断函数名**，所以精确的帧名与权重另出一张表：
`data/cost/e3_flamegraph_frames.csv`（`scripts/analyze_flamegraph.py` 生成，
每个图导出前 400 个帧的 inclusive/self 权重 + 按类别汇总）。

### 8.3 结论：Rust 帧落在哪（self-time 占比，各帧 self 求和 = 100%）

| 类别 | encode | render(chat+tools) | detokenize |
|---|---|---|---|
| `rust-tokenizers`（`tokenizers::*`） | **15.9%** | 5.8% | 13.6% |
| `rust-stdlib-or-dep`（core/alloc/hashbrown/regex/serde…） | **48.1%** | 12.6% | 26.1% |
| CPython | 9.5% | 14.9% | **34.3%** |
| libc 动态符号 + malloc/free | 6.8% | **54.9%** | 10.0% |
| 其它 | 19.7% | 11.8% | 16.0% |

**三条可引用的结论**：

1. **encode 是 Rust 主导的**：Rust 两类合计 **64%**，CPython 只占 9.5%。
   但 Rust 里最大头**不是 BPE merge**——`BPE::tokenize` 的 inclusive 只有 ~1%。
   真正花时间的是**词表/缓存查找与正则预分词**：
   `core::hash::BuildHasher::hash_one`（8.5% self）、
   `hashbrown::raw::RawTable::reserve_rehash`（5.4%）、
   `hashbrown::map::HashMap::insert`（4.9%）、
   `match_at`/`search_in_range`（正则匹配，5.1%+）、
   `tokenizers::utils::cache::Cache::get`（3.2%）。
   ⇒ 这一层还有优化空间（更便宜的哈希 / 预分配 / 关缓存换内存）。
2. **chat 渲染与 tokenizer 无关的部分占绝对多数**：libc + malloc/free 合计
   **约 55%**，CPython 15%，Rust 只有 18%。Jinja 拼字符串与
   `json.dumps` tools 定义是堆分配大户（图上能看到
   `encoder_listencode_obj`/`_dict`/`_key_value` 这条 serde_json 链）。
   与 §3.1 的常数级 Jinja 时间互相印证。
3. **detokenize 里 Python 侧占比最高**（CPython 34%），
   Rust 侧的成本集中在 `ModelWrapper::id_to_token`（4.2% self）与
   `AddedVocabulary::is_special_token`（2.6% self）——即**每步都要做
   「id → token 串 + 特判」**。这与 §4.2 的 1.37 µs/token 稳定成本吻合。

## 9. 探针代价（硬要求）

历史上这两个数：49 个探针 **+9.5%**、pystack 采样器 **+24%**。本线的对照：

| 测量装置 | 是否引入额外探针 | 代价 |
|---|---|---|
| `bench_paths.py`（本文 §2–§4 主数据） | **不加**：只用 `time.perf_counter_ns()` 包住被测调用 | 每次 `perf_counter_ns` 约 20–70 ns，对 283 µs 起的量级 < 0.03% |
| LiteScope 插桩（`bench_litescope.py` 交叉校验） | 每个 scope 两次取时钟 + 一次 `open/append/close` | 见下 |
| perf 采样（E3.1） | `cpu-clock` @99 Hz + dwarf 16 KiB | 见 `--cost-control` 输出与 REPORT |

**LiteProfiler 自身的开销（实测对照）**：同一进程、同一语料，把
`LiteScope` 打开 vs 只用自己的 `perf_counter` 计时，**同一进程、同一语料、
同一轮**里直接对比（`data/cost/e2_litescope.json`，各 300 次采样，ISL 混合）：

| scope | 真实 LiteScope 读数 | 本 harness 自测 | 相对差 |
|---|---|---|---|
| `tokenizer: encode` | **3947.8 µs** | 4008.4 µs | **1.5%** |
| `tokenizer: render_messages` | **4922.8 µs** | 4813.5 µs | **2.3%** |
| `tokenizer: decode` | **762.6 µs** | 785.0 µs | **2.9%** |

**这是本文最重要的一条自证**：本 harness 的 scope 边界与上游 **原版**
`record_function_or_nullcontext` 插桩点在 **3% 以内**一致，
所以 §2–§4 用自测 `perf_counter` 得到的数字可以按 LiteScope 的口径引用；
同时它也说明**LiteScope 本身在这个量级上的开销 < 3%**（不是历史那种 +24%）。

log 里还可以直接读出线程证据：三个 `tokenizer:` scope 的 tid 全为 **1**、
`tokenizer_scopes_single_frontend_thread = true`——本装置是单线程跑的，
与 §1.1 讨论的多线程形态不冲突（那里是 API server 的实际线程布局）。

## 10. 怎么复跑

```bash
# 1) 三个 scope 的主数据（约 60–90 s）
COST_CORES=4-5 /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh \
  ./scripts/c_cost_run.sh harness/python/bench_paths.py \
  --tasks encode,render,decompose --out /workspace/data/cost/e2_paths.json

# 2) 生成期流式解码
COST_CORES=4-5 /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh \
  ./scripts/c_cost_run.sh harness/python/bench_detokenize.py \
  --osl 32,256,1024 --out /workspace/data/cost/e2_detokenize.json

# 3) 并发曲线
COST_CORES=4-7 /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh \
  ./scripts/c_cost_run.sh harness/python/bench_concurrency.py \
  --worker-counts 1,4,8 --out /workspace/data/cost/e2_concurrency.json

# 4) 真实 LiteScope 交叉校验（会先构建插桩 overlay）
COST_CORES=4-5 /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh \
  ./scripts/c_cost_run.sh --liteprof harness/python/bench_litescope.py

# 5) 占比组装 + 出图
python3 scripts/compose_e2_4.py --cost-json data/cost/e2_paths.json \
  --cost-json data/cost/e2_detokenize.json --anchor-run <run_dir> ...
python3 scripts/make_figures.py
```

每个脚本都支持 `--help`。manifest（commit / 镜像 id / 模型 revision / 脚本 sha256 /
时间戳 / 绑核 / load average）写在每份 JSON 的 `manifest` 字段里。

## 11. 未测与待验证

| 项 | 状态 |
|---|---|
| 远端 29 个 `liteprof_wave*` run 的 scope 回收 | **未测**（a3-22 链路不通，脚本与 CSV 结构已就绪） |
| 真机 chat 负载锚点（a3-22 跑一个 chat 点测 `render_messages`） | **未测**（同上） |
| 批量/多请求同时 encode 的摊销 | **未测**（本文全是单请求） |
| `VLLM_USE_FASTOKENS=1` 下的三个 scope | **未测**（B 线范围） |
| `renderer_num_workers > 8`、NUMA 绑定下的曲线 | **未测** |
| gigatoken 适配层下的 scope 成本 | **未测** |
| `decode: prompt_reverse` 在真实服务里的触发频率 | **未测**（需要带 `needs_detokenization` 的流量） |
| 非 Qwen3 系 tokenizer（BPE vs tekken vs tiktoken）的这三条成本 | **未测** |
