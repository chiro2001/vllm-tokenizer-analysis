# vLLM tokenizer 调研 —— 执行计划

> 版本 1.0 / 2026-09-24。配合 `COORDINATION.md` 使用。

## 0. 一句话定位

**不追求把每条线都跑一遍，而是交付一份"够用且可信"的调研**：把 tokenizer 的
边界、成本、多后端成色、下一代形态讲清楚，每组数字都有出处。

规模目标：**7 篇文档 / 1500–2000 行 / 10–15 张图**（上个项目是 10 篇 4000+ 行 43 图）。

## 1. 交付物

| # | 文档 | 回答什么 | 负责 |
|---|---|---|---|
| 00 | `docs/00-INDEX.md` | 一分钟结论 + 导航 | 根代理 |
| 01 | `docs/01-code-logic.md` | 边界在哪、两代链路、进程模型、`TokenizerLike` 协议 | A |
| 02 | `docs/02-cost-and-share.md` | 花多少、占 TTFT/TPOT 多少、随什么变化 | C |
| 03 | `docs/03-backend-matrix.md` | **主表**：Rust 六后端 + Python 三实现 | B |
| 04 | `docs/04-next-gen-rust-frontend.md` | `vllm-rs` 是什么、搬了什么进 Rust、值不值 | D |
| 05 | `docs/05-gigatoken-audit.md` | 它的"快 1000×"到底在说什么 | B |
| 06 | `docs/06-conclusions.md` | 现状判断 + 选型建议 + 待验证项 | 根代理 |

### 1.1 两条主线结论（本次调研的核心价值）

1. **vLLM 0.26.0 已自带 Rust 前端**（`rust/`，`VLLM_USE_RUST_FRONTEND=1`），
   其中 `vllm-tokenizer` crate 把 **fastokens 作为一等后端**，
   并提供 tiktoken-rs / riptoken / tekken 多后端抽象。多数人还在争论
   Python 侧要不要开 `VLLM_USE_FASTOKENS`，而换代已经在进行。
2. **gigatoken 的"~1000×"是批量文件吞吐口径**：其官方 benchmark 对 HF 只喂
   100 MB 且按文档切碎（20342 个文档），对 gigatoken 喂整个 11.9 GB 当单文档。
   输入量差 119×、并行粒度不同、token 计数也不同。必须解剖后按 vLLM 的负载重测。

## 2. 评估对象全集

### 2.1 下一代：`rust/src/tokenizer` 的六个后端

| # | 家族 | 后端 / 构造函数 | 工件 |
|---|---|---|---|
| 1 | HF | `HuggingFaceTokenizer::new_hf` | `tokenizer.json` |
| 2 | HF | `HuggingFaceTokenizer::new_fastokens` | `tokenizer.json` |
| 3 | HF | `Backend::FastokensByteLevel`（自动判定，无独立入口） | `tokenizer.json` |
| 4 | Tiktoken | `TiktokenTokenizer::new_tiktoken_rs` | `tiktoken.model` |
| 5 | Tiktoken | `TiktokenTokenizer::new_riptoken` | `tiktoken.model` |
| 6 | Tekken | `TekkenTokenizer::new` | `tekken.json` |
| — | 共享 | `incremental.rs::DecodeStream` | 六个后端共用 |

已有上游 bench：`rust/src/tokenizer/benches/hf.rs`（1 vs 2）、`benches/tiktoken.rs`（4 vs 5）。
**缺**：3 与 6 没有 bench；且现有 bench 从 `hf-hub` 联网拉模型，要改成**本地路径**。

### 2.2 当代：Python 前端的三个实现

| # | 实现 | 开启方式 |
|---|---|---|
| 1 | HF `TokenizersBackend`（`tokenizers` 0.22.2） | 默认 |
| 2 | fastokens（`fastokens.patch_transformers()`） | `VLLM_USE_FASTOKENS=1` |
| 3 | gigatoken（`Tokenizer(...).as_hf()`） | 无 vLLM 集成，需自建适配 |

## 3. 实验清单（只有三组）

### E1 微基准矩阵（主力，本地无卡）

一个自建 Rust harness（criterion），**同条件**跑上表 2.1 的六后端 + gigatoken（sdist 源码 vendor）。

| 维度 | 取值 |
|---|---|
| 输入长度 | 128 / 1k / 8k tokens 等价文本 |
| 语料类型 | 中文 / 英文 / 代码 / 中英混合 |
| 操作 | 单条 encode、批量 encode、decode、增量 decode |
| 线程 | 1（在线路径）与默认并行（批量路径）分别测 |
| 正确性 | **同家族内断言 token ids 逐一致**（HF 家族 1/2/3；tiktoken 家族 4/5） |

**桥梁校准**：fastokens 与 HF `tokenizers` 在 Rust 与 Python 两个装置各测一次，
吻合才能把两侧数字放进同一张表。

### E2 成本与占比

- **本地**：最小 harness 直接调 vLLM 的 renderer/tokenizer 代码路径（不启引擎、不需要 NPU），
  把 `tokenizer: encode` / `decode` / `render_messages` **三个 scope 测全**——后两个历史上零实测。
- **真机锚点**：a3-22 跑**一个** chat 负载点做锚定（可选但推荐）。
- **历史回收**：29 个远端 + 3 个本地 run 的 `tokenizer: encode`，汇总成 CSV + 口径校准。

### E3 火焰图（2 张）

1. Python 前端（`renderers` + `tokenizers` 路径）——看 Rust 帧落在哪。
2. Rust 前端（`vllm-tokenizer` 内部）——看时间在预分词器还是 BPE merge。

工具：现成的 `cargo-flamegraph` + perf，不新造轮子。

## 4. 明确不做（避免范围蔓延）

| 不做 | 原因 |
|---|---|
| 全平台矩阵（x86 + aarch64 全覆盖） | aarch64 只用现成历史数据说明差异方向 |
| 复现 gigatoken 的 10 GB 级 benchmark | 读其 `measure.py` + `results.json` 解剖口径已足够定性 |
| 找 Mistral 真模型测 tekken | 用小 fixture 验路径，标注局限 |
| `vllm-rs` 端到端完整跑通 | 从官方 wheel 抽二进制做**一次启动验证**；跑不通降级为源码级分析并如实标注 |
| PD 分离、多模态、对抗性安全测试 | 文档里记一句"存在但未评估" |
| 全量逐字节 detokenize 比对 | 只比代表案例 |

## 5. 验收判据

| 维度 | 判据 |
|---|---|
| 后端覆盖 | 六个 Rust 后端 + 三个 Python 实现**全部有数**；未测的必须显式标注原因 |
| 正确性 | 同家族内 token ids 一致；不一致的要给出反例与影响面 |
| 跨装置 | 桥梁校准两个实现在两装置的相对差 ≤ 15%，否则不给跨装置结论 |
| 占比 | `encode`/`decode`/`render_messages` 三个 scope 都有数，且写清分子分母 |
| gigatoken | 必须同时给出"其口径下的数"与"vLLM 负载下的数" |
| 可复现 | 每条实验都能从 `scripts/` 或 `harness/` 一键复跑 |

## 6. 环境提示（省得每个 agent 重新踩）

```bash
# 容器（无卡，纯 CPU 跑 vLLM tokenizer 代码路径）
docker run --rm -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64 \
  -v /home/chiro/models:/models:ro -v $PWD:/work -w /work \
  local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922 <cmd>

# 容器内直接走 tokenizer，绕开平台插件
python3 -c "from vllm.tokenizers.hf import CachedHfTokenizer; ..."

# 本地 Rust
export PATH="$HOME/.cargo/bin:$PATH"   # rustc 1.98 / cargo / flamegraph

# 本地 pip 走清华源（已配）
~/miniforge3/bin/pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple <pkg>
```

已知坑：`~/miniforge3` 里**没有** transformers/tokenizers；容器里 `import vllm` 顶层会触发
torch_npu 加载。vLLM 源码是**浅克隆**（只有 0.26.0 一个 commit），历史考古要去远端或上游。

## 7. 时间盒

每条线**独立可交付**。若某线超时，优先保证：主表（03）有数、占比（02）有数、
下一代（04）有定性结论。火焰图（E3）是加分项，可降级为一张。
