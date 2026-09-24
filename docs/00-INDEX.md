# vLLM 0.26.0 tokenizer 链路调研

> 分析对象：**vLLM 0.26.0**（commit `568afb3a13806beb53bb2e6bd518269357b237c0`）
> 运行时：Python 3.12.13 / **transformers 5.14.1** / **tokenizers 0.22.2** / glibc 2.38
> 采集环境：x86_64 开发机（无卡，纯 CPU）+ aarch64 Kunpeng 920B 历史数据
> 采集时间：2026-09-24（Asia/Shanghai）

> **状态**：四条线（A 边界 / B 后端 / C 成本 / D 下一代）**全部完成并合并**。
> 本文为收口版。

---

## 0. 一分钟结论

**1. 链路只有三段真实工作量，且全在 API server 前端进程。**
入站 encode、出站流式 decode、停止串判定；engine core 不在这条链路上
（唯一例外是 structured output 会在 engine core 另加载一份 tokenizer）。
见 [`01-code-logic.md`](01-code-logic.md) §1。

**2. 成本按 ISL 分层：encode 线性、Jinja 常数。**（Qwen3-0.6B / 单请求 / x86 2 核）

| 环节 | @220 tok | @1k | @8k | 边际成本 |
|---|---:|---:|---:|---:|
| `tokenizer: encode`（completion 路径） | 286 µs | 1 100 µs | 10 537 µs | **≈1.47 µs/token**（R²=0.99975） |
| `render_messages`（chat + tools，**含模板内 encode**） | 1 015 µs | 2 075 µs | 12 231 µs | 同上 |
| ├ 其中 Jinja 渲染 | 63 µs | 69 µs | 63 µs | **常数级** |
| └ 其中模板内 encode | 603 µs | 1 403 µs | 10 009 µs | 随 ISL 线性 |
| `decode: prompt_reverse`（历史口径） | — | — | — | 0.17–0.19 µs/token |
| `detokenize: stream`（生成期解码） | — | — | — | **1.39–1.59 µs/token**（另有首步 216–292 µs） |

⇒ **优化 chat 请求的前端成本，矛头应指向编码速度，不是 Jinja**
（Jinja 在 128→8k 全程都是同一个量级，换模板引擎收益有限）。
占 TTFT 的比例：**1k prompt ≈0.3–1.2%、8k prompt ≈5.3–10.4%**（跨装置上界口径）；
**8k 是分水岭**，再往上 tokenizer 开始进入 TTFT 的两位数百分比。
短 prompt（<64 token）**有 25–45 µs 固定成本**，线性外推会给出负数。
见 [`02-cost-and-share.md`](02-cost-and-share.md)。

⚠️ **三条必读的口径修正**（C 线与 A 线独立发现，推翻了此前的直觉）：

1. **chat 请求永远不会触发 `tokenizer: encode` scope**——chat 的编码发生在
   `apply_chat_template(tokenize=True)` 内部，被 `render_messages` 包住。
   两个 scope 是**互补负载**，不存在"可以相加"的请求形态。
2. **`tokenizer: decode` 不是生成期解码**，它只是 prompt token ids 反解；
   生成期逐 token 解码在 `vllm/v1/engine/detokenizer.py`，**LiteProfiler 没有插桩它**。
3. **`tokenizer_pool_size` 在 0.26.0 不存在**；真实参数是 `renderer_num_workers`，
   池空时**不阻塞**而是现场 deepcopy 并让池无上限增长（A 线静态发现）。
   **C 线实测补充**：默认 async 路径上**池空分支命中 0 次**——因为池 = `N+1` 份
   而工作线程只有 `N`，天然不会取空；但代价转移到**启动期**——
   `renderer_num_workers=8` 时渲染器构造要 **9.5 s**（池预建 9 份，0.35–0.65 s/份），
   而强制触发现场 deepcopy 平均要 **3.6 s**。⇒ 调大这个旋钮是
   "换吞吐但付启动时间"，不是免费的（`02` §6）。

**3. 多后端成色：fastokens 的加速是真的，但要分清是哪一段。**
根代理与 B 线用**两套独立装置**互证：

| | HF `tokenizers` | fastokens | 倍数 |
|---|---:|---:|---:|
| encode（短） | 130 µs | **14.0 µs** | **9.3×** |
| encode（长） | 1080 µs | **80.3 µs** | **13.5×** |
| decode（Python 侧） | 17.0 µs | 15.7 µs | 1.09× |
| decode（**Rust 侧旁路**） | 20.1 µs | **2.5 µs** | **8.0×** |

⇒ **encode 的 8–13× 两端都有**（fastokens 引擎本身的能力，Python 用户开
`VLLM_USE_FASTOKENS=1` 同样受益）；**decode 的 8× 只在 Rust 侧**，来自
`Backend::FastokensByteLevel` 旁路（跳过 `Vec<String>`/`join`），
**那是 `vllm-tokenizer` crate 自己写的，不是 fastokens 提供的**。
见 [`03-backend-matrix.md`](03-backend-matrix.md)、[`../data/crosscheck/README.md`](../data/crosscheck/README.md)。

**4. 下一代不是"更快的 tokenizer"，是北向服务层整体换语言。**
vLLM 0.26.0 已自带 Rust 前端（`VLLM_USE_RUST_FRONTEND=1`）：API server、OpenAI 协议、
chat 模板（Jinja→minijinja）、tokenize/detokenize、parser 全部进 Rust 进程，
Python 侧只剩 engine core。**Python 前端要手动开的 `VLLM_USE_FASTOKENS=1`，
在 Rust 前端里是默认路径。** 官方 wheel 自带 `vllm-rs` 二进制（零编译可运行）。
见 [`04-next-gen-rust-frontend.md`](04-next-gen-rust-frontend.md)。

**5. gigatoken 的"~1000×"是批量文件吞吐口径，不是在线服务口径。**
其官方 benchmark 对 HF 只喂 100 MB 且按文档切碎（20 342 个文档），
对 gigatoken 喂整个 11.9 GB 当**单文档**（内部 chunk + rayon 并行）。
拆解结果：**"~1000×" = 119× 输入量差 × 约 2.4× 单位吞吐差**；
同条件（同 4 MiB / 同文档边界 / 同线程）收敛到 **12.5–30×**。
在线单条它确实最快（mixed-1k mixed **11.9 µs** vs HF 1 021 µs），
**但缺 vLLM 依赖的流式 decode 与 `skip_special_tokens`**。
⇒ 落点是**离线批量**。见 [`05-gigatoken-audit.md`](05-gigatoken-audit.md)。

**6. 三个"会坏"而非"慢"的上游问题（本次最有实操价值的产出）。**

| # | 问题 | 后果 |
|---|---|---|
| a | **fastokens 0.2.1 越界 panic**（`split.rs:419`，`.min(pcre2.len())` 后接 `.max(2)` 导致 `pcre2[1]` 越界） | **panic 不是 `Err`，vLLM 的回落逻辑拦不住 ⇒ 8k 级中文 prompt 直接崩掉前端进程** |
| b | **tekken-rs 0.1.1 对不完整 UTF-8 报错**（而非返回替换字符） | vLLM 的 `DecodeStream` 依赖替换字符语义 ⇒ **6/12 语料组的流式 decode 不可用（中文/中英混合）** |
| c | **Rust 前端 7 个参数"接受但无效"**（Noop） | 传了不报错也不生效，静默不一致；比"未实现"更危险 |

**7. 火焰图显示热点不在 BPE merge。** encode 的 Rust 帧里
`BPE::tokenize` inclusive 只有 **~1%**，真正花时间的是**词表/缓存查找与正则预分词**
（`hash_one` 8.5%、`RawTable::reserve_rehash` 5.4%、正则匹配 5.1%）。
chat 渲染里 **libc+malloc/free ≈55%**（Jinja 拼串 + serde_json 序列化 tools），
detokenize 里 **CPython 占 34%**。见 [`02`](02-cost-and-share.md) §8 与
`figures/e3-1{a,b,c,d}-python-frontend.svg`。

---

## 1. 文档地图

| 文档 | 内容 | 状态 |
|---|---|---|
| [`01-code-logic.md`](01-code-logic.md) | 边界定义、两代调用图、进程模型、`TokenizerLike` 协议 | ✅ 完成（350 行） |
| [`02-cost-and-share.md`](02-cost-and-share.md) | encode / render_messages / 生成期解码三块成本与占比 | ✅ 完成（522 行） |
| [`03-backend-matrix.md`](03-backend-matrix.md) | **主表**：Rust 六后端 + Python 三实现 + 正确性 + 桥梁校准 | ✅ 完成（466 行） |
| [`04-next-gen-rust-frontend.md`](04-next-gen-rust-frontend.md) | `vllm-rs` 形态、抽取与 E2E 验证、值不值 | ✅ 完成（300 行） |
| [`05-gigatoken-audit.md`](05-gigatoken-audit.md) | 其"快 1000×"声明的解剖与同条件重测（P1/P2/P3 三档） | ✅ 完成（381 行） |
| [`06-conclusions.md`](06-conclusions.md) | 现状判断、选型建议、口径纪律、待验证项 | ✅ 完成 |

合计约 **2 300 行**中文正文 + 8 张图 + 12 份带 manifest 的数据文件。

数据与图：

| 目录 | 内容 |
|---|---|
| `data/backends/` | 后端矩阵原始数据 + manifest + 正确性断言 |
| `data/cost/` | 三块成本的原始测量 + manifest |
| `data/historical/` | 历史 `tokenizer: encode` 回收与口径校准 |
| `data/crosscheck/` | 根代理对 B 线的独立复核（双装置校准） |
| `data/nextgen/` | `vllm-rs` 抽取、启动、E2E 证据 |
| `harness/` | Rust（criterion）与 Python 两套 harness |
| `figures/` | 图（SVG/PNG） |

---

## 2. 口径纪律（引用任何数字前必读）

1. **进程归属**：`tokenizer:` 三个 scope 跑在 **API server 前端进程**，
   与 engine core 的 scope 混在同一份 `lite.log` 里，必须按 tid/pid 区分。
2. **`tokenizer: decode` 不是生成期解码**：它只覆盖"prompt token ids 反解回文本"，
   生成期流式解码在 `vllm/v1/engine/detokenizer.py`。两者必须分开命名。
   （历史采集几乎全是直发 token_ids 的 completion，所以该 scope 计数为 0 ——
   **不能据此说"不耗时"**。）
3. **`tokenizer_pool_size` 在 0.26.0 不存在**（全仓 0 命中）。
   真实参数是 `renderer_num_workers`，它决定线程池 worker 数与 tokenizer deepcopy 份数
   （`workers + 1`），且池空时不阻塞而是现场 deepcopy 并让池无上限增长。
4. **gigatoken 的倍数必须带前提**：输入量、文档边界、并行粒度三者都不同。
5. **跨装置比较必须经桥梁校准**：Rust criterion 与 Python wall clock 的数字，
   用 fastokens / HF `tokenizers` 两个双端都存在的实现校准后才可同表。
6. **`vllm-rs` 的参数支持度**：50 个未实现 + 7 个"接受但无效"，后者更危险
   （静默不一致）。

---

## 3. 复现入口

```bash
# 无卡容器（跑 vLLM 的 Python tokenizer / renderer 路径），自带资源上限
scripts/docker_run.sh -c 'from vllm.tokenizers.hf import CachedHfTokenizer; ...'
scripts/docker_run.sh --shell

# 资源纪律（本机脆弱，所有重活必过）—— 见 plan/COORDINATION.md §9
scripts/limit.sh <命令...>              # 绑 4 核 + 8 GiB + 4 并行编译
scripts/heavy_lock.sh <命令...>         # 全项目唯一重活锁
scripts/heavy_lock.sh --status

# 后端矩阵（Rust criterion）
cd harness/rust && ../../scripts/heavy_lock.sh ../../scripts/limit.sh cargo bench

# Rust 前端抽取与验证
harness/rust-frontend/verify_vllm_rs.sh --no-fetch
harness/rust-frontend/e2e_mock_engine.sh
```

完整流程见 [`06-conclusions.md`](06-conclusions.md) §复现 与各文档内的一键命令。

---

## 4. 资源纪律（本机约束）

本机 `LOCAL_HOST` 是共享开发机（12 核 / 29 GiB）。用户明确要求
**不要长时间占用过多 CPU（>75%）与内存**。所有 CPU/内存密集操作必须经
`scripts/limit.sh`（绑核 + 8 GiB 上限）与 `scripts/heavy_lock.sh`（全局唯一重活锁）。
详见 [`../plan/COORDINATION.md`](../plan/COORDINATION.md) §9。
