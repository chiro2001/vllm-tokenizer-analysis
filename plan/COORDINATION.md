# vLLM tokenizer 调研 —— 协作约定

> 建立：2026-09-24（Asia/Shanghai）。**所有 agent 动手前先读本文件与 `EXECUTION.md`。**

## 1. 目标（一句话）

说明 vLLM 0.26.0 里 tokenizer 链路的**边界、成本、多后端成色、以及下一代（Rust 前端）形态**，
每组数字都有出处，不做无证据的推断。

## 2. 唯一写入范围（硬约束）

| 位置 | 路径 |
|---|---|
| 本仓库（主工作区） | `/home/chiro/projects/vllm/tokenizer` |
| 各 agent 工作区 | `/home/chiro/projects/vllm/tokenizer-wt/<line>`（git worktree） |
| 本地只读参考 | `/home/chiro/projects/vllm/HIST_PROJECT/vllm`（vLLM 0.26.0 源码） |
| 本地只读参考 | `/home/chiro/projects/vllm/preparing-input-phase`（上个项目，方法论与历史数据） |
| 远端（可选） | `a3-22:~/projects/vllm/tokenizer` |

**禁止**：写 `HIST_PROJECT/`、`preparing-input-phase/`、`OTHER_PROJECT/` 等其他项目目录；
动宿主机系统配置；碰任何 NPU 设备（本调研纯 CPU）。

## 3. 统一身份（不要把版本混池）

| 组件 | 值 |
|---|---|
| vLLM | 0.26.0，commit `568afb3a13806beb53bb2e6bd518269357b237c0` |
| 本地源码 | `/home/chiro/projects/vllm/HIST_PROJECT/vllm`（**只读**） |
| 容器镜像 | `local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922`（id `d0e26eb95616`） |
| 容器内版本 | Python 3.12.13 / transformers 5.14.1 / tokenizers 0.22.2 / numpy 1.26.4 / glibc 2.38 |
| 本地工具链 | rustc 1.98.0 / cargo 1.98.0 / cargo-flamegraph 0.6.14 / perf 7.1.6 |
| 本机 | x86_64，12 核，29 GB RAM |
| 模型 | `/home/chiro/models/Qwen3-0.6B`（本地已有）；a3-22 有 Qwen3.5-0.8B/2B |

**容器使用注意**：镜像内 `torch_npu` 会因缺 `libascend_hal.so` 而 import 失败，
需加大 `LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64`
（该 .so 在镜像内存在），并设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
纯 tokenizer 路径其实**不需要 import vllm 顶层**，直接 `from vllm.tokenizers.hf import ...`
即可绕开平台插件。

**本机无 transformers/tokenizers**（miniforge 里只有 numpy/tiktoken）；
Python 侧实验统一在容器内跑，或先 `pip install tokenizers transformers`（清华源可用）。

## 4. 目录约定

```
plan/          计划与协调（仅根代理可改）
docs/          最终交付文档（中文，00-INDEX.md 为入口）
figures/       图（SVG/PNG），文件名 = 文档编号 + 语义
data/          结构化数据（CSV/JSON）；原始 perf.data 不进包
harness/       可复现 harness（rust/ 与 python/ 两个子目录）
scripts/       采集/解析/绘图/同步脚本，均需 --help
agents/<line>/REPORT.md   每个 agent 的交接报告（结论 + 失败路径 + 踩坑）
refs/          源码快照（只读）
```

## 5. 数据口径（引用任何数字前必读）

1. **进程归属**：`tokenizer: encode` / `decode` / `render_messages` 三个 scope 跑在
   **API server 前端进程**，与 engine core 的 scope 混在同一份 `lite.log` 里，
   必须按 tid/pid 区分（历史数据里前端进程 tid 各不相同）。
2. **负载形态**：历史采集绝大多数是**直发 prompt_token_ids 的 completion**，
   所以 `render_messages` 与 `decode` 计数为 0——**不能据此说它们不耗时**。
3. **gigatoken 口径**：其官方 benchmark 对 HF 只喂 100 MB 且按文档切碎，
   对 gigatoken 喂整个 11.9 GB 单文档，两者**输入量与并行粒度都不同**。
   引用其"1000×"必须同时给出这个前提。
4. **跨装置可比性**：Rust 侧（criterion）与 Python 侧（wall clock）的数字，
   必须用 `fastokens` 与 HF `tokenizers` 这两个**双端都存在的实现**做桥梁校准后
   才能放进同一张表；未校准的数不许跨装置比较。
5. **每条实验必须写 manifest**：commit、镜像 id、模型 revision、脚本哈希、时间戳、绑核。

## 6. 分工（4 条并行线）

| 线 | 内容 | 主产出 |
|---|---|---|
| `A-boundary` | 边界与静态链路 | `docs/01-code-logic.md` |
| `B-backends` | Rust harness + 六后端 + gigatoken | `docs/03-backend-matrix.md`、`docs/05-gigatoken-audit.md` |
| `C-cost` | 成本占比 + 火焰图 | `docs/02-cost-and-share.md`、`figures/` |
| `D-nextgen` | `vllm-rs` 抽取与验证 | `docs/04-next-gen-rust-frontend.md` |

根代理负责：`docs/00-INDEX.md`、`docs/06-conclusions.md`、合并、审阅、收口。

**冲突规则**：`docs/` 下每个文件**只有一个 agent 写**；`data/` 按子目录分
（`data/backends/`、`data/cost/`、`data/historical/`）；`harness/` 同理。
`plan/` 只有根代理能改。

## 7. Git 约定

- 每个 agent 在自己的 worktree 里提交，分支 `agent/<line>`。
- commit message 用中文，格式：`<line>: <做了什么>`。
- **禁止** `git push`、`git rebase`、`git reset --hard`、改 `main` 分支。
- 完成后在 `agents/<line>/REPORT.md` 写交接报告，然后通知根代理合并。

## 8. 红线

- 不碰 NPU / 不占卡（本调研纯 CPU，**不需要 chip 锁**）。
- 不删别人数据；不覆盖他人目录。
- 大文件（>50 MB）不进 git，用 `.gitignore` 挡住。
- 结论必须有据：**没测的就写"未测"，推断的必须标"推断"**。
  上个项目的教训是——推断被当成实测引用，后面要花大力气纠正。

## 9. 资源纪律（本机脆弱，硬性要求）

本机 `LOCAL_HOST` 是**共享开发机**，12 核 / 29 GiB，同时还有其他用户与会话。
用户明确要求：**不要长时间占用过多 CPU（>75%）与过大内存**。

### 9.1 强制上限

| 项 | 上限 | 说明 |
|---|---|---|
| CPU 占用 | **≤ 75%（即 ≤ 9 核）**，且**任意时刻全项目只允许 1 个 CPU 密集任务** | 基准测量本身建议只用 **2–4 核** |
| 内存 | 单任务 **≤ 8 GiB**；docker 另加 `--memory=8g --memory-swap=8g` | 本机 `available` 常态 ~17 GiB |
| 并行编译 | cargo `-j ≤ 4`；禁止全核 `make -j` | Rust + LTO 是内存大户 |
| 容器 | `--cpus=6 --memory=8g --shm-size=2g` | 上个项目同款配方 |

### 9.2 统一入口

**所有 CPU/内存密集操作都必须经 `scripts/limit.sh`**（默认绑 4 核 + 8 GiB）：

```bash
scripts/limit.sh cargo bench -p vllm-tokenizer
CORES=2 scripts/limit.sh ./target/release/bench     # 跑基准时收窄到 2 核更稳
scripts/limit.sh cargo build --release              # 编译同样受限
```

容器操作：

```bash
docker run --rm --cpus=6 --memory=8g --memory-swap=8g --shm-size=2g ...
```

### 9.3 基准测量的额外要求

- 绑到**固定核**（建议 `4-7`，把 `0-3`、`8-11` 留给交互），并在文档里写明用的哪几核。
- 记录**测量期间的系统负载**（`uptime` 的 load average + `free -g`），
  写进 manifest，便于判断噪声。
- 同一组对比实验**串行跑**，不要并行跑多个基准（会互相污染）。
- 若发现 load average > 8 或 `available` < 6 GiB，**暂停并等**，不要硬跑。

### 9.4 违反的后果

本机是他人的工作机。超过上限可能导致别人的会话卡死或 OOM。
发现自己在跑全核任务时，**立即 `nice`/收窄或终止**，并在 REPORT 里记录。
