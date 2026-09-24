# vLLM 0.26.0 tokenizer 链路调研

> 阅读入口：**[`docs/00-INDEX.md`](docs/00-INDEX.md)**（一分钟结论 + 文档地图）。

## 一分钟结论

1. **链路只有三段真实工作量，全在 API server 前端进程**：入站 encode、
   出站流式 decode、停止串判定。engine core 不在这条链路上。
2. **8k 是分水岭**：encode 约 **1.47 µs/token**（线性，R²=0.9998），
   8k 以下 tokenizer 占 TTFT <2%，8k 以上进入 **5.7–11.7%**。
   **chat 模板（Jinja）是常数级（47–147 µs），不是瓶颈**——要优化就优化编码。
3. **fastokens 的加速是真的，但要分清哪一段**：encode 快 **8–13×**（Python/Rust
   两端都拿到）；decode 的 **8×** 只来自 Rust 前端专有的
   `FastokensByteLevel` 旁路（**流式只有 2.3×**）。
4. **下一代是"北向服务层整体换语言"**：vLLM 0.26.0 已自带 Rust 前端 `vllm-rs`，
   tokenizer 默认走 fastokens，Python 侧只剩 engine core。官方 wheel 自带二进制，
   零编译可跑（本项目已做 E2E 验证）。
5. **gigatoken 的"~1000×" = 119× 输入量差 × 约 2.4× 单位吞吐差**；
   同条件收敛到 12.5–30×。落点是**离线批量**。
6. **三个"会坏"而非"慢"的上游问题**：fastokens 0.2.1 越界 panic（**会崩前端进程**）、
   tekken-rs 中文流式解码不可用、Rust 前端 7 个参数"接受但无效"。

详细结论与口径见 [`docs/00-INDEX.md`](docs/00-INDEX.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 7 篇正文（约 2 500 行），入口 `00-INDEX.md` |
| `figures/` | 8 张图（火焰图 + 曲线，SVG） |
| `data/` | 结构化数据 + manifest（绑定核/线程/loadavg/sha256） |
| `harness/` | Rust（criterion，六后端）与 Python 两套可复现 harness |
| `scripts/` | 采集 / 解析 / 资源限额 / 净化发布脚本 |
| `plan/` | 执行计划、协作约定、实验矩阵 |
| `agents/*/REPORT.md` | 四条线的交接报告（含失败路径与踩坑） |

## 资源纪律

本项目在**共享开发机**上完成，所有 CPU/内存密集操作经
`scripts/limit.sh`（绑核 + 8 GiB 上限）与 `scripts/heavy_lock.sh`（全局唯一重活锁）
执行，详见 [`plan/COORDINATION.md`](plan/COORDINATION.md) §9。

## 口径纪律（引用任何数字前必读）

见 [`plan/COORDINATION.md`](plan/COORDINATION.md) §5。要点：

1. `tokenizer:` 三个 scope 跑在 **API server 前端进程**，与 engine core 混在同一份日志里；
2. 历史数据几乎全是直发 token_ids 的 completion ⇒ `render_messages`/`decode` 计数为 0，
   **不能据此说它们不耗时**；
3. gigatoken 的"1000×"是**批量文件吞吐口径**，输入量与并行粒度都与 HF 臂不同；
4. Rust 侧与 Python 侧数字必须经**桥梁校准**（fastokens、HF `tokenizers`）才能同表比较。
