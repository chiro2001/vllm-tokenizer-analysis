# 实验矩阵

> 配套 `EXECUTION.md`。每条实验必须写 manifest（commit / 镜像 id / 模型 revision / 脚本哈希 / 时间戳）。

## E1 后端微基准

固定：vLLM 0.26.0 `568afb3a1`；本地 x86_64；无卡。

| ID | 后端 | 工件 | 输入 | 操作 | 线程 | 判据 |
|---|---|---|---|---|---|---|
| E1.1 | `new_hf` | Qwen3-0.6B `tokenizer.json` | 128 / 1k / 8k | encode | 1 | 基线 |
| E1.2 | `new_fastokens` | 同上 | 同上 | encode | 1 | vs E1.1 |
| E1.3 | `FastokensByteLevel` | 同上 | 同上 | decode | 1 | 旁路是否生效 |
| E1.4 | `new_tiktoken_rs` | `tiktoken.model` | 同上 | encode | 1 | 基线 |
| E1.5 | `new_riptoken` | 同上 | 同上 | encode | 1 | vs E1.4 |
| E1.6 | `TekkenTokenizer` | 小 fixture `tekken.json` | 短文本 | encode/decode | 1 | 仅路径验证 |
| E1.7 | gigatoken（Rust） | sdist vendor | 同上 | encode | 1 | vs E1.1 |
| E1.8 | 六后端 | — | 1k | **增量 decode** | 1 | `min_bytes_to_buffer` 影响 |
| E1.9 | 六后端 | — | 8k | 批量 encode | default | 并行收益 |

**语料**：中文 / 英文 / 代码 / 中英混合各一份，长度对齐。
**正确性**：E1.1 vs E1.2 vs E1.3 断言 ids 一致；E1.4 vs E1.5 断言 ids 一致；不一致要留反例。

## E2 成本与占比

| ID | 内容 | 装置 | 判据 |
|---|---|---|---|
| E2.1 | `tokenizer: encode` 成本 | 本地 harness（vLLM renderer 路径） | µs/req，按 ISL 分组 |
| E2.2 | `tokenizer: render_messages` 成本 | 同上（**历史零实测**） | µs/req，chat 带 tools |
| E2.3 | `tokenizer: decode` 成本 | 同上（**历史零实测**） | µs/token，按 OSL 分组 |
| E2.4 | 三者占 TTFT/TPOT | 本地 + 真机锚点（a3-22 一个点） | 明确分子分母 |
| E2.5 | 并发曲线 | 本地 1→64 | 前端进程饱和点 |
| E2.6 | 历史数据回收 | 远端 29 run + 本地 3 run | CSV + 口径校准 |

## E3 火焰图

| ID | 内容 | 工具 | 产出 |
|---|---|---|---|
| E3.1 | Python 前端 tokenizer 路径 | `perf record -g` + flamegraph | SVG，含 Rust 帧 |
| E3.2 | Rust `vllm-tokenizer` 内部 | `cargo flamegraph` | SVG，预分词器 vs BPE merge |

## 附：gigatoken 专用档

| 档 | 目的 | 做法 |
|---|---|---|
| P1 | 复现其声明 | 读 `benchmarks/compare/measure.py` + `results.json`，**不重跑大语料**；给出其口径下的数与前提 |
| P2 | 同条件对比 | 同字节、同文档边界、同线程；分离"并行策略差异"与"算法速度差异" |
| P3 | vLLM 相关域 | 在线单条（128/1k/8k）、小批量、流式 decode；**这是结论落点** |

缺口必须量化：无流式 decode API、rayon 全局线程池与绑核的相互影响。
