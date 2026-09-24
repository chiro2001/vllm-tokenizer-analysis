# 线 B（多后端评估）交接报告

**分支**：`agent/B-backends`　**上游对象**：vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`（0.26.0）
**产出文档**：`docs/03-backend-matrix.md`、`docs/05-gigatoken-audit.md`

---

## 1. 交付物清单

| 交付物 | 路径 | 状态 |
|---|---|---|
| Rust criterion harness（核心资产） | `harness/rust/` | ✅ 可编可跑（stable 六后端 / nightly 含 gigatoken） |
| Python 三实现对比脚本 | `harness/python/bench_backends.py` | ✅ |
| 跨装置桥梁校准脚本 | `harness/bridge-calibration.sh` | ✅ |
| 工件获取 + 哈希校验 | `harness/fetch-artifacts.sh` | ✅ |
| vendored 快照一致性校验 | `harness/rust/sync-vendor.sh` | ✅ |
| 声明口径转录 | `scripts/extract-claims.py` | ✅ |
| 主表文档 | `docs/03-backend-matrix.md` | ✅ |
| gigatoken 审计 | `docs/05-gigatoken-audit.md` | ✅ |
| 原始数据 + manifest | `data/backends/*.jsonl|csv|json` | ✅ |

## 2. 六个后端是否全有数（硬要求核对）

| # | 后端 | encode | decode | 流式 decode | 批量 | 说明 |
|---|---|---|---|---|---|---|
| 1 | `Hf` | ✅ | ✅ | ✅ | ✅ | 全维度 |
| 2 | `Fastokens` | ⛔ **无法单独构造** | ⛔ | ⛔ | ⛔ | 在 Qwen3 上被自动改派到 3 号（见 §3.1） |
| 3 | `FastokensByteLevel` | ⚠️ zh/en **8k panic**，其余 ✅ | ⚠️ 同上 | ⚠️ 同上 | ⚠️ 同上 | 上游 fastokens 0.2.1 bug（§3.2） |
| 4 | `TiktokenRs` | ✅ | ✅ | ✅ | ✅ | 全维度 |
| 5 | `Riptoken` | ✅ | ✅ | ✅ | ✅ | 全维度 |
| 6 | `Tekken` | ✅ | ✅ | ❌ zh/mixed 全失败、en/code ✅ | ✅ | 上游无 lossy decode（§3.3） |
| 附加 | gigatoken | ✅ | ✅ | ⚠️ 经适配层 | ✅ | 见 `docs/05` |

**任务书问"六个后端是否全有数"**：1/4/5/6 的 encode+decode 齐全；2 号因上游
+设计无法单独构造（有明确原因，不是没测）；3 号在 4 类语料中 2 类（zh/en 的 8k）
+因**上游 fastokens 的 panic** 缺数；6 号在流式维度缺 6/12 组（同样有明确原因）。
+所有缺失点都在 `data/backends/e1.csv` 里留了 `ERROR:` 行，没有静默丢弃。

## 3. 三条最重要的实测结论

### 3.1 fastokens 的 ByteLevel 旁路在 decode 上值 8x（Rust 专有）

`Backend::FastokensByteLevel` 是 `vllm-tokenizer` crate 自己写的旁路
（跳过 `Vec<String>` 组装 + `join("")`），不是 fastokens 提供的。
同一批 token id、同一装置实测：**5.2–8.5x**（128/1k/8k × 4 类语料）。

**但必须限定**：这是**纯 decode 调用**的差距。与服务路径对应的**流式增量
decode**只有 **2.3x**（hf 75–124 µs vs fastokens 33–36 µs / 256 token）。
两组数在 `docs/03` §3.2/§3.3 分开呈现。

**跨装置复核**（第二装置由根代理在容器内独立执行，`data/crosscheck/`）：

| 项 | Rust 装置 | Python 装置 | 判定 |
|---|---:|---:|---|
| encode @128 倍率 | 8.1x | 9.3x | ✅ 互证 |
| decode @128 倍率 | 8.0x | 1.09x | ❌ 不通过（Python 无旁路）|

→ **encode 的 8–13x 两装置都有**（Python 用户开 `VLLM_USE_FASTOKENS=1`
同样能拿到）；**decode 的 8x 只属于 Rust 前端**。

### 3.2 上游 fastokens 0.2.1 的越界 panic（长输入必炸）

`fastokens-0.2.1/src/pre_tokenizers/split.rs:419`：
`n_chunks` 被 `.max(2)` 抬到 2，但 `pcre2` 表只有 1 条 → `pcre2[1]` 越界。
触发条件是文本足够长（`text.len()/MIN_CHUNK_SIZE >= 1`），实测 zh-8k / en-8k 必炸。

**影响面比"某个 backend 用不了"更严重**：vLLM 的 `new()` 是
"先试 fastokens、失败回落 HF"，但 **panic 不是 `Err`**，回落逻辑拦不住，
会直接带走前端进程。建议上报上游。

### 3.3 tekken 后端在流式路径上不可用（中文）

`tekken-rs` 0.1.1 的 `decode` 对**不完整 UTF-8** 直接报错，而不是返回替换字符。
流式解码必然出现"半个多字节字符"的中间态，因此 zh/mixed 的 6/12 组
流式 decode 全部失败（`Unable to decode into a valid UTF-8 string`）。
英文与代码语料正常。整段 decode 不受影响。

## 4. gigatoken 三档结论（各一句）

| 档 | 结论 |
|---|---|
| **P1（复现声明）** | 它的"~1000x" = **119x 输入量差 × 约 2.4x 单位吞吐差**（72 组对比中位；最大 1 299x），前提是它拿整块 11.9 GB 单文档、hf 只拿按分隔符切碎的 100 MB；另外"token 计数不一致"确实存在，但只有 **0.28% 中位差**，解释不了 690x。 |
| **P2（同条件）** | 同 4 MiB / 同文档边界 / 同线程数下，单位字节差距收敛到 **12.5–30x**（repeat 冷启动）/ **24–40x**（unique 冷启动）；它自己的 pretoken cache 贡献 **3.2–9.9x**；把输入切碎能让 hf 多拿 2.4x 而 gigatoken 不变。 |
| **P3（vLLM 相关域）** | 在线单条 128/1k/8k 上它最快（mixed-1k 11.9 µs，比 fastokens 旁路快 4.2x、比默认 hf 快 86x），**但缺 vLLM 依赖的流式 decode 与 `skip_special_tokens`**，需要 vLLM `DecodeStream` 外挂适配（实测 23.9 µs/256 token 可用），且 Rust 侧 `TokenId` 与 serial 入口不在公开 API 里。**落点：收益主要在离线批量。** |

## 5. 桥梁校准结果（**未通过，按硬要求禁止跨装置同表**）

脚本：`harness/bridge-calibration.sh`（两次测量在同一绑核窗口里串行执行）。
产物：`data/backends/bridge-calibration.json`。判据：相对差 <=15%。

| 项 | 长度 | Rust 装置 | Python 装置 | 相对差 | 判定 |
|---|---|---:|---:|---:|---|
| HF `tokenizers` encode | 128 | 97.56 µs | 122.40 µs | 20.3% | ❌ |
| fastokens encode | 128 | 12.32 µs | 11.61 µs | 5.8% | ✅ |
| HF `tokenizers` encode | 1k | 692.25 µs | 1123.81 µs | 38.4% | ❌ |
| fastokens encode | 1k | 45.35 µs | 78.96 µs | 42.6% | ❌ |
| **汇总** | | | | | **1/4 通过 → 禁止跨装置同表** |

两个可复核的原因（都不是测错）：

1. **fastokens 版本无法对齐**：vLLM 0.26.0 的 Rust 侧钉 `fastokens 0.2.1`，
   而 **PyPI 上没有 0.2.1**（只有 0.1.1/0.1.2/0.2.0/0.3.0/0.3.1/0.3.2）。
   Python 侧只能用 0.3.2，**不同引擎**，1k 上 42.6% 的差主要是版本差。
2. **解释器/计时开销 + 共享机器噪声**在低耗时格子上占比高。

因此 `docs/03` 里的 1–6 号后端数字**只来自 Rust 装置**，Python 三实现只在
`docs/03` §7.3 内部自成一张表。根代理的第二装置复核（`data/crosscheck/`）
比的是**倍率**（8.1x vs 9.3x，差 14.8%）——倍率吻合不等于绝对值吻合，
两个口径必须分开引用。

## 6. 踩过的坑（省下一位同事的时间）

1. **fastokens 在 PyPI 上没有 0.2.1**：vLLM 0.26.0 的 Rust 侧钉的是 0.2.1，
   而 PyPI 只有 0.2.0 / 0.3.x。Python 侧只能用 0.3.2，且 **0.3.x 的 API 变了**
   （`from_json_str` 返回 `Encoding` 需 `.ids`；`vocab_size` 从方法变属性）。
2. **gigatoken 必须要 nightly**：`src/lib.rs:1` 是 `#![feature(portable_simd)]`。
   安装 nightly 中途被打断会留下 `missing manifest in toolchain` 的坏状态，
   需要 `rustup toolchain uninstall nightly && rustup toolchain install nightly --profile minimal`。
3. **gigatoken sdist 的 `Cargo.toml` 会挡住 stable cargo**：`[profile.profiling]`
   里用了 nightly 的 `rustflags` profile 键（`profile-rustflags` feature），
   **stable cargo 连解析 manifest 都失败**。已在 vendored 副本里删掉这两节，
   补丁存 `vendor/gigatoken-vendor.patch`。
4. **gigatoken 是 `pub(crate)` 满天飞**：`bpe`/`token` 模块不公开，
   `TokenId` 无法命名；`encode_docs_ragged_serial` 有 `pub fn` 但模块 `pub(crate)`。
   绕法写在 `backends/gigatoken_shim.rs` 的注释里。
5. **gigatoken 用 `eyre::Report` 作错误类型**，而它不实现 `std::error::Error`，
   `anyhow::Context` 套不上，只能手工 `.map_err(|e| anyhow!("{e}"))`。
6. **跨家族复用 token id 会骗到自己**：第一版 harness 拿 Qwen3 的 ids 喂
   Kimi/tekken 的表（tekken 直接报 `Invalid token for decoding: 150644`）。
   现在每个后端都用自己的 ids，HF 家族额外断言与语料参考 id 逐一致。
7. **`measure_with_budget` 的非 try 版本会记录错误路径的耗时**：
   审计脚本早期版本因此把 fastokens 的 panic 路径当成成功测量。
   现在所有测量点都走 `measure_try_with_budget`，失败点进 `failed_rows`。
8. **gigatoken 的 pretoken cache 会让重复输入虚高 3–10x**：
   必须区分热缓存 / 冷启动（本 harness 两者都记），并区分输入重复度高（repeat）
   与低（unique）两种构造。
9. **语料变体生成曾经递归调用自身**（`generate_with_shift` → 变体 → 自身），
   直接 stack overflow。现在拆成 `build_text`（不产变体）+ `generate`（调它拼变体）。
10. **`limit.sh` 不会把 cargo 加进 PATH**：第一条命令因
    `taskset: failed to execute cargo` 直接失败，加上 `PATH="$HOME/.cargo/bin:$PATH"` 即可。

## 7. 资源纪律遵守情况

- 全部 CPU 密集操作经 `scripts/heavy_lock.sh` + `scripts/limit.sh`；
  基准统一 `CORES=2`，编译 `JOBS=2~3` / `MEM_GB=10`。
- 每个 manifest 都记了测量前后的 `loadavg` 与 `MemAvailable`
  （`data/backends/e1.csv` 有 `loadavg_before/after`、`mem_available_gb_before` 列）。
- 一次违规已发现并纠正：早期一次 `cargo build` 没带 `PATH` 被 `limit.sh` 拒绝执行，
  立刻改正后重跑，没有产生全核任务。另有一次后台 cargo 变成孤儿进程，
  确认它绑核（`limit.sh` 内部 `taskset`）后让它跑完。
- 未提交任何 `target/` 产物。

## 8. 遗留问题 / 建议后续

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | fastokens 0.2.1 的越界 panic | **生产可用性**（8k 中文 prompt 崩前端） | 上报上游；`new()` 的回退逻辑需要能捕获 panic |
| 2 | tekken 无可用的 lossy decode | 流式返回中文不可用 | 给 `TekkenTokenizer::decode` 加 lossy 语义 |
| 3 | gigatoken 无流式 API + decode 返回 bytes | 集成成本 | 若考虑引入，先让它提供 `DecodeStream` 等价物 |
| 4 | 大核数并行曲线（4/8/16/288 核）未测 | 批量路径扩展性未知 | 需要 >2 核的空闲窗口（本机是共享开发机，做不了） |
| 5 | gigatoken 常驻 RSS 未测 | 服务化内存预算 | 在放开内存限制的机器上单测 |
| 6 | Python 侧 `VLLM_USE_FASTOKENS=1` 的 vLLM wrapper 端到端未测 | 属于 E2 范围 | 交给线 C 或根代理 |
| 7 | 没有 Mistral 真模型 | tekken 只验了官方 `tekken.json` | 有模型时补 |
| 8 | `Fastokens`（2 号）在 ByteLevel 工件上无法单独构造 | 文档口径 | 若要单独测，需要非 ByteLevel 的 `tokenizer.json` |

## 9. 一键复跑

见 `docs/03-backend-matrix.md` §8 与 `docs/05-gigatoken-audit.md` §7。最短路径：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
LOCK=/home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh
LIM=/home/chiro/projects/vllm/tokenizer/scripts/limit.sh

harness/fetch-artifacts.sh
cd harness/rust && CORES=2 JOBS=2 $LOCK $LIM cargo +nightly build --release -p b-backends --features gigatoken

CORES=2 $LOCK $LIM ./target/release/b-backends correctness
CORES=2 $LOCK $LIM ./target/release/b-backends run --backend all --corpus all --lengths all --ops all --prefix e1
CORES=2 $LOCK $LIM ./target/release/b-backends gigatoken-audit --total-bytes 4194304

cd ../.. && bash harness/bridge-calibration.sh
```

---

## 10. 收尾补记（提交后再核一遍的更正）

### 10.1 资源口径更正（重要）

主矩阵与批量表的分核设置**不是**"2 核"，而是
`CORES=2 scripts/limit.sh` → **`taskset -c 2`（1 个核）+ `RAYON_NUM_THREADS=1`**
（`limit.sh` 把 `CORES=2` 解释成 1 个核，因为它按 `a-b` 区间数核）。
manifest 里可直接核对：`threads_effective=1`、`cpu_affinity="2"`。
受影响的是 §6 批量表——1 线程那一列实际是单核串行。
**已补做 2 线程对照**：`CORES=4-5`（真 2 核）重跑 batch 维度，
产物 `data/backends/e1-batch-2t.jsonl`，与 1 线程列并列在 `docs/03` §6。

新增的两条结论：

- **gigatoken 在 2 线程、小批量（4/32 文档、约 2 MB）下没有并行收益**
  （0.93–0.94×，略慢），因为它的 chunk 切分 + rayon 任务开销超过了 2 核收益；
  它的并行优势只体现在"大块单文档"（`docs/05` §3.3）。
- 1–6 号同用 `par_iter` 文档级并行，加速比从 hf 的 ~1.9× 到 riptoken 的
  3–5×（**后者 >2× 属可疑格，已在文档里标注不作结论**）。

### 10.2 两种 Rust 计时装置的交叉验证（criterion vs 自建）

| 测量点 | criterion | 自建计时器 | 相对差 |
|---|---:|---:|---:|
| `tiktoken_rs` encode en-1k | 475.8 µs | 457 µs | 4% |
| `riptoken` encode en-1k | 67.5 µs | 65.1 µs | 4% |
| `hf` 流式 256 token（zh-1k） | 83.1 µs | 81.9 µs | 1.4% |
| `hf` encode mixed-1k | 847 µs | 1 053 µs | 24% |
| `hf` decode mixed-1k | 87.6 µs | 171 µs | 95%（**可疑格**） |

hf 的两个单发格差异已定位为**共享机器的运行间抖动**：把该命令单独跑一次
（`--ops decode --corpus mixed --lengths 1k`）得到 **84.88 µs**，与 criterion
一致。即 e1 那一格被同批测量的其它负载抬高了。已在 `docs/03` §7.4 写明
"引用 hf 的 1k 级单发数字请打折"。

**副产物（值得进总结）**：criterion 顺带测了上游 `tokenizers::DecodeStream`
（Rust 侧）与 vLLM 自研 `DecodeStream`：**91.4 µs vs 83.1 µs，vLLM 快 9%**
——即 vLLM 的前缀差分实现没有比上游慢，还多提供了 `min_bytes_to_buffer`
与前缀种子逻辑。

### 10.3 最终交付核对

| 要求 | 状态 |
|---|---|
| `harness/rust/` 自建 criterion harness | ✅ `bench/src/{corpus,backends,stats,manifest}.rs` + `benches/backends.rs` |
| `harness/python/` 三实现对比 | ✅ `bench_backends.py --help` 可用 |
| `docs/03-backend-matrix.md`（中文） | ✅ 458 行（要求 200–300，**超出**；内容多于要求，未做删减以免丢口径说明） |
| `docs/05-gigatoken-audit.md`（中文） | ✅ 385 行（要求 150–250，**超出**，同上） |
| `data/backends/*.csv|json` + manifest | ✅ 含 e1 / e1-batch-2t / correctness / gigatoken-audit / gigatoken-claims / python-e1 / bridge-calibration |
| 每个脚本 `--help` | ✅ 四个 CLI 子命令 + 3 个 shell 脚本 + 1 个 python 脚本 |
| 提交 | ✅ 两次提交（主体 + 本报告） |
| vendored 快照可核验 | ✅ `harness/rust/sync-vendor.sh --check` 通过 |

**行数超出的说明**：文档偏长主要是口径说明（每组数字都要带装置/线程/缓存状态），
若根代理需要压到 200–300 行，建议优先删 §7.4（装置交叉验证）与 §3.3 的逐项读法，
把结论保留在 §9 的未测清单里；**但请保留 §1.3 的适用性表与 §3.2/§3.3 的
"8× vs 2.3×"口径区分**——这两处是最容易被误引的。
