# C 线交接报告 · 成本与占比 + 火焰图

> 分支 `agent/C-cost`，工作区 `/home/chiro/projects/vllm/tokenizer-wt/C-cost`。
> 正文文档：[`docs/02-cost-and-share.md`](../../docs/02-cost-and-share.md)。

## 1. 一句话交付

三个 scope **全部有数**（`encode` / `render_messages` / `decode`），
并且把历史上从没被采过的 **生成期流式解码** 单独补上；
另外发现并量化了一条会让人测错的口径问题：
**chat 请求根本不触发 `tokenizer: encode`**。

## 2. 交付物清单

| 类型 | 路径 |
|---|---|
| 正文 | `docs/02-cost-and-share.md`（约 420 行，中文） |
| harness | `harness/python/{common,corpora,bench_paths,bench_detokenize,bench_concurrency,bench_litescope,flame_target,probe_env}.py` |
| 脚本 | `scripts/{c_cost_run,build_liteprof_overlay,collect_historical,compose_e2_4,make_figures,run_flamegraph,analyze_flamegraph,sync_remote_runs,fetch_remote_bg}.sh` |
| 数据 | `data/cost/{e2_paths,e2_detokenize,e2_concurrency,e2_litescope,e2_share}.json`、`e2_share.csv`、`e3_flamegraph_frames.csv` |
| 历史 | `data/historical/historical_tokenizer_scopes.csv`、`historical_runs.json` |
| 图 | `figures/fig-02-*.svg`（4 张）+ `figures/e3-1{a,b,c,d}-python-frontend.svg`（4 张） |

## 3. 三个 scope 各自的量级（本机 x86 容器，2 核，Qwen3-0.6B）

| scope | 128 目标 ISL | 1k | 8k | 备注 |
|---|---|---|---|---|
| `tokenizer: encode` | **265–303 µs**（实测 220 token） | **1106–1262 µs** | **11830 µs** | 128–8k 线性，**1.47 µs/token**，R²=0.99975；**<64 token 有 ~45 µs 固定成本** |
| `tokenizer: render_messages`（chat + tools） | **805 µs**（渲染 519 token） | **1675 µs** | **12292 µs** | 含模板内 encode；Jinja 只占 ~50–70 µs 且**不随 ISL 变** |
| `tokenizer: decode`（历史口径 = prompt 反解） | 43 µs（220 token） | 177 µs | 1393 µs | **0.17–0.20 µs/token**，真实服务里极少触发 |
| `detokenize: stream`（**生成期，无插桩**） | — | — | — | **1.39 µs/token**（fast）/ 1.55（slow），另加首步 216–292 µs |

短 prompt 曲线（`bench_encode_short.py`，固定成本的直接证据）：

| 实测 token | 4 | 10 | 32 | 64 | 220 | 990 |
|---|---|---|---|---|---|---|
| mean (µs) | **43.5** | 47.4 | 65.2 | 84.3 | 303.6 | 1261.8 |

4–16 token 之间几乎是常数 ⇒ **固定成本 ≈45 µs**，`µs/token` 从 10.9 降到 1.28。

## 4. 占 TTFT / TPOT 的比例（分子分母见文档 §1.5）

**同装置可比**（同一 run 内）：

| 分子 | 分母 | 占比 |
|---|---|---|
| `tokenizer: encode` 524.9 µs | `http: create_completion` 583.2 ms | **0.090%** |
| `tokenizer: encode` 524.9 µs | `Step:Model` prefill 199.2 ms | **0.264%** |
| `tokenizer: encode` 391.6 µs | `Step:Model` prefill 101.0 ms | **0.388%** |

**跨装置上界**（分子=本机 x86 容器，分母=真机 NPU 的 `Step:Model`）：

| 分子 | 占 prefill 199.2 ms | 占 prefill 101.0 ms |
|---|---|---|
| `encode` @220 token | 0.14% | 0.28% |
| `encode` @990 token | 0.61% | 1.20% |
| `encode` @8140 token | **5.94%** | **11.71%** |
| `detokenize: stream` 1.372 µs/token | 0.325%（占 decode 步 422 µs） | 0.575%（占 240 µs） |

**关键结论**：几百 token 的短 prompt 上 tokenizer 占 TTFT < 1%（与历史数据
"163–613 µs"的量级判断一致）；**8k prompt 上升到 5.9–11.7%**，这是分水岭。
生成期解码占 TPOT **千分之几**。

## 5. 图的路径

| 图 | 内容 |
|---|---|
| `figures/fig-02-scope-cost-vs-isl.svg` | 三个 scope 成本 vs 输入规模 |
| `figures/fig-02-detokenize-segments.svg` | 流式解码的三段成本结构 |
| `figures/fig-02-frontend-concurrency.svg` | E2.5 并发曲线（吞吐 + p50 延迟） |
| `figures/fig-02-share-of-ttft-tpot.svg` | E2.4 占比（蓝=同装置，橙=跨装置上界） |
| `figures/e3-1a-python-frontend.svg` | E3.1 混合路径火焰图 |
| `figures/e3-1b/c/d-python-frontend.svg` | encode / render / detokenize 各自的火焰图 |

## 6. 发现的"会让人测错"的口径问题（最重要）

### 6.1 chat 请求不触发 `tokenizer: encode`

打桩实测（`BaseRenderer._tokenize_prompt` 调用计数）：

| 调用 | `_tokenize_prompt` 次数 |
|---|---|
| `render_chat(messages)`（chat 模板默认 `tokenize=True`） | **0** |
| `render_cmpl([{"prompt": text}])` | **1** |
| `render_cmpl([{"prompt_token_ids": ids}])` | **0** |

⇒ `encode` 与 `render_messages` **只对互补的负载形态生效，不存在两段相加的请求**。
历史日志里两者之一必然为 0，不能据此推断"某项不耗时"。

### 6.2 `tokenizer: decode` ≠ 生成期解码

历史 scope 只包 `BaseRenderer._decode`（prompt 反解）；生成期解码在
`vllm/v1/engine/detokenizer.py`，**LiteProfiler 完全没覆盖**。
两者量级与摊销方式都不同（0.18 µs/token 每请求 vs 1.37 µs/token 每步），
混在一起整个占比表就串味。

### 6.3 进程归属：同 pid、不同 tid

实测 `liteprof_v1_torch_uni_*`：`tokenizer: encode` 在 `tid=1169, pid=1`，
`http: create_completion` 在 `tid=1, pid=1`，engine core 在 `pid=131`。
tokenize 被 `make_async` 丢进了 ThreadPoolExecutor 的 worker 线程。
**按 tid 分组与按 pid 归组都对，但含义不同**，别混用。

### 6.4 流式解码的"三段结构"（只测一端会测错）

| 段 | fast | slow |
|---|---|---|
| `init`（`from_new_request`） | 17.1 µs | 16.4 µs |
| 第 1 次 `update()`（**惰性**灌 prompt） | **231.0 µs** | 262.9 µs |
| 稳定期 `update` + `get_next_output_text` | **1.372 µs/token** | 1.466 µs/token |

只测构造函数会低估（漏掉 231 µs）；只测"整请求 ÷ OSL"会在小 OSL 上高估
（OSL=32 时算出 9.7 µs/token）。

## 7. 并发与池（E2.5）

| `renderer_num_workers` | 池 | 饱和并发 | 饱和吞吐 |
|---|---|---|---|
| 1（默认） | 2 | **4** | ~700 req/s |
| 4 | 5 | **16** | ~2250 req/s |
| 8 | 9 | 8（4 核限制） | ~1755 req/s |

- **`renderer_num_workers` 是唯一真正影响前端吞吐的旋钮**（`tokenizer_pool_size`
  在 0.26.0 不存在）。
- **"池空 ⇒ 现场 deepcopy 且池无上限增长"在默认 async 路径上测到 0 次**
  （21 个配置点全 0）。原因：池大小 = `renderer_num_workers + 1`，
  executor 线程数 = `renderer_num_workers`，并发借用数最多 N < N+1，
  池永远不会空。**这条不是"随并发劣化"的现实风险**。
- 但该路径**代码里确实存在**，用 8 线程打 3 份池的受控实验强制触发：
  **5 次现场 deepcopy，平均 3.64 s/次**（同参数 3 线程触发 0 次）。
- 启动代价：`HfRenderer.__init__` 按 `workers+1` 预建池，实测
  workers=1 → 2.75 s / 7 次 deepcopy；workers=8 → 9.54 s / 14 次。
  每个 deepcopy 350–655 ms。

## 8. E3 火焰图：工具链的真实情况

### 8.1 镜像里没有采样工具

`docker run <image> which perf` 为空、无 flamegraph。所以采样改在**宿主机**做
（perf 7.1.6 + inferno-collapse-perf + inferno-flamegraph，都在 `~/.cargo/bin`），
被采样进程仍在容器里跑同一代码路径。

### 8.2 踩过的坑（都已在脚本注释里固化）

| 坑 | 现象 | 处理 |
|---|---|---|
| 容器进程在宿主机上是 root | `perf record -p` 报 `Failure to open event 'cpu/cycles/Pu'` | 用 `sudo -n perf` |
| `-g` 与 `--call-graph` 同时给 | perf 打印 help 并失败 | 只用 `--call-graph` |
| `--call-graph dwarf,65536` | 被拒（上限 < 32768） | 用 `dwarf,16384` |
| `pgrep -f flame_target.py` | 命中 **docker 客户端**的命令行 | 扫 `/proc` + 校验 cgroup + `comm` |
| 只校验 cmdline | 命中**外层 bash 包装**（在等子进程）⇒ 13 s 只采到 61 个样本 | 同上，要求 `comm` 是 `python3` |
| `/proc/<pid>/exe` | root 进程读不到（EACCES），`readlink -f` 静默给空 | 改用 `/proc/<pid>/comm` |
| Python 3.12 帧在堆上 | perf 看不到 `py::` 帧 | 本机 perf **无 JIT 接口选项**，放弃 Python 帧；Python 层归属由微基准给出 |
| `sys.stderr.write(..., flush=True)` | `TypeError`，正好在 READY 之后崩 ⇒ perf 采了个死进程 | 改 `print(..., file=sys.stderr, flush=True)` |
| 折叠栈权重单位 | 误当 µs 会得到 46454 s 这种荒唐值 | 是 **ns**（4.6455e10 ns = 46.5 s，与 70 s 窗口吻合） |
| `inferno` 截断函数名 | SVG 里只剩 `core::hash::B..` | 另出 `data/cost/e3_flamegraph_frames.csv` 给全名 + 权重 |
| inclusive 汇总 | 父子重叠，cpython 一类"占 184%" | 类别汇总**一律用 self** |

### 8.3 火焰图结论（self-time 占比）

| 类别 | encode | render(chat+tools) | detokenize |
|---|---|---|---|
| `tokenizers::*` | 15.9% | 5.8% | 13.6% |
| Rust 依赖（core/alloc/hashbrown/regex/serde） | **48.1%** | 12.6% | 26.1% |
| CPython | 9.5% | 14.9% | **34.3%** |
| libc 符号 + malloc/free | 6.8% | **54.9%** | 10.0% |

1. **encode 是 Rust 主导**（合计 64%），但最大头**不是 BPE merge**
   （`BPE::tokenize` inclusive 仅 ~1%），而是**哈希/词表查找与正则预分词**：
   `hash_one` 8.5%、`RawTable::reserve_rehash` 5.4%、`HashMap::insert` 4.9%、
   `match_at` 5.1%、`Cache::get` 3.2%。
2. **chat 渲染与 tokenizer 无关的部分占绝对多数**：libc + malloc/free ≈ **55%**
   （Jinja 拼串 / serde_json 序列化 tools），Rust 只有 18%。与 §3.1 的
   "Jinja 常数级 50–70 µs" 互相印证。
3. **detokenize 里 Python 侧占比最高**（34%），Rust 侧集中在
   `id_to_token`（4.2%）与 `is_special_token`（2.6%）——即每步的
   "id→串 + 特判"。

## 9. 探针代价对照（硬要求）

| 装置 | 代价 |
|---|---|
| 主数据 harness | 只用 `perf_counter_ns()` 包住调用，每次 20–70 ns；相对 283 µs 起 < 0.03% |
| LiteScope 插桩 | 每 scope 两次取时钟 + 一次 open/append/close；**已用同进程对照臂交叉校验**（`e2_litescope.json`） |
| perf 采样 | `-e cpu-clock -F 99 --call-graph dwarf,16384`；脚本提供 `--cost-control` 跑"无采样/有采样"各 3 轮 |

## 10. 复跑

```bash
COST_CORES=4-5 /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh \
  ./scripts/c_cost_run.sh --liteprof harness/python/bench_paths.py \
  --tasks encode,render,decompose --out /workspace/data/cost/e2_paths.json
```

全部脚本支持 `--help`；每条实验的 manifest（commit / 镜像 id / 模型 revision /
脚本 sha256 / 时间戳 / 绑核 / load average / 容器限制）写在产出 JSON 的
`manifest` 字段里。

## 11. 仍未做（如实标注）

| 项 | 原因 |
|---|---|
| 真机 chat 负载锚点（a3-22 上跑一个 chat 点测 `render_messages`） | 未做 |
| E3.2 Rust `vllm-tokenizer` 内部火焰图 | 依赖 B 线 `harness/rust/`，本次未拿到 |
| 批量/多请求同时 encode 的摊销 | 未做（本文全是单请求） |
| `VLLM_USE_FASTOKENS=1` 下的三个 scope | B 线范围 |
| 非 Qwen3 系 tokenizer 的这三条成本 | 未做 |
| **10–11 token 下 162–461 µs 的归因** | 需要在真机上做同装置复核（关/开 LiteProfiler 各测一遍）才能定 |

## 11.1 E2.6 已完成：52/52 run

远端 48 个 + 本地 4 个**全部回收**（详见 `docs/02-cost-and-share.md` §7）。
16 个 41 次采样的稳态样本：**162.3 – 461.0 µs**，中位数 275.6 µs，
与任务书给的"163–461 µs"吻合。

**但顺手挖出一个口径问题（重要）**：稳态 run 的响应体 `usage` 显示
**prompt_tokens 只有 10–11**，而本机 x86 量 10 token 只要 **47 µs**。
同一量级 prompt 相差 3.4–9.7 倍，且本机固定成本只有 ~45 µs，无法解释。
三类候选原因（机器差异 / LiteProfiler 插桩开销 / 池冷启动）**本文不判定**，
需要同装置复核。⇒ **引用"163–461 µs"时必须同时说明它是 10–11 token 的
prompt 且开着 LiteProfiler**，不要当成"几百 token 的典型成本"。

## 12. 给根代理的三条提醒

1. 引用占比时务必带上"分母是 `Step:Model`（模型执行段），所以是**上界**"；
   跨装置的行不要与同装置的行混表。
2. 不要写"`render_messages` 与 `encode` 可以直接相加"——**chat 请求根本不走
   `tokenizer: encode`**，两者是互补负载，相加没有对应的请求形态。
3. `tokenizer: decode` 必须与 `detokenize: stream` 分开命名，前者是 prompt 反解。
