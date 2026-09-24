# data/ 目录说明（C 线）

## `cost/` —— 本机实测

所有 JSON 都带 `manifest`（commit / 镜像 id / 模型 revision / 脚本 sha256 /
时间戳 / 绑核 / load average / 容器限制），字段含义见 `harness/python/common.py`。

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `e2_paths.json` | E2.1 `encode` + E2.2 `render_messages`（3 形态）+ E2.3a `decode: prompt_reverse` + `render_decompose`（Jinja vs encode 拆分） | `scripts/c_cost_run.sh --liteprof harness/python/bench_paths.py --tasks encode,render,decompose` |
| `e2_encode_short.json` | E2.1 补充：4–1024 token 的 `encode` 曲线（量固定成本） | `scripts/c_cost_run.sh harness/python/bench_encode_short.py` |
| `e2_detokenize.json` | E2.3b 生成期流式 detokenize（Fast / Slow 两条路，三段结构） | `scripts/c_cost_run.sh harness/python/bench_detokenize.py` |
| `e2_concurrency.json` | E2.5 前端并发曲线（`renderer_num_workers` 1/4/8 × 并发 1–64）+ deepcopy 计数 | `scripts/c_cost_run.sh harness/python/bench_concurrency.py` |
| `e2_litescope.json` | 交叉校验：用**真实 LiteScope 插桩**读同一批 scope + 同进程对照臂 | `scripts/c_cost_run.sh --liteprof harness/python/bench_litescope.py` |
| `e2_share.json` / `e2_share.csv` | E2.4 占比表（分子分母定义写在 JSON 的 `definitions` 里） | `scripts/compose_e2_4.py` |
| `e3_flamegraph_frames.csv` | E3.1 火焰图的**精确帧名 + 权重**（SVG 会截断函数名，这张表才是可引用的） | `scripts/analyze_flamegraph.py --folded ...` |

## `historical/` —— 历史回收（只读）

| 文件 | 内容 |
|---|---|
| `historical_tokenizer_scopes.csv` | 52 个 run 各一行：负载形态（端点/是否有 chat/usage 里的 prompt_tokens）+ `encode`/`decode`/`render_messages` 的 count/avg/min/max + 前端 tid |
| `historical_runs.json` | 同样的内容 + 每个 run 的全部 scope 名、全部 tid、按 pid 归属的前端证据 |

来源：本地 `HIST_PROJECT/runs/liteprof_v*_uni_*`（3）、
`preparing-input-phase/data/profiles/real-b1`（1）、
远端 a3-22 `~/projects/vllm/HIST_PROJECT/runs/liteprof_wave*`（48，只读拉取）。
**生成脚本 `scripts/collect_historical.py` 全程只读**，不修改任何历史目录。

原始远端 run 目录（每个约 56 KB）未进 git，复跑见 `scripts/fetch_remote_bg.sh`。
