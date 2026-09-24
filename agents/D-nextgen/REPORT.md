# D 线交接报告：下一代 Rust 前端（`vllm-rs`）

> 分支 `agent/D-nextgen`。vLLM 0.26.0 / commit `568afb3a13806beb53bb2e6bd518269357b237c0`。
> 结论标注 **【实测】/【源码】/【推断】**。

## 1. 交付物

| 路径 | 内容 |
|---|---|
| `docs/04-next-gen-rust-frontend.md` | 主文档，300 行 |
| `harness/rust-frontend/fetch_vllm_rs.py` | HTTP Range 抽 wheel 里的 `vllm-rs`，有 `--help` / `--self-test` |
| `harness/rust-frontend/trim_workspace.py` | 把 vLLM rust workspace 裁成 mock-engine 依赖链，有 `--help` |
| `harness/rust-frontend/verify_vllm_rs.sh` | 抽取 + `--help` 枚举 + 真启动验证，有 `--help` |
| `harness/rust-frontend/e2e_mock_engine.sh` | 全链路 E2E（前端 + mock engine），有 `--help` |
| `data/nextgen/manifest.json` | 口径 manifest（commit / 二进制 sha256 / 脚本哈希 / 负载） |
| `data/nextgen/` 其余 | wheel 抽取 JSON、两个架构的 help 文本、E2E 日志与响应样本 |

## 2. 核心结论（3 条）

1. **形态：北向整层换语言，南北向边界不动。** Rust 把 axum HTTP、OpenAI 协议、
   minijinja 模板渲染、fastokens/HF/tiktoken/tekken tokenizer、`DecodeStream`
   增量 detokenizer、reasoning & tool parser 全搬进一个进程；Python 侧只剩
   engine core。**【源码】**`rust/src/managed-engine/src/process.rs:57-75` 用
   `python -m vllm.entrypoints.cli.main serve ... --headless` 拉起引擎，
   `vllm/entrypoints/cli/serve.py:173-235` 的 `run_headless()` 只建 EngineCore，
   **Python 进程里不再有 tokenizer/detokenizer**。前端发出的已经是
   `prompt_token_ids`（`rust/src/text/src/lib.rs:143-148`）。
2. **分发：官方 wheel 自带真 ELF 二进制，零编译可用。**【实测】HTTP Range 只下
   5.7%（x86_64，17.4 MB）/ 6.1%（aarch64，18.3 MB）的 wheel 就抽出
   `vllm/vllm-rs`（42.79 MB / 39.60 MB，ELF64 PIE，not stripped）与
   `vllm/_rust_tool_parser.abi3.so`。**不需要容器、不需要 torch_npu**，
   `ldd` 只依赖 libgcc/libc/libm/libdl/libpthread。
   `VLLM_RUST_FRONTEND_PATH` 只在"用别处的二进制"时才需要；默认 `auto` 找不到
   `<pkg>/vllm-rs` 会**直接抛错**（`vllm/envs.py:550-580`）。
3. **成熟度：实验性，但差距是可枚举的。**【实测】`vllm-rs serve --help` 末尾有
   独立一节列出 **50 个未实现参数 + 7 个接受但无效（`Noop`）**，对比 33 个已支持
   参数。tokenizer 相关的硬缺口是 `--tokenizer`、`--tokenizer-revision`、
   `--skip-tokenizer-init`、`--hf-overrides`、`--trust-request-chat-template`，
   传了会**直接拒绝启动**；更危险的是 `--enable-tokenizer-info-endpoint` 等 7 个
   `Noop` 参数"接受但不生效"。

## 3. 实测结果（成功项）

### 3.1 抽取 + `--help` 验证：**成功**

| 架构 | wheel | 实际下载 | 占比 | `vllm-rs` | sha256 |
|---|---|---|---|---|---|
| x86_64 | 303.7 MB | 17.4 MB | 5.7% | 42.79 MB | `0597bfc9…d78f807` |
| aarch64 | 298.3 MB | 18.3 MB | 6.1% | 39.60 MB | `cae05321…540dc0b` |

子命令只有 `frontend` / `serve`；两个架构的 `--help` 与 `serve --help`
输出**逐字节相同**（`diff -q`）。aarch64 二进制在本机用
`qemu-aarch64 -L /usr/aarch64-linux-gnu` 可直接跑。

### 3.2 启动验证：**成功**

`vllm-rs serve /home/chiro/models/Qwen3-0.6B --data-parallel-size-local 0`
（纯 Rust、无 Python、无 NPU）。日志证明**真的加载了 tokenizer**：

```
[main.rs:112] running Rust frontend without a managed local Python engine ...
[hf.rs:112]   loading tokenizer with fastokens path=.../Qwen3-0.6B/tokenizer.json
[hf/mod.rs:71] loaded text backend with Hugging Face model files ...
[hf.rs:78]    loaded chat backend with Hugging Face model files ... renderer=hf
[transport.rs:155] waiting for engines to connect ...
```

HTTP 端口要等引擎 ZMQ 握手成功才 bind（`rust/src/server/src/lib.rs:182-185, 223`），
所以没有引擎时端口不通 —— 这是**预期行为，不是失败**。

### 3.3 全链路 E2E：**成功**

用上游自带的 `rust/src/mock-engine`（协议模拟器）+ wheel 里的 `vllm-rs`，
真实 ZMQ + msgpack。9 条断言全过：加载 tokenizer、握手、`starting OpenAI server`、
`/health`、`/v1/models`、`/tokenize`、`/v1/chat/completions`（`prompt_tokens: 12`）、
流式 6 个 content delta。一次覆盖 **minijinja 模板 → fastokens encode →
ZMQ → mock 采样 → `DecodeStream` 增量 detokenize → OpenAI SSE**。

### 3.4 tokenizer 正确性交叉核对：**ids 完全一致**

输入 `你好，Hello!`：Rust `POST /tokenize` 给 `[108386, 3837, 9707, 0]`，
容器内 Python `tokenizers` 0.22.2 也给 `[108386, 3837, 9707, 0]`（Python 侧回环
解码得原文）。**注意**：这里比的是"Rust 前端 tokenizer 与 HF `tokenizers`
实现"，**不是**"Rust 前端 vs Python 前端"的端到端对照。

## 4. 没做成 / 未验证（如实记录）

| 项 | 状态 | 卡在哪 |
|---|---|---|
| Python 前端 vs Rust 前端同负载对照 | **未做** | 本机无 vllm 安装；容器是 ascend 镜像，本调研禁止占 NPU |
| a3-22 真机验证 | **未做** | `scp` 40 MB 到 a3-22 超时（>120 s，两次），改走本地 `qemu-aarch64` 交叉验证 |
| 真 `vllm-rs serve` + Python headless engine | **未做** | 需要能真正推理的 vLLM 引擎（无卡环境起不来），用 mock engine 替代 |
| deepseek_v32/v4、harmony、inkling 渲染器 | **未测** | 没有对应模型 |
| 第 3、6 个后端（ByteLevel 旁路、Tekken）的耗时 | **未测** | B 线负责，交付时其 worktree 尚无 `data/` |
| 跨版本协议兼容性 | **未测** | 只验证了 0.26.0 内部一致 |

## 5. 踩坑与给后续线的提示

1. **wheel 不用全下**：`fetch_vllm_rs.py` 用 `urllib` 手写 Range 分块 + 一个
   `RawIOBase` 适配器喂给 `zipfile`。关键教训：**HTTP 响应必须校验 `206`**，
   否则 CDN 忽略 Range 时会静默把 300 MB 全灌进来。
2. **`vllm-rs` 的 HTTP 端口在引擎握手前不通**。`--data-parallel-size-local 0`
   只能验证到"加载 tokenizer + 等握手"；要看 HTTP 必须配引擎（或 mock engine）。
3. **构建 mock engine 不需要整个 workspace**。完整 workspace 有 13 个成员
   （axum / tonic / minijinja / pyo3 …），只为 `vllm-mock-engine` 解析全部依赖
   纯属浪费。裁成 3 个成员后 `cargo build --release -p vllm-mock-engine`
   约 60 秒（`-j4`，走 `limit.sh`）。
4. **`/tmp` 只有 1.1 GB 可用**，Rust 构建要放 `~/.cache/d-nextgen-mock`。
5. **本机 python3 没有 vllm / tokenizers**，Python 侧比对要走
   `scripts/docker_run.sh`（根代理提供的无卡容器执行器）。

## 6. 一键复跑

```bash
export PATH="$HOME/.cargo/bin:$PATH"
python3 harness/rust-frontend/fetch_vllm_rs.py --arch x86_64 --out /tmp/d-nextgen-wheel
python3 harness/rust-frontend/fetch_vllm_rs.py --self-test
harness/rust-frontend/verify_vllm_rs.sh --no-fetch
EVIDENCE_DIR=data/nextgen harness/rust-frontend/e2e_mock_engine.sh
```

第 1 条约 40 s；第 3 条约 55 s；第 4 条首次含编译约 2 min
（编译走 `heavy_lock.sh` + `limit.sh`，绑 4-7 核、`-j4`）。

## 7. 给根代理的合并提示

* 本文档与 B 线的 `docs/03-backend-matrix.md` 有明确分工：**我只讲形态与架构，
  不重复他的六后端微基准**；正文 §3.3 已注明"落笔时 B 线结果未出"。
* `docs/06-conclusions.md` 若引用"值不值"，请引用正文 §4 的判断与差距清单
  （两处都标了【推断】，请勿升级为实测）。
* `data/nextgen/manifest.json` 里的 `scripts_sha256_16` 是提交时的脚本哈希；
  若后续改脚本需重跑 `e2e_mock_engine.sh` 并更新。
