# 04 · 下一代 Rust 前端（vllm-rs）：搬走了什么、值不值

> 线 D（`D-nextgen`）。vLLM 0.26.0 / commit `568afb3a13806beb53bb2e6bd518269357b237c0`。
> 代码引用形如 `rust/src/...:行号`，相对只读源码 `/home/chiro/projects/vllm/HIST_PROJECT/vllm`。
> 每条结论标注 **【实测】/【源码】/【推断】**；未验证的一律写明。

## 0. 一页结论

**Rust 前端不是一个更快的 tokenizer，而是"北向服务层整体换语言"的重写。**
API server、OpenAI 协议、chat 模板渲染、tokenize/detokenize、reasoning & tool 解析
全部进 Rust 进程；Python 侧**只剩 engine core**（调度 + 模型执行），两端仍用
**ZMQ + MessagePack** —— 南北向边界没动，边界以北全部重写。

对本调研的意义：Python 前端里要用户手动开的 `VLLM_USE_FASTOKENS=1`，在 Rust 前端里
直接是默认路径（`rust/src/tokenizer/src/hf.rs:129-141`：先试 fastokens，失败回落 HF）。

1. **tokenizer 相关那部分整体搬走了**：encode、增量 decode、chat template
   （Jinja→minijinja）、parser 都在 Rust 侧，Python engine core 收到的已经是
   `prompt_token_ids`。【源码】`rust/src/text/src/lib.rs:143-148`、
   `rust/src/managed-engine/src/process.rs:57-75`
2. **官方 wheel 自带真 ELF 的 `vllm-rs`，零编译即可抽取运行**。【实测】
   x86_64 42.8 MB / aarch64 39.6 MB；HTTP Range 只下 5.7% / 6.1% 的 wheel 字节。
3. **成熟度仍是实验性**：`vllm-rs serve` 显式承认 **50 个参数未实现 + 7 个接受但无效**，
   含 tokenizer 的 `--tokenizer` / `--tokenizer-revision` / `--skip-tokenizer-init`。
   【源码+实测】`rust/src/cmd/src/cli/unsupported.rs:14-52, 86-101`

## 1. 形态：搬了什么进 Rust，南北向边界在哪

分层（上游 `rust/README.md:9-31`）：`vllm-cmd`(CLI) → `vllm-server`(axum HTTP) →
`vllm-chat`(模板/解析) → `vllm-text`(**tokenizer + 增量 detokenizer**) →
`vllm-llm`(token-in/token-out 门面) → `vllm-engine-core-client`(ZMQ+msgpack)。
**【实测】**二进制只有两个子命令：`frontend`（Python 监管的 worker，socket fd 与
ZMQ 地址由 Python 传入）与 `serve`（Rust 托管：自己 spawn Python headless engine）。

`serve` 真正执行的命令（【源码】`rust/src/managed-engine/src/process.rs:57-75`）：

```
python3 -m vllm.entrypoints.cli.main serve <MODEL> --headless \
  --data-parallel-address <host> --data-parallel-rpc-port <port> --data-parallel-size <n>
```

关键在 `--headless`：Python 侧走 `run_headless()`
（`vllm/entrypoints/cli/serve.py:173-235`），**只创建 EngineCore 进程**
（`CoreEngineProcManager`），不启 API server，因此
**tokenizer / detokenizer 在 Python 进程里不再存在**。
另一条路径 `VLLM_USE_RUST_FRONTEND=1 vllm serve ...` 由 Python 当监管者：
`vllm/v1/utils.py:325-401` 的 `RustFrontendProcessManager` 把监听 socket fd、
输入/输出 ZMQ 地址和一份 `--args-json` 交给 `vllm-rs frontend`。

> **【推断】**"Python 侧只剩 engine core"由 `--headless` 代码路径推出，未起真
> vLLM Python engine 验证（见 §6）。但 mock engine E2E 证实前端发出的确实是 token id。

与 Python 前端逐项对照：HTTP 层 FastAPI/uvicorn → axum；chat 模板 Jinja2 →
**minijinja** + `minijinja-contrib` pycompat（`rust/src/chat/src/renderer/hf/template.rs:14,29-37`）；
tokenizer `vllm/tokenizers/hf.py`（要 `VLLM_USE_FASTOKENS=1` 才走 fastokens）→ 多后端
抽象且 **fastokens 优先**；detokenizer `detokenizer_utils.py` →
`rust/src/tokenizer/src/incremental.rs` 的 `DecodeStream`；parser Python 注册表 →
`vllm-parser` + reasoning/tool-parser crate；与引擎通信两边都是 ZMQ + msgpack。
**【源码】**`rust/src/tokenizer` 是唯一的 tokenizer 实现地，`vllm-text` 只做编排，
唯一构造分发点是 `rust/src/text/src/backend/hf/mod.rs:24-30`。

## 2. 抽取与启动验证（【实测】，可复跑）

wheel 是 zip、中央目录在尾部，因此用 HTTP Range 只取需要的几十 MB。抽取脚本
`harness/rust-frontend/fetch_vllm_rs.py` 有 `--help`、离线 `--self-test`。

| 架构 | wheel | 实际下载 | 占比 | `vllm/vllm-rs` | `_rust_tool_parser` |
|---|---|---|---|---|---|
| x86_64 | 303.7 MB | 17.4 MB | 5.7% | 42.79 MB ELF64 PIE `0597bfc9…807` | 0.99 MB ELF64 DYN |
| aarch64 | 298.3 MB | 18.3 MB | 6.1% | 39.60 MB ELF64 PIE `cae05321…dc0b` | 1.07 MB ELF64 DYN |

详数见 `data/nextgen/wheel-extract-{x86_64,aarch64}.json`。二进制 **not stripped**、
`Type: DYN`(PIE)，`ldd` 只依赖 libgcc/libc/libm/libdl/libpthread ——
**没有 libzmq、没有 libtorch、没有 torch_npu**（ZMQ 是纯 Rust 实现）。
两个架构的 `--help` / `serve --help` 输出**逐字节相同**（`diff -q` 无差异）。

**启动验证：成功。** 【实测】`vllm-rs serve /home/chiro/models/Qwen3-0.6B --data-parallel-size-local 0`
（`data/nextgen/verify-serve-x86_64.log.txt`）：

```
[cmd/src/main.rs:112] running Rust frontend without a managed local Python engine ...
[hf.rs:112]    loading tokenizer with fastokens path=.../Qwen3-0.6B/tokenizer.json
[hf/mod.rs:71] loaded text backend with Hugging Face model files ...
[hf.rs:78]     loaded chat backend with Hugging Face model files ... renderer=hf
[transport.rs:155] waiting for engines to connect ...
```

→ **真的加载了 tokenizer、真的建好了 chat renderer**，随后在 ZMQ 上等引擎握手。
**【源码】**HTTP 监听要等 `EngineCoreClient::connect()` 成功后才 bind
（`rust/src/server/src/lib.rs:182-185, 223`），这解释了没有引擎时端口不通。
aarch64 二进制在本机用 `qemu-aarch64 -L /usr/aarch64-linux-gnu` 可直接跑；
a3-22 真机因 scp 40 MB 超时（>120 s）未用上。

```bash
harness/rust-frontend/verify_vllm_rs.sh --no-fetch              # 复用已抽出的二进制
harness/rust-frontend/verify_vllm_rs.sh                         # 全流程（含 Range 抽取）
harness/rust-frontend/verify_vllm_rs.sh --arch aarch64 --qemu    # aarch64 交叉验证
```

**需不需要 `VLLM_RUST_FRONTEND_PATH`？**【源码】`vllm/envs.py:550-580`：
`VLLM_USE_RUST_FRONTEND=1` 且路径为默认 `auto` 时在 `vllm/` 包目录下找 `vllm-rs`，
**找不到直接 `FileNotFoundError`**（提示 "Build with setuptools-rust"）；想用别处的
二进制（如我们从 wheel 抽的）**必须显式设** `VLLM_RUST_FRONTEND_PATH=/path/to/vllm-rs`。
我们的镜像 `vllm/` 下**没有** `vllm-rs` 与 `_rust_tool_parser.abi3.so`，所以镜像内
直接用 `VLLM_USE_RUST_FRONTEND=1` 会立刻抛错。

## 3. 端到端对照（【实测】做到了，但不是"对 Python 前端"的对照）

本调研纯 CPU、不占 NPU，但上游有现成的引擎协议模拟器 `rust/src/mock-engine`：
**官方 wheel 的 `vllm-rs` 当前端 + 自己编译的 mock engine 当引擎**，走真实
ZMQ + msgpack。

```bash
harness/rust-frontend/e2e_mock_engine.sh --help
EVIDENCE_DIR=data/nextgen harness/rust-frontend/e2e_mock_engine.sh   # 含编译约 2 分钟
```

**【实测】**9 条断言全过（`data/nextgen/2026-09-24-native-e2e-*.log.txt`）：前端加载
tokenizer ✓、ZMQ 握手 `engines connected` ✓、`starting OpenAI server` ✓、
`GET /health` 200、`GET /v1/models` 200（`owned_by: vllm-frontend-rs`）、
`POST /tokenize` 200、`POST /v1/chat/completions` 200（`prompt_tokens: 12`）、
流式 SSE 6 个 content delta。一次覆盖 **minijinja 渲染 HF 模板 → fastokens encode →
ZMQ → mock 采样 → `DecodeStream` 增量 detokenize → OpenAI SSE**，全程无 Python/NPU。

**tokenizer 正确性交叉核对：ids 完全一致。**【实测】同输入 `你好，Hello!`：
Rust `POST /tokenize`（fastokens 后端）给 `[108386, 3837, 9707, 0]`，
容器内 Python `tokenizers` 0.22.2 的 `Tokenizer.from_file(...)` 也给
`[108386, 3837, 9707, 0]`，**逐 id 相同**（Python 侧回环解码得 `你好，Hello!`）。

**与 B 线的分工**：`rust/src/tokenizer` 的 criterion 微基准归 **B 线**
（`harness/rust/`、`docs/03-backend-matrix.md`），本文不重复测量。
**【实测】**落笔时 B 线尚未提交结果（其 worktree 只有 vendored 源码、无 `data/`），
因此本文只补两件 B 线不做的：① 上游 bench 的口径——`benches/hf.rs:61-72` 与
`benches/tiktoken.rs:58-...` 用 `hf_hub` **联网拉模型**（`Qwen/Qwen3.5-0.8B`、
`moonshotai/Kimi-K2.5`），本地复跑必须先改本地路径；② 覆盖缺口——
`benches/hf.rs` 只覆盖六后端的 1/2、`benches/tiktoken.rs` 覆盖 4/5，
**第 3 个（FastokensByteLevel）与第 6 个（Tekken）没有 bench**，且旁路只影响 `decode`。

## 4. 值不值：判断与依据

**换来了什么**：① tokenizer 最佳实践被固化——Python 侧要用户手动开
`VLLM_USE_FASTOKENS=1`，Rust 侧 fastokens 是默认首选、HF 只是兜底；② 一条北向链路
进一个进程，模板渲染/encode/流式 decode/parser 同地址空间，省掉跨进程跨线程对象与
GIL 开销（**【推断】**净收益需 C 线的 TTFT 占比判定）；③ 分发成本几乎为零；④ 一个 trait 统一六后端，加后端是加 impl 而非加分支。

**功能覆盖度差距清单。** 上游 `rust/README.md:5` 自述 "experimental, and is not
feature-complete"。我们把它具体化：`vllm-rs serve --help` 末尾有独立一节
`Options not implemented in Rust frontend yet`
（`data/nextgen/vllm-rs-serve-help-x86_64.txt:158+`），源码里对应
`UnsupportedArgs`（传了就报错退出）与 `Noop`（接受但无效果）两种标记类型
（`rust/src/cmd/src/cli/unsupported.rs:14-52, 86-101`）。
**【实测】计数：50 个参数未实现 + 7 个接受但无效，对比 33 个已支持参数。**

与 tokenizer 直接相关的缺口：

| 缺什么 | 参数 | 影响 |
|---|---|---|
| 不能指定独立 tokenizer | `--tokenizer` | 只能从模型目录找 `tokenizer.json`/`tiktoken.model`/`tekken.json`，指向别处**直接报错** |
| 不能钉版本 / 独立 config | `--tokenizer-revision` `--hf-config-path` | HF revision 不可控；非标准目录受限 |
| 没有 token-in/token-out 模式 | `--skip-tokenizer-init` | 想完全跳过 tokenizer（纯 bench/离线 tokens）做不到 |
| HF 鉴权只有环境变量 | `--hf-token` | `model_files.rs:13` 直接读 `HF_TOKEN`，无 CLI / `hf auth login` 语义 |
| 无 `hf_overrides` / `generation-config` | `--hf-overrides` `--generation-config` | 不能改影响 tokenizer 的 config；采样默认值不能按三模式切换 |
| prompt_embeds / `token_id:{id}` 缺失 | `--enable-prompt-embeds` `--return-tokens-as-token-ids` | embedding 输入路径没有；`logprobs` 里不可 JSON 编码的 token 无法安全表示 |
| **接受但无效（`Noop`）** | `--enable-tokenizer-info-endpoint` `--structured-outputs-config` | 不报错也不生效，静默不一致最危险；另有 `--max-log-len`（未实现）使 prompt 日志截断不可控 |

间接相关也缺：多模态一族（`--limit-mm-per-prompt`/`--mm-processor-kwargs`/
`--media-io-kwargs` …）、LoRA（`--lora-modules`）、插件机制
（`--tool-parser-plugin`/`--reasoning-parser-plugin`/`--io-processor-plugin`）、
`--trust-request-chat-template`（**请求级 chat 模板**）、`--response-role`、
`--stream-interval`、`--config`(YAML)、`--api-server-count`（Noop，多 API server
不可用）、`--grpc`（服务模式；`--grpc-port` 单端口支持）、`middleware`/`root_path`/
`uvicorn-log-level`/`enable-ssl-refresh` 等运维面参数。

**判断（【推断】，依据上述清单）**：

1. **标准 HF chat 服务已在可用区间**：单模型、标准目录、模板在
   `tokenizer_config.json`、不需要 LoRA/多模态 —— E2E 恰好落在交集里，全链路通过。
2. **生产里的怪配置还早**：一旦用到 `--tokenizer`/`--skip-tokenizer-init`/
   `--trust-request-chat-template`/`--hf-overrides`，Rust 前端不是降级而是
   **直接拒绝启动**（`unsupported.rs:23-35` 的 `from_str` 返回 "argument is not
   implemented in Rust frontend yet"）。这比静默降级好，但**灰度必须按参数集合
   做准入判断**。
3. **真该警惕的是 `Noop` 而非 `Unsupported`**：7 个"接受但无效果"里，
   `--enable-tokenizer-info-endpoint`、`--structured-outputs-config`、
   `--enable-auto-tool-choice` 都是客户端/运维会依赖的行为开关。
   建议升级检查把这 7 个参数列成白名单校验。

**落点**：Rust 前端不是"另一个 tokenizer 后端"，而是 Python 侧
`VLLM_USE_FASTOKENS=1` 这条路的正式继任者。上不上取决于业务参数面是否落在
上面的已支持子集里，而不是取决于 encode 快多少。

## 5. 源码级分析

### 5.1 `Tokenizer` trait：六后端如何统一

`rust/src/tokenizer/src/lib.rs:24-67` 的 `pub trait Tokenizer: Send + Sync`
必需四个方法（`encode`/`decode`/`token_to_id`/`id_to_token`），默认
`vocab_size()`（默认 `usize::MAX`）、`is_special_id()`（默认 `false`）与
`create_decode_stream(...)`（默认返回 `DecodeStream`）；`:69` 是
`pub type DynTokenizer = Arc<dyn Tokenizer>`。

六个后端的落点：① HF `tokenizers` → `HuggingFaceTokenizer::new_hf`（`hf.rs:120-126`）；
② fastokens → `HuggingFaceTokenizer::new_fastokens`（`hf.rs:111-117`）；
③ **fastokens byte-level 旁路 → `Backend::FastokensByteLevel`，无独立入口**，
由 `from_fastokens_backend` 自动判定（`hf.rs:98-103`）；
④/⑤ riptoken / tiktoken-rs → `TiktokenTokenizer::new_riptoken` / `::new_tiktoken_rs`
（`tiktoken.rs:280-306` / `:307-...`）；⑥ tekken-rs → `TekkenTokenizer::new`
（`tekken.rs:18-28`）。

统一靠三件事：① **同一 trait + `Arc<dyn Tokenizer>`**，`vllm-text` 只持有
`DynTokenizer`（`rust/src/text/src/backend/mod.rs:61`），不认识具体后端；
② **"优先 + 回落"的构造约定**——`HuggingFaceTokenizer::new()` 先 fastokens、失败
warn 后回落 HF（`hf.rs:129-141`），`TiktokenTokenizer::new()` 先 riptoken、失败回落
tiktoken-rs 并可用 `VLLM_RS_DISABLE_RIPTOKEN` 强制跳过（`tiktoken.rs:33, 261-283`）。
**这是比 Python 侧多出来的一层工程化**：Python 的 `VLLM_USE_FASTOKENS` 是全局单选项，
Rust 是逐模型自动降级；③ **特殊 id 与词表大小的统一语义**——HF 把 special id
预排序成 `Arc<[u32]>` 用 `binary_search`（`hf.rs:196-198`），tiktoken 把
`[num_base_tokens, vocab_upper_bound)` 全注册成占位 special token 保证 decode 不 panic
（`tiktoken.rs:121-150`），tekken 直接委托 `is_special_token`（`tekken.rs:66-68`）。
**已知不统一**：tiktoken 的 `encode` **忽略 `add_special_tokens`**（`tiktoken.rs:456-464`
注释明说）；错误类型是字符串化的 `TokenizerError(String)`（`error.rs:9-12`）。

### 5.2 `incremental.rs::DecodeStream`：六后端共享的增量解码

`IncrementalDecoder`（`incremental.rs:9-25`）只有四个动作：`push_token(id)->新增字节数` /
`next_chunk()` / `flush(truncate_output_to)` / `output()`。默认实现由 trait 默认方法返回
（`lib.rs:54-66`），所以**六个后端不花一行代码就都有流式解码**。算法是
**滑动窗口 + 前缀 diff**，注释明说与 `tokenizers::DecodeStream` 同思路
（`incremental.rs:27-29`）：

1. 持有 `ids`（当前窗口）、`prefix`（上次解出的文本）、`prefix_index`。
2. `push_token`：id 压窗 → 解码整窗 → 与 `prefix` 比长度取增量 →
   **`drain(..prefix_index)` 丢掉已定型的 id**，只把剩余重解码成新 `prefix`
   （`incremental.rs:128-147`）。窗口不随生成长度线性增长。
3. 新文本比 prefix 短、或以 U+FFFD 结尾 → 返回 0，等下一个 token
   （`incremental.rs:137-139`）。

三个易被忽略的细节：① **prompt 侧容错**——prompt id 可能来自模型词表而非本
tokenizer，`decode_prompt_context` 在严格 decode 失败时**过滤映射不出的 id
再重试**（`incremental.rs:78-95`），该容错**只给 prompt、不给生成**（生成侧未知
id 仍报错，测试 `generated_unknown_ids_still_return_decode_error`）；
② **CJK 友好的前缀播种**——只试末尾 4–6 个 id 的后缀，要求解出的文本不含
U+FFFD 且过滤后仍 ≥4 个 id，否则退回整段 prompt（`incremental.rs:67-68, 102-124`），
上限取 6 是因为再放大就退化成全量解码、白白多花钱；③ **UTF-8 边界**——
`push_token`/`next_chunk`/`flush` 三处都用 `floor_char_boundary`
（`incremental.rs:141, 152, 168`）。这不是洁癖：
`next_chunk_cutoff_respects_char_boundary`（`incremental.rs:473-493`）对应的正是
"stop string 触发 hold-back 时把 CJK/emoji 切一半"的 panic，
`non_monotonic_decode_does_not_panic`（`incremental.rs:435-448`）复现上游
vllm-project/vllm#17448 那类"加一个 token 让前面文本变短"的场景。

`min_bytes_to_buffer` 是**为 stop string 服务的 hold-back**，调用点在
`rust/src/text/src/output/decoded.rs:121-131`：`include_stop_str_in_output=false`
时回退字节数 = 最长 stop string 长度 − 1，避免流式输出先把 stop string 的开头吐出去。

### 5.3 `byte_level_decode.rs`：为什么能跳过 `Vec<String>` / `join`

模块注释写得很直白（`byte_level_decode.rs:4-6`）：fastokens 通用
`Decoder::decode` 管线每级都要拼 `Vec<String>` 再 `join`，而**纯 GPT-2
ByteLevel 的语义其实就是"字符→原始字节→UTF-8"**，没有中间字符串。实现只有 60 行：
① **编译期建表** `const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte()`
（`byte_level_decode.rs:10, 16-29`）——GPT-2 的 byte↔char 映射只落在
U+0000..U+0143，扁平数组足够，`const fn` 编译期算完、运行时零初始化；
② **一次分配** `Vec::with_capacity(lower.saturating_mul(4))`
（`:48`，每字符最多 4 字节）；③ **逐字符回写字节**——命中表就
`push(CHAR_TO_BYTE[cp])`，否则（DeepSeek 的 U+FF5C、U+2581 这类**非 GPT-2
codepoint**）按 UTF-8 原样透传（`:49-59`）；④ **一次 UTF-8 校验**
`String::from_utf8(bytes)`、失败才 lossy（`:61`）。

→ 全程只有**一个 `Vec<u8>` + 一个 `String`**，而非"每级一个 `Vec<String>`、
每 token 一个 `String`、最后 `join`"。调用侧同样省：`decode_fastokens_byte_level`
先把 id 映成 `Vec<&str>`（借用，不复制）再交给 `decode_byte_level`
（`hf.rs:40-54`、`byte_level_decode.rs:45`）。**启用条件是加载期一次性判定**：
`is_byte_level_only` 统计 decoder 树里的 `ByteLevel` 叶子是否为 1（`Fuse` 在
fastokens 里是空 `Sequence`、算 no-op），命中则设 backend 为 `FastokensByteLevel`
（`hf.rs:30-38, 98-103`）——所以 Qwen3 这类纯 GPT-2 byte-level 词表**默认走旁路**。
正确性有对拍兜底：`fast_byte_level_matches_fastokens_decode`（`hf.rs:362-381`）对
5 组 token 序列 × `skip_special_tokens` 两取值断言旁路输出与 fastokens 自己的
`decode` **逐字节相同**；`byte_level_decode.rs:84-121` 另有全 256 字节 round-trip、
空格标记 Ġ、多字节 €、非 GPT-2 字符透传四个单测。

## 6. 复现清单与遗留问题

| 目的 | 命令 | 耗时 |
|---|---|---|
| 抽取 `vllm-rs` | `python3 harness/rust-frontend/fetch_vllm_rs.py --arch x86_64 --out /tmp/d-nextgen-wheel` | ~40 s |
| 启动验证 | `harness/rust-frontend/verify_vllm_rs.sh` | ~55 s |
| 启动验证（aarch64 交叉） | `harness/rust-frontend/verify_vllm_rs.sh --arch aarch64 --qemu` | ~90 s |
| 全链路 E2E | `EVIDENCE_DIR=data/nextgen harness/rust-frontend/e2e_mock_engine.sh` | ~2 min |

资源纪律：编译一律经 `scripts/heavy_lock.sh` + `scripts/limit.sh`（绑 4-7 核、`-j4`）；
HTTP/启动实验都是秒级轻进程，不占 NPU。实测起始 `load average 1.33 / 2.08 / 2.06`。

**遗留问题（未验证，勿当结论）**：① **没做 Python 前端的同负载对照**——本机无 vllm
安装、容器是 ascend 镜像（禁止占 NPU），"Rust 前端省了多少"**没有数**，只有链路证明；
② **a3-22 真机未验证**——scp 40 MB 超时，改用本地 `qemu-aarch64`，**aarch64 原生跑通
仍属未测**；③ **renderer 只验证了 `auto`→`hf`**，deepseek_v32/deepseek_v4/harmony/
inkling 四种渲染器没有对应模型可测；④ **第 3、6 个后端的实测数据仍缺**（ByteLevel 旁路、
Tekken），等 B 线结论；⑤ **跨版本绑定强度未查**，只验证了 0.26.0 内部一致；
⑥ **容器镜像的修复路径未端到端验证**（抽 wheel 二进制 + `VLLM_RUST_FRONTEND_PATH`）。
