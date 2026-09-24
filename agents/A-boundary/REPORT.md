# A 线交接报告：边界与静态链路

> 交付物：[`docs/01-code-logic.md`](../../docs/01-code-logic.md)（350 行，中文）
> 状态：**已完成**。纯静态分析，未跑实验/基准（符合线 A 约定）。
> 口径：vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`（0.26.0），
> 只读源 `/home/chiro/projects/vllm/HIST_PROJECT/vllm`。

## 1. 结论摘要

1. **链路只有三段真实工作量，全在 API server 前端进程**：入站 encode、出站流式 decode、
   停止串判定。**engine core 不在这条链路上**——唯一例外是 structured output：
   engine core 为语法编译另加载一份 tokenizer
   （`vllm/v1/structured_output/__init__.py:79`，构造点 `vllm/v1/engine/core.py:137`）。
2. **架构分层已固化**：`vllm/tokenizers/`（tokenizer 对象 + `TokenizerLike` 协议）与
   `vllm/renderers/`（渲染 + 编码编排）互不继承，Renderer 持有 TokenizerLike
   （`vllm/renderers/registry.py:68`）。讲"tokenizer 链路"必须跨这两层。
3. **`tokenizer_pool_size` 这个配置项在 0.26.0 不存在**（全仓 0 命中）；任务描述里的该名字
   应理解为"池大小"概念。实际由 `renderer_num_workers`（`vllm/config/model.py:337`）决定
   线程池 worker 数、由 `renderer_num_workers + 1` 决定 tokenizer deepcopy 份数
   （`vllm/renderers/hf.py:920-922`）。池空时**不阻塞**，现场 deepcopy 并让池无上限增长
   （`vllm/tokenizers/hf.py:51-55`）——这是链路里唯一"随并发劣化"的隐藏项。
4. **历史日志三个 `tokenizer:` scope 是 LiteProfiler 补丁加的，不是上游代码**
   （全仓 0 命中；出处
   `HIST_PROJECT/.research/liteprofiler-patches/liteprofiler-vllm.patch:331/341/351/360/375/397`）。
   由此推出两条口径：**①** 前端 `tokenizer: decode` 只是"prompt token ids 反解回文本"，
   **不覆盖生成期解码**（后者在 `vllm/v1/engine/detokenizer.py`）；
   **②** `render_messages` 的 scope 体不含 `tokenize_prompts`，两段**可直接相加**。
5. **下一代把链路整体搬进 Rust 进程**（axum → minijinja → `vllm-tokenizer` → ZMQ/msgpack），
   流式解码与停止串判定也在 Rust 侧（`rust/src/text/src/output/decoded.rs:176`、`:309`）；
   **engine core 原封不动留在 Python**。南向 DTO `EngineCoreSamplingParams` 明确不含 `stop`
   字符串（`rust/src/engine-core-client/src/protocol/sampling.rs:51-56`）
   ⇒ 停止串判定天然属于前端。

## 2. 对各线有用的发现

### 2.1 给 C 线（成本占比）

- **分子分母要分清**：`tokenizer: encode` 的 scope 体就是 `tokenizer(...)` 一次调用
  （`vllm/renderers/base.py:481`）；`render_messages` scope 只包 `render_messages` 调用列表
  （patch `:375-380`），**不含**其后的 `tokenize_prompts`。两者可加，但不能把
  `render_messages` 当作"chat 全流程"。
- **生成期 decode 没有历史 scope**：`FastIncrementalDetokenizer.decode_next`（`:210`）与
  `SlowIncrementalDetokenizer.decode_next`（`:291`）在
  `OutputProcessor.process_outputs`（`vllm/v1/engine/output_processor.py:653`）里被调用。
  E2.3 若复用历史 `tokenizer: decode` 口径，测到的是**另一个东西**（prompt 反解），必须区分命名。
- **decode 走哪条路在启动期就定死**：`USE_FAST_DETOKENIZER`（`tokenizers >= 0.22.0`）**且**
  tokenizer 是 `TokenizersBackend` 才走 Fast，否则 Slow
  （`vllm/v1/engine/detokenizer.py:24/56/60/65`）。容器内 0.22.2 + transformers v5 ⇒ Qwen3 走 Fast。
- **线程池是三条 scope 之外的隐藏串扰源**：并发 > `renderer_num_workers+1` 时每个超额请求
  付一次 deepcopy（`vllm/tokenizers/hf.py:54`），E2.5 的并发曲线要把它单独标出来。
- 历史 lite.log（`preparing-input-phase/data/profiles/real-b1/lite.log`）里
  `tokenizer: encode` 的 tid=712698 ≠ 主线程 700907、pid 相同 ⇒ 三条 scope 确实在前端线程池，
  可作旁证。

### 2.2 给 B 线（后端矩阵）

- Python 侧三个实现的分派入口是 `get_tokenizer`（`vllm/tokenizers/registry.py:181`）：
  `VLLM_USE_FASTOKENS` 时先 `apply_fastokens_patch()`（`:191-196`），是**进程级幂等
  monkey-patch**（`vllm/tokenizers/fastokens.py:20`），会**同时重绑
  `tokenizers.decoders.DecodeStream`**——Fast 解码器刻意"从模块属性查"该符号来兼容它
  （`vllm/v1/engine/detokenizer.py:180-182`）。所以 fastokens 的 A/B 必须在独立进程里跑。
- `maybe_make_thread_pool` 会把 tokenizer 的 `__class__` 原地换成池子类
  （`vllm/tokenizers/hf.py:101`）；基准若经过 renderer 构造路径，量到的就不是裸 tokenizer。
- `TokenizersBackend` 的只读属性（`vocab_size`/`all_special_ids`/`max_chars_per_token`/…）
  在 `get_cached_tokenizer`（`:107-195`）里被缓存成实例字段；不经过它则每请求重算。

### 2.3 给 D 线（下一代 Rust 前端）

- Rust 侧 tokenizer 三态在 `rust/src/tokenizer/src/hf.rs:19-25`：`Backend::{Hf, Fastokens,
  FastokensByteLevel}`；`new()` 的选择顺序是 **fastokens 优先、失败回退 HF**（`:129-141`），
  `FastokensByteLevel` 由 decoder 是否"纯 ByteLevel"自动判定（`:30-38`、`:98-103`）。
- Rust 前端的增量解码是**第三种算法**（不是 Fast 的 `DecodeStream`，也不是 Slow 的前缀偏移）：
  每 token 全量 `decode(ids)` 取增量 + 4~6 token 短后缀播种
  （`rust/src/tokenizer/src/incremental.rs:67-124`、`:135-146`）。跨实现比 decode 必须说明这一点。
- 南北向：Python 侧用 `RustFrontendProcessManager` 启动 `vllm-rs frontend`，传 `--listen-fd`
  与 ZMQ 地址（`vllm/v1/utils.py:328-360`）；`VLLM_USE_RUST_FRONTEND` 默认 **0**
   （`vllm/envs.py:155`、`:1339`）。

## 3. 口径陷阱（务必别踩）

| # | 坑 | 事实 |
|---|---|---|
| 1 | 以为 `tokenizer_pool_size` 是配置项 | 0.26.0 **没有**这个符号；池大小 = `renderer_num_workers + 1` |
| 2 | 以为 `tokenizer: decode` 是生成期解码 | 它只包 `_decode`/`_detokenize_prompt*`（prompt 反解）；生成期解码在 `vllm/v1/engine/detokenizer.py` |
| 3 | 以为 `render_messages` scope 含编码 | 不含，`tokenize_prompts` 在 `with` 之外 |
| 4 | 以为三个 scope 是上游代码 | 是 LiteProfiler 补丁加的（patch `:331` 等）；上游只有 `VLLM_CUSTOM_SCOPES_FOR_PROFILING` 开关（`vllm/v1/utils.py:755`） |
| 5 | 以为 engine core 会加载 tokenizer | 纯文本模型下 `gpu_worker.py`/`gpu_model_runner.py` 对 tokenizer **零命中**；只有 structured output 与少数 MM 模型自用 |
| 6 | 把 `TokensPrompt` 当 engine 侧结构 | 它先转 `TokensInput`（`vllm/inputs/engine.py:29`）再转 `EngineCoreRequest`（`vllm/v1/engine/__init__.py:88`）；**offsets 到 `EngineCoreRequest` 就丢了** |
| 7 | 在基准里复用 renderer 构造路径 | 会拿到被换过 `__class__` 的池化 tokenizer |

## 4. 未解项 / 需后续验证

1. **三个 scope 的插桩开销未测**：`record_function_or_nullcontext` 在开关关闭时是
   `nullcontext`（`vllm/v1/utils.py:755-759`），打开时的成本须实测（属 C 线）。
2. **`copies = renderer_num_workers + 1` 的 `+1` 意图是推断**：依据是"池并发度 = worker 数 +
   主线程 1"，无上游注释直接说明。
3. **单次 `copy.deepcopy` 的 µs 成本未测**，池空扩容在并发拐点上的占比亦未测（属 E2.5）。
4. **Rust 前端未做运行验证**：本文只有源码级结论，`VLLM_USE_RUST_FRONTEND` 实际启用后的行为
   未验证（属 D 线）。
5. **模型自带 tokenizer 的每步成本未测**（如 `qwen3_vl_moe.py:220`、`voxtral.py:323`）：
   仅据调用点判定"不在 `execute_model` 每步路径上"，属推断。
6. 本仓库源码为**浅克隆**（单 commit `568afb3a1`），无法做历史考古；结论只对 0.26.0 成立。

## 5. 交付与自查

- 行号核对：文档内 **52 条** `路径:行号` 引用全部用 `rg`/`sed` 逐条回读验证（含越界检查），
  无凭记忆书写；推断/未证实处以 **推断**、**未验证**、**未测** 显式标注。
- 形态：350 行中文；图 3 张——两代并排 ASCII 调用图（§3.1）、Mermaid 进程/线程图（§6）、
  进程拓扑 ASCII（§2.1）。
- 提交：`A: 交付 docs/01-code-logic.md（tokenizer 链路边界与静态分析）`；本报告另起一次提交。
