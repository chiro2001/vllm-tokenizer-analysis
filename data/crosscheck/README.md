# 根代理交叉复核：fastokens vs HF tokenizers（Python 侧）

**目的**：B 线在 Rust harness 里测出 `fastokens_byte_level` 比 `hf` 快 8–19×。
该数字会进结论，因此由根代理用**独立装置**（Python 侧、容器内、不同文本）复核。

**装置**：`scripts/docker_run.sh` 同款镜像；fastokens 0.3.2 的 wheel 解包后以
`sys.path` 注入（容器内没装它）；文本为自造中英混合。

**结果**（2026-09-24，x86_64，绑核 4 核）：

| 项 | HF `tokenizers` | fastokens `_TokenizerShim` | 倍数 |
|---|---:|---:|---:|
| encode 短文（160 tokens / 472 chars） | 129.6 µs | **13.98 µs** | **9.3×** |
| encode 长文（1280 tokens） | 1080.0 µs | **80.3 µs** | **13.5×** |
| decode（160 ids → 文本） | 17.01 µs | 15.66 µs | **1.09×** |
| token ids 一致性 | — | `ids_identical: true` | ✅ |

**与 B 线对照**（`harness/rust/bench`，Rust criterion）：

| 项 | B 线（Rust） | 根代理（Python） | 判定 |
|---|---:|---:|---|
| encode @128 | 118 → 14.5 µs（8.1×） | 130 → 14.0 µs（9.3×） | ✅ 吻合 |
| encode @1k | 959 → 50.9 µs（18.8×） | 1080 → 80.3 µs（13.5×） | ✅ 同量级 |
| decode @128 | 20.1 → 2.53 µs（**8.0×**） | 17.0 → 15.7 µs（**1.09×**） | ❌ **显著不同** |

**关键结论（三条）**：

1. **encode 的 8–13× 加速在两端都存在** ⇒ 这是 fastokens 引擎本身的能力，
   不是 Rust 前端专有；Python 用户开 `VLLM_USE_FASTOKENS=1` 也能拿到。
2. **decode 的 8× 只出现在 Rust 侧** ⇒ 它来自 `Backend::FastokensByteLevel`
   这条**旁路**（跳过 `Vec<String>` 组装 + `join("")`），而 Python 侧用的是
   通用 decode 路径。**这个旁路是 `vllm-tokenizer` crate 自己写的，不是 fastokens 提供的。**
3. **两条独立装置在 encode 上互证**，构成 B 线 `03-backend-matrix` 的跨装置校准依据。

**局限**：文本自造、长度不完全对齐；fastokens 经 `sys.path` 注入而非正式安装；
Python 侧测的是 `_TokenizerShim`（fastokens 的兼容层），与 Rust 侧原生
`fastokens::Tokenizer` 是同一库的不同绑定。
