# harness/ —— 线 B（多后端评估）的复现装置

两套装置 + 三个辅助脚本。**所有 CPU 密集操作都必须经**
`/home/chiro/projects/vllm/tokenizer/scripts/{heavy_lock,limit}.sh`
（本机是共享开发机，见 `plan/COORDINATION.md` §9）。

```
harness/
├── rust/                     # 装置 1：Rust（criterion + 自建计时器）
│   ├── Cargo.toml            # workspace（bench + vendored vllm-tokenizer）
│   ├── bench/                # 本线自建 harness（核心资产）
│   ├── vendor/vllm-tokenizer # vLLM 0.26.0 的 tokenizer crate 快照（逐字节复制）
│   ├── vendor/gigatoken/     # gigatoken 0.10.0 sdist 解包（含 vendor 补丁）
│   └── sync-vendor.sh        # 校验/更新上面的快照
├── python/bench_backends.py  # 装置 2：Python 侧三实现（hf / fastokens / gigatoken）
├── bridge-calibration.sh     # 跨装置桥梁校准（Rust vs Python）
└── fetch-artifacts.sh        # 下载并校验本地工件（tiktoken / tekken）
```

## 最短上手

```bash
export PATH="$HOME/.cargo/bin:$PATH"
ROOT=/home/chiro/projects/vllm/tokenizer-wt/B-backends
LOCK=/home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh
LIM=/home/chiro/projects/vllm/tokenizer/scripts/limit.sh

cd $ROOT
./harness/fetch-artifacts.sh                       # 取 tiktoken.model / tekken.json
cd harness/rust
CORES=2 JOBS=2 $LOCK $LIM cargo +nightly build --release -p b-backends --features gigatoken

./target/release/b-backends --help                 # 四个子命令都有 --help
CORES=2 $LOCK $LIM ./target/release/b-backends correctness
CORES=2 $LOCK $LIM ./target/release/b-backends run --backend all --corpus all --lengths all --ops all --prefix e1
CORES=2 $LOCK $LIM ./target/release/b-backends gigatoken-audit --total-bytes 4194304
CORES=2 $LOCK $LIM cargo +nightly bench --features gigatoken    # criterion 交叉验证
```

## 为什么 gigatoken 需要 nightly

`vendor/gigatoken/src/lib.rs:1` 是 `#![feature(portable_simd)]`。
不带 `--features gigatoken` 时用 stable 即可（六个 vLLM 后端）。

## 为什么 vendored 的 gigatoken Cargo.toml 与上游不同

上游 sdist 的 `[profile.profiling]` 用了 nightly 的 `rustflags` profile 键
（`profile-rustflags` feature），**stable cargo 连解析 manifest 都会失败**，
会连带整个 workspace 无法构建。已删除这两节，补丁见
`vendor/gigatoken-vendor.patch`（只影响它的本地 profiling 工作流，不影响库产物）。

## 已知的上游问题（实测复现，详见 `docs/03` §4/§5）

- **fastokens 0.2.1**：长中文/长英文输入会在 `pre_tokenizers/split.rs:419`
  越界 panic（`n_chunks` 被 `.max(2)` 抬到 2、`pcre2` 表只有 1 条）。
  vLLM 的"fastokens 失败回落 HF"逻辑拦不住 panic。
- **tekken-rs 0.1.1**：`decode` 对不完整 UTF-8 直接报错（而非返回替换字符），
  使中文流式解码不可用。

本 harness 用 `stats::guard_panic`（保留 `panic = unwind`）隔离这两类问题，
把失败记成数据行而不是让整轮矩阵崩掉。
