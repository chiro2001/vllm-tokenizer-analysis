# 发布记录

## 2026-09-25 · 首次发布

| 项 | 值 |
|---|---|
| 仓库 | **https://github.com/chiro2001/vllm-tokenizer-analysis**（private） |
| 分支 | `main`，单提交 `a84906d` |
| 导出目录 | `/home/chiro/projects/vllm/tokenizer-publish`（本工作树之外） |
| 提交者 | `Chiro <41908064+chiro2001@users.noreply.github.com>`（GitHub noreply） |
| 内容 | 306 个文件 / 9.7 MB |

### 净化处置

由 `scripts/export_publish.sh` 自动完成：rsync 导出 → 替换内部标识 →
**重建为单一起点的历史**（不携带内部提交历史）→ 复核 → `gh repo create --private --push`。

| 原值类别 | 占位符 | 命中 |
|---|---|---:|
| 内部分析项目目录名 | `HIST_PROJECT` | 20 文件 |
| 另一内部项目目录名 | `OTHER_PROJECT` | 2 文件 |
| 本地开发机主机名 | `LOCAL_HOST` | 11 文件 |
| 内部员工号 | `REMOTE_USER` | 1 文件（仅映射表） |
| 发布者邮箱 | `PUBLISHER_EMAIL` | 1 文件（仅映射表） |

**发布前的安全检查**（全部为 0）：明文凭据（`ghp_` / `PRIVATE KEY` /
`BEGIN RSA` / `secretid` / `secretkey`）、内网 IP（`192.168`）、
内部镜像源（`quay.nju`）。

### 整体排除的内容

| 路径 | 原因 |
|---|---|
| `refs/` | upstream vLLM 源码副本（Apache-2.0），非本项目产出；按 `refs/README.md` 记录的 commit 自行 clone |
| `harness/**/target/` | Rust 编译产物（约 1.8 GB），可重建 |
| `.sanitize-map.tsv` | 净化映射表本身（含真实值），**只存在于内部工作树** |
| `.locks/` | 运行时锁文件 |
| `data/profiles/` | 原始 perf 采样 |

### 可逆性与复现

净化**可逆**：`bash scripts/sanitize_for_publish.sh --revert`
（在导出目录上执行，需能访问内部的 `.sanitize-map.tsv`）。

重新发布（内容更新后）：

```bash
scripts/export_publish.sh --no-push    # 先干跑，看复核是否通过
scripts/export_publish.sh              # 实际导出 + 强推更新
```

> 导出目录已是 git 仓库且已设 `origin`；脚本会检测到仓库已存在并直接 `push`，
> 因此重复执行会**重写为单提交历史**（`git init` 只在无 `.git` 时执行，
> 已有仓库会追加提交）。如需保持"单提交"形态，先删除导出目录的 `.git`。

### 与内部工作树的关系

- 内部工作树 `/home/chiro/projects/vllm/tokenizer` 保留**完整历史**
  （四条线的 worktree 分支 `agent/*`、全部中间提交、原始 manifest 与映射表）。
- 发布副本是**净化后的快照**，不含内部标识与内部历史。
- 两者内容同步靠 `scripts/export_publish.sh`，不要手工改发布副本。
