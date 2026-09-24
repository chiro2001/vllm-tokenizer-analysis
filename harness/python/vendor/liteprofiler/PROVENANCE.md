# 来源与指纹

| 文件 | 来源（只读引用，未改动） | sha256 |
|---|---|---|
| `liteprofiler-vllm.patch` | `/home/chiro/projects/vllm/HIST_PROJECT/.research/liteprofiler-patches/liteprofiler-vllm.patch` | `f283fb121bc099f531f433b5fff3431a2f3f92a168b6717168a7a01828520066` |
| `minimal.patch` | 上面整份 patch 中 **只保留** `vllm/renderers/base.py`、`vllm/v1/utils.py`、`vllm/utils/lite_profiler.py` 三个文件后的子集 | 见 `MANIFEST.json` |

`minimal.patch` 的作用：让本地 CPU-only 容器（源码未插桩）也能产出与历史
`lite.log` **同格式、同边界**的 scope 行。抽取方式见
`scripts/build_liteprof_overlay.sh` 的头部注释；重放命令：
`python3 -c "import re,sys; ..."`（见 `harness/python/vendor/liteprofiler/extract.py`）。
