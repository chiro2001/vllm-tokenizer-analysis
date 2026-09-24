#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build_liteprof_overlay.sh —— 在容器内构建「带 LiteProfiler 插桩」的 vllm 覆盖层
#
# 为什么需要它：本次调研用的镜像
#   local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922
# 里的 /vllm-workspace/vllm 是 **未插桩** 的 0.26.0 基线；历史 lite.log 则来自
# LiteProfiler 插桩镜像。为了让 C 线的三个 scope 与历史数据口径一致，这里把
# 上游 patch（vllm/utils/lite_profiler.py + vllm/v1/utils.py + vllm/renderers/base.py
# 三个文件）叠到镜像源码的 **符号链接覆盖层** 上，然后用 PYTHONPATH 前置该覆盖层。
#
# 覆盖层做法：cp -al 把镜像里的 vllm 包整树做成**硬链接**副本（秒级、零额外
# 空间），只把需要改写的文件先 rm 再 cp 成真实副本（从而断开硬链接）再打 patch。
# 这样既不会写坏镜像源码（容器 --rm，改动本就随容器消失），也不会把源码拷进 git。
#
# 用法（在容器内，工作区挂在 /work）：
#   PYTHON=/usr/local/python3.12.13/bin/python3 bash /work/scripts/build_liteprof_overlay.sh \
#       --dest /tmp/c-overlay --patch /work/harness/python/vendor/liteprofiler/minimal.patch
# ---------------------------------------------------------------------------
set -euo pipefail

usage() {
    cat <<'EOF'
用法: build_liteprof_overlay.sh [选项]

  --dest DIR       覆盖层输出目录（默认 /tmp/c-overlay；放在容器 /tmp 下是为了
                   避开 git worktree 检测——覆盖层只是容器内的临时物）
  --src DIR        镜像内 vllm 源码根（默认 /vllm-workspace/vllm）
  --patch FILE     只包含三个文件的 LiteProfiler patch（必填）
  --files LIST     覆盖（真实拷贝）的文件列表，逗号分隔
                   默认 vllm/renderers/base.py,vllm/v1/utils.py,vllm/envs.py
                   （lite_profiler.py 是 patch 新建文件，不需要预先拷贝）
  --force          目标目录已存在时先删除重建
  -h, --help       显示本帮助

退出码: 0 成功；2 参数错误；1 构建失败
EOF
}

DEST=/tmp/c-overlay
SRC=/vllm-workspace/vllm
PATCH=
FILES=vllm/renderers/base.py,vllm/v1/utils.py,vllm/envs.py
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dest)  DEST=$2; shift 2 ;;
        --src)   SRC=$2; shift 2 ;;
        --patch) PATCH=$2; shift 2 ;;
        --files) FILES=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$PATCH" ] || { echo "缺少 --patch" >&2; usage >&2; exit 2; }
[ -d "$SRC" ] || { echo "源码目录不存在: $SRC" >&2; exit 1; }
[ -f "$PATCH" ] || { echo "patch 不存在: $PATCH" >&2; exit 1; }

if [ -e "$DEST" ]; then
    if [ "$FORCE" = 1 ]; then
        rm -rf "$DEST"
    else
        echo "目标已存在（加 --force 重建）: $DEST" >&2
        exit 1
    fi
fi

mkdir -p "$DEST"
# 1) 整树硬链接副本：$DEST/vllm 必须是包目录本身（不要写成 cp -a src dest
#    那种「dest 不存在时 dest 就是副本」的形式，否则 $DEST/vllm 不存在）。
cp -al "$SRC/vllm" "$DEST/vllm"
[ -f "$DEST/vllm/__init__.py" ] || { echo "覆盖层结构不对: $DEST/vllm/__init__.py 缺失" >&2; exit 1; }

# 2) 需要改写的文件替换为真实副本
IFS=',' read -r -a file_list <<<"$FILES"
for f in "${file_list[@]}"; do
    rm -f "$DEST/$f"
    mkdir -p "$(dirname "$DEST/$f")"
    cp "$SRC/$f" "$DEST/$f"
    echo "overlay real file: $f"
done

# 3) 打 patch。必须 cd 到覆盖层再 apply：如果从 /work（git worktree）里调用，
#    git 会去找 .git 而容器内没有对应的 worktree 元数据，直接报
#    "fatal: not a git repository"。覆盖层在 /tmp 下、不在任何仓库里，纯工作区
#    改写不需要仓库（不带 --index/--cached）。
(
    cd "$DEST"
    git apply --verbose -p1 "$PATCH" 2>&1 | sed 's/^/  /'
) \
    || { echo "git apply 失败" >&2; exit 1; }

# 4) 校验：三个 scope 字符串、模块可读
PY="${PYTHON:-/usr/local/python3.12.13/bin/python3}"
for name in "tokenizer: encode" "tokenizer: decode" "tokenizer: render_messages"; do
    grep -q "$name" "$DEST/vllm/renderers/base.py" \
        || { echo "缺少 scope: $name" >&2; exit 1; }
done
grep -q "class LiteScope" "$DEST/vllm/utils/lite_profiler.py" \
    || { echo "lite_profiler.py 未落地" >&2; exit 1; }
grep -q "LiteScope" "$DEST/vllm/v1/utils.py" \
    || { echo "v1/utils.py 未接线" >&2; exit 1; }

# 5) 语法/结构自检。注意：**不要**在这里 import vllm 顶层——镜像里 import vllm
#    会走平台插件 + torch_npu 加载，只有经过 /opt/va26-scripts 的 CPU 桩 wrapper
#    才成立。这里只做 py_compile 与静态检查，真正的 import 冒烟放在 harness 里。
"$PY" -m py_compile \
    "$DEST/vllm/renderers/base.py" \
    "$DEST/vllm/v1/utils.py" \
    "$DEST/vllm/envs.py" \
    "$DEST/vllm/utils/lite_profiler.py" \
    || { echo "py_compile 失败" >&2; exit 1; }
grep -q "VLLM_LITE_PROFILER_LOG_PATH" "$DEST/vllm/envs.py" \
    || { echo "envs.py 未接入 VLLM_LITE_PROFILER_LOG_PATH" >&2; exit 1; }
echo "overlay ready: $DEST"
