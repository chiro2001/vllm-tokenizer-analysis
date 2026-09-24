#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# c_cost_run.sh —— C 线容器执行器
#
# 在无卡 x86 上跑 vLLM 0.26.0 的 renderer/tokenizer 代码路径。
# 与主仓库的 scripts/docker_run.sh 是同一套底座（都用镜像自带的 CPU 桩
# wrapper /opt/va26-scripts/run_real_test_official.sh），额外提供：
#   --liteprof   在容器内先构建 LiteProfiler 覆盖层（真实 LiteScope 插桩），
#                再以 PYTHONPATH=/tmp/c-overlay 运行，使产出的 scope 行
#                与历史 lite.log 同格式、同边界。
#
# **资源纪律**（plan/COORDINATION.md §9）：本机 LOCAL_HOST 是共享开发机。
# 本脚本强制：
#   docker  --cpus=6 --memory=8g --memory-swap=8g --shm-size=2g
#   cpuset  默认 4-7（COST_CORES 可覆盖），把 0-3 / 8-11 留给交互与其他会话
#   线程池  RAYON/OMP/MKL/NUMEXPR_NUM_THREADS = cpuset 核数
#   负载门禁 跑之前检查 load average 与 MemAvailable，不达标就拒绝启动
#
# 用法:
#   scripts/c_cost_run.sh harness/python/bench_paths.py --out data/cost/x.json
#   scripts/c_cost_run.sh --liteprof harness/python/bench_paths.py
#   COST_CORES=4-5 scripts/c_cost_run.sh harness/python/bench_paths.py
#   COST_SKIP_GUARD=1 scripts/c_cost_run.sh ...        # 跳过负载门禁（不建议）
#   TOKENIZER_ROOT=/path/to/wt scripts/c_cost_run.sh --shell
#
# 重活锁（同一时刻全项目只允许一个重活）：
#   scripts/heavy_lock.sh 用主仓库那份：
#   /home/chiro/projects/vllm/tokenizer/scripts/heavy_lock.sh scripts/c_cost_run.sh ...
# ---------------------------------------------------------------------------
set -euo pipefail

IMAGE="${TOKENIZER_IMAGE:-local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922}"
MODELS="${TOKENIZER_MODELS:-$HOME/models}"
HOST_ROOT="${TOKENIZER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OVERLAY="${COST_OVERLAY:-/tmp/c-overlay}"
COST_CORES="${COST_CORES:-4-7}"
COST_CPUS="${COST_CPUS:-6}"
COST_MEM="${COST_MEM:-8g}"
COST_SHM="${COST_SHM:-2g}"
MAX_LOAD="${COST_MAX_LOAD:-8}"
MIN_AVAIL_GB="${COST_MIN_AVAIL_GB:-6}"
LITEPROF=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --liteprof) LITEPROF=1; shift ;;
        --overlay) OVERLAY=$2; shift 2 ;;
        -h|--help|"") usage ;;
        *) break ;;
    esac
done

[ $# -gt 0 ] || { echo "缺少要执行的脚本/代码" >&2; usage >&2; }

NCHOSEN=0
IFS=',' read -r -a _parts <<<"$COST_CORES"
for p in "${_parts[@]}"; do
    if [[ $p == *-* ]]; then
        a=${p%-*}; b=${p#*-}; NCHOSEN=$((NCHOSEN + b - a + 1))
    else
        NCHOSEN=$((NCHOSEN + 1))
    fi
done

# --- 负载门禁：load average 与可用内存（§9.3） ----------------------------
if [ "${COST_SKIP_GUARD:-0}" != "1" ]; then
    read -r _l1 _l5 _l15 _ _ < /proc/loadavg
    avail_gb=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    echo "[guard] load=${_l1}/${_l5}/${_l15} mem_available=${avail_gb}GiB" >&2
    bad=0
    awk -v l="$_l1" -v m="$MAX_LOAD" 'BEGIN {exit !(l > m)}' && {
        echo "[guard] load average ${_l1} > ${MAX_LOAD}，暂停不跑（§9.3）" >&2; bad=1; }
    [ "$avail_gb" -lt "$MIN_AVAIL_GB" ] && {
        echo "[guard] available ${avail_gb}GiB < ${MIN_AVAIL_GB}GiB，暂停不跑（§9.3）" >&2; bad=1; }
    [ "$bad" = 0 ] || exit 3
fi

# 把测量期间的系统负载写进容器可见的位置，由 harness 收进 manifest
{
    date -Is
    echo "loadavg $(cat /proc/loadavg)"
    awk '/^MemTotal:|^MemAvailable:/ {print}' /proc/meminfo
    echo "cpuset ${COST_CORES} docker_cpus=${COST_CPUS} mem=${COST_MEM}"
} > /tmp/c-cost-system-snapshot.txt

PRELUDE=""
if [ "$LITEPROF" = 1 ]; then
    PRELUDE=$(cat <<EOF
bash /workspace/scripts/build_liteprof_overlay.sh --dest $OVERLAY \\
    --patch /workspace/harness/python/vendor/liteprofiler/minimal.patch --force >/tmp/overlay.log 2>&1 \\
    || { cat /tmp/overlay.log; exit 1; }
export PYTHONPATH=$OVERLAY
export COST_OVERLAY=$OVERLAY
EOF
)
fi
PRELUDE="$PRELUDE
export COST_IMAGE='$IMAGE'
export COST_CORES='$COST_CORES'
export COST_NCPUS='$NCHOSEN'
export COST_HOST_ROOT='$HOST_ROOT'
export COST_WORKTREE_HEAD='$(git -C "$HOST_ROOT" rev-parse HEAD 2>/dev/null || echo "")'
export COST_WORKTREE_BRANCH='$(git -C "$HOST_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")'
export RAYON_NUM_THREADS='$NCHOSEN' OMP_NUM_THREADS='$NCHOSEN' MKL_NUM_THREADS='$NCHOSEN'
export NUMEXPR_NUM_THREADS='$NCHOSEN' TOKENIZERS_PARALLELISM=false
"

# 镜像 id 与 vllm 源码 commit 在宿主机侧采集好，写进同一个快照文件
# （容器里看不到 docker，也看不到 worktree 的 .git）。
IMAGE_ID="$(docker inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || echo "")"
VLLM_SRC="${VLLM_SRC:-/home/chiro/projects/vllm/HIST_PROJECT/vllm}"
VLLM_SRC_COMMIT="$(git -C "$VLLM_SRC" rev-parse HEAD 2>/dev/null || echo "")"
{
    echo "image=$IMAGE"
    echo "image_id=$IMAGE_ID"
    echo "worktree_head=$(git -C "$HOST_ROOT" rev-parse HEAD 2>/dev/null || echo "")"
    echo "worktree_branch=$(git -C "$HOST_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
    echo "vllm_src=$VLLM_SRC"
    echo "vllm_src_commit=$VLLM_SRC_COMMIT"
} >> /tmp/c-cost-system-snapshot.txt

DOCKER_LIMITS=(
    --cpus="$COST_CPUS"
    --cpuset-cpus="$COST_CORES"
    --memory="$COST_MEM"
    --memory-swap="$COST_MEM"
    --shm-size="$COST_SHM"
)

if [ "${1:-}" = "--shell" ]; then
    exec docker run --rm -it "${DOCKER_LIMITS[@]}" \
        -v "$MODELS:/models:ro" -v "$HOST_ROOT:/workspace" \
        --entrypoint bash "$IMAGE" -lc "$PRELUDE bash -l"
fi

if [ "${1:-}" = "-c" ]; then
    TMP_CODE="$(mktemp -t c_cost_code_XXXXXX.py)"
    printf '%s\n' "${2:?missing code}" > "$TMP_CODE"
    trap 'rm -f "$TMP_CODE"' EXIT
    exec docker run --rm "${DOCKER_LIMITS[@]}" \
        -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/harness/python \
        -v "$MODELS:/models:ro" -v "$HOST_ROOT:/workspace" \
        -v "/tmp/c-cost-system-snapshot.txt:/workspace/system-snapshot.txt:ro" \
        -v "$TMP_CODE:/workspace/_inline_code.py:ro" \
        --entrypoint bash "$IMAGE" -lc "$PRELUDE cd /workspace && bash /opt/va26-scripts/run_real_test_official.sh /workspace/_inline_code.py"
fi

SCRIPT="$1"; shift
exec docker run --rm "${DOCKER_LIMITS[@]}" \
    -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/harness/python \
    -v "$MODELS:/models:ro" -v "$HOST_ROOT:/workspace" \
    -v "/tmp/c-cost-system-snapshot.txt:/workspace/system-snapshot.txt:ro" \
    --entrypoint bash "$IMAGE" -lc "$PRELUDE cd /workspace && bash /opt/va26-scripts/run_real_test_official.sh /workspace/${SCRIPT} $*"
