#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_flamegraph.sh —— E3.1：Python 前端 tokenizer 路径的火焰图
#
# ## 为什么不是「在容器里跑 perf」
# 镜像里 **没有 perf、没有 flamegraph**（实测：`which perf` 为空）。
# 所以采样在**宿主机**做（perf 7.1.6 + inferno-collapse-perf + inferno-flamegraph
# 都在本机 ~/.cargo/bin），被采样的进程仍在容器里跑（纯 CPU 的 tokenizer 路径，
# 不需要 NPU）。挂载用的还是同一套镜像，代码路径与 E2 完全一致。
#
# ## 为什么必须开 PYTHONPERFSUPPORT=1
# CPython 3.12 的帧是**堆分配**的 `_PyInterpreterFrame`，不在 C 栈上，
# 所以 `perf` 默认只能看到 `_PyEval_EvalFrameDefault` 这一个 C 帧，
# 看不到 Python 函数名。CPython 3.12 提供了 perf 跳板（trampoline）：
# 设 `PYTHONPERFSUPPORT=1`（或 `sys.activate_stack_trampoline("perf")`）后，
# perf 会把它记为 `py::<module>::<function>` 的 JIT 帧。
# 这样一张图里能同时出现：
#   py::...            —— Python 层（Jinja 渲染、renderer 包装、池借用）
#   tokenizers::...    —— Rust `tokenizers` crate 内部（真正的 BPE/解码）
#   _PyEval_.../libc   —— 解释器与内存分配
#
# ## 采样器代价（硬要求：用采样器必须做代价对照）
# `--cost-control` 会跑「无采样 / 有采样」各 3 轮同负载，把 wall 时间差打出来。
#
# ## 资源纪律（plan/COORDINATION.md §9）
# 容器：`--cpus=6 --cpuset-cpus=4-5 --memory=8g --shm-size=2g`；
# perf 采样频率 199 Hz（默认 997 太密，会明显抬高开销）。
#
# 用法:
#   scripts/run_flamegraph.sh                    # 默认 199Hz / 800 次迭代
#   FREQ=99 scripts/run_flamegraph.sh
#   scripts/run_flamegraph.sh --cost-control
# ---------------------------------------------------------------------------
set -euo pipefail

IMAGE="${TOKENIZER_IMAGE:-local/vllm-ascend-stub-x86:v0.26.0rc1-a3-cpuonly-20260922}"
MODELS="${TOKENIZER_MODELS:-$HOME/models}"
HOST_ROOT="${TOKENIZER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COST_CORES="${COST_CORES:-4-5}"
FREQ="${FREQ:-199}"
DURATION="${DURATION:-90}"     # 采样窗口（秒）：起火前必须已经进循环
TASKS="${TASKS:-encode,render,detokenize}"
TAG="${TAG:-e3-1}"
TMP=/tmp/c-e3

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cost-control) MODE=cost; shift ;;
        --duration) DURATION=$2; shift 2 ;;
        --tasks) TASKS=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
    esac
done
MODE="${MODE:-record}"

# --- 负载门禁（§9.3）-----------------------------------------------
if [ "${COST_SKIP_GUARD:-0}" != "1" ]; then
    read -r l1 _ < <(cut -d' ' -f1-2 /proc/loadavg)
    avail=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
    echo "[guard] load=$l1 mem_available=${avail}GiB" >&2
    awk -v l="$l1" 'BEGIN{exit !(l>8)}' && { echo "[guard] load 过高，退出" >&2; exit 3; }
    [ "$avail" -ge 6 ] || { echo "[guard] 内存不足，退出" >&2; exit 3; }
fi

mkdir -p "$HOST_ROOT/figures" "$HOST_ROOT/data/cost" "$TMP"

run_container() {  # $1=额外环境变量串  $2=python 参数  $3=容器名
    docker run --rm -d --name "$3" \
        --cpus=6 --cpuset-cpus="$COST_CORES" --memory=8g --memory-swap=8g --shm-size=2g \
        -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/harness/python \
        -e PYTHONPERFSUPPORT=1 $1 \
        -v "$MODELS:/models:ro" -v "$HOST_ROOT:/workspace" \
        --entrypoint bash "$IMAGE" -lc \
        "cd /workspace && bash /opt/va26-scripts/run_real_test_official.sh \
         /workspace/harness/python/flame_target.py $2" >/dev/null
}

wait_ready() {  # 容器名 —— 等 flame_target 打印 READY（import 完成、进入循环）
    local name="$1" i
    for i in $(seq 1 240); do
        if docker logs "$name" 2>&1 | grep -q "^\[flame_target\] READY"; then
            return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            echo "容器提前退出：" >&2; docker logs "$name" 2>&1 | tail -5 >&2
            return 1
        fi
        sleep 1
    done
    echo "等待 READY 超时" >&2
    return 1
}

host_pid_of() {  # 容器名 → 容器内 python 的宿主机 pid
    # 两个坑都踩过：
    #   1. `pgrep -f flame_target.py` 会命中 **docker 客户端**的命令行（它把整条
    #      bash -lc 脚本作为参数），而那个进程马上退出；
    #   2. 只要求「cmdline 含 flame_target.py」还会命中**外层 bash 包装脚本**
    #      （它的 cmdline 里也有脚本路径），而 bash 只是在等子进程 ——
    #      采它就等于采一个 sleep 的进程（实测：13s 只拿到 61 个样本）。
    # 所以要求：cmdline 命中 + cgroup 属于该容器 + **/proc/pid/exe 是 python**。
    local name="$1" cid pid exe
    cid=$(docker inspect --format '{{.Id}}' "$name" 2>/dev/null || true)
    [ -n "$cid" ] || return 1
    for _ in $(seq 1 90); do
        for pid in $(ls /proc | grep -E '^[0-9]+$'); do
            [ -r "/proc/$pid/cmdline" ] || continue
            tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "flame_target.py" || continue
            grep -q "$cid" "/proc/$pid/cgroup" 2>/dev/null || continue
            # 注意：`/proc/<pid>/exe` 对 root 拥有的进程**读不到**（EACCES），
            # 而 `readlink -f` 会静默给空串 —— 这条也踩过。
            # `/proc/<pid>/comm` 是世界可读的，拿到的就是可执行文件名。
            exe=$(cat "/proc/$pid/comm" 2>/dev/null || true)
            case "$exe" in
                python3*) : ;;
                *) continue ;;
            esac
            echo "$pid"; return 0
        done
        sleep 1
    done
    return 1
}

wait_pid() {  # 容器名
    local name="$1" pid
    pid=$(docker inspect --format '{{.State.Pid}}' "$name" 2>/dev/null || echo 0)
    [ "${pid:-0}" != "0" ] || return 1
    echo "$pid"
}

if [ "$MODE" = cost ]; then
    echo "[e3.1] 采样器代价对照：每臂 3 轮，负载固定 duration=20s"
    for arm in noperf perf; do
        for rep in 1 2 3; do
            t0=$(date +%s.%N)
            if [ "$arm" = noperf ]; then
                docker run --rm --cpus=6 --cpuset-cpus="$COST_CORES" --memory=8g \
                    -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/harness/python \
                    -v "$MODELS:/models:ro" -v "$HOST_ROOT:/workspace" \
                    --entrypoint bash "$IMAGE" -lc \
                    "cd /workspace && bash /opt/va26-scripts/run_real_test_official.sh \
                     /workspace/harness/python/flame_target.py --duration 20" >/dev/null 2>&1
            else
                cname="c-e3-cost-$rep"
                docker rm -f "$cname" >/dev/null 2>&1 || true
                run_container "" "--duration 20" "$cname"
                pid=$(host_pid_of "$cname") || { echo "找不到进程"; exit 1; }
                wait_ready "$cname"
                sudo -n perf record -o "/tmp/c-e3-cost-$rep.data" -e cpu-clock \
                    -F "$FREQ" --call-graph "dwarf,16384" -p "$pid" -- sleep 20 \
                    >/dev/null 2>&1 || true
                docker wait "c-e3-cost-$rep" >/dev/null 2>&1 || true
                docker rm -f "$cname" >/dev/null 2>&1 || true
            fi
            t1=$(date +%s.%N)
            echo "COST arm=$arm rep=$rep wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')s"
        done
    done
    echo "[e3.1] 对照含义：perf 臂比 noperf 臂多出的 wall 时间 = 探针自身代价"
    exit 0
fi

# --- 正式记录 ------------------------------------------------------
NAME=c-$TAG
echo "[e3.1] 启动容器（PYTHONPERFSUPPORT=1, freq=${FREQ}Hz, duration=${DURATION}s）"
docker rm -f "$NAME" >/dev/null 2>&1 || true
run_container "" "--duration $((DURATION + 30)) --tasks $TASKS" "$NAME"

echo "[e3.1] 等容器内 python 进程出现…"
pid=$(host_pid_of "$NAME") || {
    echo "失败：找不到容器内 python 进程" >&2
    docker logs "$NAME" 2>&1 | tail
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    exit 1
}
echo "[e3.1] 宿主机 pid=$pid，等 READY（import 完成、进入 tokenizer 循环）"
wait_ready "$NAME" || { docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }
echo "[e3.1] READY：开始 perf record，窗口 ${DURATION}s"
# 用 `-- sleep N` 做计时器（`-- sleep 0` 会立刻返回，是个坑）。
# perf 需要 sudo：容器进程在宿主机上是 root（uid 0），paranoid=2 下
# 非 root 的 perf 无法给别人的进程开事件。
# 注意：本机 perf 7.1.6 **同时给 -g 和 --call-graph 会报错并打印帮助**
# （实测：「-g --call-graph dwarf,...」被拒）。只用 --call-graph 即可，
# 它本身就启用调用图。
sudo -n perf record -o "$TMP/$TAG.perf.data" -e cpu-clock -F "$FREQ" \
    --call-graph "dwarf,16384" -p "$pid" -- sleep "$DURATION" 2>&1 | tail -4
docker logs "$NAME" 2>&1 | tail -4
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[e3.1] 折叠 + 出图"
export PATH="$HOME/.cargo/bin:$PATH"
sudo -n perf script -i "$TMP/$TAG.perf.data" > "$TMP/$TAG.perf.script" 2>/dev/null
sudo -n chown "$(id -u):$(id -g)" "$TMP/$TAG.perf.script" "$TMP/$TAG.perf.data" 2>/dev/null || true
inferno-collapse-perf --all < "$TMP/$TAG.perf.script" > "$TMP/$TAG.folded"
inferno-flamegraph \
    --title "E3.1 vLLM 0.26.0 Python frontend tokenizer path [$TASKS]" \
    --subtitle "perf -F ${FREQ} --call-graph dwarf,16384 -e cpu-clock（宿主机 perf 7.1.6，PYTHONPERFSUPPORT=1）；x86_64 容器 2 核；Qwen3-0.6B；ISL=1024 / OSL=256" \
    --width 1800 --height 1000 \
    < "$TMP/$TAG.folded" > "$HOST_ROOT/figures/$TAG-python-frontend.svg"

echo "[e3.1] → figures/$TAG-python-frontend.svg"
wc -c "$HOST_ROOT/figures/$TAG-python-frontend.svg" "$TMP/$TAG.folded"
echo "[e3.1] 样本数 = $(wc -l < "$TMP/$TAG.folded") 个折叠行"
