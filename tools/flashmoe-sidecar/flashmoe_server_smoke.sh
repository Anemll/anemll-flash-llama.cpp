#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL_FILE_OR_PACKAGE_DIR [run_flashmoe_server args...]" >&2
    exit 1
fi

target=$1
shift

bind_host=${HOST:-127.0.0.1}

# The smoke client must connect to a routable address, not a wildcard bind.
client_host=${CLIENT_HOST:-$bind_host}
case "$client_host" in
    0.0.0.0)
        client_host=127.0.0.1
        ;;
    ::|\[::\])
        client_host=::1
        ;;
esac
url_host=$client_host
if [[ "$url_host" == *:* && "$url_host" != \[*\] ]]; then
    url_host="[$url_host]"
fi
port=${PORT:-8080}
url="http://${url_host}:${port}/completion"
ready_urls=("http://${bind_host}:${port}")
if [[ "$bind_host" == *:* && "$bind_host" != \[*\] ]]; then
    ready_urls+=("http://[$bind_host]:${port}")
fi
prompt_label=${PROMPT_LABEL:-4k}
n_predict=${N_PREDICT:-64}
id_slot=${ID_SLOT:-0}
ready_timeout=${READY_TIMEOUT_SEC:-600}
request_timeout=${REQUEST_TIMEOUT_SEC:-600}
python_bin=${PYTHON_BIN:-python3}
memory_check_sec=${MEMORY_CHECK_SEC:-5}
free_mem_abort_percent=${FREE_MEM_ABORT_PERCENT:-20}
startup_only=${STARTUP_ONLY:-0}
ready_stabilize_sec=${READY_STABILIZE_SEC:-3}

timestamp=$(date +"%Y%m%d-%H%M%S")
target_name=$(basename "$target")
target_name=${target_name//[^[:alnum:]._-]/_}
default_log_root="$script_dir/logs"
log_root=${LOG_ROOT:-$default_log_root}
mkdir -p "$log_root"
log_dir=${LOG_DIR:-"$log_root/${timestamp}-${target_name}-p${port}"}
server_log=${SERVER_LOG:-"$log_dir/server.log"}
client_log=${CLIENT_LOG:-"$log_dir/client.log"}
progress_log=${PROGRESS_LOG:-"$log_dir/progress.log"}
keep_logs=${KEEP_LOGS:-1}
abort_flag="$log_dir/aborted-by-memory-guard"

mkdir -p "$log_dir"

server_pid=
client_pid=
monitor_pid=
log_note() {
    local msg=$1
    local now
    now=$(date +"%Y-%m-%d %H:%M:%S")
    printf "[%s] %s\n" "$now" "$msg" | tee -a "$progress_log"
}

system_free_percent() {
    local pct
    pct=$(memory_pressure -Q 2>/dev/null | awk '/System-wide memory free percentage:/ { gsub("%", "", $5); print $5 }' | tail -n 1)
    if [[ -n "$pct" ]]; then
        printf '%s\n' "$pct"
        return 0
    fi

    python3 - <<'PY'
import os
import re
import subprocess
import sys

memsize = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
vm_stat = subprocess.check_output(["vm_stat"], text=True)
page_size = 16384
match = re.search(r"page size of (\d+) bytes", vm_stat)
if match:
    page_size = int(match.group(1))
pages = {}
for line in vm_stat.splitlines():
    m = re.match(r"([^:]+):\s+(\d+)\.", line)
    if m:
        pages[m.group(1)] = int(m.group(2))
freeish = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
free_pct = int((freeish * page_size * 100) / memsize)
print(free_pct)
PY
}

monitor_memory() {
    while true; do
        local free_pct
        free_pct=$(system_free_percent || true)
        if [[ -n "$free_pct" ]]; then
    log_note "memory: system_free_pct=${free_pct}% threshold=${free_mem_abort_percent}%"
            if (( free_pct <= free_mem_abort_percent )); then
                log_note "memory guard: threshold reached, interrupting client/server"
                : > "$abort_flag"
                if [[ -n "${client_pid:-}" ]] && kill -0 "$client_pid" 2>/dev/null; then
                    kill -INT "$client_pid" 2>/dev/null || true
                fi
                if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
                    kill -INT "$server_pid" 2>/dev/null || true
                fi
                return
            fi
        else
            log_note "memory: unable to read system free percentage"
        fi

        if [[ -n "${server_pid:-}" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
            return
        fi
        sleep "$memory_check_sec"
    done
}

cleanup() {
    if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "${client_pid:-}" ]] && kill -0 "$client_pid" 2>/dev/null; then
        kill -INT "$client_pid" 2>/dev/null || true
        wait "$client_pid" 2>/dev/null || true
    fi
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ "$keep_logs" == "0" ]]; then
        rm -rf "$log_dir"
    fi
}
trap cleanup EXIT INT TERM

stop_pid() {
    local pid=$1
    local name=$2
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill -INT "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    log_note "${name}: escalating to TERM"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    log_note "${name}: escalating to KILL"
    kill -KILL "$pid" 2>/dev/null || true
}

log_note "server_log=$server_log"
log_note "client_log=$client_log"
log_note "target=$target"
log_note "bind_host=$bind_host client_host=$client_host url=$url"
log_note "request_timeout_sec=$request_timeout ready_timeout_sec=$ready_timeout free_mem_abort_percent=$free_mem_abort_percent startup_only=$startup_only"

HOST=$bind_host PORT=$port \
    bash "$script_dir/run_flashmoe_server.sh" "$target" "$@" >"$server_log" 2>&1 &
server_pid=$!
monitor_memory &
monitor_pid=$!

deadline=$(( $(date +%s) + ready_timeout ))
while true; do
    for ready_url in "${ready_urls[@]}"; do
        if rg -Fq "server is listening on ${ready_url}" "$server_log"; then
            log_note "server ready at $url"
            break 2
        fi
    done
    if [[ -f "$abort_flag" ]]; then
        echo "aborted by memory guard before server became ready" >&2
        cat "$progress_log" >&2
        exit 1
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited before becoming ready" >&2
        cat "$server_log" >&2
        exit 1
    fi
    if (( $(date +%s) >= deadline )); then
        echo "timed out waiting for server readiness" >&2
        cat "$server_log" >&2
        exit 1
    fi
    sleep 1
done

if [[ "$startup_only" == "1" ]]; then
    sleep "$ready_stabilize_sec"
    stop_pid "$server_pid" "server"
    wait "$server_pid" 2>/dev/null || true
    server_pid=

    if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    monitor_pid=

    if [[ -f "$abort_flag" ]]; then
        echo "aborted by memory guard" >&2
        cat "$progress_log" >&2
        exit 1
    fi

    echo
    echo "== startup-only summary =="
    cat "$progress_log"
    echo
    echo "== startup memory lines =="
    rg -n "llama_kv_cache: size|compute buffer size|slot-bank reserve estimate|shared bank reserve|server is listening" "$server_log" || true
    exit 0
fi

client_args=(
    "$script_dir/flashmoe_server_turn_test.py"
    --url "$url"
    --n-predict "$n_predict"
    --id-slot "$id_slot"
    --request-timeout "$request_timeout"
)

if [[ -n "${PROMPT_FILE:-}" ]]; then
    client_args+=(--prompt-file "$PROMPT_FILE")
else
    client_args+=(--prompt-label "$prompt_label")
fi

if [[ -n "${FOLLOWUP_TEXT:-}" ]]; then
    client_args+=(--followup "$FOLLOWUP_TEXT")
fi

"$python_bin" "${client_args[@]}" >"$client_log" 2>&1 &
client_pid=$!
client_status=0
if ! wait "$client_pid"; then
    client_status=$?
fi
client_pid=

stop_pid "$server_pid" "server"
wait "$server_pid" 2>/dev/null || true
server_pid=

if [[ -n "${monitor_pid:-}" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
fi
monitor_pid=

if [[ -f "$abort_flag" ]]; then
    echo "aborted by memory guard" >&2
    cat "$progress_log" >&2
    exit 1
fi

if (( client_status != 0 )); then
    echo "client request failed" >&2
    cat "$client_log" >&2
    exit "$client_status"
fi

echo
echo "== client timings =="
cat "$client_log"

echo
echo "== server prompt timings =="
rg -n "prompt eval time|eval time =|total time =" "$server_log" || true

echo
echo "== layer-major summary =="
rg -n "src=prefill-layer-major|prefill dedup|prefill pacing|Read bandwidth|Max refs/expert|Max tokens/expert" "$server_log" || true

echo
echo "== decode summary =="
rg -n "src=pread-slot-bank|slot-bank cached expert hit rate|predict-top1-prev overlap" "$server_log" || true
