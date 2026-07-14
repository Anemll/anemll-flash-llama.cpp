#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)

usage() {
    cat >&2 <<EOF
usage: $0 [PACKAGE_DIR]

Runs the DeepSeek V4 Flash-MoE layer-major benchmark matrix:
  top-k:        ${TOPKS:-6 4 2}
  prompt files: ${PROMPT_LABELS:-128 1k 4k}

Defaults target the native HF-exported full DSv4 package:
  MODEL_PATH    ${MODEL_PATH:-$HOME/Models/DeepSeek-V4-SSD/dense/model-dense.gguf}
  SIDECAR_PATH  ${SIDECAR_PATH:-$HOME/Models/DeepSeek-V4-SSD/sidecar}

Override with env vars, for example:
  TOPKS="6 4 2" PROMPT_LABELS="128 1k 4k" $0
  RESULTS_DIR=/tmp/dsv4-sweep STOP_ON_FAILURE=0 $0

If PACKAGE_DIR is supplied, MODEL_PATH and SIDECAR_PATH default to:
  PACKAGE_DIR/dense/model-dense.gguf
  PACKAGE_DIR/sidecar
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
fi

package_dir=${1:-}
if [[ -n "$package_dir" ]]; then
    MODEL_PATH="${MODEL_PATH:-$package_dir/dense/model-dense.gguf}"
    SIDECAR_PATH="${SIDECAR_PATH:-$package_dir/sidecar}"
else
    MODEL_PATH="${MODEL_PATH:-$HOME/Models/DeepSeek-V4-SSD/dense/model-dense.gguf}"
    SIDECAR_PATH="${SIDECAR_PATH:-$HOME/Models/DeepSeek-V4-SSD/sidecar}"
fi

PROMPT_DIR="${PROMPT_DIR:-$script_dir/prompts/coding}"
PROMPT_LABELS="${PROMPT_LABELS:-128 1k 4k}"
TOPKS="${TOPKS:-6 4 2}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/flashmoe-results/dsv4-topk-prompt-sweep/$(date +%Y%m%d-%H%M%S)}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

MOE_CACHE_IO_SPLIT="${MOE_CACHE_IO_SPLIT:-8}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-1}"
CTX="${CTX:-90000}"
SEED="${SEED:-123}"
TEMP="${TEMP:-0}"
MOE_SLOT_BANK="${MOE_SLOT_BANK:-32}"
MOE_PREFILL_BATCH="${MOE_PREFILL_BATCH:-8192}"
MOE_PREFILL_MICRO_BATCH="${MOE_PREFILL_MICRO_BATCH:-auto}"
MOE_PREFILL_IO_SPLIT="${MOE_PREFILL_IO_SPLIT:-8}"
MOE_PREFILL_BANKS="${MOE_PREFILL_BANKS:-4}"
N_PREDICT="${N_PREDICT:-500}"
SAMPLE_TEMP="${SAMPLE_TEMP:-1.0}"
TOP_P="${TOP_P:-1.0}"

[[ -f "$MODEL_PATH" ]] || {
    echo "error: missing model: $MODEL_PATH" >&2
    exit 1
}
[[ -e "$SIDECAR_PATH" ]] || {
    echo "error: missing sidecar: $SIDECAR_PATH" >&2
    exit 1
}

mkdir -p "$RESULTS_DIR"
summary_csv="$RESULTS_DIR/summary.csv"
printf "status,topk,prompt,prefill_tokens,decode_tokens,prompt_tps,generation_tps,prefill_src,prefill_uniq,prefill_dedup_pct,decode_hit_pct,decode_bytes_gib,log\n" > "$summary_csv"

prompt_for_label() {
    local label=$1
    local path="$PROMPT_DIR/coding_${label}.txt"
    [[ -f "$path" ]] || {
        echo "error: missing prompt for label '$label': $path" >&2
        exit 1
    }
    printf "%s\n" "$path"
}

extract_last() {
    local pattern=$1
    local log_path=$2
    sed -n "$pattern" "$log_path" | tail -1
}

extract_field() {
    local line=$1
    local pattern=$2
    printf "%s\n" "$line" | sed -n "$pattern"
}

append_summary() {
    local status=$1
    local topk=$2
    local label=$3
    local log_path=$4

    local prompt_tps generation_tps prefill_tokens decode_tokens
    local prefill_line decode_line dedup_line prefill_src prefill_uniq prefill_dedup_pct decode_hit_pct decode_bytes_gib

    prompt_tps=$(extract_last 's/.*Prompt: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    generation_tps=$(extract_last 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    prefill_tokens=$(extract_last 's/.*tokens - prefill: \([-0-9][0-9-]*\), decode: [-0-9][0-9-]*.*/\1/p' "$log_path")
    decode_tokens=$(extract_last 's/.*tokens - prefill: [-0-9][0-9-]*, decode: \([-0-9][0-9-]*\).*/\1/p' "$log_path")

    prefill_line=$(grep 'Flash-MoE routed src=prefill-layer-major' "$log_path" | tail -1 || true)
    decode_line=$(grep 'Flash-MoE routed src=pread-slot-bank' "$log_path" | tail -1 || true)
    dedup_line=$(grep 'Flash-MoE prefill dedup' "$log_path" | tail -1 || true)

    prefill_src=$(extract_field "$prefill_line" 's/.*src=\([^ ]*\).*/\1/p')
    prefill_uniq=$(extract_field "$prefill_line" 's/.*uniq=\([0-9][0-9]*\).*/\1/p')
    prefill_dedup_pct=$(extract_field "$dedup_line" 's/.*saved=[0-9][0-9]* (\([0-9.][0-9.]*\)%).*/\1/p')
    decode_hit_pct=$(extract_field "$decode_line" 's/.*hit=\([0-9.][0-9.]*\)%.*/\1/p')
    decode_bytes_gib=$(extract_field "$decode_line" 's/.*bytes=\([0-9.][0-9.]*\) GiB.*/\1/p')

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$status" "$topk" "$label" "${prefill_tokens:-}" "${decode_tokens:-}" \
        "${prompt_tps:-}" "${generation_tps:-}" "${prefill_src:-}" "${prefill_uniq:-}" \
        "${prefill_dedup_pct:-}" "${decode_hit_pct:-}" "${decode_bytes_gib:-}" "$log_path" \
        >> "$summary_csv"
}

print_summary() {
    python3 - "$summary_csv" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
if not rows:
    print("summary: no completed rows")
    sys.exit(0)

columns = [
    ("status", "status"),
    ("topk", "topk"),
    ("prompt", "prompt"),
    ("prefill_tps", "prompt_tps"),
    ("decode_tps", "generation_tps"),
    ("prefill_tok", "prefill_tokens"),
    ("decode_tok", "decode_tokens"),
    ("prefill_uniq", "prefill_uniq"),
    ("dedup_%", "prefill_dedup_pct"),
    ("decode_hit_%", "decode_hit_pct"),
    ("decode_GiB", "decode_bytes_gib"),
]

def clean(value: str) -> str:
    return value if value not in (None, "") else "-"

widths = []
for title, key in columns:
    widths.append(max(len(title), *(len(clean(row.get(key, ""))) for row in rows)))

print()
print("Sweep summary")
print(" ".join(title.ljust(width) for (title, _), width in zip(columns, widths)))
print(" ".join("-" * width for width in widths))
for row in rows:
    print(" ".join(clean(row.get(key, "")).ljust(width) for (_, key), width in zip(columns, widths)))

ok_rows = [row for row in rows if row.get("status") == "ok"]
if ok_rows:
    def as_float(row: dict[str, str], key: str) -> float:
        try:
            return float(row.get(key) or "nan")
        except ValueError:
            return float("nan")

    best_prefill = max(ok_rows, key=lambda row: as_float(row, "prompt_tps"))
    best_decode = max(ok_rows, key=lambda row: as_float(row, "generation_tps"))
    print()
    print(
        "Best prefill: "
        f"topk={best_prefill.get('topk')} prompt={best_prefill.get('prompt')} "
        f"{best_prefill.get('prompt_tps') or '-'} t/s"
    )
    print(
        "Best decode:  "
        f"topk={best_decode.get('topk')} prompt={best_decode.get('prompt')} "
        f"{best_decode.get('generation_tps') or '-'} t/s"
    )

failed = [row for row in rows if row.get("status") != "ok"]
if failed:
    print()
    print("Failures:")
    for row in failed:
        print(f"  {row.get('status')} topk={row.get('topk')} prompt={row.get('prompt')} log={row.get('log')}")
PY
}

run_case() {
    local topk=$1
    local label=$2
    local prompt_path log_path status

    prompt_path=$(prompt_for_label "$label")
    log_path="$RESULTS_DIR/dsv4_topk${topk}_coding_${label}.log"

    echo "== topk=${topk} prompt=coding_${label} log=$log_path"

    set +e
    DISPLAY_PROMPT=0 \
    SIMPLE_IO=1 \
    RAW_COMPLETION=1 \
    MOE_TOPK="$topk" \
    MOE_CACHE_IO_SPLIT="$MOE_CACHE_IO_SPLIT" \
    BATCH="$BATCH" \
    UBATCH="$UBATCH" \
    CTX="$CTX" \
    SEED="$SEED" \
    TEMP="$TEMP" \
    MOE_SLOT_BANK="$MOE_SLOT_BANK" \
    LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS:-1}" \
    bash "$script_dir/run_dsv4_flash_profile.sh" --temp "$SAMPLE_TEMP" --top-p "$TOP_P" \
        -m "$MODEL_PATH" \
        --moe-sidecar "$SIDECAR_PATH" \
        -f "$prompt_path" \
        --moe-predict-top1-prev \
        --moe-topk "$topk" \
        --moe-prefill-layer-major \
        --moe-prefill-batch "$MOE_PREFILL_BATCH" \
        --moe-prefill-micro-batch "$MOE_PREFILL_MICRO_BATCH" \
        --moe-prefill-io-split "$MOE_PREFILL_IO_SPLIT" \
        --moe-prefill-banks "$MOE_PREFILL_BANKS" \
        -n "$N_PREDICT" -st \
        > "$log_path" 2>&1
    status=$?
    set -e

    if (( status == 0 )); then
        append_summary ok "$topk" "$label" "$log_path"
    else
        append_summary "fail:$status" "$topk" "$label" "$log_path"
        echo "failed: topk=${topk} prompt=coding_${label} status=${status} log=$log_path" >&2
        if [[ "$STOP_ON_FAILURE" != "0" ]]; then
            exit "$status"
        fi
    fi
}

echo "results: $RESULTS_DIR"
echo "model:   $MODEL_PATH"
echo "sidecar: $SIDECAR_PATH"
echo "topks:   $TOPKS"
echo "prompts: $PROMPT_LABELS"

for label in $PROMPT_LABELS; do
    prompt_for_label "$label" >/dev/null
done

for topk in $TOPKS; do
    for label in $PROMPT_LABELS; do
        run_case "$topk" "$label"
    done
done

print_summary
echo "summary: $summary_csv"
