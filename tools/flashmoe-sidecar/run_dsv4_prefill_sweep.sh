#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)

RESULTS_DIR="${RESULTS_DIR:-$repo_root/flashmoe-results/dsv4-prefill-sweep/$(date +%Y%m%d-%H%M%S)}"
PROMPT_LABELS="${PROMPT_LABELS:-1k 4k 8k 64k}"
BASELINE_LABELS="${BASELINE_LABELS-1k}"
PREFILL_BATCHES="${PREFILL_BATCHES:-1024 4096 8192 16384}"
RUN_BASELINE="${RUN_BASELINE:-1}"

MODEL_PATH="${MODEL_PATH:-/Volumes/SN8100/DS/DeepSeek-V4-Flash-FP4-FP8-SSD/dense/model-dense.gguf}"
SIDECAR_PATH="${SIDECAR_PATH:-/Volumes/SN8100/DS/DeepSeek-V4-Flash-FP4-FP8-SSD/sidecar}"

MOE_SLOT_BANK="${MOE_SLOT_BANK:-96}"
MOE_TOPK="${MOE_TOPK:-6}"
MOE_CACHE_IO_SPLIT="${MOE_CACHE_IO_SPLIT:-8}"
MOE_PREFILL_IO_SPLIT="${MOE_PREFILL_IO_SPLIT:-$MOE_CACHE_IO_SPLIT}"
MOE_PREFILL_BANKS="${MOE_PREFILL_BANKS:-4}"
MOE_PREFILL_MICRO_BATCH="${MOE_PREFILL_MICRO_BATCH:-auto}"

BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-1}"
N_PREDICT="${N_PREDICT:-1}"
SEED="${SEED:-123}"
TEMP="${TEMP:-0}"

mkdir -p "$RESULTS_DIR"
summary_csv="$RESULTS_DIR/summary.csv"

ctx_for_label() {
    case "$1" in
        1k)  printf "4096\n" ;;
        4k)  printf "8192\n" ;;
        8k)  printf "12000\n" ;;
        64k) printf "80000\n" ;;
        *)   printf "%s\n" "${CTX:-12000}" ;;
    esac
}

prompt_for_label() {
    local label=$1
    local path="$script_dir/prompts/coding/coding_${label}.txt"
    [[ -f "$path" ]] || {
        printf "error: missing prompt for label '%s': %s\n" "$label" "$path" >&2
        exit 1
    }
    printf "%s\n" "$path"
}

extract_last() {
    local pattern=$1
    local log_path=$2
    sed -n "$pattern" "$log_path" | tail -1
}

extract_summary() {
    local mode=$1
    local label=$2
    local ctx=$3
    local prefill_batch=$4
    local log_path=$5

    local prompt_tps generation_tps prefill_tokens decode_tokens routed_line dedup_line src calls refs uniq bytes_gib dedup_pct
    prompt_tps=$(extract_last 's/.*Prompt: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    generation_tps=$(extract_last 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    prefill_tokens=$(extract_last 's/.*tokens - prefill: \([0-9][0-9]*\), decode: [0-9][0-9]*.*/\1/p' "$log_path")
    decode_tokens=$(extract_last 's/.*tokens - prefill: [0-9][0-9]*, decode: \([0-9][0-9]*\).*/\1/p' "$log_path")

    routed_line=$(grep 'Flash-MoE routed src=' "$log_path" | tail -1 || true)
    dedup_line=$(grep 'Flash-MoE prefill dedup' "$log_path" | tail -1 || true)

    src=$(printf "%s\n" "$routed_line" | sed -n 's/.*src=\([^ ]*\).*/\1/p')
    calls=$(printf "%s\n" "$routed_line" | sed -n 's/.*calls=\([0-9][0-9]*\).*/\1/p')
    refs=$(printf "%s\n" "$routed_line" | sed -n 's/.*refs=\([0-9][0-9]*\).*/\1/p')
    uniq=$(printf "%s\n" "$routed_line" | sed -n 's/.*uniq=\([0-9][0-9]*\).*/\1/p')
    bytes_gib=$(printf "%s\n" "$routed_line" | sed -n 's/.*bytes=\([0-9.][0-9.]*\) GiB.*/\1/p')
    dedup_pct=$(printf "%s\n" "$dedup_line" | sed -n 's/.*saved=[0-9][0-9]* (\([0-9.][0-9.]*\)%).*/\1/p')

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$mode" "$label" "${prefill_tokens:-}" "$decode_tokens" "$ctx" "$UBATCH" "$prefill_batch" \
        "${prompt_tps:-}" "${generation_tps:-}" "${src:-}" "${calls:-}" "${refs:-}" "${uniq:-}" \
        "${dedup_pct:-}" "${bytes_gib:-}" "$log_path" >> "$summary_csv"
}

run_case() {
    local mode=$1
    local label=$2
    local prefill_batch=$3
    local prompt_path ctx log_path
    prompt_path=$(prompt_for_label "$label")
    ctx=$(ctx_for_label "$label")

    if [[ "$mode" == "baseline" ]]; then
        log_path="$RESULTS_DIR/baseline_${label}_ub${UBATCH}.log"
        printf "== baseline label=%s ubatch=%s log=%s\n" "$label" "$UBATCH" "$log_path"
        DISPLAY_PROMPT=0 \
        SIMPLE_IO=1 \
        RAW_COMPLETION=1 \
        MOE_TOPK="$MOE_TOPK" \
        MOE_CACHE_IO_SPLIT="$MOE_CACHE_IO_SPLIT" \
        BATCH="$BATCH" \
        UBATCH="$UBATCH" \
        CTX="$ctx" \
        SEED="$SEED" \
        TEMP="$TEMP" \
        MOE_SLOT_BANK="$MOE_SLOT_BANK" \
        LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS:-0}" \
        bash "$script_dir/run_dsv4_flash_profile.sh" \
            -m "$MODEL_PATH" \
            --moe-sidecar "$SIDECAR_PATH" \
            -f "$prompt_path" \
            --moe-predict-top1-prev \
            --moe-topk "$MOE_TOPK" \
            --no-moe-prefill-layer-major \
            -n "$N_PREDICT" -st \
            > "$log_path" 2>&1
        extract_summary "$mode" "$label" "$ctx" "" "$log_path"
        return 0
    fi

    log_path="$RESULTS_DIR/layer_${label}_pb${prefill_batch}_ub${UBATCH}.log"
    printf "== layer label=%s prefill_batch=%s ubatch=%s log=%s\n" "$label" "$prefill_batch" "$UBATCH" "$log_path"
    DISPLAY_PROMPT=0 \
    SIMPLE_IO=1 \
    RAW_COMPLETION=1 \
    MOE_TOPK="$MOE_TOPK" \
    MOE_CACHE_IO_SPLIT="$MOE_CACHE_IO_SPLIT" \
    BATCH="$BATCH" \
    UBATCH="$UBATCH" \
    CTX="$ctx" \
    SEED="$SEED" \
    TEMP="$TEMP" \
    MOE_SLOT_BANK="$MOE_SLOT_BANK" \
    LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS:-0}" \
    bash "$script_dir/run_dsv4_flash_profile.sh" \
        -m "$MODEL_PATH" \
        --moe-sidecar "$SIDECAR_PATH" \
        -f "$prompt_path" \
        --moe-predict-top1-prev \
        --moe-topk "$MOE_TOPK" \
        --moe-prefill-layer-major \
        --moe-prefill-batch "$prefill_batch" \
        --moe-prefill-micro-batch "$MOE_PREFILL_MICRO_BATCH" \
        --moe-prefill-io-split "$MOE_PREFILL_IO_SPLIT" \
        --moe-prefill-banks "$MOE_PREFILL_BANKS" \
        -n "$N_PREDICT" -st \
        > "$log_path" 2>&1
    extract_summary "$mode" "$label" "$ctx" "$prefill_batch" "$log_path"
}

printf "mode,label,prefill_tokens,decode_tokens,ctx,ubatch,prefill_batch,prompt_tps,generation_tps,src,calls,refs,uniq,dedup_saved_pct,bytes_gib,log\n" > "$summary_csv"

if [[ "$RUN_BASELINE" != "0" ]]; then
    for label in $BASELINE_LABELS; do
        run_case baseline "$label" ""
    done
fi

for label in $PROMPT_LABELS; do
    for prefill_batch in $PREFILL_BATCHES; do
        run_case layer "$label" "$prefill_batch"
    done
done

printf "summary: %s\n" "$summary_csv"
