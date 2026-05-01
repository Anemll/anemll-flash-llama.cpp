#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
default_package_dir="/Volumes/SN8100/DS/DeepSeek-V4-Flash-FP4-FP8-SSD"
default_prompt_dir="$repo_root/tools/flashmoe-sidecar/prompts/coding"

note() {
    printf "note: %s\n" "$1" >&2
}

warn() {
    printf "warning: %s\n" "$1" >&2
}

die() {
    printf "error: %s\n" "$1" >&2
    exit 1
}

env_truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON)
            return 0
            ;;
    esac
    return 1
}

usage() {
    cat >&2 <<EOF
usage: $0 [PACKAGE_DIR] [llama-cli args...]

DeepSeek V4 Flash SSD Flash-MoE profiling wrapper.

Defaults:
  PACKAGE_DIR     ${default_package_dir}
  MODEL_PATH      PACKAGE_DIR/dense/model-dense.gguf
  SIDECAR_PATH    PACKAGE_DIR/sidecar
  MOE_SLOT_BANK   96
  MOE_TOPK        sidecar manifest expert_used_count, fallback 6
  CTX             7000
  BATCH           2048
  UBATCH          24
  N_PREDICT       256

Override dense and sidecar either with env vars:
  MODEL_PATH=/path/model-dense.gguf SIDECAR_PATH=/path/sidecar $0 ...

or with llama-cli args:
  $0 -m /path/model-dense.gguf --moe-sidecar /path/sidecar ...
EOF
}

resolve_input_path() {
    local raw_path=$1

    if [[ "$raw_path" = /* ]]; then
        printf "%s\n" "$raw_path"
        return 0
    fi

    if [[ -e "$raw_path" ]]; then
        python3 - "$raw_path" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve())
PY
        return 0
    fi

    if [[ -e "$repo_root/$raw_path" ]]; then
        python3 - "$repo_root" "$raw_path" <<'PY'
from pathlib import Path
import sys

print((Path(sys.argv[1]) / sys.argv[2]).resolve())
PY
        return 0
    fi

    printf "%s\n" "$raw_path"
}

resolve_prompt_label_path() {
    local label=$1
    local candidate="$default_prompt_dir/coding_${label}.txt"

    if [[ -f "$candidate" ]]; then
        printf "%s\n" "$candidate"
        return 0
    fi

    die "unknown BENCHMARK_PROMPT_LABEL '$label' (expected a tools/flashmoe-sidecar/prompts/coding/coding_<label>.txt file)"
}

is_builtin_coding_prompt_path() {
    local candidate=$1
    [[ "$candidate" == "$default_prompt_dir"/coding_*.txt ]]
}

looks_like_package_dir() {
    local candidate=$1
    [[ -d "$candidate/dense" || -f "$candidate/dense/model-dense.gguf" || -d "$candidate/sidecar" ]]
}

parse_sidecar_manifest_defaults() {
    local manifest=$1
    python3 - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

model = manifest.get("model") or {}
entries = manifest.get("entries") or []
families = sorted({entry.get("tensor_family", "") for entry in entries if entry.get("tensor_family")})
bytes_per_slot_all_layers = sum(int(entry.get("bytes_per_expert") or 0) for entry in entries)

print(model.get("expert_used_count", 6))
print(model.get("expert_count", 256))
print(model.get("arch", "unknown"))
print(len(entries))
print(bytes_per_slot_all_layers)
print(",".join(families))
PY
}

human_gib() {
    python3 - "$1" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024**3:.2f}")
PY
}

package_dir="${DSV4_FLASH_PACKAGE_DIR:-$default_package_dir}"
if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            ;;
        *)
            package_dir=$1
            shift
            ;;
    esac
fi

llama_bin=${LLAMA_BIN:-"$repo_root/build/bin/llama-cli"}
llama_bin_name=$(basename "$llama_bin")
model_path=${MODEL_PATH:-"$package_dir/dense/model-dense.gguf"}
sidecar_path=${SIDECAR_PATH:-"$package_dir/sidecar"}
sidecar_manifest_path=${SIDECAR_MANIFEST_PATH:-}

if [[ -z "$sidecar_manifest_path" && -d "$sidecar_path" ]]; then
    sidecar_manifest_path="$sidecar_path/manifest.json"
elif [[ -z "$sidecar_manifest_path" && -f "$sidecar_path" ]]; then
    sidecar_manifest_path="$sidecar_path"
fi

[[ -x "$llama_bin" ]] || die "missing executable llama-cli: $llama_bin"

declare -a forwarded_args
forwarded_args=()
had_forwarded_args=0
saw_model_arg=0
saw_sidecar_arg=0
saw_moe_mode_arg=0
saw_slot_bank_arg=0
saw_topk_arg=0
saw_cache_io_split_arg=0
saw_ngl_arg=0
saw_ctx_arg=0
saw_batch_arg=0
saw_ubatch_arg=0
saw_perf_arg=0
saw_prompt_source_arg=0
saw_n_predict_arg=0
saw_st_arg=0
saw_fit_arg=0
saw_temp_arg=0
saw_raw_completion_arg=0
saw_conversation_mode_arg=0
saw_prefill_layer_major_arg=0
selected_builtin_coding_prompt=0
prefill_batch_arg_index=-1
prefill_batch_eq_arg_index=-1
requested_prefill_batch_arg=""

benchmark_prompt_label=${BENCHMARK_PROMPT_LABEL:-${PROMPT_LABEL:-}}
if [[ -n "${PROMPT_LABEL:-}" && -z "${BENCHMARK_PROMPT_LABEL:-}" ]]; then
    warn "PROMPT_LABEL is accepted as a legacy alias; prefer BENCHMARK_PROMPT_LABEL"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            model_path=$(resolve_input_path "$2")
            saw_model_arg=1
            shift 2
            ;;
        --moe-sidecar)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            sidecar_path=$(resolve_input_path "$2")
            saw_sidecar_arg=1
            if [[ -d "$sidecar_path" ]]; then
                sidecar_manifest_path="$sidecar_path/manifest.json"
            else
                sidecar_manifest_path="$sidecar_path"
            fi
            shift 2
            ;;
        --moe-mode)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            forwarded_args+=("$1" "$2")
            saw_moe_mode_arg=1
            had_forwarded_args=1
            shift 2
            ;;
        --moe-slot-bank)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            MOE_SLOT_BANK=$2
            saw_slot_bank_arg=1
            shift 2
            ;;
        --moe-topk)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            MOE_TOPK=$2
            saw_topk_arg=1
            shift 2
            ;;
        --moe-cache-io-split)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            MOE_CACHE_IO_SPLIT=$2
            saw_cache_io_split_arg=1
            shift 2
            ;;
        -ngl|--gpu-layers|--n-gpu-layers)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            N_GPU_LAYERS=$2
            saw_ngl_arg=1
            shift 2
            ;;
        -c|--ctx-size)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            CTX=$2
            saw_ctx_arg=1
            shift 2
            ;;
        -b|--batch-size)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            BATCH=$2
            saw_batch_arg=1
            shift 2
            ;;
        -ub|--ubatch-size)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            UBATCH=$2
            saw_ubatch_arg=1
            shift 2
            ;;
        --perf)
            saw_perf_arg=1
            shift
            ;;
        --no-perf)
            forwarded_args+=("$1")
            saw_perf_arg=1
            had_forwarded_args=1
            shift
            ;;
        -fit|--fit)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            FIT=$2
            saw_fit_arg=1
            shift 2
            ;;
        --temp|--temperature)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            forwarded_args+=("$1" "$2")
            had_forwarded_args=1
            saw_temp_arg=1
            shift 2
            ;;
        -f|--file|-bf|--binary-file)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            resolved_input=$(resolve_input_path "$2")
            forwarded_args+=("$1" "$resolved_input")
            had_forwarded_args=1
            saw_prompt_source_arg=1
            if [[ "$1" == "-f" || "$1" == "--file" ]] && is_builtin_coding_prompt_path "$resolved_input"; then
                selected_builtin_coding_prompt=1
            fi
            shift 2
            ;;
        -p|--prompt)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            forwarded_args+=("$1" "$2")
            had_forwarded_args=1
            saw_prompt_source_arg=1
            shift 2
            ;;
        -n|--predict|--n-predict)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            forwarded_args+=("$1" "$2")
            had_forwarded_args=1
            saw_n_predict_arg=1
            shift 2
            ;;
        -st|--single-turn)
            forwarded_args+=("$1")
            had_forwarded_args=1
            saw_st_arg=1
            shift
            ;;
        -cnv|--conversation|-no-cnv|--no-conversation)
            forwarded_args+=("$1")
            had_forwarded_args=1
            saw_conversation_mode_arg=1
            shift
            ;;
        --moe-trace-harness)
            forwarded_args+=("$1")
            had_forwarded_args=1
            saw_raw_completion_arg=1
            shift
            ;;
        --moe-prefill-layer-major)
            forwarded_args+=("$1")
            had_forwarded_args=1
            saw_prefill_layer_major_arg=1
            shift
            ;;
        --moe-prefill-batch)
            [[ $# -ge 2 ]] || die "missing argument for $1"
            forwarded_args+=("$1" "$2")
            prefill_batch_arg_index=$((${#forwarded_args[@]} - 1))
            requested_prefill_batch_arg=$2
            had_forwarded_args=1
            shift 2
            ;;
        --moe-prefill-batch=*)
            forwarded_args+=("$1")
            prefill_batch_eq_arg_index=$((${#forwarded_args[@]} - 1))
            requested_prefill_batch_arg=${1#*=}
            had_forwarded_args=1
            shift
            ;;
        --no-moe-prefill-layer-major)
            forwarded_args+=("$1")
            had_forwarded_args=1
            saw_prefill_layer_major_arg=0
            shift
            ;;
        *)
            forwarded_args+=("$1")
            had_forwarded_args=1
            shift
            ;;
    esac
done

[[ -f "$model_path" ]] || die "missing dense model: $model_path"
[[ -e "$sidecar_path" ]] || die "missing sidecar path: $sidecar_path"
[[ -f "$sidecar_manifest_path" ]] || die "missing sidecar manifest: $sidecar_manifest_path"

manifest_defaults=()
while IFS= read -r manifest_line; do
    manifest_defaults+=("$manifest_line")
done < <(parse_sidecar_manifest_defaults "$sidecar_manifest_path")
manifest_topk=${manifest_defaults[0]:-6}
manifest_expert_count=${manifest_defaults[1]:-256}
manifest_arch=${manifest_defaults[2]:-unknown}
manifest_entries=${manifest_defaults[3]:-0}
manifest_bytes_per_slot_all_layers=${manifest_defaults[4]:-0}
manifest_families=${manifest_defaults[5]:-}

requested_topk=${MOE_TOPK:-$manifest_topk}
effective_slot_bank=${MOE_SLOT_BANK:-96}
cache_io_split=${MOE_CACHE_IO_SPLIT:-4}
requested_ubatch=${UBATCH:-24}
requested_batch=${BATCH:-2048}
requested_ctx=${CTX:-7000}
requested_ngl=${N_GPU_LAYERS:-999}
fit_mode=${FIT:-on}
n_predict=${N_PREDICT:-256}
max_estimated_model_gib=${MAX_ESTIMATED_MODEL_GIB:-112}
dsv4_long_ctx_threshold=${DSV4_PREFILL_BATCH_LONG_CTX_THRESHOLD:-65536}
dsv4_safe_prefill_batch=${DSV4_PREFILL_BATCH_SAFE_MAX:-8192}
dsv4_adaptive_prefill=0
if [[ "$requested_ctx" =~ ^[0-9]+$ &&
      "$dsv4_long_ctx_threshold" =~ ^[0-9]+$ &&
      "$requested_ctx" -ge "$dsv4_long_ctx_threshold" ]]; then
    dsv4_safe_prefill_batch=${DSV4_PREFILL_BATCH_LONG_CTX_SAFE_MAX:-2048}
fi

if ! [[ "$effective_slot_bank" =~ ^[0-9]+$ && "$requested_topk" =~ ^[0-9]+$ && "$requested_ubatch" =~ ^[0-9]+$ ]]; then
    die "MOE_SLOT_BANK, MOE_TOPK, and UBATCH must be integers"
fi

if ! [[ "$dsv4_long_ctx_threshold" =~ ^[0-9]+$ && "$dsv4_long_ctx_threshold" -gt 0 ]]; then
    die "DSV4_PREFILL_BATCH_LONG_CTX_THRESHOLD must be a positive integer"
fi

if ! [[ "$dsv4_safe_prefill_batch" =~ ^[0-9]+$ && "$dsv4_safe_prefill_batch" -gt 0 ]]; then
    die "DSV4_PREFILL_BATCH_SAFE_MAX must be a positive integer"
fi

if [[ "$manifest_arch" == "deepseek4" &&
      "$requested_ctx" =~ ^[0-9]+$ &&
      "$requested_ctx" -ge "$dsv4_long_ctx_threshold" ]] &&
      env_truthy "${DSV4_PREFILL_ADAPTIVE:-1}" &&
      ! env_truthy "${LLAMA_FLASH_MOE_DSV4_ALLOW_UNSAFE_PREFILL_BATCH:-0}" &&
      ! env_truthy "${LLAMA_FLASH_MOE_DSV4_ALLOW_PREFILL_BATCH_GT_8K:-0}"; then
    dsv4_adaptive_prefill=1
    dsv4_safe_prefill_batch=${DSV4_PREFILL_BATCH_ADAPTIVE_MAX:-8192}
    export LLAMA_FLASH_MOE_DSV4_ADAPTIVE_PREFILL="${LLAMA_FLASH_MOE_DSV4_ADAPTIVE_PREFILL:-1}"
    if ! [[ "$dsv4_safe_prefill_batch" =~ ^[0-9]+$ && "$dsv4_safe_prefill_batch" -gt 0 ]]; then
        die "DSV4_PREFILL_BATCH_ADAPTIVE_MAX must be a positive integer"
    fi
fi

if [[ "$manifest_arch" == "deepseek4" &&
      "$requested_prefill_batch_arg" =~ ^[0-9]+$ &&
      "$requested_prefill_batch_arg" -gt "$dsv4_safe_prefill_batch" ]] &&
      ! env_truthy "${LLAMA_FLASH_MOE_DSV4_ALLOW_UNSAFE_PREFILL_BATCH:-0}" &&
      ! env_truthy "${LLAMA_FLASH_MOE_DSV4_ALLOW_PREFILL_BATCH_GT_8K:-0}"; then
    warn "DeepSeek V4 Flash layer-major prefill above ${dsv4_safe_prefill_batch} is unsafe for ctx=${requested_ctx}; clamping requested --moe-prefill-batch=${requested_prefill_batch_arg}. Set LLAMA_FLASH_MOE_DSV4_ALLOW_UNSAFE_PREFILL_BATCH=1 to debug the unsafe path."
    if (( prefill_batch_arg_index >= 0 )); then
        forwarded_args[$prefill_batch_arg_index]=$dsv4_safe_prefill_batch
    elif (( prefill_batch_eq_arg_index >= 0 )); then
        forwarded_args[$prefill_batch_eq_arg_index]="--moe-prefill-batch=${dsv4_safe_prefill_batch}"
    fi
fi

if (( manifest_expert_count > 0 && effective_slot_bank > manifest_expert_count )); then
    die "MOE_SLOT_BANK=$effective_slot_bank exceeds manifest expert_count=$manifest_expert_count"
fi

if (( requested_topk > manifest_topk )); then
    warn "MOE_TOPK=$requested_topk exceeds manifest expert_used_count=$manifest_topk; runtime may reject this"
fi

required_decode_slots=$(( requested_topk * requested_ubatch ))
if (( saw_prefill_layer_major_arg == 0 && requested_ubatch > 1 && required_decode_slots > effective_slot_bank )); then
    warn "MOE_SLOT_BANK=$effective_slot_bank is below MOE_TOPK*UBATCH=$required_decode_slots; decode may thrash or fail for multi-token ubatches"
fi

dense_bytes=$(stat -f %z "$model_path")
slot_bank_bytes=$(( manifest_bytes_per_slot_all_layers * effective_slot_bank ))
estimated_model_bytes=$(( dense_bytes + slot_bank_bytes ))
estimated_model_gib=$(human_gib "$estimated_model_bytes")
if python3 - "$estimated_model_bytes" "$max_estimated_model_gib" <<'PY'
import sys
estimated = int(sys.argv[1])
limit = float(sys.argv[2]) * 1024**3
raise SystemExit(0 if estimated > limit else 1)
PY
then
    if [[ "${ALLOW_HIGH_MEMORY:-0}" == "0" ]]; then
        die "estimated dense+slot-bank model memory is ${estimated_model_gib} GiB, above MAX_ESTIMATED_MODEL_GIB=${max_estimated_model_gib}. Lower MOE_SLOT_BANK or set ALLOW_HIGH_MEMORY=1."
    fi
    warn "estimated dense+slot-bank model memory is ${estimated_model_gib} GiB; ALLOW_HIGH_MEMORY=1 is set"
fi

export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT:-65536}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB:-0}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES="${LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES:-1}"

declare -a cmd
cmd=("$llama_bin")

if [[ "$saw_perf_arg" == "0" ]]; then
    cmd+=(--perf)
fi

if [[ "$saw_model_arg" == "0" ]]; then
    cmd+=(-m "$model_path")
else
    cmd+=(-m "$model_path")
fi

if [[ "$saw_moe_mode_arg" == "0" ]]; then
    cmd+=(--moe-mode slot-bank)
fi

if [[ "$saw_sidecar_arg" == "0" ]]; then
    cmd+=(--moe-sidecar "$sidecar_path")
else
    cmd+=(--moe-sidecar "$sidecar_path")
fi

cmd+=(
    --moe-slot-bank "$effective_slot_bank"
    --moe-topk "$requested_topk"
    --moe-cache-io-split "$cache_io_split"
    -fit "$fit_mode"
    -ngl "$requested_ngl"
    -c "$requested_ctx"
    -b "$requested_batch"
    -ub "$requested_ubatch"
    --no-warmup
)

if [[ "${MOE_PREFETCH_TEMPORAL:-1}" != "0" ]]; then
    cmd+=(--moe-prefetch-temporal)
fi

if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi

if [[ -n "${TEMP:-}" && "$saw_temp_arg" == "0" ]]; then
    cmd+=(--temp "$TEMP")
fi

if [[ "${DISPLAY_PROMPT:-1}" == "0" ]]; then
    cmd+=(--no-display-prompt)
fi

if [[ "${SIMPLE_IO:-0}" != "0" ]]; then
    cmd+=(--simple-io)
fi

if [[ "${RAW_COMPLETION:-0}" != "0" && "$saw_raw_completion_arg" == "0" ]]; then
    cmd+=(--moe-trace-harness)
fi

if [[ -n "${PROMPT_FILE:-}" && "$saw_prompt_source_arg" == "0" ]]; then
    resolved_prompt_file=$(resolve_input_path "$PROMPT_FILE")
    forwarded_args+=(-f "$resolved_prompt_file")
    had_forwarded_args=1
    saw_prompt_source_arg=1
    if is_builtin_coding_prompt_path "$resolved_prompt_file"; then
        selected_builtin_coding_prompt=1
    fi
fi

if [[ -n "$benchmark_prompt_label" && "$saw_prompt_source_arg" == "0" ]]; then
    resolved_benchmark_prompt=$(resolve_prompt_label_path "$benchmark_prompt_label")
    note "BENCHMARK_PROMPT_LABEL=$benchmark_prompt_label selects $resolved_benchmark_prompt"
    forwarded_args+=(-f "$resolved_benchmark_prompt")
    had_forwarded_args=1
    saw_prompt_source_arg=1
    selected_builtin_coding_prompt=1
fi

if [[ "$selected_builtin_coding_prompt" == "1" && "$saw_conversation_mode_arg" == "0" && "$saw_raw_completion_arg" == "0" && -z "${RAW_COMPLETION:-}" && "$llama_bin_name" == "llama-cli" ]]; then
    note "auto-enabling --moe-trace-harness for built-in coding prompt"
    note "pass -cnv or set RAW_COMPLETION=0 to keep chat-template mode for this benchmark prompt"
    forwarded_args+=(--moe-trace-harness)
    saw_raw_completion_arg=1
fi

if [[ "$saw_prompt_source_arg" == "0" ]]; then
    forwarded_args+=(-p "Make a game of Space Invaders in PyGame")
fi

if [[ "$saw_n_predict_arg" == "0" ]]; then
    forwarded_args+=(-n "$n_predict")
fi

if [[ "$saw_st_arg" == "0" ]]; then
    forwarded_args+=(-st)
fi

cmd+=("${forwarded_args[@]}")

note "DeepSeek V4 Flash profile"
note "arch=${manifest_arch} entries=${manifest_entries} families=${manifest_families} expert_used=${manifest_topk} experts=${manifest_expert_count}"
note "dense=$model_path"
note "sidecar=$sidecar_path"
note "slot_bank=${effective_slot_bank} topk=${requested_topk} cache_io_split=${cache_io_split} ctx=${requested_ctx} batch=${requested_batch} ubatch=${requested_ubatch}"
if (( dsv4_adaptive_prefill )); then
    note "dsv4 adaptive prefill=on schedule=0-16k:8192,16k-40k:4096,40k+:2048 max_arg=${dsv4_safe_prefill_batch}"
fi
note "estimated dense+slot-bank model memory=${estimated_model_gib} GiB (limit=${max_estimated_model_gib} GiB, override with ALLOW_HIGH_MEMORY=1)"

printf "running:"
printf " %q" "${cmd[@]}"
printf "\n"

exec "${cmd[@]}"
