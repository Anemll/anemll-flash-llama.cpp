#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)

if [[ $# -lt 1 ]]; then
    echo "usage: $0 MODEL_FILE_OR_PACKAGE_DIR [llama-server args...]" >&2
    exit 1
fi

target=$1
shift

llama_server_bin=${LLAMA_SERVER_BIN:-./build/bin/llama-server}

package_dir=
if [[ -d "$target" ]]; then
    package_dir=$target
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    model_path=$MODEL_PATH
elif [[ -f "$target" ]]; then
    model_path=$target
else
    model_path="$target/model-dense.gguf"
fi

sidecar_path=${SIDECAR_PATH:-}
if [[ -z "$sidecar_path" && -n "$package_dir" ]]; then
    sidecar_path="$package_dir/sidecar"
fi

secondary_sidecar_path=${SECONDARY_SIDECAR_PATH:-}
tertiary_sidecar_path=${TERTIARY_SIDECAR_PATH:-}
prefetch_sidecar=${PREFETCH_SIDECAR:-}
model_alias=${MODEL_ALIAS:-}
chat_template_file=${CHAT_TEMPLATE_FILE:-}
package_json=${PACKAGE_JSON:-}
if [[ -z "$package_json" && -n "$package_dir" ]]; then
    package_json="$package_dir/flashmoe-package.json"
fi

package_topk=4
package_slot_bank=64
package_cache_io_split=4
package_prefetch_temporal=1
package_expert_count=0
sidecar_manifest_path=${SIDECAR_MANIFEST_PATH:-}
if [[ -z "$sidecar_manifest_path" && -n "$sidecar_path" && -d "$sidecar_path" ]]; then
    sidecar_manifest_path="$sidecar_path/manifest.json"
fi

if [[ -n "$package_json" && -f "$package_json" ]]; then
    read -r package_topk package_slot_bank package_cache_io_split package_prefetch_temporal package_expert_count < <(
        python3 - "$package_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    package = json.load(handle)

hint = package.get("runtime_hint") or {}
model = package.get("model") or {}
print(
    hint.get("moe_topk", 4),
    hint.get("moe_slot_bank", 64),
    hint.get("moe_cache_io_split", 4),
    1 if hint.get("moe_prefetch_temporal", True) else 0,
    model.get("expert_count", 0),
)
PY
    )
fi

if [[ "$package_expert_count" == "0" && -n "$sidecar_manifest_path" && -f "$sidecar_manifest_path" ]]; then
    package_expert_count=$(python3 - "$sidecar_manifest_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

model = manifest.get("model") or {}
print(model.get("expert_count", 0))
PY
    )
fi

if [[ ! -f "$model_path" ]]; then
    echo "missing dense model: $model_path" >&2
    exit 1
fi

if [[ -z "$sidecar_path" || ! -e "$sidecar_path" ]]; then
    echo "missing sidecar path: ${sidecar_path:-<unset>}" >&2
    exit 1
fi

if [[ -n "$secondary_sidecar_path" && ! -e "$secondary_sidecar_path" ]]; then
    echo "missing secondary sidecar path: $secondary_sidecar_path" >&2
    exit 1
fi

if [[ -n "$tertiary_sidecar_path" && ! -e "$tertiary_sidecar_path" ]]; then
    echo "missing tertiary sidecar path: $tertiary_sidecar_path" >&2
    exit 1
fi

export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT:-65536}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB:-0}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES="${LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES:-1}"
export LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS:-1}"
export LLAMA_FLASH_MOE_PERF_PREFILL_LAYER_STATS="${LLAMA_FLASH_MOE_PERF_PREFILL_LAYER_STATS:-0}"

requested_topk=${MOE_TOPK:-$package_topk}
requested_ubatch=${UBATCH:-32}
server_profile=${FLASHMOE_SERVER_PROFILE:-default}
server_perf=${FLASHMOE_SERVER_PERF:-0}
profile_slot_bank_floor=0

case "$server_profile" in
    ""|default)
        server_profile=default
        ;;
    highmem-decode|decode-highmem|high-memory-decode)
        server_profile=highmem-decode
        profile_slot_bank_floor=128
        ;;
    *)
        echo "unknown FLASHMOE_SERVER_PROFILE '$server_profile' (expected: default or highmem-decode)" >&2
        exit 1
        ;;
esac

if [[ -n "${MOE_SLOT_BANK:-}" ]]; then
    effective_slot_bank=$MOE_SLOT_BANK
else
    effective_slot_bank=$package_slot_bank
    required_slots=$(( requested_topk * requested_ubatch ))
    if (( requested_ubatch > 1 )); then
        required_slots=$(( required_slots + 32 ))
    fi
    if (( required_slots > effective_slot_bank )); then
        effective_slot_bank=$(( ((required_slots + 31) / 32) * 32 ))
    fi
    if (( profile_slot_bank_floor > effective_slot_bank )); then
        effective_slot_bank=$profile_slot_bank_floor
    fi
    if (( package_expert_count > 0 && effective_slot_bank > package_expert_count )); then
        effective_slot_bank=$package_expert_count
    fi
fi

declare -a cmd

cmd_has_flag() {
    local needle=$1
    local arg
    for arg in "${cmd[@]}"; do
        if [[ "$arg" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

cmd_opt_value() {
    local needle=$1
    local arg
    local take_next=0
    local value=
    for arg in "${cmd[@]}"; do
        if [[ "$take_next" == 1 ]]; then
            value=$arg
            take_next=0
            continue
        fi
        if [[ "$arg" == "$needle" ]]; then
            take_next=1
        fi
    done
    if [[ -n "$value" ]]; then
        printf "%s" "$value"
    fi
}

has_explicit_perf_args=0
if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
        case "$arg" in
            --perf|--no-perf)
                has_explicit_perf_args=1
                break
                ;;
        esac
    done
fi

cmd=(
    "$llama_server_bin"
    --host "${HOST:-127.0.0.1}"
    --port "${PORT:-8080}"
    --model "$model_path"
    --moe-mode slot-bank
    --moe-sidecar "$sidecar_path"
    --moe-slot-bank "$effective_slot_bank"
    --moe-topk "$requested_topk"
    --moe-cache-io-split "${MOE_CACHE_IO_SPLIT:-$package_cache_io_split}"
    -fit on
    -ub "$requested_ubatch"
    -b "${BATCH:-2048}"
    -ngl "${N_GPU_LAYERS:-999}"
    -c "${CTX:-8192}"
    --parallel "${N_PARALLEL:-1}"
)

if [[ "$has_explicit_perf_args" == "0" ]]; then
    if [[ "$server_perf" != "0" ]]; then
        cmd=( "${cmd[0]}" --perf "${cmd[@]:1}" )
    else
        cmd=( "${cmd[0]}" --no-perf "${cmd[@]:1}" )
    fi
fi

if [[ -n "$model_alias" ]]; then
    cmd+=(--alias "$model_alias")
fi

if [[ "${NO_WEBUI:-1}" != "0" ]]; then
    cmd+=(--no-webui)
fi

if [[ -n "$secondary_sidecar_path" ]]; then
    cmd+=(--moe-secondary-sidecar "$secondary_sidecar_path")
fi

if [[ -n "$tertiary_sidecar_path" ]]; then
    cmd+=(--moe-tertiary-sidecar "$tertiary_sidecar_path")
fi

if [[ "${MOE_PREFETCH_TEMPORAL:-$package_prefetch_temporal}" != "0" ]]; then
    cmd+=(--moe-prefetch-temporal)
fi

if [[ -n "$prefetch_sidecar" ]]; then
    cmd+=(--moe-prefetch-sidecar "$prefetch_sidecar")
fi

if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi

if [[ -n "${TEMP:-}" ]]; then
    cmd+=(--temp "$TEMP")
fi

if [[ $# -gt 0 ]]; then
    has_explicit_template_args=0
    for arg in "$@"; do
        case "$arg" in
            --chat-template|--chat-template-file)
                has_explicit_template_args=1
                break
                ;;
        esac
    done
else
    has_explicit_template_args=0
fi

if [[ -z "$chat_template_file" && "$has_explicit_template_args" == "0" ]]; then
    local_minimax_template="$script_dir/../../models/templates/MiniMax-M2.jinja"
    if [[ -f "$local_minimax_template" ]]; then
        minimax_hint="${model_alias} ${model_path} ${package_dir:-} ${target}"
        if [[ "$minimax_hint" == *MiniMax-M2* || "$minimax_hint" == *minimax-m2* ]]; then
            chat_template_file="$local_minimax_template"
        fi
    fi
fi

if [[ -n "$chat_template_file" && "$has_explicit_template_args" == "0" ]]; then
    cmd+=(--jinja --chat-template-file "$chat_template_file")
fi

if [[ $# -gt 0 ]]; then
    cmd+=("$@")
fi

resolved_host=$(cmd_opt_value --host)
resolved_port=$(cmd_opt_value --port)
resolved_alias=$(cmd_opt_value --alias)
resolved_ctx=$(cmd_opt_value -c)
resolved_batch=$(cmd_opt_value -b)
resolved_ubatch=$(cmd_opt_value -ub)
resolved_topk=$(cmd_opt_value --moe-topk)
resolved_slot_bank=$(cmd_opt_value --moe-slot-bank)
resolved_cache_io_split=$(cmd_opt_value --moe-cache-io-split)
resolved_temp=$(cmd_opt_value --temp)
resolved_seed=$(cmd_opt_value --seed)
resolved_prefill_batch=$(cmd_opt_value --moe-prefill-batch)
resolved_prefill_micro_batch=$(cmd_opt_value --moe-prefill-micro-batch)
resolved_prefill_io_split=$(cmd_opt_value --moe-prefill-io-split)
resolved_prefill_banks=$(cmd_opt_value --moe-prefill-banks)
resolved_perf=default
if cmd_has_flag --perf; then
    resolved_perf=on
elif cmd_has_flag --no-perf; then
    resolved_perf=off
fi

printf "resolved server settings:\n"
printf "  host=%s port=%s alias=%s perf=%s\n" \
    "${resolved_host:-<unset>}" "${resolved_port:-<unset>}" "${resolved_alias:-<none>}" "$resolved_perf"
printf "  ctx=%s batch=%s ubatch=%s topk=%s slot_bank=%s cache_io_split=%s\n" \
    "${resolved_ctx:-<unset>}" "${resolved_batch:-<unset>}" "${resolved_ubatch:-<unset>}" \
    "${resolved_topk:-<unset>}" "${resolved_slot_bank:-<unset>}" "${resolved_cache_io_split:-<unset>}"
printf "  prefill_batch=%s prefill_micro_batch=%s prefill_io_split=%s prefill_banks=%s\n" \
    "${resolved_prefill_batch:-<unset>}" "${resolved_prefill_micro_batch:-<unset>}" \
    "${resolved_prefill_io_split:-<unset>}" "${resolved_prefill_banks:-<unset>}"
printf "  seed=%s temp=%s\n" \
    "${resolved_seed:-<unset>}" "${resolved_temp:-<model-default>}"

printf "running:"
printf " %q" "${cmd[@]}"
printf "\n"

exec "${cmd[@]}"
