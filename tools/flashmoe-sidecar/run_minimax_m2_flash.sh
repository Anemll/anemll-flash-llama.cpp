#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)
default_prompt_dir="$repo_root/tools/flashmoe-sidecar/prompts/coding"

note() {
    printf "note: %s\n" "$1" >&2
}

warn() {
    printf "warning: %s\n" "$1" >&2
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

    echo "unknown PROMPT_LABEL '$label' (expected one of: 1k, 4k, 16k, 22k)" >&2
    exit 1
}

is_builtin_coding_prompt_path() {
    local candidate=$1
    [[ "$candidate" == "$default_prompt_dir"/coding_*.txt ]]
}

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PACKAGE_DIR [llama-cli args...]" >&2
    exit 1
fi

package_dir=$1
shift

llama_bin=${LLAMA_BIN:-./build/bin/llama-cli}
llama_bin_name=$(basename "$llama_bin")
model_path=${MODEL_PATH:-"$package_dir/model-dense.gguf"}
sidecar_path=${SIDECAR_PATH:-"$package_dir/sidecar"}
secondary_sidecar_path=${SECONDARY_SIDECAR_PATH:-}
tertiary_sidecar_path=${TERTIARY_SIDECAR_PATH:-}
prefetch_sidecar=${PREFETCH_SIDECAR:-}
package_json=${PACKAGE_JSON:-"$package_dir/flashmoe-package.json"}

package_topk=4
package_chat_topk=8
package_slot_bank=64
package_cache_io_split=4
package_prefetch_temporal=1
package_reasoning=off
package_reasoning_budget=0
package_ubatch=8
package_expert_count=0
sidecar_manifest_path=${SIDECAR_MANIFEST_PATH:-}
if [[ -z "$sidecar_manifest_path" && -d "$sidecar_path" ]]; then
    sidecar_manifest_path="$sidecar_path/manifest.json"
fi

if [[ -f "$package_json" ]]; then
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

if [[ "$package_expert_count" == "0" && -f "$sidecar_manifest_path" ]]; then
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

if [[ ! -e "$sidecar_path" ]]; then
    echo "missing sidecar path: $sidecar_path" >&2
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

requested_topk=${MOE_TOPK:-${MINIMAX_CHAT_TOPK:-$package_chat_topk}}
requested_ubatch=${UBATCH:-$package_ubatch}

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
    if (( package_expert_count > 0 && effective_slot_bank > package_expert_count )); then
        effective_slot_bank=$package_expert_count
    fi
fi

declare -a cmd
cmd=(
    "$llama_bin"
    --perf
    -m "$model_path"
    --moe-mode slot-bank
    --moe-sidecar "$sidecar_path"
    --moe-slot-bank "$effective_slot_bank"
    --moe-topk "$requested_topk"
    --moe-cache-io-split "${MOE_CACHE_IO_SPLIT:-$package_cache_io_split}"
    -fit on
    -ub "$requested_ubatch"
    -b "${BATCH:-64}"
    -ngl "${N_GPU_LAYERS:-999}"
    -c "${CTX:-4096}"
    -rea "${REASONING_MODE:-$package_reasoning}"
    --reasoning-budget "${REASONING_BUDGET:-$package_reasoning_budget}"
)

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

if [[ "${DISPLAY_PROMPT:-1}" == "0" ]]; then
    cmd+=(--no-display-prompt)
fi

if [[ "${SIMPLE_IO:-0}" != "0" ]]; then
    cmd+=(--simple-io)
fi

if [[ "${RAW_COMPLETION:-0}" != "0" ]]; then
    cmd+=(--moe-trace-harness)
fi

declare -a forwarded_args
forwarded_args=()
had_forwarded_args=0
saw_prompt_file_arg=0
saw_prompt_binary_file_arg=0
saw_prompt_text_arg=0
saw_conversation_mode_arg=0
saw_raw_completion_arg=0
effective_reasoning_mode=${REASONING_MODE:-$package_reasoning}
saw_reasoning_format_arg=0
selected_builtin_coding_prompt=0

if [[ -n "${PROMPT_LABEL:-}" && -n "${BENCHMARK_PROMPT_LABEL:-}" && "${PROMPT_LABEL}" != "${BENCHMARK_PROMPT_LABEL}" ]]; then
    echo "PROMPT_LABEL and BENCHMARK_PROMPT_LABEL disagree; set only one of them" >&2
    exit 1
fi

benchmark_prompt_label=${BENCHMARK_PROMPT_LABEL:-${PROMPT_LABEL:-}}
benchmark_prompt_label_source=
if [[ -n "${BENCHMARK_PROMPT_LABEL:-}" ]]; then
    benchmark_prompt_label_source="BENCHMARK_PROMPT_LABEL"
elif [[ -n "${PROMPT_LABEL:-}" ]]; then
    benchmark_prompt_label_source="PROMPT_LABEL"
    warn "PROMPT_LABEL is a legacy alias here; it injects a built-in synthetic coding benchmark prompt file, not a generic prompt-length knob. Prefer BENCHMARK_PROMPT_LABEL for clarity."
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file|-bf|--binary-file)
            if [[ $# -lt 2 ]]; then
                echo "missing argument for $1" >&2
                exit 1
            fi
            resolved_input=$(resolve_input_path "$2")
            forwarded_args+=("$1" "$resolved_input")
            had_forwarded_args=1
            if [[ "$1" == "-f" || "$1" == "--file" ]]; then
                saw_prompt_file_arg=1
                if is_builtin_coding_prompt_path "$resolved_input"; then
                    selected_builtin_coding_prompt=1
                fi
            else
                saw_prompt_binary_file_arg=1
            fi
            shift 2
            ;;
        -p|--prompt)
            if [[ $# -lt 2 ]]; then
                echo "missing argument for $1" >&2
                exit 1
            fi
            forwarded_args+=("$1" "$2")
            had_forwarded_args=1
            saw_prompt_text_arg=1
            shift 2
            ;;
        -n|--predict|--n-predict)
            if [[ $# -lt 2 ]]; then
                echo "missing argument for $1" >&2
                exit 1
            fi
            value=$2
            if [[ "$value" =~ ^(-?[0-9]+)-st$ ]]; then
                forwarded_args+=("$1" "${BASH_REMATCH[1]}" "-st")
                note "normalized merged '$value' into '${BASH_REMATCH[1]}' and '-st' after $1"
            else
                forwarded_args+=("$1" "$value")
            fi
            had_forwarded_args=1
            shift 2
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
        --reasoning-format)
            if [[ $# -lt 2 ]]; then
                echo "missing argument for $1" >&2
                exit 1
            fi
            forwarded_args+=("$1" "$2")
            had_forwarded_args=1
            saw_reasoning_format_arg=1
            shift 2
            ;;
        *)
            forwarded_args+=("$1")
            had_forwarded_args=1
            shift
            ;;
    esac
done

if [[ -n "${PROMPT_FILE:-}" && "$saw_prompt_file_arg" == "0" && "$saw_prompt_binary_file_arg" == "0" && "$saw_prompt_text_arg" == "0" ]]; then
    resolved_prompt_file=$(resolve_input_path "$PROMPT_FILE")
    forwarded_args+=(-f "$resolved_prompt_file")
    had_forwarded_args=1
    saw_prompt_file_arg=1
    if is_builtin_coding_prompt_path "$resolved_prompt_file"; then
        selected_builtin_coding_prompt=1
    fi
elif [[ -n "${PROMPT_FILE:-}" ]]; then
    note "ignoring PROMPT_FILE because an explicit prompt source was already provided"
fi

if [[ -n "${PROMPT_BINARY_FILE:-}" && "$saw_prompt_file_arg" == "0" && "$saw_prompt_binary_file_arg" == "0" && "$saw_prompt_text_arg" == "0" ]]; then
    forwarded_args+=(-bf "$(resolve_input_path "$PROMPT_BINARY_FILE")")
    had_forwarded_args=1
    saw_prompt_binary_file_arg=1
elif [[ -n "${PROMPT_BINARY_FILE:-}" ]]; then
    note "ignoring PROMPT_BINARY_FILE because an explicit prompt source was already provided"
fi

if [[ -n "$benchmark_prompt_label" && "$saw_prompt_file_arg" == "0" && "$saw_prompt_binary_file_arg" == "0" && "$saw_prompt_text_arg" == "0" ]]; then
    resolved_benchmark_prompt=$(resolve_prompt_label_path "$benchmark_prompt_label")
    note "${benchmark_prompt_label_source}=$benchmark_prompt_label selects the built-in synthetic coding benchmark prompt file: $resolved_benchmark_prompt"
    note "remove ${benchmark_prompt_label_source} or pass -p/--prompt to run a normal ad-hoc generation instead"
    forwarded_args+=(-f "$resolved_benchmark_prompt")
    had_forwarded_args=1
    saw_prompt_file_arg=1
    selected_builtin_coding_prompt=1
elif [[ -n "$benchmark_prompt_label" ]]; then
    note "ignoring ${benchmark_prompt_label_source} because an explicit prompt source was already provided"
fi

if [[ "$selected_builtin_coding_prompt" == "1" && "$saw_conversation_mode_arg" == "0" && "$saw_raw_completion_arg" == "0" && -z "${RAW_COMPLETION:-}" ]]; then
    if [[ "$llama_bin_name" == "llama-cli" ]]; then
        note "auto-enabling --moe-trace-harness for built-in coding benchmark prompts so llama-cli runs a single raw completion instead of wrapping the prompt in the model chat template"
        note "pass -cnv to keep chat-template mode for this benchmark prompt"
        forwarded_args+=(--moe-trace-harness)
    else
        note "built-in coding benchmark prompt detected, but auto raw-completion mode is only enabled for llama-cli; pass the appropriate raw-completion flag manually for '$llama_bin_name' if needed"
    fi
fi

if [[ "$effective_reasoning_mode" == "off" && "$saw_reasoning_format_arg" == "0" ]]; then
    # With reasoning disabled, leaving the parser in its default DeepSeek-style
    # extraction mode can hide the entire answer if the model still emits a
    # reasoning-shaped stream. Force plain content parsing unless the caller
    # explicitly asked for a different reasoning format.
    forwarded_args+=(--reasoning-format none)
fi

if [[ "$had_forwarded_args" == "0" ]]; then
    cmd+=(
        -p "Explain why Flash-MoE helps large routed MiniMax models."
        -n "${N_PREDICT:-128}"
        -st
    )
else
    cmd+=("${forwarded_args[@]}")
fi

printf "running:"
printf " %q" "${cmd[@]}"
printf "\n"

exec "${cmd[@]}"
