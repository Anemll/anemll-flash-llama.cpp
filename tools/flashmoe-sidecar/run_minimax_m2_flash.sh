#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PACKAGE_DIR [llama-cli args...]" >&2
    exit 1
fi

package_dir=$1
shift

llama_bin=${LLAMA_BIN:-./build/bin/llama-cli}
model_path=${MODEL_PATH:-"$package_dir/model-dense.gguf"}
sidecar_path=${SIDECAR_PATH:-"$package_dir/sidecar"}
package_json=${PACKAGE_JSON:-"$package_dir/flashmoe-package.json"}

package_topk=4
package_slot_bank=64
package_cache_io_split=4
package_prefetch_temporal=1

if [[ -f "$package_json" ]]; then
    read -r package_topk package_slot_bank package_cache_io_split package_prefetch_temporal < <(
        python3 - "$package_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    package = json.load(handle)

hint = package.get("runtime_hint") or {}
print(
    hint.get("moe_topk", 4),
    hint.get("moe_slot_bank", 64),
    hint.get("moe_cache_io_split", 4),
    1 if hint.get("moe_prefetch_temporal", True) else 0,
)
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

export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY:-1}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT:-65536}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB="${LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB:-0}"
export LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES="${LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES:-1}"

declare -a cmd
cmd=(
    "$llama_bin"
    --perf
    -m "$model_path"
    --moe-mode slot-bank
    --moe-sidecar "$sidecar_path"
    --moe-slot-bank "${MOE_SLOT_BANK:-$package_slot_bank}"
    --moe-topk "${MOE_TOPK:-$package_topk}"
    --moe-cache-io-split "${MOE_CACHE_IO_SPLIT:-$package_cache_io_split}"
    -fit on
    -ub "${UBATCH:-1}"
    -b "${BATCH:-64}"
    -ngl "${N_GPU_LAYERS:-999}"
    -c "${CTX:-4096}"
)

if [[ "${MOE_PREFETCH_TEMPORAL:-$package_prefetch_temporal}" != "0" ]]; then
    cmd+=(--moe-prefetch-temporal)
fi

if [[ -n "${SEED:-}" ]]; then
    cmd+=(--seed "$SEED")
fi

if [[ -n "${TEMP:-}" ]]; then
    cmd+=(--temp "$TEMP")
fi

if [[ $# -eq 0 ]]; then
    cmd+=(
        -p "Explain why Flash-MoE helps large routed MiniMax models."
        -n "${N_PREDICT:-128}"
        -st
    )
else
    cmd+=("$@")
fi

printf "running:"
printf " %q" "${cmd[@]}"
printf "\n"

exec "${cmd[@]}"
