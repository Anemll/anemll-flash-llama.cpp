#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LLAMA_CLI="${LLAMA_CLI:-${LLAMA_DIR}/build/bin/llama-cli}"
MODEL="${MODEL:-${HOME}/Models/Qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
PROMPT_FILE="${PROMPT_FILE:-${SCRIPT_DIR}/prompts/coding/coding_22k.txt}"
RESULTS_DIR="${RESULTS_DIR:-${LLAMA_DIR}/flashmoe-results/in-memory-prefill}"

THRESHOLDS="${THRESHOLDS:-0 8 32 64 128}"
MICRO_BATCHES="${MICRO_BATCHES:-512}"
BANKS="${BANKS:-1 2}"
FP16_ACT="${FP16_ACT:-1}"
DEBUG_IN_MEMORY="${DEBUG_IN_MEMORY:-0}"
LAYER_STATS="${LAYER_STATS:-1}"
INLINE_PROGRESS="${INLINE_PROGRESS:-1}"
GREEDY_BUCKETS="${GREEDY_BUCKETS:-0}"

BATCH_SIZE="${BATCH_SIZE:-4096}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
MOE_PREFILL_BATCH="${MOE_PREFILL_BATCH:-32000}"
N_PREDICT="${N_PREDICT:-100}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
SEED_VALUE="${SEED_VALUE:-123}"
TEMP_VALUE="${TEMP_VALUE:-0}"

if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "missing llama-cli: ${LLAMA_CLI}" >&2
    echo "build first with: cmake --build ${LLAMA_DIR}/build -j8 --target llama-cli" >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "missing model: ${MODEL}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "missing prompt file: ${PROMPT_FILE}" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

for threshold in ${THRESHOLDS}; do
    for micro in ${MICRO_BATCHES}; do
        for banks in ${BANKS}; do
            label="m5-${threshold}_micro-${micro}_banks-${banks}_fp16-${FP16_ACT}_ub-${UBATCH_SIZE}"
            log_path="${RESULTS_DIR}/${label}.log"

            echo
            echo "===== ${label} ====="
            echo "log: ${log_path}"

            env \
                LLAMA_FLASH_MOE_EXPERIMENTAL_IN_MEMORY_FP16_ACTIVATIONS="${FP16_ACT}" \
                LLAMA_FLASH_MOE_EXPERIMENTAL_IN_MEMORY_GREEDY_BUCKETS="${GREEDY_BUCKETS}" \
                LLAMA_FLASH_MOE_EXPERIMENTAL_M5_EXPERT_MM_MIN_TOKENS="${threshold}" \
                LLAMA_FLASH_MOE_DEBUG_IN_MEMORY_PREFILL="${DEBUG_IN_MEMORY}" \
                LLAMA_FLASH_MOE_PERF_PREFILL_LAYER_STATS="${LAYER_STATS}" \
                LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${INLINE_PROGRESS}" \
                TEMP="${TEMP_VALUE}" \
                SEED="${SEED_VALUE}" \
                "${LLAMA_CLI}" \
                    -m "${MODEL}" \
                    -f "${PROMPT_FILE}" \
                    -ngl "${N_GPU_LAYERS}" \
                    -n "${N_PREDICT}" \
                    --no-warmup \
                    --perf \
                    -st \
                    --simple-io \
                    --no-display-prompt \
                    -b "${BATCH_SIZE}" \
                    -ub "${UBATCH_SIZE}" \
                    --moe-prefill-layer-major \
                    --moe-prefill-batch "${MOE_PREFILL_BATCH}" \
                    --moe-prefill-micro-batch "${micro}" \
                    --moe-prefill-banks "${banks}" \
                2>&1 | tee "${log_path}"
        done
    done
done

