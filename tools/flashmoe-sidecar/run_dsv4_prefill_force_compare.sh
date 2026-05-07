#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir=$(cd "$(dirname "$0")" && pwd -P)
repo_root=$(cd "$script_dir/../.." && pwd -P)

usage() {
    cat >&2 <<EOF
usage: $0 [PACKAGE_DIR]

Runs a paired DeepSeek V4 layer-major prefill comparison:
  baseline: requested --moe-prefill-batch, no force, normal safety clamp/adaptive split
  forced:   same requested --moe-prefill-batch plus --force-moe-prefill-batch

Every run log starts with an explicit "comparison:" header and the wrapper also
emits note: comparison=<id> role=<role>, so pasted logs identify the case.

Defaults:
  MODEL_PATH          PACKAGE_DIR/dense/model-dense.gguf or DeepSeek-V4-SSD
  SIDECAR_PATH        PACKAGE_DIR/sidecar or DeepSeek-V4-SSD
  PROMPT_FILE         tools/flashmoe-sidecar/prompts/coding/coding_16k.txt
  CTX                 30000
  MOE_PREFILL_BATCH   16384
  MOE_TOPK            4
  MOE_SLOT_BANK       32
  N_PREDICT           64

Useful overrides:
  RESULTS_DIR=/tmp/dsv4-compare CTX=90000 N_PREDICT=128 $0 /path/package
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

PROMPT_FILE="${PROMPT_FILE:-$script_dir/prompts/coding/coding_16k.txt}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/flashmoe-results/dsv4-prefill-force-compare/$(date +%Y%m%d-%H%M%S)}"
COMPARE_ID="${COMPARE_ID:-dsv4-prefill-force}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

MOE_TOPK="${MOE_TOPK:-4}"
MOE_CACHE_IO_SPLIT="${MOE_CACHE_IO_SPLIT:-8}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-1}"
CTX="${CTX:-30000}"
SEED="${SEED:-123}"
TEMP="${TEMP:-0}"
MOE_SLOT_BANK="${MOE_SLOT_BANK:-32}"
MOE_PREFILL_BATCH="${MOE_PREFILL_BATCH:-16384}"
MOE_PREFILL_MICRO_BATCH="${MOE_PREFILL_MICRO_BATCH:-auto}"
MOE_PREFILL_IO_SPLIT="${MOE_PREFILL_IO_SPLIT:-8}"
MOE_PREFILL_BANKS="${MOE_PREFILL_BANKS:-4}"
N_PREDICT="${N_PREDICT:-64}"
SAMPLE_TEMP="${SAMPLE_TEMP:-}"
TOP_P="${TOP_P:-}"

[[ -f "$MODEL_PATH" ]] || {
    echo "error: missing model: $MODEL_PATH" >&2
    exit 1
}
[[ -e "$SIDECAR_PATH" ]] || {
    echo "error: missing sidecar: $SIDECAR_PATH" >&2
    exit 1
}
[[ -f "$PROMPT_FILE" ]] || {
    echo "error: missing prompt: $PROMPT_FILE" >&2
    exit 1
}

mkdir -p "$RESULTS_DIR"
summary_csv="$RESULTS_DIR/summary.csv"
summary_txt="$RESULTS_DIR/summary.txt"
printf "role,status,ctx,prefill_batch,topk,slot_bank,actual_chunk,prefill_tokens,decode_tokens,prompt_tps,generation_tps,payload_sha,payload_preview,log\n" > "$summary_csv"

extract_last() {
    local pattern=$1
    local log_path=$2
    sed -n "$pattern" "$log_path" | tail -1
}

extract_payload_meta() {
    local log_path=$1
    python3 - "$log_path" <<'PY'
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8", errors="replace")
raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
raw = raw.replace("\r", "\n")
marker = "mode       : Flash-MoE trace harness (raw completion)"
if marker in raw:
    raw = raw.split(marker, 1)[1]
lines = raw.splitlines()

skip_prefixes = (
    "comparison:",
    "case:",
    "model:",
    "sidecar:",
    "prompt:",
    "ctx:",
    "requested_prefill_batch:",
    "force_prefill_batch:",
    "note:",
    "warning:",
    "running:",
    "ggml_",
    "Flash-MoE ",
    "Loading model",
    "build      :",
    "model      :",
    "modalities :",
    "mode       :",
    "[ Prompt:",
    "Exiting...",
    "llama_memory_breakdown_print:",
    "~llama_context:",
    "log_runtime_summary:",
    "log_perf_profile_table:",
    "operator():",
    "DEPRECATED:",
    "WARNING:",
)

payload: list[str] = []
for line in lines:
    stripped = line.strip()
    if not stripped:
        continue
    if stripped.startswith(skip_prefixes):
        continue
    if re.fullmatch(r"[▄█▀ ]+", stripped):
        continue
    if re.fullmatch(r"[=\-]{20,}", stripped):
        payload.append(stripped)
        continue
    if "prefill layer-major progress:" in stripped:
        continue
    payload.append(line)

text = "\n".join(payload).strip()
sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
preview = re.sub(r"\s+", " ", text)[:160]
equals_ratio = (text.count("=") / len(text)) if text else 0.0
print(sha)
print(preview)
print(f"{equals_ratio:.4f}")
PY
}

append_summary() {
    local role=$1
    local status=$2
    local log_path=$3

    local prompt_tps generation_tps prefill_tokens decode_tokens actual_chunk
    local payload_sha payload_preview equals_ratio

    prompt_tps=$(extract_last 's/.*Prompt: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    generation_tps=$(extract_last 's/.*Generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$log_path")
    prefill_tokens=$(extract_last 's/.*tokens - prefill: \([-0-9][0-9-]*\), decode: [-0-9][0-9-]*.*/\1/p' "$log_path")
    decode_tokens=$(extract_last 's/.*tokens - prefill: [-0-9][0-9-]*, decode: \([-0-9][0-9-]*\).*/\1/p' "$log_path")
    actual_chunk=$(extract_last 's/.*chunk \([0-9][0-9]*\).*/\1/p' "$log_path")

    payload_sha=""
    payload_preview=""
    equals_ratio="0"
    {
        IFS= read -r payload_sha || true
        IFS= read -r payload_preview || true
        IFS= read -r equals_ratio || true
    } < <(extract_payload_meta "$log_path")
    if python3 - "$equals_ratio" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.80 else 1)
PY
    then
        payload_preview="[mostly '=' output] ${payload_preview}"
    fi

    payload_preview=${payload_preview//\"/\"\"}
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\"%s\",%s\n" \
        "$role" "$status" "$CTX" "$MOE_PREFILL_BATCH" "$MOE_TOPK" "$MOE_SLOT_BANK" \
        "${actual_chunk:-}" "${prefill_tokens:-}" "${decode_tokens:-}" \
        "${prompt_tps:-}" "${generation_tps:-}" "${payload_sha:-}" "$payload_preview" "$log_path" \
        >> "$summary_csv"
}

run_case() {
    local role=$1
    local force_flag=$2
    local log_path="$RESULTS_DIR/${role}_ctx${CTX}_prefill${MOE_PREFILL_BATCH}.log"
    local status

    echo "comparison: $COMPARE_ID role=$role log=$log_path"

    set +e
    {
        printf "comparison: %s\n" "$COMPARE_ID"
        printf "case: %s\n" "$role"
        printf "model: %s\n" "$MODEL_PATH"
        printf "sidecar: %s\n" "$SIDECAR_PATH"
        printf "prompt: %s\n" "$PROMPT_FILE"
        printf "ctx: %s\n" "$CTX"
        printf "requested_prefill_batch: %s\n" "$MOE_PREFILL_BATCH"
        printf "force_prefill_batch: %s\n" "$force_flag"

        declare -a profile_cmd
        profile_cmd=("$script_dir/run_dsv4_flash_profile.sh")
        if [[ -n "$SAMPLE_TEMP" ]]; then
            profile_cmd+=(--temp "$SAMPLE_TEMP")
        fi
        if [[ -n "$TOP_P" ]]; then
            profile_cmd+=(--top-p "$TOP_P")
        fi
        profile_cmd+=(
            -m "$MODEL_PATH"
            --moe-sidecar "$SIDECAR_PATH"
            -f "$PROMPT_FILE"
            --moe-predict-top1-prev
            --moe-topk "$MOE_TOPK"
            --moe-prefill-layer-major
            --moe-prefill-batch "$MOE_PREFILL_BATCH"
        )
        if [[ "$force_flag" == "1" ]]; then
            profile_cmd+=(--force-moe-prefill-batch)
        fi
        profile_cmd+=(
            --moe-prefill-micro-batch "$MOE_PREFILL_MICRO_BATCH"
            --moe-prefill-io-split "$MOE_PREFILL_IO_SPLIT"
            --moe-prefill-banks "$MOE_PREFILL_BANKS"
            -n "$N_PREDICT" -st
        )

        DSV4_COMPARISON_LABEL="$COMPARE_ID" \
        DSV4_COMPARISON_ROLE="$role" \
        DISPLAY_PROMPT=0 \
        SIMPLE_IO=1 \
        RAW_COMPLETION=1 \
        MOE_TOPK="$MOE_TOPK" \
        MOE_CACHE_IO_SPLIT="$MOE_CACHE_IO_SPLIT" \
        BATCH="$BATCH" \
        UBATCH="$UBATCH" \
        CTX="$CTX" \
        SEED="$SEED" \
        TEMP="$TEMP" \
        MOE_SLOT_BANK="$MOE_SLOT_BANK" \
        LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS="${LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS:-1}" \
        bash "${profile_cmd[@]}"
    } > "$log_path" 2>&1
    status=$?
    set -e

    append_summary "$role" "$status" "$log_path"
    if (( status != 0 )); then
        echo "failed: comparison=$COMPARE_ID role=$role status=$status log=$log_path" >&2
        if [[ "$STOP_ON_FAILURE" != "0" ]]; then
            exit "$status"
        fi
    fi
}

run_case baseline_no_force 0
run_case forced_prefill 1

python3 - "$summary_csv" "$summary_txt" "$COMPARE_ID" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
txt_path = Path(sys.argv[2])
compare_id = sys.argv[3]
rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
by_role = {row["role"]: row for row in rows}
base = by_role.get("baseline_no_force")
forced = by_role.get("forced_prefill")

lines = []
lines.append(f"comparison: {compare_id}")
lines.append(f"summary_csv: {csv_path}")
for row in rows:
    lines.append(
        "case={role} status={status} chunk={actual_chunk} "
        "prefill_tps={prompt_tps} decode_tps={generation_tps} sha={payload_sha} log={log}".format(**row)
    )
    preview = row.get("payload_preview") or ""
    if preview:
        lines.append(f"  preview: {preview}")

if base and forced:
    same = base.get("payload_sha") == forced.get("payload_sha")
    lines.append(f"payload_compare: {'MATCH' if same else 'DIFFER'}")
    if not same:
        lines.append("payload_compare_note: baseline and forced generated different visible text.")

text = "\n".join(lines) + "\n"
txt_path.write_text(text, encoding="utf-8")
print(text, end="")
PY

echo "summary: $summary_txt"
