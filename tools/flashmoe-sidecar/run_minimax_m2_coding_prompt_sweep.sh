#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 PACKAGE_DIR [run_minimax_m2_flash.sh args...]" >&2
    exit 1
fi

package_dir=$1
shift

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
prompt_dir=${PROMPT_DIR:-"$script_dir/prompts/coding"}
prompt_labels=${PROMPT_LABELS:-"1k 4k 16k 22k"}

if [[ ! -d "$prompt_dir" ]]; then
    echo "missing prompt directory: $prompt_dir" >&2
    echo "generate samples first with: python3 $script_dir/make_coding_prompts.py" >&2
    exit 1
fi

for label in $prompt_labels; do
    prompt_file="$prompt_dir/coding_${label}.txt"
    if [[ ! -f "$prompt_file" ]]; then
        echo "missing prompt file: $prompt_file" >&2
        exit 1
    fi

    echo
    echo "===== coding_${label} ====="
    if [[ $# -eq 0 ]]; then
        bash "$script_dir/run_minimax_m2_flash.sh" \
            "$package_dir" \
            -f "$prompt_file" \
            -n "${N_PREDICT:-1}" \
            -st
    else
        bash "$script_dir/run_minimax_m2_flash.sh" \
            "$package_dir" \
            -f "$prompt_file" \
            "$@"
    fi
done
