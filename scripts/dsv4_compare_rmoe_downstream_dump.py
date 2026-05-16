#!/usr/bin/env python3
"""Compare routed-MoE replacement downstream JSONL dumps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


STAGES = [
    "ffn_input",
    "topk_ids",
    "topk_weights",
    "final_ffn",
    "hc_post_input",
    "hc_post_output",
    "layer_output_after_ffn",
    "next_layer_input",
    "next_layer_attn_norm_input",
    "next_layer_ffn_input",
    "result_hc",
    "logits_input",
]

STAT_KEYS = ["shape", "dtype", "stride", "op", "sum", "sumsq", "min", "max", "first_8_values", "last_8_values"]


def load(path: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["token"])] = row
    return out


def stage_stats(row: dict[str, Any], stage: str) -> Any:
    if stage == "ffn_input":
        return row.get("ffn_input")
    if stage == "topk_ids":
        return row.get("topk_ids")
    if stage == "topk_weights":
        return row.get("topk_weights")
    if stage == "final_ffn":
        return row.get("stages", {}).get("final_ffn") or row.get("final")
    if stage == "hc_post_input":
        return row.get("downstream", {}).get("hc_post_input") or row.get("hc_ffn_post_input")
    if stage == "hc_post_output":
        return row.get("downstream", {}).get("hc_post_output") or row.get("hc_ffn_post_output")
    return row.get("downstream", {}).get(stage)


def stage_value(row: dict[str, Any], stage: str) -> Any:
    stats = stage_stats(row, stage)
    if isinstance(stats, dict):
        return stats.get("hash")
    return stats


def stage_stat_value(row: dict[str, Any], stage: str) -> Any:
    stats = stage_stats(row, stage)
    if isinstance(stats, dict):
        return {key: stats.get(key) for key in STAT_KEYS if key in stats}
    return stats


def summarize(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("tensor_name", "op", "hash", "shape", "dtype", "stride", "sum", "sumsq", "min", "max", "first_8_values"):
            if key in value:
                parts.append(f"{key}={value[key]}")
        return "; ".join(parts)
    return repr(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dsv4_compare_rmoe_downstream_dump.py BASELINE.jsonl REPLACE.jsonl", file=sys.stderr)
        return 2

    baseline = load(sys.argv[1])
    replace = load(sys.argv[2])
    common = sorted(set(baseline) & set(replace))

    print(f"baseline rows: {len(baseline)}")
    print(f"replace rows: {len(replace)}")
    print(f"common tokens: {len(common)}")

    first_token: int | None = None
    first_stage: str | None = None
    first_stat_token: int | None = None
    first_stat_stage: str | None = None

    for token in common:
        for stage in STAGES:
            av = stage_value(baseline[token], stage)
            bv = stage_value(replace[token], stage)
            if av != bv:
                first_token = token
                first_stage = stage
                break
        if first_token is not None:
            break

    for token in common:
        for stage in STAGES:
            av = stage_stat_value(baseline[token], stage)
            bv = stage_stat_value(replace[token], stage)
            if av != bv:
                first_stat_token = token
                first_stat_stage = stage
                break
        if first_stat_token is not None:
            break

    if first_token is None:
        print("first differing token: none")
        print("first differing stage: none")
    else:
        print(f"first differing token: {first_token}")
        print("first differing layer: 0")
        print(f"first differing stage: {first_stage}")
        print(f"first differing tensor: {first_stage}")

    if first_stat_token is None:
        print("first differing token by stats: none")
        print("first differing stage by stats: none")
    else:
        print(f"first differing token by stats: {first_stat_token}")
        print(f"first differing stage by stats: {first_stat_stage}")

    print("")
    print("stage summary:")
    for stage in STAGES:
        mismatch_token = None
        for token in common:
            if stage_value(baseline[token], stage) != stage_value(replace[token], stage):
                mismatch_token = token
                break
        if mismatch_token is None:
            print(f"- {stage}: exact")
            continue
        print(f"- {stage}: mismatch token {mismatch_token}")
        print(f"  baseline: {summarize(stage_stats(baseline[mismatch_token], stage))}")
        print(f"  replace:  {summarize(stage_stats(replace[mismatch_token], stage))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
