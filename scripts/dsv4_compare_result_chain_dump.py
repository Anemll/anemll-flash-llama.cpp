#!/usr/bin/env python3
"""Compare routed-MoE result-chain JSONL dumps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


STAGES = [
    "layer0_next_input",
    "layer1_input",
    "layer1_after_attn",
    "layer1_after_ffn",
    "last_layer_input",
    "last_layer_output",
    "result_hc",
    "result_norm",
    "logits_input",
]

STAGE_LAYER = {
    "layer0_next_input": 0,
    "layer1_input": 1,
    "layer1_after_attn": 1,
    "layer1_after_ffn": 1,
    "last_layer_input": 42,
    "last_layer_output": 42,
    "result_hc": -1,
    "result_norm": -1,
    "logits_input": -1,
}

STAT_KEYS = [
    "tensor_name",
    "op",
    "shape",
    "dtype",
    "stride",
    "view_src",
    "storage_offset",
    "src0",
    "src1",
    "sum",
    "sumsq",
    "min",
    "max",
    "first_8_values",
    "last_8_values",
]


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
    chain = row.get("result_chain", {})
    downstream = row.get("downstream", {})
    if stage in chain:
        return chain[stage]
    if stage == "layer0_next_input":
        return downstream.get("next_layer_input")
    if stage == "result_hc":
        return downstream.get("result_hc")
    if stage in ("result_norm", "logits_input"):
        return downstream.get("logits_input")
    return None


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
        for key in (
            "tensor_name",
            "op",
            "hash",
            "shape",
            "dtype",
            "stride",
            "view_src",
            "src0",
            "src1",
            "sum",
            "sumsq",
            "min",
            "max",
            "first_8_values",
        ):
            if key in value:
                parts.append(f"{key}={value[key]}")
        return "; ".join(parts)
    return repr(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dsv4_compare_result_chain_dump.py BASELINE.jsonl REPLACE.jsonl", file=sys.stderr)
        return 2

    baseline = load(sys.argv[1])
    replace = load(sys.argv[2])
    common = sorted(set(baseline) & set(replace))

    print(f"baseline rows: {len(baseline)}")
    print(f"replace rows: {len(replace)}")
    print(f"common tokens: {len(common)}")
    print(f"token alignment confirmed: {'yes' if common and len(common) == len(baseline) == len(replace) else 'no'}")

    first_token: int | None = None
    first_stage: str | None = None
    for token in common:
        for stage in STAGES:
            if stage_value(baseline[token], stage) != stage_value(replace[token], stage):
                first_token = token
                first_stage = stage
                break
        if first_token is not None:
            break

    first_stat_token: int | None = None
    first_stat_stage: str | None = None
    for token in common:
        for stage in STAGES:
            if stage_stat_value(baseline[token], stage) != stage_stat_value(replace[token], stage):
                first_stat_token = token
                first_stat_stage = stage
                break
        if first_stat_token is not None:
            break

    if first_token is None:
        print("first differing token: none")
        print("first differing layer: none")
        print("first differing stage: none")
        print("first differing tensor: none")
    else:
        print(f"first differing token: {first_token}")
        print(f"first differing layer: {STAGE_LAYER.get(first_stage or '', 'unknown')}")
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
