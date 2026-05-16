#!/usr/bin/env python3
"""Compare routed-MoE SwiGLU byte/value dumps across baseline/shadow/replace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = ("gate", "up", "swiglu")
VALUE_KEYS = ("checksum", "sum", "sumsq", "min", "max", "first_16_values", "last_16_values")


def load(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "swiglu_byte_dump" in row:
                rows[int(row["token"])] = row
    return rows


def slot(row: dict[str, Any], stage: str, index: int) -> dict[str, Any] | None:
    slots = row.get("swiglu_byte_dump", {}).get(stage, [])
    if index >= len(slots):
        return None
    return slots[index]


def slot_value(row: dict[str, Any], stage: str, index: int) -> Any:
    item = slot(row, stage, index)
    if item is None:
        return None
    return {key: item.get(key) for key in VALUE_KEYS}


def first_stage_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    stage: str,
) -> tuple[int | None, int | None]:
    for token in sorted(set(left) & set(right)):
        for index in range(6):
            if slot_value(left[token], stage, index) != slot_value(right[token], stage, index):
                return token, index
    return None, None


def first_any_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> tuple[int | None, str | None, int | None]:
    for token in sorted(set(left) & set(right)):
        for stage in STAGES:
            for index in range(6):
                if slot_value(left[token], stage, index) != slot_value(right[token], stage, index):
                    return token, stage, index
    return None, None, None


def describe(row: dict[str, Any], stage: str, index: int) -> str:
    item = slot(row, stage, index)
    if item is None:
        return "missing"
    keys = [
        "expert_id",
        "topk_weight",
        "checksum",
        "shape",
        "parent_shape",
        "parent_stride",
        "slot_offset_bytes",
        "sum",
        "sumsq",
        "min",
        "max",
        "first_16_values",
        "first_64_bytes",
    ]
    return "; ".join(f"{key}={item.get(key)}" for key in keys if key in item)


def print_stage_pair(
    name: str,
    left_name: str,
    left: dict[int, dict[str, Any]],
    right_name: str,
    right: dict[int, dict[str, Any]],
) -> None:
    print(f"{name}:")
    token, stage, index = first_any_diff(left, right)
    if token is None:
        print("  first differing token: none")
        print("  exact: yes")
        return
    print(f"  first differing token: {token}")
    print(f"  first differing stage: {stage}")
    print(f"  first differing slot: {index}")
    for s in STAGES:
        stoken, sslot = first_stage_diff(left, right, s)
        print(f"  first {s} mismatch: token={stoken if stoken is not None else 'none'} slot={sslot if sslot is not None else 'none'}")
    assert stage is not None and index is not None
    print(f"  {left_name} topk_ids: {left[token].get('topk_ids')}")
    print(f"  {right_name} topk_ids: {right[token].get('topk_ids')}")
    print(f"  {left_name} topk_weights: {left[token].get('topk_weights')}")
    print(f"  {right_name} topk_weights: {right[token].get('topk_weights')}")
    print(f"  {left_name} swiglu_parent: {left[token].get('swiglu_parent')}")
    print(f"  {right_name} swiglu_parent: {right[token].get('swiglu_parent')}")
    print(f"  {left_name} slot: {describe(left[token], stage, index)}")
    print(f"  {right_name} slot: {describe(right[token], stage, index)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--replace", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    shadow = load(args.shadow)
    replace = load(args.replace)

    print(f"baseline rows with swiglu byte dump: {len(baseline)}")
    print(f"shadow rows with swiglu byte dump: {len(shadow)}")
    print(f"replace rows with swiglu byte dump: {len(replace)}")
    print(f"common baseline/shadow/replace tokens: {len(set(baseline) & set(shadow) & set(replace))}")
    print("")
    print_stage_pair("baseline_vs_shadow", "baseline", baseline, "shadow", shadow)
    print("")
    print_stage_pair("baseline_vs_replace", "baseline", baseline, "replace", replace)
    print("")
    print_stage_pair("shadow_vs_replace", "shadow", shadow, "replace", replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
