#!/usr/bin/env python3
"""Compare routed-MoE gate/up byte-level dumps across baseline/shadow/replace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = ("gate", "up")
STAT_KEYS = ("sum", "sumsq", "min", "max", "first_16_values", "last_16_values")


def load(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            token = int(row["token"])
            if "gate_up_byte_dump" in row:
                rows[token] = row
    return rows


def slot(row: dict[str, Any], stage: str, index: int) -> dict[str, Any] | None:
    slots = row.get("gate_up_byte_dump", {}).get(stage, [])
    if index >= len(slots):
        return None
    return slots[index]


def checksum_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.get("checksum") == b.get("checksum") and a.get("first_64_bytes") == b.get("first_64_bytes")


def stat_equal(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return {k: a.get(k) for k in STAT_KEYS} == {k: b.get(k) for k in STAT_KEYS}


def pair_status(a: dict[str, Any] | None, b: dict[str, Any] | None) -> str:
    if checksum_equal(a, b) and stat_equal(a, b):
        return "exact"
    if stat_equal(a, b):
        return "checksum_only"
    return "stats_differ"


def first_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> tuple[int | None, str | None, int | None, str]:
    for token in sorted(set(left) & set(right)):
        for stage in STAGES:
            for index in range(6):
                a = slot(left[token], stage, index)
                b = slot(right[token], stage, index)
                status = pair_status(a, b)
                if status != "exact":
                    return token, stage, index, status
    return None, None, None, "exact"


def topk_summary(row: dict[str, Any] | None) -> tuple[Any, Any]:
    if row is None:
        return None, None
    return row.get("topk_ids"), row.get("topk_weights")


def describe_slot(row: dict[str, Any], stage: str, index: int) -> str:
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
    return "; ".join(f"{k}={item.get(k)}" for k in keys if k in item)


def print_pair(name: str, left_name: str, left: dict[int, dict[str, Any]], right_name: str, right: dict[int, dict[str, Any]]) -> None:
    token, stage, index, status = first_diff(left, right)
    print(f"{name}:")
    if token is None:
        print("  first differing token: none")
        print("  exact: yes")
        return
    print(f"  first differing token: {token}")
    print(f"  first differing stage: {stage}")
    print(f"  first differing slot: {index}")
    print(f"  status: {status}")
    left_ids, left_weights = topk_summary(left.get(token))
    right_ids, right_weights = topk_summary(right.get(token))
    print(f"  {left_name} topk_ids: {left_ids}")
    print(f"  {right_name} topk_ids: {right_ids}")
    print(f"  {left_name} topk_weights: {left_weights}")
    print(f"  {right_name} topk_weights: {right_weights}")
    assert stage is not None and index is not None
    print(f"  {left_name} slot: {describe_slot(left[token], stage, index)}")
    print(f"  {right_name} slot: {describe_slot(right[token], stage, index)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--replace", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    shadow = load(args.shadow)
    replace = load(args.replace)

    print(f"baseline rows with gate/up byte dump: {len(baseline)}")
    print(f"shadow rows with gate/up byte dump: {len(shadow)}")
    print(f"replace rows with gate/up byte dump: {len(replace)}")
    print(f"common baseline/shadow/replace tokens: {len(set(baseline) & set(shadow) & set(replace))}")
    print("")

    print_pair("baseline_vs_shadow", "baseline", baseline, "shadow", shadow)
    print("")
    print_pair("baseline_vs_replace", "baseline", baseline, "replace", replace)
    print("")
    print_pair("shadow_vs_replace", "shadow", shadow, "replace", replace)

    token, stage, index, status = first_diff(shadow, replace)
    print("")
    print("summary:")
    print(f"  first differing token: {token if token is not None else 'none'}")
    print(f"  first differing stage: {stage if stage is not None else 'none'}")
    print(f"  first differing slot: {index if index is not None else 'none'}")
    print(f"  shadow_vs_replace status: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
