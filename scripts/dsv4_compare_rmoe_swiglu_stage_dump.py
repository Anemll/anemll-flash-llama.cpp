#!/usr/bin/env python3
"""Compare routed-MoE SwiGLU CLAMP/SILU/MUL stage dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    "gate_raw",
    "gate_clamp_pre_silu",
    "silu_out",
    "silu_clamp_post",
    "up_raw",
    "up_clamp_pre_mul",
    "mul_out",
)

VALUE_KEYS = (
    "checksum",
    "sum",
    "sumsq",
    "min",
    "max",
    "first_16_values",
    "last_16_values",
    "first_64_bytes",
)

LAYOUT_KEYS = (
    "shape",
    "parent_shape",
    "stride",
    "parent_stride",
    "slot_offset_bytes",
    "op",
    "view_src",
    "storage_offset",
    "contiguous",
    "contiguously_allocated",
)


def load(path: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "swiglu_stage_dump" not in row:
                continue
            token = int(row["token"])
            rows[token] = row
    return rows


def stage_slots(row: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    stages = row.get("swiglu_stage_dump", {})
    if stage == "silu_clamp_post" and stage not in stages:
        return stages.get("silu_out", [])
    return stages.get(stage, [])


def stage_parent(row: dict[str, Any], stage: str) -> dict[str, Any]:
    parents = row.get("swiglu_stage_parent", {})
    if stage == "silu_clamp_post" and stage not in parents:
        return parents.get("silu_out", {})
    return parents.get(stage, {})


def slot(row: dict[str, Any], stage: str, index: int) -> dict[str, Any] | None:
    slots = stage_slots(row, stage)
    if index >= len(slots):
        return None
    return slots[index]


def pick(item: dict[str, Any] | None, keys: tuple[str, ...]) -> Any:
    if item is None:
        return None
    return {key: item.get(key) for key in keys}


def value_exact(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    return pick(a, VALUE_KEYS) == pick(b, VALUE_KEYS)


def layout_exact(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    return pick(a, LAYOUT_KEYS) == pick(b, LAYOUT_KEYS)


def byte_exact(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.get("checksum") == b.get("checksum") and a.get("first_64_bytes") == b.get("first_64_bytes")


def status(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, bool]:
    return {
        "value_exact": value_exact(a, b),
        "layout_exact": layout_exact(a, b),
        "byte_exact": byte_exact(a, b),
    }


def first_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> tuple[int | None, str | None, int | None, dict[str, bool]]:
    for token in sorted(set(left) & set(right)):
        for stage in STAGES:
            for index in range(6):
                a = slot(left[token], stage, index)
                b = slot(right[token], stage, index)
                s = status(a, b)
                if not (s["value_exact"] and s["layout_exact"] and s["byte_exact"]):
                    return token, stage, index, s
    return None, None, None, {"value_exact": True, "layout_exact": True, "byte_exact": True}


def first_stage_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    stage: str,
) -> tuple[int | None, int | None, dict[str, bool]]:
    for token in sorted(set(left) & set(right)):
        for index in range(6):
            a = slot(left[token], stage, index)
            b = slot(right[token], stage, index)
            s = status(a, b)
            if not (s["value_exact"] and s["layout_exact"] and s["byte_exact"]):
                return token, index, s
    return None, None, {"value_exact": True, "layout_exact": True, "byte_exact": True}


def first_stage_value_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    stage: str,
) -> tuple[int | None, int | None]:
    for token in sorted(set(left) & set(right)):
        for index in range(6):
            if not value_exact(slot(left[token], stage, index), slot(right[token], stage, index)):
                return token, index
    return None, None


def first_value_diff(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
) -> tuple[int | None, str | None, int | None]:
    for token in sorted(set(left) & set(right)):
        for stage in STAGES:
            for index in range(6):
                if not value_exact(slot(left[token], stage, index), slot(right[token], stage, index)):
                    return token, stage, index
    return None, None, None


def describe(row: dict[str, Any], stage: str, index: int) -> str:
    item = slot(row, stage, index)
    if item is None:
        return "missing"
    keys = (
        "stage",
        "slot",
        "expert_id",
        "topk_weight",
        "op",
        "checksum",
        "shape",
        "parent_shape",
        "stride",
        "parent_stride",
        "slot_offset_bytes",
        "view_src",
        "storage_offset",
        "contiguous",
        "sum",
        "sumsq",
        "min",
        "max",
        "first_16_values",
        "first_64_bytes",
    )
    return "; ".join(f"{key}={item.get(key)}" for key in keys if key in item)


def print_pair(
    name: str,
    left_name: str,
    left: dict[int, dict[str, Any]],
    right_name: str,
    right: dict[int, dict[str, Any]],
) -> None:
    print(f"{name}:")
    print(f"  common tokens: {len(set(left) & set(right))}")
    token, stage, index, s = first_diff(left, right)
    if token is None:
        print("  first differing token: none")
        print("  first differing stage: none")
        print("  exact: yes")
    else:
        print(f"  first differing token: {token}")
        print(f"  first differing slot: {index}")
        print(f"  first differing stage: {stage}")
        print(f"  value_exact: {int(s['value_exact'])}")
        print(f"  layout_exact: {int(s['layout_exact'])}")
        print(f"  byte_exact: {int(s['byte_exact'])}")
    v_token, v_stage, v_index = first_value_diff(left, right)
    print(f"  first value-differing token: {v_token if v_token is not None else 'none'}")
    print(f"  first value-differing stage: {v_stage if v_stage is not None else 'none'}")
    print(f"  first value-differing slot: {v_index if v_index is not None else 'none'}")

    for st in STAGES:
        st_token, st_slot, st_status = first_stage_diff(left, right, st)
        val_token, val_slot = first_stage_value_diff(left, right, st)
        print(
            f"  {st}: token={st_token if st_token is not None else 'none'} "
            f"slot={st_slot if st_slot is not None else 'none'} "
            f"value_token={val_token if val_token is not None else 'none'} "
            f"value_slot={val_slot if val_slot is not None else 'none'} "
            f"value_exact={int(st_status['value_exact'])} "
            f"layout_exact={int(st_status['layout_exact'])} "
            f"byte_exact={int(st_status['byte_exact'])}"
        )

    if token is not None and stage is not None and index is not None:
        print(f"  {left_name} topk_ids: {left[token].get('topk_ids')}")
        print(f"  {right_name} topk_ids: {right[token].get('topk_ids')}")
        print(f"  {left_name} topk_weights: {left[token].get('topk_weights')}")
        print(f"  {right_name} topk_weights: {right[token].get('topk_weights')}")
        print(f"  {left_name} parent: {stage_parent(left[token], stage)}")
        print(f"  {right_name} parent: {stage_parent(right[token], stage)}")
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

    print(f"baseline rows with swiglu stage dump: {len(baseline)}")
    print(f"shadow rows with swiglu stage dump: {len(shadow)}")
    print(f"replace rows with swiglu stage dump: {len(replace)}")
    print(f"common baseline/shadow/replace tokens: {len(set(baseline) & set(shadow) & set(replace))}")
    print("")

    print_pair("baseline_vs_shadow", "baseline", baseline, "shadow", shadow)
    print("")
    print_pair("baseline_vs_replace", "baseline", baseline, "replace", replace)
    print("")
    print_pair("shadow_vs_replace", "shadow", shadow, "replace", replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
