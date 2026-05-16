#!/usr/bin/env python3
"""Analyze routed-MoE layer-executor fixture payload semantics."""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path


def payload_path(fixture: Path, rec: dict) -> Path | None:
    payload_file = rec.get("payload_file") or ""
    if not payload_file:
        return None
    path = Path(payload_file)
    if path.is_absolute():
        return path
    local = fixture.parent / path
    if local.exists():
        return local
    return fixture.parent.parent / path


def read_values(fixture: Path, rec: dict) -> list[float] | list[int]:
    path = payload_path(fixture, rec)
    if path is None or not path.exists():
        return []
    data = path.read_bytes()
    dtype = rec.get("dtype", "")
    if dtype in ("f32", "fp32") and len(data) % 4 == 0:
        return list(struct.unpack(f"<{len(data)//4}f", data))
    if dtype in ("i32", "int32") and len(data) % 4 == 0:
        return list(struct.unpack(f"<{len(data)//4}i", data))
    return list(data)


def same_payload_groups(records: list[dict]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        checksum = rec.get("payload_checksum") or ""
        label = rec.get("semantic_label") or rec.get("tensor") or ""
        if checksum:
            groups[checksum].append(label)
    return [labels for labels in groups.values() if len(labels) > 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.fixture.read_text().splitlines() if line.strip()]
    by_label = {rec.get("semantic_label") or rec.get("tensor"): rec for rec in records}

    labels = [
        "rmoe_ffn_input",
        "rmoe_topk_ids",
        "rmoe_topk_weights",
        "rmoe_expert_gate",
        "rmoe_expert_up",
        "rmoe_expert_swiglu",
        "rmoe_expert_down",
        "rmoe_routed_sum",
        "rmoe_shared_gate",
        "rmoe_shared_up",
        "rmoe_shared_swiglu",
        "rmoe_shared_down",
        "rmoe_final_ffn_reference",
    ]

    print("dsv4_rmoe_payload_semantics:")
    print("stage_availability:")
    for label in labels:
        rec = by_label.get(label)
        if rec is None:
            print(f"  {label}=missing")
            continue
        payload = rec.get("payload_kind", "unknown")
        availability = rec.get("availability", "unknown")
        shape = rec.get("shape", [])
        stride = rec.get("stride", [])
        print(f"  {label}={availability} payload={payload} shape={shape} stride={stride}")

    topk_ids = read_values(args.fixture, by_label.get("rmoe_topk_ids", {}))
    topk_weights = read_values(args.fixture, by_label.get("rmoe_topk_weights", {}))
    print(f"topk_ids={topk_ids[:6] if topk_ids else []}")
    print(f"topk_weights={topk_weights[:6] if topk_weights else []}")
    print(f"topk_weights_sum={sum(float(v) for v in topk_weights[:6]) if topk_weights else 0.0}")
    print(f"expert_slot_order={topk_ids[:6] if topk_ids else []}")

    print("alias_groups:")
    groups = same_payload_groups(records)
    if groups:
        for group in groups:
            print("  " + ",".join(group))
    else:
        print("  none")

    routed = by_label.get("rmoe_routed_sum")
    shared = by_label.get("rmoe_shared_down")
    final = by_label.get("rmoe_final_ffn_reference")
    print(f"routed_sum == final_ffn? {1 if routed and final and routed.get('payload_checksum') == final.get('payload_checksum') else 0}")
    print(f"shared_down == final_ffn? {1 if shared and final and shared.get('payload_checksum') == final.get('payload_checksum') else 0}")
    print(f"weighted_down availability={by_label.get('rmoe_expert_down', {}).get('availability', 'missing')}")
    print(f"shared_down availability={by_label.get('rmoe_shared_down', {}).get('availability', 'missing')}")
    print(f"final reference availability={by_label.get('rmoe_final_ffn_reference', {}).get('availability', 'missing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
