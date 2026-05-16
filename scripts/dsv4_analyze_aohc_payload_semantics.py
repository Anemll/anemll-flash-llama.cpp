#!/usr/bin/env python3
"""Inspect AOHC / DSV4_HC_EXPAND fixture payload semantics."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


PAIRS = [
    ("src0", "aohc_hcexpand_src0_block", "aohc_hcexpand_dispatch_src0"),
    ("src1", "aohc_hcexpand_src1_residual", "aohc_hcexpand_dispatch_src1"),
    ("src2", "aohc_hcexpand_src2_post", "aohc_hcexpand_dispatch_src2"),
    ("src3", "aohc_hcexpand_src3_comb", "aohc_hcexpand_dispatch_src3"),
    ("output", "aohc_after_attn_hc_reference", "aohc_hcexpand_dispatch_output"),
]


def load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tensor = rec.get("tensor", "")
        if tensor and rec.get("availability") == "available":
            records[tensor] = rec
    return records


def payload_path(fixture: Path, rec: dict) -> Path | None:
    source = rec.get("metadata", {}).get("payload_source", "")
    if source:
        p = Path(source)
        if p.exists():
            return p
    payload_file = rec.get("payload_file", "")
    if not payload_file:
        return None
    p = Path(payload_file)
    if p.is_absolute() and p.exists():
        return p
    for base in (fixture.parent, fixture.parent.parent, Path("/tmp")):
        candidate = base / p
        if candidate.exists():
            return candidate
    return None


def read_bytes(fixture: Path, rec: dict) -> bytes:
    p = payload_path(fixture, rec)
    return p.read_bytes() if p is not None else b""


def first_values(data: bytes, count: int = 16) -> list[float]:
    n = min(count, len(data) // 4)
    if n <= 0:
        return []
    return [float(v) for v in struct.unpack(f"<{n}f", data[: n * 4])]


def first_diff(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def describe(rec: dict | None) -> str:
    if rec is None:
        return "missing"
    md = rec.get("metadata", {})
    return (
        f"shape={rec.get('shape', [])} stride={rec.get('stride', [])} "
        f"op={rec.get('producer', {}).get('op', '')} "
        f"view_src={rec.get('view_src', md.get('view_src', ''))} "
        f"view_offs={rec.get('storage_offset', md.get('storage_offset', 0))} "
        f"checksum={rec.get('payload_checksum', '')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    args = ap.parse_args()

    records = load_records(args.fixture)
    print("dsv4_aohc_payload_semantics:")
    print("payload checksum table:")
    for name in [
        "aohc_hcexpand_src0_block",
        "aohc_hcexpand_dispatch_src0",
        "aohc_hcexpand_src1_residual",
        "aohc_hcexpand_dispatch_src1",
        "aohc_hcexpand_src2_post",
        "aohc_hcexpand_dispatch_src2",
        "aohc_hcexpand_src3_comb",
        "aohc_hcexpand_dispatch_src3",
        "aohc_after_attn_hc_reference",
        "aohc_hcexpand_dispatch_output",
    ]:
        print(f"  {name}: {describe(records.get(name))}")

    first_differing = "none"
    for label, post_name, dispatch_name in PAIRS:
        post = records.get(post_name)
        dispatch = records.get(dispatch_name)
        post_data = read_bytes(args.fixture, post) if post else b""
        dispatch_data = read_bytes(args.fixture, dispatch) if dispatch else b""
        equal = bool(post_data) and post_data == dispatch_data
        if not equal and first_differing == "none":
            first_differing = label
        diff = first_diff(post_data, dispatch_data) if post_data and dispatch_data else -1
        print(f"{label} post_eval == dispatch_time? {1 if equal else 0}")
        print(f"  {label}_first_diff_byte={diff}")
        print(f"  {label}_post_eval_first_16={first_values(post_data)}")
        print(f"  {label}_dispatch_time_first_16={first_values(dispatch_data)}")

    print(f"which src first differs: {first_differing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
