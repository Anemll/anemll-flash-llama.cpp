#!/usr/bin/env python3
"""Analyze HC_PRE_NORM layer-executor payload fixture semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def payload_path(fixture: Path, record: dict) -> Path | None:
    payload_file = record.get("payload_file") or ""
    if not payload_file:
        return None
    path = Path(payload_file)
    return path if path.is_absolute() else fixture.parent / path


def payload_info(fixture: Path, record: dict) -> dict:
    info = {
        "checksum": record.get("payload_checksum", ""),
        "bytes": record.get("payload_bytes", 0),
        "numel": record.get("payload_numel", 0),
        "first_8": record.get("first_8_values", []),
        "last_8": record.get("last_8_values", []),
    }
    path = payload_path(fixture, record)
    if path is None or not path.exists():
        return info
    data = path.read_bytes()
    info["checksum"] = hashlib.sha256(data).hexdigest()
    info["bytes"] = len(data)
    if record.get("dtype") == "f32" and len(data) % 4 == 0:
        values = struct.unpack(f"<{len(data)//4}f", data)
        info["numel"] = len(values)
        info["first_8"] = [float(v) for v in values[:8]]
        info["last_8"] = [float(v) for v in values[-8:]]
    else:
        info["numel"] = len(data)
        info["first_8"] = [int(v) for v in data[:8]]
        info["last_8"] = [int(v) for v in data[-8:]]
    return info


def find(records: list[dict], *names: str) -> dict | None:
    for name in names:
        for record in records:
            if record.get("tensor") == name or record.get("semantic_label") == name:
                return record
    return None


def same(fixture: Path, a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return False
    return payload_info(fixture, a).get("checksum") == payload_info(fixture, b).get("checksum")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()

    records = [r for r in load_records(args.fixture) if r.get("stage") == "hc_pre_norm"]
    by_checksum: dict[str, list[str]] = defaultdict(list)

    print("dsv4_hcnorm_payload_semantics:")
    print(f"  fixture={args.fixture}")
    print(f"  records={len(records)}")

    print("  semantic_label_table:")
    for record in records:
        label = record.get("semantic_label") or record.get("tensor", "")
        info = payload_info(args.fixture, record)
        checksum = info.get("checksum", "")
        if checksum:
            by_checksum[checksum].append(label)
        print(
            "    "
            f"label={label} tensor={record.get('tensor','')} availability={record.get('availability','')} "
            f"capture_mode={record.get('metadata', {}).get('capture_mode', '')} "
            f"payload_kind={record.get('payload_kind','')} dtype={record.get('dtype','')} "
            f"shape={record.get('shape', [])} stride={record.get('stride', [])} "
            f"view_src={record.get('view_src', '')} storage_offset={record.get('storage_offset', '')} "
            f"payload_file={record.get('payload_file', '')} payload_numel={info.get('numel', 0)} "
            f"payload_bytes={info.get('bytes', 0)} checksum={checksum[:16]} "
            f"first_8={info.get('first_8', [])} last_8={info.get('last_8', [])} "
            f"same_payload_as={record.get('same_payload_as', '')} "
            f"same_tensor_ptr_as={record.get('same_tensor_ptr_as', '')} "
            f"same_view_src_as={record.get('same_view_src_as', '')} "
            f"same_storage_offset_as={record.get('same_storage_offset_as', '')}"
        )

    print("  alias_groups:")
    for checksum, labels in sorted(by_checksum.items()):
        if len(labels) > 1:
            print(f"    checksum={checksum[:16]} labels={','.join(labels)}")

    ref_cur = find(records, "hc_pre_weighted_cur_reference", "reference_cur")
    ref_norm = find(records, "hc_pre_norm_reference", "reference_norm")
    ref_post = find(records, "hc_pre_post_reference", "reference_post")
    main = find(records, "hc_pre_norm", "summary")
    input_hc = find(records, "hc_pre_input_hc_original", "input_hc_original_residual")
    split_pre = find(records, "hc_pre_split_pre", "split_pre")
    norm_weight = find(records, "hc_pre_norm_weight", "norm_weight")

    print(f"  reference_cur == reference_norm? {1 if same(args.fixture, ref_cur, ref_norm) else 0}")
    print(f"  reference_cur == reference_post? {1 if same(args.fixture, ref_cur, ref_post) else 0}")
    print(f"  reference_norm == reference_post? {1 if same(args.fixture, ref_norm, ref_post) else 0}")
    print(f"  main_hc_pre_norm == reference_cur? {1 if same(args.fixture, main, ref_cur) else 0}")
    print(f"  input_hc == reference_cur? {1 if same(args.fixture, input_hc, ref_cur) else 0}")
    print(f"  split_pre shape/layout: shape={split_pre.get('shape', []) if split_pre else []} stride={split_pre.get('stride', []) if split_pre else []}")
    print(f"  norm_weight shape/layout: shape={norm_weight.get('shape', []) if norm_weight else []} stride={norm_weight.get('stride', []) if norm_weight else []}")
    available = all(r is not None and r.get("availability") == "available" for r in [input_hc, split_pre, norm_weight, ref_cur, ref_norm, ref_post])
    print(f"  candidate_formula_inputs_available={1 if available else 0}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
