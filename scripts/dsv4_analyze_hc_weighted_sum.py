#!/usr/bin/env python3
"""Analyze DSV4 HC_PRE_NORM weighted-sum fixture payloads."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path


LABELS = [
    "hc_ws_input_inpL_raw",
    "hc_ws_input_inpL_view",
    "hc_ws_input_inpL_contiguous",
    "hc_ws_split_full_raw",
    "hc_ws_split_pre_raw",
    "hc_ws_split_pre_view",
    "hc_ws_split_pre_contiguous",
    "hc_ws_reference_cur",
    "hc_ws_reference_cur_pre_reshape",
    "hc_ws_reference_cur_post_reshape",
    "hc_pre_input_hc_original",
    "hc_pre_split_pre",
    "hc_pre_split_post",
    "hc_pre_split_comb",
    "hc_pre_weighted_cur_reference",
]


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def by_label(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        label = record.get("semantic_label") or record.get("tensor")
        if label:
            out[label] = record
    return out


def payload_path(fixture: Path, record: dict) -> Path | None:
    payload_file = record.get("payload_file") or ""
    if not payload_file:
        return None
    path = Path(payload_file)
    if path.is_absolute():
        return path
    return fixture.parent / path


def payload_f32(fixture: Path, record: dict) -> list[float]:
    path = payload_path(fixture, record)
    if path is None or not path.exists():
        return []
    data = path.read_bytes()
    if len(data) % 4 != 0:
        return []
    return list(struct.unpack(f"<{len(data)//4}f", data))


def checksum(fixture: Path, record: dict) -> str:
    path = payload_path(fixture, record)
    if path is None or not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt_values(values: list[float], n: int = 16) -> str:
    return "[" + ",".join(f"{v:.9g}" for v in values[:n]) + "]"


def row_col_sums(weights: list[float], n_hc: int = 4) -> tuple[list[float], list[float]]:
    if len(weights) < n_hc:
        return [], []
    if len(weights) == n_hc:
        return [sum(weights)], weights[:]
    rows = []
    cols = [0.0] * n_hc
    n_rows = len(weights) // n_hc
    for r in range(n_rows):
        row = weights[r*n_hc:(r + 1)*n_hc]
        rows.append(sum(row))
        for c, v in enumerate(row):
            cols[c] += v
    return rows, cols


def looks_like_weights(values: list[float]) -> bool:
    return bool(values) and all(math.isfinite(v) and -1e-5 <= v <= 2.1 for v in values)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <hc_pre_norm_l0_n16.jsonl>", file=sys.stderr)
        return 2

    fixture = Path(sys.argv[1])
    records = load_records(fixture)
    labels = by_label(records)

    print("dsv4_hc_weighted_sum_analysis:")
    print("  payload_checksum_table:")
    checksum_groups: dict[str, list[str]] = {}
    for label in LABELS:
        record = labels.get(label)
        if record is None:
            print(f"    {label}: missing")
            continue
        values = payload_f32(fixture, record)
        csum = checksum(fixture, record)
        checksum_groups.setdefault(csum, []).append(label)
        print(
            f"    {label}: availability={record.get('availability')} payload_kind={record.get('payload_kind')} "
            f"shape={record.get('shape')} stride={record.get('stride')} view_src={record.get('view_src')} "
            f"storage_offset={record.get('storage_offset')} bytes={record.get('payload_bytes', 0)} "
            f"numel={len(values)} checksum={csum[:16]}"
        )

    print("  checksum_groups:")
    for csum, group in sorted(checksum_groups.items(), key=lambda item: item[0]):
        if csum and len(group) > 1:
            print(f"    {csum[:16]}: {','.join(group)}")

    split = payload_f32(fixture, labels.get("hc_ws_split_pre_raw") or labels.get("hc_pre_split_pre") or {})
    split_view = payload_f32(fixture, labels.get("hc_ws_split_pre_view") or {})
    split_full = payload_f32(fixture, labels.get("hc_ws_split_full_raw") or {})
    input_hc = payload_f32(fixture, labels.get("hc_ws_input_inpL_raw") or labels.get("hc_pre_input_hc_original") or {})
    ref_cur = payload_f32(fixture, labels.get("hc_ws_reference_cur") or labels.get("hc_pre_weighted_cur_reference") or {})

    print("  candidate_layouts:")
    print("    source_contract: x[d*nb0 + h*nb1 + t*nb2], weights[h*nb0 + t*nb1]")
    print("    contiguous_expected_input_layout: hidden_major index=h*n_embd+e")
    print("    contiguous_expected_split_layout: pre[h + token*n_hc]")
    print("    expected_output_shape: [4096,1]")

    rows, cols = row_col_sums(split)
    print(f"  split_pre: values={fmt_values(split)} row_sums={fmt_values(rows)} col_sums={fmt_values(cols)}")
    rows_full, cols_full = row_col_sums(split_full)
    print(f"  split_full: first_16={fmt_values(split_full)} row_sums_first_rows={fmt_values(rows_full[:4])} col_sums={fmt_values(cols_full)}")
    print(f"  split_pre_view_equals_raw={1 if split and split == split_view else 0}")
    print(f"  split_pre_rows_sum_to_1={1 if any(abs(v - 1.0) < 1e-3 for v in rows) else 0}")
    print(f"  split_pre_columns_sum_to_1={1 if any(abs(v - 1.0) < 1e-3 for v in cols) else 0}")
    print(f"  split_pre_values_look_like_mixture_weights={1 if looks_like_weights(split) else 0}")
    print(f"  input_hc_payload_numel={len(input_hc)} expected_numel=16384")
    print(f"  input_hc_matches_expected_n_embd_x_n_hc={1 if len(input_hc) == 4096*4 else 0}")
    print(f"  first_token_input_hc_first_16={fmt_values(input_hc)}")
    if len(input_hc) >= 4096*4:
        print(f"  input_hc_hidden0_first_16={fmt_values(input_hc[:16])}")
        print(f"  input_hc_hidden1_first_16={fmt_values(input_hc[4096:4096+16])}")
    print(f"  reference_cur_first_16={fmt_values(ref_cur)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
