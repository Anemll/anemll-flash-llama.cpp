#!/usr/bin/env python3
"""Summarize compressor/update layer-executor fixture payload semantics."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


RECOMPUTE_INPUTS = {
    "compressed_norm",
    "compressed_norm_weight",
    "compressed_norm_weighted",
}

ROPE_INPUTS = {
    "cupd_rope_input",
    "cupd_rope_reference",
    "cupd_rope_position",
    "cupd_rope_n_rot",
}

ROPE_METADATA_LABELS = {
    "cupd_rope_op_params",
    "cupd_rope_position",
    "cupd_rope_cache_position",
    "cupd_rope_n_rot",
    "cupd_rope_width",
    "cupd_rope_freq_base",
    "cupd_rope_freq_scale",
    "cupd_rope_mode",
    "cupd_rope_tail_offset",
    "cupd_rope_metadata",
}


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <compressor_update.jsonl>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    records = load_records(path)
    stage_records = [r for r in records if r.get("stage") == "compressor_update"]
    by_label = {r.get("semantic_label") or r.get("tensor"): r for r in stage_records}

    print("dsv4_cupd_payload_semantics:")
    print(f"  fixture={path}")
    print(f"  records={len(records)}")
    print("  available_payloads:")
    for label in sorted(by_label):
        rec = by_label[label]
        print(
            "    "
            f"{label}: availability={rec.get('availability','')} "
            f"payload_kind={rec.get('payload_kind','')} "
            f"dtype={rec.get('dtype','')} "
            f"shape={rec.get('shape', [])} "
            f"bytes={rec.get('payload_bytes', 0)} "
            f"checksum={rec.get('payload_checksum','')[:16]}"
        )

    checksum_groups: dict[str, list[str]] = defaultdict(list)
    view_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for label, rec in by_label.items():
        checksum = rec.get("payload_checksum", "")
        if checksum:
            checksum_groups[checksum].append(label)
        view_src = rec.get("metadata", {}).get("view_src", "") or rec.get("view_src", "")
        storage_offset = int(rec.get("metadata", {}).get("storage_offset", rec.get("storage_offset", 0)) or 0)
        if view_src and view_src not in ("(nil)", "0x0"):
            view_groups[(view_src, storage_offset)].append(label)

    print("  alias_groups:")
    emitted = False
    for labels in checksum_groups.values():
        if len(labels) > 1:
            emitted = True
            print(f"    checksum: {','.join(sorted(labels))}")
    for (_, _), labels in view_groups.items():
        if len(labels) > 1:
            emitted = True
            print(f"    view: {','.join(sorted(labels))}")
    if not emitted:
        print("    none")

    print("  shape_layout_table:")
    for label in sorted(by_label):
        rec = by_label[label]
        md = rec.get("metadata", {})
        shape = md.get("ne", rec.get("shape", []))
        stride = md.get("nb", rec.get("stride", []))
        print(
            "    "
            f"{label}: op={rec.get('producer', {}).get('op', '')} "
            f"tensor_name={rec.get('producer', {}).get('tensor_name', '')} "
            f"shape={shape} stride={stride} "
            f"view_src={md.get('view_src_name', md.get('view_src', ''))} "
            f"storage_offset={md.get('storage_offset', rec.get('storage_offset', 0))}"
        )

    categories = {
        "state": ["state_kv", "state_score"],
        "pool": ["pool_input", "pooled_compressed_row"],
        "norm": ["compressed_norm", "compressed_norm_weight", "compressed_norm_weighted"],
        "rope": [
            "compressed_rope",
            "cupd_norm_weighted",
            "cupd_rope_input",
            "cupd_rope_reference",
            "cupd_rope_op_params",
            "cupd_rope_position",
            "cupd_rope_cache_position",
            "cupd_rope_n_rot",
            "cupd_rope_width",
            "cupd_rope_freq_base",
            "cupd_rope_freq_scale",
            "cupd_rope_mode",
            "cupd_rope_tail_offset",
            "cupd_rope_metadata",
            "cupd_rope_cos",
            "cupd_rope_sin",
        ],
        "quant": ["compressed_quant_fp8_or_nfp4", "downstream_compressed_kv", "downstream_kv"],
    }
    print("  availability_by_region:")
    for region, labels in categories.items():
        available = [label for label in labels if by_label.get(label, {}).get("availability") == "available"]
        print(f"    {region}: {len(available)}/{len(labels)} available ({','.join(available) if available else 'none'})")

    missing = [label for label in sorted(RECOMPUTE_INPUTS) if by_label.get(label, {}).get("availability") != "available"]
    rope_missing = [label for label in sorted(ROPE_INPUTS) if by_label.get(label, {}).get("availability") != "available"]
    byte_rows = [
        label for label, rec in by_label.items()
        if rec.get("payload_kind") == "byte_values" and rec.get("availability") == "available"
    ]
    print(f"  enough_for_norm_weighted_recompute={0 if missing else 1}")
    print(f"  missing_norm_weighted_inputs={','.join(missing) if missing else 'none'}")
    print(f"  norm_weighted_available={1 if by_label.get('cupd_norm_weighted', by_label.get('compressed_norm_weighted', {})).get('availability') == 'available' else 0}")
    print(f"  rope_input_available={1 if by_label.get('cupd_rope_input', {}).get('availability') == 'available' else 0}")
    print(f"  rope_reference_available={1 if by_label.get('cupd_rope_reference', by_label.get('compressed_rope', {})).get('availability') == 'available' else 0}")
    print(f"  position_metadata_available={1 if by_label.get('cupd_rope_position', {}).get('availability') == 'available' else 0}")
    print(f"  cache_position_metadata_available={1 if by_label.get('cupd_rope_cache_position', {}).get('availability') == 'available' else 0}")
    print(f"  n_rot_metadata_available={1 if by_label.get('cupd_rope_n_rot', {}).get('availability') == 'available' else 0}")
    print(f"  freq_base_available={1 if by_label.get('cupd_rope_freq_base', {}).get('availability') == 'available' else 0}")
    print(f"  freq_scale_available={1 if by_label.get('cupd_rope_freq_scale', {}).get('availability') == 'available' else 0}")
    print(f"  rope_mode_available={1 if by_label.get('cupd_rope_mode', {}).get('availability') == 'available' else 0}")
    print(f"  tail_offset_available={1 if by_label.get('cupd_rope_tail_offset', {}).get('availability') == 'available' else 0}")
    print(f"  cos_available={1 if by_label.get('cupd_rope_cos', {}).get('availability') == 'available' else 0}")
    print(f"  sin_available={1 if by_label.get('cupd_rope_sin', {}).get('availability') == 'available' else 0}")
    meta_records = [
        r for r in stage_records
        if (r.get("semantic_label") or r.get("tensor")) in ROPE_METADATA_LABELS
        and r.get("availability") == "available"
    ]
    positions = sorted({
        int(r.get("metadata", {}).get("position", 0))
        for r in meta_records
        if "position" in r.get("metadata", {})
    })
    n_rots = sorted({
        int(r.get("metadata", {}).get("n_rot", 0))
        for r in meta_records
        if "n_rot" in r.get("metadata", {})
    })
    freq_bases = sorted({
        float(r.get("metadata", {}).get("freq_base", 0.0))
        for r in meta_records
        if "freq_base" in r.get("metadata", {})
    })
    freq_scales = sorted({
        float(r.get("metadata", {}).get("freq_scale", 0.0))
        for r in meta_records
        if "freq_scale" in r.get("metadata", {})
    })
    rope_modes = sorted({
        str(r.get("metadata", {}).get("rope_mode", ""))
        for r in meta_records
        if r.get("metadata", {}).get("rope_mode", "") != ""
    })
    tail_offsets = sorted({
        int(r.get("metadata", {}).get("tail_offset", 0))
        for r in meta_records
        if "tail_offset" in r.get("metadata", {})
    })
    nonzero_positions = [p for p in positions if p != 0]
    cos_sin_formula = any(r.get("metadata", {}).get("cos_sin_formula_source") for r in meta_records)
    print(f"  cos_sin_materialized_or_formula_derived={1 if by_label.get('cupd_rope_cos', {}).get('availability') == 'available' or by_label.get('cupd_rope_sin', {}).get('availability') == 'available' or cos_sin_formula else 0}")
    print(f"  nonzero_position_records_count={len(nonzero_positions)}")
    print(f"  position_list={','.join(str(p) for p in positions) if positions else 'none'}")
    print(f"  n_rot_list={','.join(str(n) for n in n_rots) if n_rots else 'none'}")
    print(f"  freq_base_list={','.join(f'{v:g}' for v in freq_bases) if freq_bases else 'none'}")
    print(f"  freq_scale_list={','.join(f'{v:g}' for v in freq_scales) if freq_scales else 'none'}")
    print(f"  rope_mode_list={','.join(rope_modes) if rope_modes else 'none'}")
    print(f"  tail_offset_list={','.join(str(v) for v in tail_offsets) if tail_offsets else 'none'}")
    print(f"  enough_for_rope_recompute={0 if rope_missing else 1}")
    print(f"  missing_rope_inputs={','.join(rope_missing) if rope_missing else 'none'}")
    norm_weighted_checksum = by_label.get("cupd_norm_weighted", by_label.get("compressed_norm_weighted", {})).get("payload_checksum", "")
    rope_input_checksum = by_label.get("cupd_rope_input", {}).get("payload_checksum", "")
    rope_ref_checksum = by_label.get("cupd_rope_reference", by_label.get("compressed_rope", {})).get("payload_checksum", "")
    print(f"  norm_weighted_equals_rope_reference={1 if norm_weighted_checksum and norm_weighted_checksum == rope_ref_checksum else 0}")
    print(f"  rope_input_equals_rope_reference={1 if rope_input_checksum and rope_input_checksum == rope_ref_checksum else 0}")
    print(f"  byte_rows_available={','.join(sorted(byte_rows)) if byte_rows else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
