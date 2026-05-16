#!/usr/bin/env python3
"""Normalize DSV4 layer-executor side-probe summaries into JSONL fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path


DEFAULT_FILES = {
    "hc_pre_norm": "hc_pre_norm_l0_n16.jsonl",
    "routed_moe_final_output": "routed_moe_final_output_l0_n16.jsonl",
    "aohc_boundary": "aohc_boundary_l0_n16.jsonl",
    "compressor_update": "compressor_update_l2_n16.jsonl",
    "kv_cache_finalizer": "kv_cache_finalizer_l0_n16.jsonl",
}

STAGE_TURNS = {
    "hc_pre_norm": 77,
    "routed_moe_final_output": 78,
    "aohc_boundary": 79,
    "compressor_update": 80,
    "kv_cache_finalizer": 81,
}

STAGE_LAYERS = {
    "hc_pre_norm": 0,
    "routed_moe_final_output": 0,
    "aohc_boundary": 0,
    "compressor_update": 2,
    "kv_cache_finalizer": 0,
}

STAGE_PAYLOAD_META = {
    "hc_pre_norm": ("f32", [4096, 1, 1, 1]),
    "routed_moe_final_output": ("f32", [4096, 1, 1, 1]),
    "aohc_boundary": ("f32", [16384, 1, 1, 1]),
    "compressor_update": ("bytes", []),
    "kv_cache_finalizer": ("bytes", []),
}

HC_PRE_NORM_REQUIRED_INPUTS = {
    "input_hc_original_residual": ("f32", []),
    "split_pre": ("f32", []),
    "norm_weight": ("f32", [4096, 1, 1, 1]),
    "reference_cur": ("f32", [4096, 1, 1, 1]),
    "reference_norm": ("f32", [4096, 1, 1, 1]),
    "reference_post": ("f32", [4096, 1, 1, 1]),
}

HC_PRE_NORM_SEMANTIC_INPUTS = {
    "hc_pre_input_hc_original": ("f32", [4096, 4, 1, 1], True),
    "hc_pre_flat_hc": ("f32", [16384, 1, 1, 1], False),
    "hc_pre_flat_hc_normed": ("f32", [16384, 1, 1, 1], False),
    "hc_pre_hc_mix": ("f32", [4, 1, 1, 1], False),
    "hc_pre_split_pre": ("f32", [4, 1, 1, 1], True),
    "hc_pre_split_post": ("f32", [4, 1, 1, 1], False),
    "hc_pre_split_comb": ("f32", [4, 4, 1, 1], False),
    "hc_pre_weighted_cur_reference": ("f32", [4096, 1, 1, 1], True),
    "hc_pre_norm_weight": ("f32", [4096, 1, 1, 1], True),
    "hc_pre_norm_reference": ("f32", [4096, 1, 1, 1], True),
    "hc_pre_post_reference": ("f32", [4096, 1, 1, 1], True),
}

HC_WEIGHTED_SUM_INPUTS = {
    "hc_ws_input_inpL_raw": ("f32", [4096, 4, 1, 1], True),
    "hc_ws_input_inpL_view": ("f32", [4096, 4, 1, 1], False),
    "hc_ws_input_inpL_contiguous": ("f32", [4096, 4, 1, 1], False),
    "hc_ws_split_full_raw": ("f32", [24, 1, 1, 1], False),
    "hc_ws_split_pre_raw": ("f32", [4, 1, 1, 1], True),
    "hc_ws_split_pre_view": ("f32", [4, 1, 1, 1], False),
    "hc_ws_split_pre_contiguous": ("f32", [4, 1, 1, 1], False),
    "hc_ws_reference_cur": ("f32", [4096, 1, 1, 1], True),
    "hc_ws_reference_cur_pre_reshape": ("f32", [4096, 1, 1, 1], False),
    "hc_ws_reference_cur_post_reshape": ("f32", [4096, 1, 1, 1], False),
}

RMOE_SOURCE_CONTRACT_INPUTS = {
    "rmoe_ffn_input": ("f32", [4096, 1, 1, 1], False),
    "rmoe_topk_ids": ("i32", [6, 1, 1, 1], False),
    "rmoe_topk_weights": ("f32", [1, 6, 1, 1], False),
    "rmoe_expert_gate": ("f32", [2048, 6, 1, 1], False),
    "rmoe_expert_up": ("f32", [2048, 6, 1, 1], False),
    "rmoe_expert_swiglu": ("f32", [2048, 6, 1, 1], False),
    "rmoe_expert_down": ("f32", [4096, 6, 1, 1], False),
    "rmoe_weighted_down_slot0": ("f32", [4096, 1, 1, 1], False),
    "rmoe_weighted_down_slot1": ("f32", [4096, 1, 1, 1], False),
    "rmoe_weighted_down_slot2": ("f32", [4096, 1, 1, 1], False),
    "rmoe_weighted_down_slot3": ("f32", [4096, 1, 1, 1], False),
    "rmoe_weighted_down_slot4": ("f32", [4096, 1, 1, 1], False),
    "rmoe_weighted_down_slot5": ("f32", [4096, 1, 1, 1], False),
    "rmoe_routed_sum": ("f32", [4096, 1, 1, 1], True),
    "rmoe_shared_gate": ("f32", [2048, 1, 1, 1], False),
    "rmoe_shared_up": ("f32", [2048, 1, 1, 1], False),
    "rmoe_shared_swiglu": ("f32", [2048, 1, 1, 1], False),
    "rmoe_shared_down": ("f32", [4096, 1, 1, 1], True),
    "rmoe_final_ffn_reference": ("f32", [4096, 1, 1, 1], True),
}

AOHC_SOURCE_CONTRACT_INPUTS = {
    "aohc_attn_core_output": ("f32", [], False),
    "aohc_attn_low": ("f32", [], False),
    "aohc_attn_out": ("f32", [4096, 1, 1, 1], True),
    "aohc_hc_post_input": ("f32", [4096, 1, 1, 1], False),
    "aohc_hc_post_residual": ("f32", [4096, 4, 1, 1], True),
    "aohc_hc_split_full": ("f32", [24, 1, 1, 1], False),
    "aohc_hc_post_weights": ("f32", [4, 1, 1, 1], True),
    "aohc_hc_comb": ("f32", [4, 4, 1, 1], True),
    "aohc_after_attn_hc_reference": ("f32", [4096, 4, 1, 1], True),
    "aohc_layer_attn_output_anchor": ("f32", [4096, 4, 1, 1], False),
    "aohc_hcexpand_src0_block": ("f32", [4096, 1, 1, 1], True),
    "aohc_hcexpand_src1_residual": ("f32", [4096, 4, 1, 1], True),
    "aohc_hcexpand_src2_post": ("f32", [4, 1, 1, 1], True),
    "aohc_hcexpand_src3_comb": ("f32", [4, 4, 1, 1], True),
    "aohc_hcexpand_dispatch_src0": ("f32", [4096, 1, 1, 1], True),
    "aohc_hcexpand_dispatch_src1": ("f32", [4096, 4, 1, 1], True),
    "aohc_hcexpand_dispatch_src2": ("f32", [4, 1, 1, 1], True),
    "aohc_hcexpand_dispatch_src3": ("f32", [4, 4, 1, 1], True),
    "aohc_hcexpand_dispatch_output": ("f32", [4096, 4, 1, 1], False),
}

CUPD_SOURCE_CONTRACT_INPUTS = {
    "state_kv": ("f32", [], False),
    "state_score": ("f32", [], False),
    "pool_input": ("f32", [], False),
    "pooled_compressed_row": ("f32", [], False),
    "compressed_pre_norm": ("f32", [], False),
    "compressed_norm": ("f32", [], False),
    "compressed_norm_weight": ("f32", [], True),
    "compressed_norm_weighted": ("f32", [], True),
    "compressed_rope": ("f32", [], False),
    "cupd_norm_weighted": ("f32", [], True),
    "cupd_rope_input": ("f32", [], True),
    "cupd_rope_reference": ("f32", [], True),
    "cupd_rope_op_params": ("metadata", [], True),
    "cupd_rope_position": ("metadata", [], True),
    "cupd_rope_cache_position": ("metadata", [], True),
    "cupd_rope_n_rot": ("metadata", [], True),
    "cupd_rope_width": ("metadata", [], True),
    "cupd_rope_freq_base": ("metadata", [], True),
    "cupd_rope_freq_scale": ("metadata", [], True),
    "cupd_rope_mode": ("metadata", [], True),
    "cupd_rope_tail_offset": ("metadata", [], True),
    "cupd_rope_metadata": ("metadata", [], True),
    "cupd_rope_cos": ("f32", [], False),
    "cupd_rope_sin": ("f32", [], False),
    "compressed_quant_fp8_or_nfp4": ("bytes", [], False),
    "downstream_compressed_kv": ("bytes", [], False),
    "downstream_kv": ("bytes", [], False),
    "cache_handoff_row_metadata": ("metadata", [], False),
}


def parse_kv(line: str, key: str, default: str = "") -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", line)
    return match.group(1) if match else default


def parse_i64_list(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        return []
    return [int(v) for v in value.split(",") if v]


def parse_payload_target_metadata(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pattern = re.compile(
        r"dsv4_lexec_payload_target: .*?tensor=(?P<tensor>\S+)"
        r".*?op=(?P<op>\S+).*?tensor_name=(?P<tensor_name>.*?) ne="
        r"(?P<ne>\[[^\]]*\]).*?nb=(?P<nb>\[[^\]]*\])"
        r".*?view_src=(?P<view_src>\S+).*?view_src_name=(?P<view_src_name>.*?) view_offs="
        r"(?P<view_offs>\d+).*?ptr=(?P<ptr>\S+)"
        r".*?capture_mode=(?P<capture_mode>\S+).*?pin_output=(?P<pin_output>\d+)"
    )
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        tensor = match.group("tensor")
        out[tensor] = {
            "producer_op": match.group("op"),
            "producer_tensor_name": match.group("tensor_name"),
            "shape": parse_i64_list(match.group("ne")),
            "stride": parse_i64_list(match.group("nb")),
            "view_src": match.group("view_src"),
            "view_src_name": match.group("view_src_name"),
            "storage_offset": int(match.group("view_offs")),
            "tensor_ptr": match.group("ptr"),
            "capture_mode": match.group("capture_mode"),
            "pin_output": int(match.group("pin_output")),
        }
    return out


def parse_cupd_rope_metadata(path: Path) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "dsv4_lexec_cupd_rope_metadata:" not in line:
            continue
        layer = int(parse_kv(line, "layer", "-1"))
        token = int(parse_kv(line, "token", "-1"))
        if layer < 0 or token < 0:
            continue
        md = {
            "position": int(parse_kv(line, "position", "0")),
            "cache_position": int(parse_kv(line, "cache_position", "0")),
            "n_rot": int(parse_kv(line, "n_rot", "0")),
            "width": int(parse_kv(line, "width", "0")),
            "freq_base": float(parse_kv(line, "freq_base", "0")),
            "freq_scale": float(parse_kv(line, "freq_scale", "0")),
            "rope_mode": parse_kv(line, "rope_mode", ""),
            "rope_type": int(parse_kv(line, "rope_type", "0")),
            "tail_offset": int(parse_kv(line, "tail_offset", "0")),
            "n_ctx_orig": int(parse_kv(line, "n_ctx_orig", "0")),
            "ext_factor": float(parse_kv(line, "ext_factor", "0")),
            "attn_factor": float(parse_kv(line, "attn_factor", "0")),
            "beta_fast": float(parse_kv(line, "beta_fast", "0")),
            "beta_slow": float(parse_kv(line, "beta_slow", "0")),
            "cos_sin_materialized": int(parse_kv(line, "cos_sin_materialized", "0")),
            "cos_sin_formula_source": parse_kv(line, "cos_sin_formula_source", ""),
            "capture_intrusive": int(parse_kv(line, "capture_intrusive", "1")),
            "used_for_fixture_only": int(parse_kv(line, "used_for_fixture_only", "1")),
            "not_hot_neutral_validation": int(parse_kv(line, "not_hot_neutral_validation", "1")),
        }
        out[(layer, token)] = md
    return out


def source_turn(path: Path) -> int:
    match = re.search(r"turn(\d+)", path.name)
    return int(match.group(1)) if match else -1


def normalize_capture(stage: str, path: Path) -> dict:
    summary = ""
    for line in path.read_text(errors="replace").splitlines():
        if "dsv4_lexec_side_probe_summary:" in line:
            summary = line
    if not summary:
        raise SystemExit(f"no dsv4_lexec_side_probe_summary found in {path}")

    layer = int(parse_kv(summary, "layer_filter", "-1"))
    token = int(parse_kv(summary, "token_min", "-1"))
    if stage == "compressor_update":
        first_token = int(parse_kv(summary, "first_compressor_token", "-1"))
        if first_token >= 0:
            token = first_token
    elif stage == "kv_cache_finalizer":
        first_token = int(parse_kv(summary, "first_swa_cache_token", "-1"))
        if first_token < 0:
            first_token = int(parse_kv(summary, "first_compressed_cache_token", "-1"))
        if first_token >= 0:
            token = first_token
    max_abs = float(parse_kv(summary, "max_abs", "0"))
    rms = float(parse_kv(summary, "max_rms", "0"))
    over_tol = int(parse_kv(summary, "over_tol", "0"))

    return {
        "schema_version": 1,
        "capture_id": f"t{source_turn(path)}_{stage}_l{layer if layer >= 0 else 'all'}_n16",
        "stage": stage,
        "layer": layer,
        "token": token,
        "tensor": "summary",
        "dtype": "metadata",
        "shape": [],
        "availability": "available",
        "payload_kind": "stats_only",
        "required": True,
        "unavailable_reason": "",
        "producer": {
            "op": "post_eval_side_probe",
            "tensor_name": "summary",
            "stage": stage,
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_identity",
            "tensor_name": "summary",
            "stage": stage,
        },
        "stats_only": True,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": max_abs,
            "rms": rms,
            "over_tol": over_tol,
        },
        "metadata": {
            "source_turn": source_turn(path),
            "source_log": str(path),
            "mode": parse_kv(summary, "mode", "post_eval_cpu_compare"),
            "capture_mode": parse_kv(summary, "payload_capture_mode", "post_eval_tensor_get"),
            "capture_intrusive": 1 if parse_kv(summary, "payload_capture_mode", "post_eval_tensor_get") in ("producer_capture", "consumer_dispatch") else 0,
            "used_for_fixture_only": 1 if parse_kv(summary, "payload_capture_mode", "post_eval_tensor_get") in ("producer_capture", "consumer_dispatch") else 0,
            "not_hot_neutral_validation": 1 if parse_kv(summary, "payload_capture_mode", "post_eval_tensor_get") in ("producer_capture", "consumer_dispatch") else 0,
            "live_graph_nodes_added": int(parse_kv(summary, "live_graph_nodes_added", "0")),
            "live_backend_dispatches": int(parse_kv(summary, "live_backend_dispatches", "0")),
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "summary_cases": int(parse_kv(summary, "cases", "0")),
            "exact_cases": int(parse_kv(summary, "exact_cases", "0")),
        },
    }


def payload_file_for(stage: str, layer: int, token: int, payload_dir: Path | None, tensor: str | None = None) -> str:
    if payload_dir is None:
        return ""
    prefix = stage if tensor is None or tensor == stage else f"{stage}_{tensor}"
    candidate = payload_dir / f"{prefix}_l{layer}_t{token}.bin"
    if not candidate.exists():
        return ""
    try:
        return str(candidate.relative_to(payload_dir.parent))
    except ValueError:
        return candidate.name


def payload_preview(payload_path: Path, dtype: str) -> dict:
    data = payload_path.read_bytes()
    out = {
        "payload_bytes": len(data),
        "payload_checksum": hashlib.sha256(data).hexdigest(),
        "first_8_values": [],
        "last_8_values": [],
    }
    if dtype == "f32" and len(data) % 4 == 0:
        values = struct.unpack(f"<{len(data)//4}f", data)
        out["payload_numel"] = len(values)
        out["first_8_values"] = [float(v) for v in values[:8]]
        out["last_8_values"] = [float(v) for v in values[-8:]]
    else:
        out["payload_numel"] = len(data)
        out["first_8_values"] = [int(v) for v in data[:8]]
        out["last_8_values"] = [int(v) for v in data[-8:]]
    return out


def payload_source_path(payload_dir: Path | None, payload_file: str) -> str:
    if payload_dir is None or not payload_file:
        return ""
    path = Path(payload_file)
    if path.is_absolute():
        return str(path)
    return str(payload_dir.parent / path)


def enrich_payload_record(record: dict, payload_dir: Path | None) -> dict:
    record.setdefault("semantic_label", record.get("tensor", ""))
    record.setdefault("producer_op", record.get("producer", {}).get("op", ""))
    record.setdefault("consumer_op", record.get("consumer", {}).get("op", ""))
    record.setdefault("stride", [])
    record.setdefault("view_src", "")
    record.setdefault("storage_offset", 0)
    record.setdefault("same_payload_as", "")
    record.setdefault("same_tensor_ptr_as", "")
    record.setdefault("same_view_src_as", "")
    record.setdefault("same_storage_offset_as", "")
    payload_file = record.get("payload_file", "")
    if payload_dir is not None and payload_file:
        path = Path(payload_file)
        payload_path = path if path.is_absolute() else payload_dir.parent / path
        if payload_path.exists():
            record.update(payload_preview(payload_path, record.get("dtype", "")))
    return record


def apply_target_metadata(record: dict, target_meta: dict[str, dict]) -> dict:
    label = record.get("semantic_label") or record.get("tensor", "")
    meta = target_meta.get(label)
    if not meta:
        return record
    if meta.get("shape"):
        record["shape"] = meta["shape"]
    record["stride"] = meta.get("stride", record.get("stride", []))
    record["view_src"] = meta.get("view_src_name") or meta.get("view_src", "")
    record["storage_offset"] = meta.get("storage_offset", record.get("storage_offset", 0))
    record["same_tensor_ptr_as"] = ""
    record["same_view_src_as"] = ""
    record["same_storage_offset_as"] = ""
    record["producer"]["op"] = meta.get("producer_op", record["producer"].get("op", ""))
    record["producer"]["tensor_name"] = meta.get("producer_tensor_name", record["producer"].get("tensor_name", ""))
    record["producer_op"] = record["producer"]["op"]
    md = record.setdefault("metadata", {})
    md["ne"] = meta.get("shape", [])
    md["nb"] = meta.get("stride", [])
    md["view_src"] = meta.get("view_src", "")
    md["view_src_name"] = meta.get("view_src_name", "")
    md["storage_offset"] = meta.get("storage_offset", 0)
    md["tensor_ptr"] = meta.get("tensor_ptr", "")
    md["pin_output"] = meta.get("pin_output", 0)
    return record


def apply_alias_diagnostics(records: list[dict]) -> None:
    first_by_checksum: dict[str, str] = {}
    first_by_ptr: dict[str, str] = {}
    first_by_view: dict[tuple[str, int], str] = {}
    for record in records:
        checksum = record.get("payload_checksum", "")
        label = record.get("semantic_label") or record.get("tensor", "")
        if not checksum:
            pass
        elif checksum not in first_by_checksum:
            first_by_checksum[checksum] = label
            record["same_payload_as"] = ""
        elif first_by_checksum[checksum] != label:
            record["same_payload_as"] = first_by_checksum[checksum]
        ptr = record.get("metadata", {}).get("tensor_ptr", "")
        if ptr:
            if ptr not in first_by_ptr:
                first_by_ptr[ptr] = label
            elif first_by_ptr[ptr] != label:
                record["same_tensor_ptr_as"] = first_by_ptr[ptr]
        view = record.get("metadata", {}).get("view_src", "")
        offs = int(record.get("storage_offset", 0))
        if view and view != "(nil)" and view != "0x0":
            key = (view, offs)
            if key not in first_by_view:
                first_by_view[key] = label
            elif first_by_view[key] != label:
                record["same_view_src_as"] = first_by_view[key]
                record["same_storage_offset_as"] = first_by_view[key]


def apply_payload_if_present(record: dict, payload_dir: Path | None) -> dict:
    payload_file = payload_file_for(record["stage"], int(record["layer"]), int(record["token"]), payload_dir)
    if not payload_file:
        return record
    record = dict(record)
    dtype, shape = STAGE_PAYLOAD_META.get(record["stage"], (record.get("dtype", "metadata"), record.get("shape", [])))
    record["dtype"] = dtype
    record["shape"] = shape
    record["payload_kind"] = "tensor_values"
    record["payload_file"] = payload_file
    record["stats_only"] = False
    record["byte_payload_available"] = False
    record["metadata"]["payload_source"] = payload_source_path(payload_dir, payload_file)
    return enrich_payload_record(record, payload_dir)


def byte_payload_record_if_present(stage: str, layer: int, token: int, payload_dir: Path | None) -> dict | None:
    payload_file = payload_file_for(stage, layer, token, payload_dir)
    if not payload_file:
        return None
    turn = STAGE_TURNS[stage]
    return {
        "schema_version": 1,
        "capture_id": f"t{turn}_{stage}_quant_cache_byte_payload",
        "stage": stage,
        "layer": layer,
        "token": token,
        "tensor": "quant_cache_byte_payload",
        "dtype": "bytes",
        "shape": [],
        "availability": "available",
        "payload_kind": "byte_values",
        "payload_file": payload_file,
        "required": False,
        "unavailable_reason": "",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": "quant_cache_byte_payload",
            "stage": stage,
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_identity",
            "tensor_name": "quant_cache_byte_payload",
            "stage": stage,
        },
        "stats_only": False,
        "byte_payload_available": True,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": turn,
            "mode": "post_eval_cpu_compare",
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "payload_source": payload_source_path(payload_dir, payload_file),
        },
    }


def hcnorm_input_record(tensor: str, layer: int, token: int, payload_dir: Path | None, capture_mode: str, target_meta: dict[str, dict]) -> dict:
    dtype, shape = HC_PRE_NORM_REQUIRED_INPUTS[tensor]
    payload_file = payload_file_for("hc_pre_norm", layer, token, payload_dir, tensor)
    available = bool(payload_file)
    return apply_target_metadata(enrich_payload_record({
        "schema_version": 1,
        "capture_id": f"t89_hc_pre_norm_{tensor}_l{layer}_t{token}",
        "stage": "hc_pre_norm",
        "layer": layer,
        "token": token,
        "tensor": tensor,
        "semantic_label": tensor,
        "dtype": dtype if available else "metadata",
        "shape": shape if available else [],
        "availability": "available" if available else "unavailable",
        "payload_kind": "tensor_values" if available else "metadata_only",
        "payload_file": payload_file,
        "required": True,
        "unavailable_reason": "" if available else "T89 HC_PRE_NORM recompute input payload was not captured",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": tensor,
            "stage": "hc_pre_norm",
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_hcnorm_recompute",
            "tensor_name": tensor,
            "stage": "hc_pre_norm",
        },
        "stats_only": False,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": 89,
            "mode": "post_eval_cpu_compare",
            "capture_mode": capture_mode,
            "capture_intrusive": 1 if capture_mode == "producer_capture" else 0,
            "used_for_fixture_only": 1 if capture_mode == "producer_capture" else 0,
            "not_hot_neutral_validation": 1 if capture_mode == "producer_capture" else 0,
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "payload_source": payload_source_path(payload_dir, payload_file) if available else "",
        },
    }, payload_dir), target_meta)


def hcnorm_semantic_record(tensor: str, layer: int, token: int, payload_dir: Path | None, capture_mode: str, target_meta: dict[str, dict]) -> dict:
    if tensor in HC_WEIGHTED_SUM_INPUTS:
        dtype, shape, required = HC_WEIGHTED_SUM_INPUTS[tensor]
    else:
        dtype, shape, required = HC_PRE_NORM_SEMANTIC_INPUTS[tensor]
    payload_file = payload_file_for("hc_pre_norm", layer, token, payload_dir, tensor)
    available = bool(payload_file)
    return apply_target_metadata(enrich_payload_record({
        "schema_version": 1,
        "capture_id": f"t90_hc_pre_norm_{tensor}_l{layer}_t{token}",
        "stage": "hc_pre_norm",
        "layer": layer,
        "token": token,
        "tensor": tensor,
        "semantic_label": tensor,
        "dtype": dtype if available else "metadata",
        "shape": shape if available else [],
        "availability": "available" if available else "unavailable",
        "payload_kind": "tensor_values" if available else "metadata_only",
        "payload_file": payload_file,
        "required": required,
        "unavailable_reason": "" if available else "T90 HC_PRE_NORM semantic payload was not captured",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": tensor,
            "stage": "hc_pre_norm",
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_hcnorm_recompute",
            "tensor_name": tensor,
            "stage": "hc_pre_norm",
        },
        "stats_only": False,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": 92 if tensor in HC_WEIGHTED_SUM_INPUTS else 90,
            "mode": "post_eval_cpu_compare",
            "capture_mode": capture_mode,
            "capture_intrusive": 1 if capture_mode == "producer_capture" else 0,
            "used_for_fixture_only": 1 if capture_mode == "producer_capture" else 0,
            "not_hot_neutral_validation": 1 if capture_mode == "producer_capture" else 0,
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "payload_source": payload_source_path(payload_dir, payload_file) if available else "",
            "n_embd": 4096,
            "n_hc": 4,
            "n_tokens": 1,
            "expected_output_shape": [4096, 1, 1, 1],
            "input_layout_candidate": "hidden_major_for_contiguous_x",
            "split_layout_candidate": "weights[h + token*n_hc]",
        },
    }, payload_dir), target_meta)


def rmoe_semantic_record(tensor: str, layer: int, token: int, payload_dir: Path | None, capture_mode: str, target_meta: dict[str, dict]) -> dict:
    dtype, shape, required = RMOE_SOURCE_CONTRACT_INPUTS[tensor]
    payload_file = payload_file_for("routed_moe_final_output", layer, token, payload_dir, tensor)
    available = bool(payload_file)
    return apply_target_metadata(enrich_payload_record({
        "schema_version": 1,
        "capture_id": f"t94_routed_moe_final_output_{tensor}_l{layer}_t{token}",
        "stage": "routed_moe_final_output",
        "layer": layer,
        "token": token,
        "tensor": tensor,
        "semantic_label": tensor,
        "dtype": dtype if available else "metadata",
        "shape": shape if available else [],
        "availability": "available" if available else "unavailable",
        "payload_kind": "tensor_values" if available else "metadata_only",
        "payload_file": payload_file,
        "required": required,
        "unavailable_reason": "" if available else "T94 routed-MoE source-contract payload was not captured",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": tensor,
            "stage": "routed_moe_final_output",
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_rmoe_recompute",
            "tensor_name": tensor,
            "stage": "routed_moe_final_output",
        },
        "stats_only": False,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": 94,
            "mode": "post_eval_cpu_compare",
            "capture_mode": capture_mode,
            "capture_intrusive": 1 if capture_mode == "producer_capture" else 0,
            "used_for_fixture_only": 1 if capture_mode == "producer_capture" else 0,
            "not_hot_neutral_validation": 1 if capture_mode == "producer_capture" else 0,
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "payload_source": payload_source_path(payload_dir, payload_file) if available else "",
            "weights_not_decoded": 1,
            "expert_weight_recompute": 0,
        },
    }, payload_dir), target_meta)


def aohc_semantic_record(tensor: str, layer: int, token: int, payload_dir: Path | None, capture_mode: str, target_meta: dict[str, dict]) -> dict:
    dtype, shape, required = AOHC_SOURCE_CONTRACT_INPUTS[tensor]
    payload_file = payload_file_for("aohc_boundary", layer, token, payload_dir, tensor)
    available = bool(payload_file)
    return apply_target_metadata(enrich_payload_record({
        "schema_version": 1,
        "capture_id": f"t98_aohc_boundary_{tensor}_l{layer}_t{token}",
        "stage": "aohc_boundary",
        "layer": layer,
        "token": token,
        "tensor": tensor,
        "semantic_label": tensor,
        "dtype": dtype if available else "metadata",
        "shape": shape if available else [],
        "availability": "available" if available else "unavailable",
        "payload_kind": "tensor_values" if available else "metadata_only",
        "payload_file": payload_file,
        "required": required,
        "unavailable_reason": "" if available else "T96 AOHC source-contract payload was not captured",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": tensor,
            "stage": "aohc_boundary",
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_aohc_recompute",
            "tensor_name": tensor,
            "stage": "aohc_boundary",
        },
        "stats_only": False,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": 98 if tensor.startswith("aohc_hcexpand_dispatch_") else 96,
            "mode": "post_eval_cpu_compare",
            "capture_mode": capture_mode,
            "capture_intrusive": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
            "used_for_fixture_only": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
            "not_hot_neutral_validation": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "payload_source": payload_source_path(payload_dir, payload_file) if available else "",
            "weights_not_decoded": 1,
        },
    }, payload_dir), target_meta)


def cupd_semantic_record(
    tensor: str,
    layer: int,
    token: int,
    payload_dir: Path | None,
    capture_mode: str,
    target_meta: dict[str, dict],
    rope_meta: dict[tuple[int, int], dict] | None = None,
) -> dict:
    dtype, shape, required = CUPD_SOURCE_CONTRACT_INPUTS[tensor]
    payload_file = payload_file_for("compressor_update", layer, token, payload_dir, tensor)
    token_rope_meta = (rope_meta or {}).get((layer, token), {})
    is_metadata = dtype == "metadata"
    available = bool(payload_file) or (is_metadata and bool(token_rope_meta) and tensor.startswith("cupd_rope_"))
    is_byte_payload = dtype == "bytes" and available
    metadata = {
        "source_turn": 102,
        "mode": "post_eval_cupd_compare",
        "capture_mode": capture_mode,
        "capture_intrusive": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
        "used_for_fixture_only": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
        "not_hot_neutral_validation": 1 if capture_mode in ("producer_capture", "consumer_dispatch") else 0,
        "live_graph_nodes_added": 0,
        "live_backend_dispatches": 0,
        "output_consumed": 0,
        "cache_mutation": "disabled",
        "payload_source": payload_source_path(payload_dir, payload_file) if payload_file else "",
        "weights_not_decoded": 1,
        "cache_mutation_by_harness": "disabled",
    }
    if token_rope_meta and tensor.startswith("cupd_rope_"):
        metadata.update(token_rope_meta)
    return apply_target_metadata(enrich_payload_record({
        "schema_version": 1,
        "capture_id": f"t99_compressor_update_{tensor}_l{layer}_t{token}",
        "stage": "compressor_update",
        "layer": layer,
        "token": token,
        "tensor": tensor,
        "semantic_label": tensor,
        "dtype": dtype if payload_file else "metadata",
        "shape": shape if available else [],
        "availability": "available" if available else "unavailable",
        "payload_kind": "byte_values" if is_byte_payload else ("tensor_values" if payload_file else "metadata_only"),
        "payload_file": payload_file,
        "required": required,
        "unavailable_reason": "" if available else "compressor/update source-contract payload or metadata was not captured",
        "producer": {
            "op": "post_eval_side_probe_payload_readback",
            "tensor_name": tensor,
            "stage": "compressor_update",
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_cupd_recompute",
            "tensor_name": tensor,
            "stage": "compressor_update",
        },
        "stats_only": False,
        "byte_payload_available": is_byte_payload,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": metadata,
    }, payload_dir), target_meta)


def append_hcnorm_input_records(records: list[dict], payload_dir: Path | None, target_meta: dict[str, dict]) -> None:
    main = records[0]
    if main.get("stage") != "hc_pre_norm":
        return
    layer = int(main["layer"])
    token = int(main["token"])
    capture_mode = main.get("metadata", {}).get("capture_mode", "post_eval_tensor_get")
    for tensor in HC_PRE_NORM_REQUIRED_INPUTS:
        records.append(hcnorm_input_record(tensor, layer, token, payload_dir, capture_mode, target_meta))
    for tensor in HC_PRE_NORM_SEMANTIC_INPUTS:
        records.append(hcnorm_semantic_record(tensor, layer, token, payload_dir, capture_mode, target_meta))
    for tensor in HC_WEIGHTED_SUM_INPUTS:
        records.append(hcnorm_semantic_record(tensor, layer, token, payload_dir, capture_mode, target_meta))
    apply_alias_diagnostics(records)


def append_rmoe_input_records(records: list[dict], payload_dir: Path | None, target_meta: dict[str, dict]) -> None:
    main = records[0]
    if main.get("stage") != "routed_moe_final_output":
        return
    layer = int(main["layer"])
    token = int(main["token"])
    capture_mode = main.get("metadata", {}).get("capture_mode", "post_eval_tensor_get")
    for tensor in RMOE_SOURCE_CONTRACT_INPUTS:
        records.append(rmoe_semantic_record(tensor, layer, token, payload_dir, capture_mode, target_meta))
    apply_alias_diagnostics(records)


def append_aohc_input_records(records: list[dict], payload_dir: Path | None, target_meta: dict[str, dict]) -> None:
    main = records[0]
    if main.get("stage") != "aohc_boundary":
        return
    layer = int(main["layer"])
    token = int(main["token"])
    capture_mode = main.get("metadata", {}).get("capture_mode", "post_eval_tensor_get")
    for tensor in AOHC_SOURCE_CONTRACT_INPUTS:
        records.append(aohc_semantic_record(tensor, layer, token, payload_dir, capture_mode, target_meta))
    apply_alias_diagnostics(records)


def append_cupd_input_records(
    records: list[dict],
    payload_dir: Path | None,
    target_meta: dict[str, dict],
    rope_meta: dict[tuple[int, int], dict] | None = None,
) -> None:
    main = records[0]
    if main.get("stage") != "compressor_update":
        return
    layer = int(main["layer"])
    main_token = int(main["token"])
    capture_mode = main.get("metadata", {}).get("capture_mode", "post_eval_tensor_get")
    tokens = {main_token}
    if payload_dir is not None and payload_dir.exists():
        for path in payload_dir.glob(f"compressor_update_*_l{layer}_t*.bin"):
            match = re.search(rf"_l{layer}_t(-?\d+)\.bin$", path.name)
            if match:
                tokens.add(int(match.group(1)))
    for token in sorted(tokens):
        for tensor in CUPD_SOURCE_CONTRACT_INPUTS:
            if token != main_token:
                dtype, _, required = CUPD_SOURCE_CONTRACT_INPUTS[tensor]
                has_payload = bool(payload_file_for("compressor_update", layer, token, payload_dir, tensor))
                has_meta = dtype == "metadata" and (rope_meta or {}).get((layer, token)) is not None and tensor.startswith("cupd_rope_")
                if required and not (has_payload or has_meta):
                    continue
            records.append(cupd_semantic_record(tensor, layer, token, payload_dir, capture_mode, target_meta, rope_meta))
    apply_alias_diagnostics(records)


def summary_fixture_record(stage: str, payload_dir: Path | None = None) -> dict:
    turn = STAGE_TURNS[stage]
    layer = STAGE_LAYERS[stage]
    record = {
        "schema_version": 1,
        "capture_id": f"t{turn}_{stage}_l{layer}_n16",
        "stage": stage,
        "layer": layer,
        "token": 1,
        "tensor": "summary",
        "dtype": "metadata",
        "shape": [],
        "availability": "available",
        "payload_kind": "stats_only",
        "required": True,
        "unavailable_reason": "",
        "producer": {
            "op": "post_eval_side_probe",
            "tensor_name": "summary",
            "stage": stage,
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_identity",
            "tensor_name": "summary",
            "stage": stage,
        },
        "stats_only": True,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": turn,
            "mode": "post_eval_cpu_compare",
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "fixture_source": "accepted_turn_summary",
        },
    }
    return apply_payload_if_present(record, payload_dir)


def unavailable_record(stage: str, tensor: str, reason: str) -> dict:
    turn = STAGE_TURNS[stage]
    layer = STAGE_LAYERS[stage]
    return {
        "schema_version": 1,
        "capture_id": f"t{turn}_{stage}_{tensor}_unavailable",
        "stage": stage,
        "layer": layer,
        "token": 1,
        "tensor": tensor,
        "dtype": "metadata",
        "shape": [],
        "availability": "unavailable",
        "payload_kind": "metadata_only",
        "required": False,
        "unavailable_reason": reason,
        "producer": {
            "op": "post_eval_side_probe",
            "tensor_name": tensor,
            "stage": stage,
        },
        "consumer": {
            "op": "dsv4_layer_executor_harness_identity",
            "tensor_name": tensor,
            "stage": stage,
        },
        "stats_only": False,
        "byte_payload_available": False,
        "stats": {
            "sum": 0.0,
            "sumsq": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
            "rms": 0.0,
            "over_tol": 0,
        },
        "metadata": {
            "source_turn": turn,
            "mode": "post_eval_cpu_compare",
            "live_graph_nodes_added": 0,
            "live_backend_dispatches": 0,
            "output_consumed": 0,
            "cache_mutation": "disabled",
            "fixture_source": "accepted_turn_summary",
        },
    }


def write_stage_fixture(out_path: Path, records: list[dict]) -> None:
    apply_alias_diagnostics(records)
    out_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--capture", action="append", default=[], help="stage:/path/to/log")
    parser.add_argument("--capture-log", action="append", default=[], help="shortcut for hc_pre_norm:/path/to/log")
    parser.add_argument("--payload-dir", type=Path, default=None, help="directory containing raw payload side-files named <stage>_l<layer>_t<token>.bin")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for capture_log in args.capture_log:
        args.capture.append(f"hc_pre_norm:{capture_log}")
    if not args.capture:
        reason = "T77-T81 accepted captures contain stats-only side-probe summaries, not full tensor payloads"
        for stage, file_name in DEFAULT_FILES.items():
            records = [summary_fixture_record(stage, args.payload_dir)]
            if records[0].get("payload_kind") != "tensor_values":
                records.append(unavailable_record(stage, "full_tensor_payload", reason))
            append_hcnorm_input_records(records, args.payload_dir, {})
            append_rmoe_input_records(records, args.payload_dir, {})
            append_aohc_input_records(records, args.payload_dir, {})
            append_cupd_input_records(records, args.payload_dir, {}, {})
            if stage == "kv_cache_finalizer":
                byte_record = byte_payload_record_if_present(stage, STAGE_LAYERS[stage], 1, args.payload_dir)
                if byte_record is not None:
                    records.append(byte_record)
                else:
                    records.append(unavailable_record(stage, "quant_cache_byte_payload", "T77-T81 accepted captures contain stats-only side-probe summaries, not raw quant/cache byte payloads"))
            out_path = args.out / file_name
            write_stage_fixture(out_path, records)
            print(f"wrote {out_path}")
        return 0

    for item in args.capture:
        if ":" not in item:
            raise SystemExit(f"bad --capture value, expected stage:path: {item}")
        stage, raw_path = item.split(":", 1)
        if stage not in DEFAULT_FILES:
            raise SystemExit(f"unknown stage {stage!r}")
        raw_log = Path(raw_path)
        target_meta = parse_payload_target_metadata(raw_log)
        cupd_rope_meta = parse_cupd_rope_metadata(raw_log) if stage == "compressor_update" else {}
        record = apply_payload_if_present(normalize_capture(stage, raw_log), args.payload_dir)
        reason = "T77-T81 accepted captures contain stats-only side-probe summaries, not full tensor payloads"
        records = [record]
        if record.get("payload_kind") != "tensor_values":
            records.append(unavailable_record(stage, "full_tensor_payload", reason))
        append_hcnorm_input_records(records, args.payload_dir, target_meta)
        append_rmoe_input_records(records, args.payload_dir, target_meta)
        append_aohc_input_records(records, args.payload_dir, target_meta)
        append_cupd_input_records(records, args.payload_dir, target_meta, cupd_rope_meta)
        if stage == "kv_cache_finalizer":
            byte_record = byte_payload_record_if_present(stage, int(record["layer"]), int(record["token"]), args.payload_dir)
            if byte_record is not None:
                records.append(byte_record)
            else:
                records.append(unavailable_record(stage, "quant_cache_byte_payload", "T77-T81 accepted captures contain stats-only side-probe summaries, not raw quant/cache byte payloads"))
        out_path = args.out / DEFAULT_FILES[stage]
        write_stage_fixture(out_path, records)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
