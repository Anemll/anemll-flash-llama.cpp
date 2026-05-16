#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional


COUNTER_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)")


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            kind = obj.get("kind")
            if kind == "metadata":
                metadata.append(obj)
            elif kind == "logits" or "topk" in obj or "logits" in obj:
                records.append(obj)
    return metadata, records


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("missing required fields:\n  " + message)


def first_present(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return None


def as_int(value: Any, name: str) -> int:
    require(value is not None, name)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"field {name} is not int-like: {value!r}") from exc


def as_float(value: Any, name: str) -> float:
    require(value is not None, name)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"field {name} is not float-like: {value!r}") from exc


def token_index(record: dict[str, Any]) -> int:
    return as_int(first_present(record, "token_index", "idx", "token_idx", "eval_index"), "token_index/idx/token_idx")


def token_id(record: dict[str, Any]) -> int:
    return as_int(
        first_present(record, "sampled_token_id", "token_id", "id", "predicted_token_id", "top1_id"),
        "sampled_token_id/token_id/id/predicted_token_id/top1_id",
    )


def token_text(record: dict[str, Any], fallback_id: Optional[int] = None) -> str:
    value = first_present(record, "sampled_token_text", "token_text", "text", "piece", "predicted_token_text")
    if value is not None:
        return str(value)
    if fallback_id is not None:
        for item in topk_items(record):
            if item["id"] == fallback_id:
                return item["text"]
    return ""


def topk_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = first_present(record, "topk", "top20", "logits")
    require(isinstance(raw, list), "topk/top20/logits list")
    items: list[dict[str, Any]] = []
    for rank, item in enumerate(raw, 1):
        if isinstance(item, dict):
            item_id = first_present(item, "id", "token_id", "tok")
            logit = first_present(item, "logit", "value")
            text = first_present(item, "text", "token", "piece", "txt")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            item_id = item[0]
            logit = item[1]
            text = item[2] if len(item) > 2 else ""
        else:
            raise SystemExit(f"unsupported top-k item schema: {item!r}")
        raw_rank = item.get("rank", rank) if isinstance(item, dict) else rank
        items.append({
            "rank": as_int(raw_rank, "topk.rank"),
            "id": as_int(item_id, "topk.id"),
            "logit": as_float(logit, "topk.logit"),
            "text": "" if text is None else str(text),
        })
    require(len(items) >= 2, "at least two top-k entries")
    return items


def topk_map(record: dict[str, Any], limit: Optional[int] = None) -> dict[int, float]:
    items = topk_items(record)
    if limit is not None:
        items = items[:limit]
    return {item["id"]: item["logit"] for item in items}


def rank_of(record: dict[str, Any], target: int) -> Optional[int]:
    for item in topk_items(record):
        if item["id"] == target:
            return item["rank"]
    return None


def top_record(record: dict[str, Any], rank: int) -> dict[str, Any]:
    items = topk_items(record)
    require(len(items) >= rank, f"top-k rank {rank}")
    return items[rank - 1]


def top_logit(record: dict[str, Any], rank: int) -> float:
    key = f"top{rank}_logit"
    if key in record:
        return as_float(record[key], key)
    return top_record(record, rank)["logit"]


def top_id(record: dict[str, Any], rank: int) -> int:
    key = f"top{rank}_id"
    if key in record:
        return as_int(record[key], key)
    return top_record(record, rank)["id"]


def top_text(record: dict[str, Any], rank: int) -> str:
    key = f"top{rank}_text"
    if key in record:
        return str(record[key])
    return top_record(record, rank)["text"]


def top_margin(record: dict[str, Any]) -> float:
    if "top1_top2_margin" in record:
        return as_float(record["top1_top2_margin"], "top1_top2_margin")
    return top_logit(record, 1) - top_logit(record, 2)


def overlap_count(a: dict[str, Any], b: dict[str, Any], limit: int) -> int:
    return len(set(topk_map(a, limit)) & set(topk_map(b, limit)))


def logit_error_metrics(a: dict[str, Any], b: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    amap = topk_map(a, limit)
    bmap = topk_map(b, limit)
    overlap = sorted(set(amap) & set(bmap))
    diffs = [abs(amap[token] - bmap[token]) for token in overlap]
    if not diffs:
        return {
            "metric_scope": f"overlapping tokens in dumped top{limit}",
            "overlap_for_error": 0,
            "max_abs_logit_err": None,
            "mean_abs_logit_err": None,
            "rms_logit_err": None,
        }
    return {
        "metric_scope": f"overlapping tokens in dumped top{limit}",
        "overlap_for_error": len(overlap),
        "max_abs_logit_err": max(diffs),
        "mean_abs_logit_err": sum(diffs) / len(diffs),
        "rms_logit_err": math.sqrt(sum(d * d for d in diffs) / len(diffs)),
    }


def active_flags(metadata: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, Any]:
    flags = record.get("active_stage_flags")
    if isinstance(flags, dict):
        return flags
    for meta in reversed(metadata):
        flags = meta.get("active_stage_flags")
        if isinstance(flags, dict):
            return flags
    return {}


def metadata_value(metadata: list[dict[str, Any]], key: str) -> Any:
    for meta in reversed(metadata):
        if key in meta:
            return meta[key]
    return None


def hotpath_neutral_summary(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hot_path_neutral_validation": bool(metadata_value(metadata, "hot_path_neutral_validation")),
        "validation_hot_path_neutral": bool(metadata_value(metadata, "validation_hot_path_neutral")),
        "hot_path_neutral_guard_ok": bool(metadata_value(metadata, "hot_path_neutral_guard_ok")),
        "hot_path_neutral_guard_error": metadata_value(metadata, "hot_path_neutral_guard_error"),
        "validation_source": metadata_value(metadata, "validation_source"),
        "wall_clock_neutral": metadata_value(metadata, "wall_clock_neutral"),
        "sampling_n_probs_forced_for_validation": metadata_value(metadata, "sampling_n_probs_forced_for_validation"),
        "graph_eval_callback_registered_for_logit_dump": metadata_value(metadata, "graph_eval_callback_registered_for_logit_dump"),
        "ggml_graph_nodes_added_for_validation": metadata_value(metadata, "ggml_graph_nodes_added_for_validation"),
        "tensor_readbacks_added_for_validation": metadata_value(metadata, "tensor_readbacks_added_for_validation"),
        "consume_path_enabled_for_validation": metadata_value(metadata, "consume_path_enabled_for_validation"),
        "cache_mutation_enabled_for_validation": metadata_value(metadata, "cache_mutation_enabled_for_validation"),
        "hot_path_neutral_intrusive_flags_enabled": metadata_value(metadata, "hot_path_neutral_intrusive_flags_enabled") or [],
        "hot_path_neutral_rejected_paths_enabled": metadata_value(metadata, "hot_path_neutral_rejected_paths_enabled") or [],
        "experimental_under_test": bool(metadata_value(metadata, "experimental_under_test")),
        "under_test_name": metadata_value(metadata, "under_test_name") or "",
        "under_test_flags": metadata_value(metadata, "under_test_flags") or [],
        "under_test_changes_graph": bool(metadata_value(metadata, "under_test_changes_graph")),
        "path_accepted": bool(metadata_value(metadata, "path_accepted")),
        "acceptance_policy": metadata_value(metadata, "acceptance_policy"),
        "under_test_allow_rejected": bool(metadata_value(metadata, "under_test_allow_rejected")),
        "under_test_ack": metadata_value(metadata, "under_test_ack") or "",
        "skip_generic": metadata_value(metadata, "skip_generic"),
        "skip_generic_mode": metadata_value(metadata, "skip_generic_mode") or "",
        "fused": metadata_value(metadata, "fused"),
        "fused_consume": metadata_value(metadata, "fused_consume"),
        "backend_fused": metadata_value(metadata, "backend_fused"),
        "layers": metadata_value(metadata, "layers") or "",
        "layer_count": metadata_value(metadata, "layer_count"),
        "projection_source": metadata_value(metadata, "projection_source") or "",
        "cache_mutation_mode": metadata_value(metadata, "cache_mutation_mode") or "",
        "candidate_cache_side_effect": metadata_value(metadata, "candidate_cache_side_effect"),
        "skip_generic_tail": metadata_value(metadata, "skip_generic_tail"),
        "backend_tail": metadata_value(metadata, "backend_tail"),
        "backend_tail_consume": metadata_value(metadata, "backend_tail_consume"),
        "backend_tail_dep_barrier": metadata_value(metadata, "backend_tail_dep_barrier"),
        "backend_tail_emit_only": metadata_value(metadata, "backend_tail_emit_only"),
        "backend_tail_attn_row_probe": metadata_value(metadata, "backend_tail_attn_row_probe"),
        "backend_tail_value_probe": metadata_value(metadata, "backend_tail_value_probe"),
        "decode_compress_internal_probe": metadata_value(metadata, "decode_compress_internal_probe"),
        "backend_tail_attn_layout_mode": metadata_value(metadata, "backend_tail_attn_layout_mode") or "",
        "rmoe_backend_op": metadata_value(metadata, "rmoe_backend_op"),
        "rmoe_backend_consume": metadata_value(metadata, "rmoe_backend_consume"),
        "rmoe_backend_replace_generic": metadata_value(metadata, "rmoe_backend_replace_generic"),
        "replace_generic": metadata_value(metadata, "replace_generic"),
        "pair_preserve": metadata_value(metadata, "pair_preserve"),
        "pair_preserve_mode": metadata_value(metadata, "pair_preserve_mode") or "",
        "shared_final_only": metadata_value(metadata, "shared_final_only"),
        "shared_final_mode": metadata_value(metadata, "shared_final_mode") or "",
        "rmoe_backend_substage": metadata_value(metadata, "rmoe_backend_substage") or "",
        "swiglu_mode": metadata_value(metadata, "swiglu_mode") or "",
        "down_input": metadata_value(metadata, "down_input") or "",
        "rmoe_backend_branch_mode": metadata_value(metadata, "rmoe_backend_branch_mode") or "",
        "rmoe_backend_branch_order": metadata_value(metadata, "rmoe_backend_branch_order") or "",
        "consume_layer": metadata_value(metadata, "consume_layer") or "",
        "lowering_parity": metadata_value(metadata, "lowering_parity"),
        "lowering_parity_layer": metadata_value(metadata, "lowering_parity_layer") or "",
        "lowering_parity_token_min": metadata_value(metadata, "lowering_parity_token_min") or "",
        "lowering_parity_token_max": metadata_value(metadata, "lowering_parity_token_max") or "",
        "lowering_parity_mode": metadata_value(metadata, "lowering_parity_mode") or "",
        "generic_ffn_built": metadata_value(metadata, "generic_ffn_built"),
        "backend_ffn_built": metadata_value(metadata, "backend_ffn_built"),
        "dsv4_layer_executor_dryrun_op": metadata_value(metadata, "dsv4_layer_executor_dryrun_op"),
        "dsv4_layer_executor_dryrun_mode": metadata_value(metadata, "dsv4_layer_executor_dryrun_mode") or "",
        "dsv4_layer_executor_side_probe": metadata_value(metadata, "dsv4_layer_executor_side_probe"),
        "dsv4_layer_executor_side_probe_stage": metadata_value(metadata, "dsv4_layer_executor_side_probe_stage") or "",
        "dsv4_layer_executor_side_probe_mode": metadata_value(metadata, "dsv4_layer_executor_side_probe_mode") or "",
        "dsv4_layer_executor_live_graph_nodes_added": metadata_value(metadata, "dsv4_layer_executor_live_graph_nodes_added"),
        "dsv4_layer_executor_live_backend_dispatches": metadata_value(metadata, "dsv4_layer_executor_live_backend_dispatches"),
        "dsv4_layer_executor_output_consumed": metadata_value(metadata, "dsv4_layer_executor_output_consumed"),
        "dsv4_layer_executor_cache_mutation": metadata_value(metadata, "dsv4_layer_executor_cache_mutation") or "",
    }


UNDER_TEST_REJECTED_FLAGS = {
    "ffnmoe_v2_full_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME",
    },
    "cupd2_fused_comp": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
    },
    "down_sum6": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6",
    },
    "aohc_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
    },
    "aohc_single_layer_skip_generic": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC",
    },
    "aohc_fused_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "aohc_backend_fused_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "aohc_backend_fused_layer_set_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "cupd3_tail_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
    },
    "cupd3_backend_tail_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
    },
    "rmoe_backend_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
    },
    "rmoe_backend_single_layer_replace_generic": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
    },
    "rmoe_backend_pair_preserve_single_layer": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
    },
    "rmoe_backend_shared_final_single_layer": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
    },
    "rmoe_generic_lowering_parity": set(),
    "dsv4_layer_executor_metadata_only": set(),
    "dsv4_layer_executor_side_probe_hcnorm": set(),
    "dsv4_layer_executor_side_probe_rmoe": set(),
    "dsv4_layer_executor_side_probe_aohc": set(),
    "dsv4_layer_executor_side_probe_compressor_update": set(),
    "dsv4_layer_executor_side_probe_kv_cache_finalizer": set(),
}

UNDER_TEST_REQUIRED_FLAGS = {
    "ffnmoe_v2_full_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME",
    },
    "cupd2_fused_comp": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
    },
    "down_sum6": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6",
    },
    "aohc_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
    },
    "aohc_single_layer_skip_generic": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC",
    },
    "aohc_fused_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "aohc_backend_fused_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "aohc_backend_fused_layer_set_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
    },
    "cupd3_tail_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
    },
    "cupd3_backend_tail_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
    },
    "rmoe_backend_single_layer_consume": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
    },
    "rmoe_backend_single_layer_replace_generic": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
    },
    "rmoe_backend_pair_preserve_single_layer": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE",
    },
    "rmoe_backend_shared_final_single_layer": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_ONLY",
    },
    "rmoe_generic_lowering_parity": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY",
    },
    "dsv4_layer_executor_metadata_only": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
    },
    "dsv4_layer_executor_side_probe_hcnorm": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
    },
    "dsv4_layer_executor_side_probe_rmoe": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
    },
    "dsv4_layer_executor_side_probe_aohc": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
    },
    "dsv4_layer_executor_side_probe_compressor_update": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
    },
    "dsv4_layer_executor_side_probe_kv_cache_finalizer": {
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
    },
}


def require_hotpath_neutral(metadata: list[dict[str, Any]], label: str, allow_under_test: Optional[str] = None) -> None:
    summary = hotpath_neutral_summary(metadata)
    problems: list[str] = []
    if not summary["hot_path_neutral_validation"]:
        problems.append("hot_path_neutral_validation missing/false")
    if not summary["hot_path_neutral_guard_ok"]:
        problems.append("hot_path_neutral_guard_ok missing/false")
    if summary["validation_source"] != "server_prob_output":
        problems.append(f"validation_source={summary['validation_source']!r}")
    if summary["graph_eval_callback_registered_for_logit_dump"] not in (False, None):
        problems.append("graph_eval_callback_registered_for_logit_dump is true")
    if summary["ggml_graph_nodes_added_for_validation"] not in (False, None):
        problems.append("ggml_graph_nodes_added_for_validation is true")
    if summary["tensor_readbacks_added_for_validation"] not in (False, None):
        problems.append("tensor_readbacks_added_for_validation is true")
    if summary["consume_path_enabled_for_validation"] not in (False, None):
        problems.append("consume_path_enabled_for_validation is true")
    if summary["cache_mutation_enabled_for_validation"] not in (False, None):
        problems.append("cache_mutation_enabled_for_validation is true")
    if summary["hot_path_neutral_intrusive_flags_enabled"]:
        problems.append("intrusive flags enabled: " + ",".join(map(str, summary["hot_path_neutral_intrusive_flags_enabled"])))
    rejected = set(map(str, summary["hot_path_neutral_rejected_paths_enabled"]))
    under_test_flags = set(map(str, summary["under_test_flags"]))
    if allow_under_test is None:
        if summary["experimental_under_test"]:
            problems.append(f"experimental_under_test is true ({summary['under_test_name']})")
        if rejected:
            problems.append("rejected paths enabled: " + ",".join(sorted(rejected)))
    else:
        if allow_under_test not in UNDER_TEST_REJECTED_FLAGS:
            problems.append(f"unsupported --allow-under-test {allow_under_test!r}")
        if not summary["experimental_under_test"]:
            problems.append("experimental_under_test missing/false")
        if summary["under_test_name"] != allow_under_test:
            problems.append(f"under_test_name={summary['under_test_name']!r}, expected {allow_under_test!r}")
        if summary["path_accepted"]:
            problems.append("path_accepted is true")
        if summary["acceptance_policy"] != "transcript_exact":
            problems.append(f"acceptance_policy={summary['acceptance_policy']!r}")
        if not summary["under_test_allow_rejected"]:
            problems.append("under_test_allow_rejected missing/false")
        if summary["under_test_ack"] != "not_accepted":
            problems.append(f"under_test_ack={summary['under_test_ack']!r}")
        rmoe_branch_mode = summary.get("rmoe_backend_branch_mode") if allow_under_test == "rmoe_backend_single_layer_consume" else ""
        expected_rejected = UNDER_TEST_REJECTED_FLAGS.get(allow_under_test, set())
        expected_required = UNDER_TEST_REQUIRED_FLAGS.get(allow_under_test, set())
        if rmoe_branch_mode:
            expected_rejected = set()
            expected_required = {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP"}
        if rejected != expected_rejected:
            problems.append("rejected paths mismatch: expected " +
                            ",".join(sorted(expected_rejected)) + " actual " + ",".join(sorted(rejected)))
        if under_test_flags != expected_required:
            problems.append("under_test_flags mismatch: expected " +
                            ",".join(sorted(expected_required)) + " actual " + ",".join(sorted(under_test_flags)))
        if allow_under_test == "aohc_single_layer_skip_generic":
            if summary["skip_generic"] is not True:
                problems.append("skip_generic metadata missing/false")
            if summary["skip_generic_mode"] != "selected_layer_only":
                problems.append(f"skip_generic_mode={summary['skip_generic_mode']!r}")
        if allow_under_test in {"aohc_fused_single_layer_consume", "aohc_backend_fused_single_layer_consume", "aohc_backend_fused_layer_set_consume"}:
            if summary.get("fused") is not True:
                problems.append("fused metadata missing/false")
            if summary.get("fused_consume") is not True:
                problems.append("fused_consume metadata missing/false")
        if allow_under_test in {"aohc_backend_fused_single_layer_consume", "aohc_backend_fused_layer_set_consume"}:
            if summary.get("backend_fused") is not True:
                problems.append("backend_fused metadata missing/false")
        if allow_under_test == "aohc_backend_fused_layer_set_consume":
            if summary.get("layers") != "0,14,28,42":
                problems.append(f"layers metadata={summary.get('layers')!r}")
            if summary.get("layer_count") != 4:
                problems.append(f"layer_count metadata={summary.get('layer_count')!r}")
        if allow_under_test == "cupd3_tail_single_layer_consume":
            if summary.get("projection_source") != "generic":
                problems.append(f"projection_source={summary.get('projection_source')!r}")
            if summary.get("cache_mutation_mode") != "generic_existing_write":
                problems.append(f"cache_mutation_mode={summary.get('cache_mutation_mode')!r}")
            if summary.get("candidate_cache_side_effect") is not False:
                problems.append("candidate_cache_side_effect metadata missing/not false")
            if summary.get("skip_generic_tail") is not True:
                problems.append("skip_generic_tail metadata missing/false")
        if allow_under_test == "cupd3_backend_tail_single_layer_consume":
            if summary.get("projection_source") != "generic":
                problems.append(f"projection_source={summary.get('projection_source')!r}")
            if summary.get("cache_mutation_mode") != "generic_existing_write":
                problems.append(f"cache_mutation_mode={summary.get('cache_mutation_mode')!r}")
            if summary.get("candidate_cache_side_effect") is not False:
                problems.append("candidate_cache_side_effect metadata missing/not false")
            if summary.get("backend_tail") is not True:
                problems.append("backend_tail metadata missing/false")
            if summary.get("backend_tail_consume") is not True:
                problems.append("backend_tail_consume metadata missing/false")
        if allow_under_test == "rmoe_backend_single_layer_consume":
            if summary.get("rmoe_backend_op") is not True:
                problems.append("rmoe_backend_op metadata missing/false")
            if not rmoe_branch_mode and summary.get("rmoe_backend_consume") is not True:
                problems.append("rmoe_backend_consume metadata missing/false")
            if summary.get("rmoe_backend_substage") != "shared":
                problems.append(f"rmoe_backend_substage={summary.get('rmoe_backend_substage')!r}")
            if summary.get("consume_layer") not in ("0", 0):
                problems.append(f"consume_layer={summary.get('consume_layer')!r}")
        if allow_under_test == "rmoe_backend_single_layer_replace_generic":
            if summary.get("rmoe_backend_op") is not True:
                problems.append("rmoe_backend_op metadata missing/false")
            if summary.get("rmoe_backend_consume") is not True:
                problems.append("rmoe_backend_consume metadata missing/false")
            if summary.get("rmoe_backend_replace_generic") is not True:
                problems.append("rmoe_backend_replace_generic metadata missing/false")
            if summary.get("replace_generic") is not True:
                problems.append("replace_generic metadata missing/false")
            if summary.get("rmoe_backend_substage") != "shared":
                problems.append(f"rmoe_backend_substage={summary.get('rmoe_backend_substage')!r}")
            if summary.get("swiglu_mode") != "generic_graph_boundary":
                problems.append(f"swiglu_mode={summary.get('swiglu_mode')!r}")
            if summary.get("down_input") != "generic_graph_boundary":
                problems.append(f"down_input={summary.get('down_input')!r}")
            if summary.get("consume_layer") not in ("0", 0):
                problems.append(f"consume_layer={summary.get('consume_layer')!r}")
        if allow_under_test == "rmoe_backend_pair_preserve_single_layer":
            if summary.get("rmoe_backend_op") is not True:
                problems.append("rmoe_backend_op metadata missing/false")
            if summary.get("rmoe_backend_consume") is not True:
                problems.append("rmoe_backend_consume metadata missing/false")
            if summary.get("pair_preserve") is not True:
                problems.append("pair_preserve metadata missing/false")
            if summary.get("pair_preserve_mode") != "down_shared_from_generic_swiglu":
                problems.append(f"pair_preserve_mode={summary.get('pair_preserve_mode')!r}")
            if summary.get("consume_layer") not in ("0", 0):
                problems.append(f"consume_layer={summary.get('consume_layer')!r}")
        if allow_under_test == "rmoe_backend_shared_final_single_layer":
            if summary.get("rmoe_backend_op") is not True:
                problems.append("rmoe_backend_op metadata missing/false")
            if summary.get("rmoe_backend_consume") is not True:
                problems.append("rmoe_backend_consume metadata missing/false")
            if summary.get("shared_final_only") is not True:
                problems.append("shared_final_only metadata missing/false")
            if summary.get("shared_final_mode") != "shared_down_plus_final_add":
                problems.append(f"shared_final_mode={summary.get('shared_final_mode')!r}")
            if summary.get("consume_layer") not in ("0", 0):
                problems.append(f"consume_layer={summary.get('consume_layer')!r}")
        if allow_under_test == "rmoe_generic_lowering_parity":
            if summary.get("lowering_parity") is not True:
                problems.append("lowering_parity metadata missing/false")
            if summary.get("lowering_parity_layer") not in ("0", 0):
                problems.append(f"lowering_parity_layer={summary.get('lowering_parity_layer')!r}")
            if summary.get("generic_ffn_built") is not True:
                problems.append("generic_ffn_built metadata missing/false")
            if summary.get("backend_ffn_built") is not False:
                problems.append("backend_ffn_built metadata missing/not false")
        if allow_under_test == "dsv4_layer_executor_metadata_only":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
        if allow_under_test == "dsv4_layer_executor_side_probe_hcnorm":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_side_probe") is not True:
                problems.append("dsv4_layer_executor_side_probe metadata missing/false")
            if summary.get("dsv4_layer_executor_side_probe_stage") != "hc_pre_norm":
                problems.append(f"dsv4_layer_executor_side_probe_stage={summary.get('dsv4_layer_executor_side_probe_stage')!r}")
            if summary.get("dsv4_layer_executor_side_probe_mode") != "post_eval_cpu_compare":
                problems.append(f"dsv4_layer_executor_side_probe_mode={summary.get('dsv4_layer_executor_side_probe_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
        if allow_under_test == "dsv4_layer_executor_side_probe_rmoe":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_side_probe") is not True:
                problems.append("dsv4_layer_executor_side_probe metadata missing/false")
            if summary.get("dsv4_layer_executor_side_probe_stage") != "routed_moe_final_output":
                problems.append(f"dsv4_layer_executor_side_probe_stage={summary.get('dsv4_layer_executor_side_probe_stage')!r}")
            if summary.get("dsv4_layer_executor_side_probe_mode") != "post_eval_cpu_compare":
                problems.append(f"dsv4_layer_executor_side_probe_mode={summary.get('dsv4_layer_executor_side_probe_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
        if allow_under_test == "dsv4_layer_executor_side_probe_aohc":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_side_probe") is not True:
                problems.append("dsv4_layer_executor_side_probe metadata missing/false")
            if summary.get("dsv4_layer_executor_side_probe_stage") != "aohc_boundary":
                problems.append(f"dsv4_layer_executor_side_probe_stage={summary.get('dsv4_layer_executor_side_probe_stage')!r}")
            if summary.get("dsv4_layer_executor_side_probe_mode") != "post_eval_cpu_compare":
                problems.append(f"dsv4_layer_executor_side_probe_mode={summary.get('dsv4_layer_executor_side_probe_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
        if allow_under_test == "dsv4_layer_executor_side_probe_compressor_update":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_side_probe") is not True:
                problems.append("dsv4_layer_executor_side_probe metadata missing/false")
            if summary.get("dsv4_layer_executor_side_probe_stage") != "compressor_update":
                problems.append(f"dsv4_layer_executor_side_probe_stage={summary.get('dsv4_layer_executor_side_probe_stage')!r}")
            if summary.get("dsv4_layer_executor_side_probe_mode") != "post_eval_cpu_compare":
                problems.append(f"dsv4_layer_executor_side_probe_mode={summary.get('dsv4_layer_executor_side_probe_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
        if allow_under_test == "dsv4_layer_executor_side_probe_kv_cache_finalizer":
            if summary.get("dsv4_layer_executor_dryrun_op") is not True:
                problems.append("dsv4_layer_executor_dryrun_op metadata missing/false")
            if summary.get("dsv4_layer_executor_dryrun_mode") != "metadata_only":
                problems.append(f"dsv4_layer_executor_dryrun_mode={summary.get('dsv4_layer_executor_dryrun_mode')!r}")
            if summary.get("dsv4_layer_executor_side_probe") is not True:
                problems.append("dsv4_layer_executor_side_probe metadata missing/false")
            if summary.get("dsv4_layer_executor_side_probe_stage") != "kv_cache_finalizer":
                problems.append(f"dsv4_layer_executor_side_probe_stage={summary.get('dsv4_layer_executor_side_probe_stage')!r}")
            if summary.get("dsv4_layer_executor_side_probe_mode") != "post_eval_cpu_compare":
                problems.append(f"dsv4_layer_executor_side_probe_mode={summary.get('dsv4_layer_executor_side_probe_mode')!r}")
            if summary.get("dsv4_layer_executor_live_graph_nodes_added") is not False:
                problems.append("dsv4_layer_executor_live_graph_nodes_added metadata missing/not false")
            if summary.get("dsv4_layer_executor_live_backend_dispatches") is not False:
                problems.append("dsv4_layer_executor_live_backend_dispatches metadata missing/not false")
            if summary.get("dsv4_layer_executor_output_consumed") is not False:
                problems.append("dsv4_layer_executor_output_consumed metadata missing/not false")
            if summary.get("dsv4_layer_executor_cache_mutation") != "disabled":
                problems.append(f"dsv4_layer_executor_cache_mutation={summary.get('dsv4_layer_executor_cache_mutation')!r}")
    if problems:
        raise SystemExit(f"{label}: dump is not hot-path-neutral:\n  " + "\n  ".join(problems))


def infer_log_path(jsonl_path: Path) -> Optional[Path]:
    candidate = jsonl_path.with_suffix(".log")
    return candidate if candidate.exists() else None


def parse_counters(log_path: Optional[Path]) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {}
    counters: dict[str, Any] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        if "ggml_metal_op_mul_mat_log_stats:" not in line and "ggml_metal_op_mul_mat_id_log_stats:" not in line:
            continue
        for key, value in COUNTER_RE.findall(line):
            if "." in value:
                counters[key] = float(value)
            else:
                counters[key] = int(value)
    return counters


def pair_summary(
    baseline: dict[str, Any],
    fused: dict[str, Any],
    baseline_meta: list[dict[str, Any]],
    fused_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    b_token = token_id(baseline)
    f_token = token_id(fused)
    return {
        "token_index": token_index(baseline),
        "position": first_present(baseline, "position", "pos"),
        "baseline_token_id": b_token,
        "baseline_token_text": token_text(baseline, b_token),
        "fused_token_id": f_token,
        "fused_token_text": token_text(fused, f_token),
        "baseline_top1_id": top_id(baseline, 1),
        "baseline_top1_text": top_text(baseline, 1),
        "baseline_top1_logit": top_logit(baseline, 1),
        "baseline_top2_id": top_id(baseline, 2),
        "baseline_top2_text": top_text(baseline, 2),
        "baseline_top2_logit": top_logit(baseline, 2),
        "baseline_top1_top2_margin": top_margin(baseline),
        "fused_top1_id": top_id(fused, 1),
        "fused_top1_text": top_text(fused, 1),
        "fused_top1_logit": top_logit(fused, 1),
        "fused_top2_id": top_id(fused, 2),
        "fused_top2_text": top_text(fused, 2),
        "fused_top2_logit": top_logit(fused, 2),
        "fused_top1_top2_margin": top_margin(fused),
        "baseline_top1_rank_in_fused": rank_of(fused, top_id(baseline, 1)),
        "fused_top1_rank_in_baseline": rank_of(baseline, top_id(fused, 1)),
        "top10_overlap": overlap_count(baseline, fused, 10),
        "top20_overlap": overlap_count(baseline, fused, 20),
        "baseline_active_flags": active_flags(baseline_meta, baseline),
        "fused_active_flags": active_flags(fused_meta, fused),
        **logit_error_metrics(baseline, fused, 20),
    }


def classify(summary: Optional[dict[str, Any]], no_divergence: bool, max_seen_rms: Optional[float]) -> str:
    if no_divergence:
        if max_seen_rms is not None and max_seen_rms < 1.0e-3:
            return "Case C: no divergence through compared records with tiny top-k-overlap logit error"
        return "Case C/D: no token divergence through compared records; inspect accumulated logit error"
    assert summary is not None
    margin = abs(float(summary["baseline_top1_top2_margin"]))
    rms = summary.get("rms_logit_err")
    max_abs = summary.get("max_abs_logit_err")
    top20 = int(summary["top20_overlap"])
    b_rank = summary["baseline_top1_rank_in_fused"]
    f_rank = summary["fused_top1_rank_in_baseline"]
    high_rank = (b_rank is not None and int(b_rank) <= 5 and f_rank is not None and int(f_rank) <= 5)
    if margin < 0.25 and high_rank and top20 >= 15 and rms is not None and float(rms) < 0.25:
        return "Case A: likely tolerance-policy issue; do not accept automatically"
    if (margin >= 1.0 or
            not high_rank or
            top20 < 12 or
            (rms is not None and float(rms) >= 1.0) or
            (max_abs is not None and float(max_abs) >= 1.0)):
        return "Case B: real numerical bug signal; path remains rejected"
    return "Case D: late/accumulated drift candidate; rejected under transcript-exact policy"


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def write_text_report(
    baseline_path: Path,
    fused_path: Path,
    baseline_meta: list[dict[str, Any]],
    fused_meta: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    fused_records: list[dict[str, Any]],
    first: Optional[dict[str, Any]],
    last: Optional[dict[str, Any]],
    baseline_counters: dict[str, Any],
    fused_counters: dict[str, Any],
    classification: str,
) -> str:
    lines: list[str] = []
    lines.append("DSV4 first-divergence report")
    lines.append(f"baseline_jsonl: {baseline_path}")
    lines.append(f"fused_jsonl: {fused_path}")
    lines.append(f"baseline_records: {len(baseline_records)}")
    lines.append(f"fused_records: {len(fused_records)}")
    lines.append("hot_path_neutral:")
    lines.append(f"  baseline: {json.dumps(hotpath_neutral_summary(baseline_meta), sort_keys=True)}")
    lines.append(f"  fused:    {json.dumps(hotpath_neutral_summary(fused_meta), sort_keys=True)}")
    lines.append("")
    if first is None:
        lines.append("first divergent token index: none within aligned records")
        if last is not None:
            lines.append("")
            lines.append("last aligned token summary:")
            append_summary(lines, last)
    else:
        lines.append(f"first divergent token index: {first['token_index']}")
        append_summary(lines, first)
    lines.append("")
    lines.append("active counters:")
    lines.append(f"  baseline: {json.dumps(baseline_counters, sort_keys=True)}")
    lines.append(f"  fused:    {json.dumps(fused_counters, sort_keys=True)}")
    lines.append("")
    lines.append(f"interpretation: {classification}")
    lines.append("metric note: max/mean/rms logit errors are over overlapping tokens in the dumped top20 lists, not full vocabulary logits.")
    return "\n".join(lines) + "\n"


def append_summary(lines: list[str], summary: dict[str, Any]) -> None:
    keys = [
        "position",
        "baseline_token_id",
        "baseline_token_text",
        "fused_token_id",
        "fused_token_text",
        "baseline_top1_id",
        "baseline_top1_text",
        "baseline_top1_logit",
        "baseline_top2_id",
        "baseline_top2_text",
        "baseline_top2_logit",
        "baseline_top1_top2_margin",
        "fused_top1_id",
        "fused_top1_text",
        "fused_top1_logit",
        "fused_top2_id",
        "fused_top2_text",
        "fused_top2_logit",
        "fused_top1_top2_margin",
        "baseline_top1_rank_in_fused",
        "fused_top1_rank_in_baseline",
        "top10_overlap",
        "top20_overlap",
        "max_abs_logit_err",
        "mean_abs_logit_err",
        "rms_logit_err",
        "metric_scope",
    ]
    for key in keys:
        lines.append(f"  {key}: {fmt(summary.get(key))}")
    lines.append(f"  baseline_active_flags: {json.dumps(summary.get('baseline_active_flags', {}), sort_keys=True)}")
    lines.append(f"  fused_active_flags: {json.dumps(summary.get('fused_active_flags', {}), sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report first token divergence between two DSV4 logit JSONL dumps.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("--baseline-log", type=Path, default=None)
    parser.add_argument("--fused-log", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="optional machine-readable summary output")
    parser.add_argument("--require-hotpath-neutral", action="store_true",
                        help="fail unless both dumps declare hot-path-neutral validation metadata")
    parser.add_argument("--allow-under-test", choices=sorted(UNDER_TEST_REJECTED_FLAGS),
                        help="permit one explicit rejected-path under-test dump on the fused side")
    args = parser.parse_args()

    baseline_meta, baseline_records = load_jsonl(args.baseline)
    fused_meta, fused_records = load_jsonl(args.fused)
    if args.require_hotpath_neutral:
        require_hotpath_neutral(baseline_meta, "baseline")
        require_hotpath_neutral(fused_meta, "fused", args.allow_under_test)
    elif args.allow_under_test is not None:
        raise SystemExit("--allow-under-test requires --require-hotpath-neutral")
    require(bool(baseline_records), f"{args.baseline}: no logit records")
    require(bool(fused_records), f"{args.fused}: no logit records")

    first = None
    last = None
    max_seen_rms: Optional[float] = None
    n = min(len(baseline_records), len(fused_records))
    for i in range(n):
        summary = pair_summary(baseline_records[i], fused_records[i], baseline_meta, fused_meta)
        last = summary
        rms = summary.get("rms_logit_err")
        if rms is not None:
            max_seen_rms = max(float(rms), max_seen_rms if max_seen_rms is not None else 0.0)
        if summary["baseline_token_id"] != summary["fused_token_id"]:
            first = summary
            break

    baseline_log = args.baseline_log or infer_log_path(args.baseline)
    fused_log = args.fused_log or infer_log_path(args.fused)
    baseline_counters = parse_counters(baseline_log)
    fused_counters = parse_counters(fused_log)
    classification = classify(first, first is None, max_seen_rms)

    report = write_text_report(
        args.baseline,
        args.fused,
        baseline_meta,
        fused_meta,
        baseline_records,
        fused_records,
        first,
        last,
        baseline_counters,
        fused_counters,
        classification,
    )
    print(report, end="")

    if args.json is not None:
        payload = {
            "baseline_jsonl": str(args.baseline),
            "fused_jsonl": str(args.fused),
            "baseline_records": len(baseline_records),
            "fused_records": len(fused_records),
            "baseline_hot_path_neutral": hotpath_neutral_summary(baseline_meta),
            "fused_hot_path_neutral": hotpath_neutral_summary(fused_meta),
            "first_divergence": first,
            "last_aligned": last,
            "baseline_counters": baseline_counters,
            "fused_counters": fused_counters,
            "classification": classification,
        }
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
