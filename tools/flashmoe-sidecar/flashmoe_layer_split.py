#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import flashmoe_sidecar as sidecar_tools
from gguf import GGUFReader  # type: ignore

GiB = float(1024 ** 3)


@dataclass
class NodeSpec:
    name: str
    mode: str
    usable_gib: float
    slot_bank: int


@dataclass
class LayerCosts:
    dense_gib: float
    expert_full_gib: float
    expert_slot_bank_gib: float
    kv_gib: float


@dataclass
class BoundaryInfo:
    from_node: str
    to_node: str
    after_layer: int
    activation_bytes_per_token: int
    activation_mib_per_token: float
    activation_bytes_per_ubatch: int
    activation_mib_per_ubatch: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan a contiguous multi-machine layer split for a Flash-MoE model. "
            "Resident nodes keep their assigned routed layers fully resident; flash nodes "
            "budget dense/shared + KV + slot-bank reserve."
        ),
    )
    parser.add_argument("--model", required=True, type=Path, help="canonical GGUF path")
    parser.add_argument("--sidecar", required=True, type=Path, help="Flash-MoE sidecar directory or manifest path")
    parser.add_argument(
        "--node",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "node spec as comma-separated key=value pairs, e.g. "
            "'name=m3u,mode=resident,usable_gib=76' or "
            "'name=m5max,mode=flash,usable_gib=100,slot_bank=96'"
        ),
    )
    parser.add_argument(
        "--ctx",
        type=int,
        help="context length to budget for; defaults to model context_length",
    )
    parser.add_argument(
        "--kv-bytes",
        type=int,
        default=2,
        help="bytes per K/V element (default: 2 for fp16/bf16-style KV)",
    )
    parser.add_argument(
        "--activation-bytes",
        type=int,
        default=2,
        help="bytes per hidden-state element exchanged at a machine boundary (default: 2)",
    )
    parser.add_argument(
        "--ubatch",
        type=int,
        default=1,
        help="micro-batch size used for boundary payload estimates (default: 1)",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "tcp", "rdma"],
        default="auto",
        help="transport kind to write into the plan (default: auto)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    return parser.parse_args()


def parse_node_spec(raw: str) -> NodeSpec:
    values: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"invalid --node entry '{raw}': expected key=value items")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()

    missing = [key for key in ("name", "mode", "usable_gib") if key not in values]
    if missing:
        raise SystemExit(f"invalid --node entry '{raw}': missing {', '.join(missing)}")

    mode = values["mode"]
    if mode not in {"resident", "flash"}:
        raise SystemExit(f"invalid node mode '{mode}' in '{raw}': expected resident or flash")

    try:
        usable_gib = float(values["usable_gib"])
    except ValueError as exc:
        raise SystemExit(f"invalid usable_gib in '{raw}': {exc}") from exc

    slot_bank = 0
    if "slot_bank" in values:
        try:
            slot_bank = int(values["slot_bank"])
        except ValueError as exc:
            raise SystemExit(f"invalid slot_bank in '{raw}': {exc}") from exc
    if mode == "resident" and slot_bank != 0:
        raise SystemExit(f"resident node '{values['name']}' must not specify slot_bank")

    return NodeSpec(
        name=values["name"],
        mode=mode,
        usable_gib=usable_gib,
        slot_bank=slot_bank,
    )


def reader_scalar(reader: GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def load_model_geometry(model_path: Path, sidecar_path: Path, ctx_override: int | None, kv_bytes: int) -> dict[str, Any]:
    model_paths = sidecar_tools.resolve_model_paths(model_path)
    reader = GGUFReader(str(model_paths[0]), "r")
    arch = str(reader_scalar(reader, "general.architecture", "unknown"))

    n_layer = int(reader_scalar(reader, f"{arch}.block_count"))
    n_layer_dense_lead = int(reader_scalar(reader, f"{arch}.leading_dense_block_count", 0))
    n_ctx = int(ctx_override or reader_scalar(reader, f"{arch}.context_length"))
    n_embd = int(reader_scalar(reader, f"{arch}.embedding_length"))
    key_length = int(reader_scalar(reader, f"{arch}.attention.key_length"))
    value_length = int(reader_scalar(reader, f"{arch}.attention.value_length"))
    expert_count = int(reader_scalar(reader, f"{arch}.expert_count"))

    layer_dense_bytes: dict[int, int] = {layer: 0 for layer in range(n_layer)}
    input_bytes = 0
    output_bytes = 0
    misc_bytes = 0

    for path in model_paths:
        shard = GGUFReader(str(path), "r")
        for tensor in shard.tensors:
            name = tensor.name
            nbytes = int(tensor.n_bytes)
            if name.startswith("blk."):
                try:
                    layer = int(name.split(".")[1])
                except (IndexError, ValueError):
                    misc_bytes += nbytes
                    continue
                layer_dense_bytes[layer] += nbytes
                continue

            if name == "token_embd.weight":
                input_bytes += nbytes
            elif name == "output.weight" or name == "output_norm.weight":
                output_bytes += nbytes
            else:
                misc_bytes += nbytes

    manifest_path, manifest = sidecar_tools.load_manifest(sidecar_path)
    entries = sidecar_tools.filter_manifest_entries(
        manifest["entries"],
        family_filter=sidecar_tools.ROUTED_FAMILIES,
    )

    expert_bytes_per_layer: dict[int, int] = {layer: 0 for layer in range(n_layer)}
    bytes_per_slot_per_layer: dict[int, int] = {layer: 0 for layer in range(n_layer)}
    for entry in entries:
        layer = int(entry["layer"])
        bpe = int(entry["bytes_per_expert"])
        bytes_per_slot_per_layer[layer] += bpe
        expert_bytes_per_layer[layer] += bpe * expert_count

    kv_bytes_per_layer = n_ctx * (key_length + value_length) * kv_bytes

    return {
        "arch": arch,
        "model_paths": [str(path) for path in model_paths],
        "manifest_path": str(manifest_path),
        "n_layer": n_layer,
        "n_layer_dense_lead": n_layer_dense_lead,
        "n_layer_routed": n_layer - n_layer_dense_lead,
        "n_ctx": n_ctx,
        "n_embd": n_embd,
        "key_length": key_length,
        "value_length": value_length,
        "expert_count": expert_count,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "misc_bytes": misc_bytes,
        "layer_dense_bytes": layer_dense_bytes,
        "expert_bytes_per_layer": expert_bytes_per_layer,
        "bytes_per_slot_per_layer": bytes_per_slot_per_layer,
        "kv_bytes_per_layer": kv_bytes_per_layer,
    }


def layer_costs_for_node(geometry: dict[str, Any], node: NodeSpec, layer: int) -> LayerCosts:
    dense_gib = geometry["layer_dense_bytes"][layer] / GiB
    kv_gib = geometry["kv_bytes_per_layer"] / GiB
    expert_full_gib = geometry["expert_bytes_per_layer"][layer] / GiB
    expert_slot_bank_gib = (geometry["bytes_per_slot_per_layer"][layer] * node.slot_bank) / GiB
    return LayerCosts(
        dense_gib=dense_gib,
        expert_full_gib=expert_full_gib,
        expert_slot_bank_gib=expert_slot_bank_gib,
        kv_gib=kv_gib,
    )


def total_layer_cost_gib(geometry: dict[str, Any], node: NodeSpec, layer: int) -> float:
    costs = layer_costs_for_node(geometry, node, layer)
    if layer < geometry["n_layer_dense_lead"]:
        return costs.dense_gib + costs.kv_gib
    if node.mode == "resident":
        return costs.dense_gib + costs.expert_full_gib + costs.kv_gib
    return costs.dense_gib + costs.expert_slot_bank_gib + costs.kv_gib


def layer_mode_label(geometry: dict[str, Any], layer: int) -> str:
    return "dense" if layer < geometry["n_layer_dense_lead"] else "routed"


def recommend_transport(kind: str) -> dict[str, str]:
    if kind == "rdma":
        return {
            "kind": "rdma",
            "note": "User-requested RDMA. Keep the runtime transport abstract; the first execution path should still preserve a TCP fallback.",
        }
    if kind == "tcp":
        return {
            "kind": "tcp",
            "note": "TCP over a private Thunderbolt IP link is the lowest-risk first transport and matches the existing RPC direction in this repo.",
        }
    return {
        "kind": "tcp",
        "note": "Auto-selected TCP for the first execution path. It is simpler to debug on macOS/Thunderbolt; keep RDMA as a future interchangeable transport.",
    }


def plan_split(geometry: dict[str, Any], nodes: list[NodeSpec], activation_bytes: int, ubatch: int, transport: str) -> dict[str, Any]:
    if not nodes:
        raise SystemExit("pass at least one --node")

    next_layer = 0
    n_layer = geometry["n_layer"]
    plan_nodes: list[dict[str, Any]] = []

    for index, node in enumerate(nodes):
        is_first = index == 0
        is_last = index == len(nodes) - 1

        fixed_gib = 0.0
        fixed_breakdown: dict[str, float] = {}
        if is_first and geometry["input_bytes"] > 0:
            fixed_gib += geometry["input_bytes"] / GiB
            fixed_breakdown["input_embed_gib"] = geometry["input_bytes"] / GiB
        if is_last and geometry["output_bytes"] > 0:
            fixed_gib += geometry["output_bytes"] / GiB
            fixed_breakdown["output_head_gib"] = geometry["output_bytes"] / GiB
        if is_first and geometry["misc_bytes"] > 0:
            fixed_gib += geometry["misc_bytes"] / GiB
            fixed_breakdown["misc_global_gib"] = geometry["misc_bytes"] / GiB

        remaining_gib = node.usable_gib - fixed_gib
        if remaining_gib < 0:
            remaining_gib = 0.0

        layer_start = next_layer
        layer_end = next_layer - 1
        layer_breakdown: list[dict[str, Any]] = []
        used_gib = fixed_gib

        while next_layer < n_layer:
            total_gib = total_layer_cost_gib(geometry, node, next_layer)
            if used_gib + total_gib > node.usable_gib:
                break

            costs = layer_costs_for_node(geometry, node, next_layer)
            layer_breakdown.append(
                {
                    "layer": next_layer,
                    "kind": layer_mode_label(geometry, next_layer),
                    "dense_gib": round(costs.dense_gib, 4),
                    "expert_full_gib": round(costs.expert_full_gib, 4),
                    "expert_slot_bank_gib": round(costs.expert_slot_bank_gib, 4),
                    "kv_gib": round(costs.kv_gib, 4),
                    "total_gib": round(total_gib, 4),
                }
            )
            used_gib += total_gib
            layer_end = next_layer
            next_layer += 1

        plan_nodes.append(
            {
                "name": node.name,
                "mode": node.mode,
                "usable_gib": round(node.usable_gib, 3),
                "slot_bank": node.slot_bank,
                "layer_start": layer_start if layer_start <= layer_end else None,
                "layer_end": layer_end if layer_start <= layer_end else None,
                "layer_count": len(layer_breakdown),
                "fixed_gib": round(fixed_gib, 4),
                "fixed_breakdown": {key: round(value, 4) for key, value in fixed_breakdown.items()},
                "used_gib": round(used_gib, 4),
                "free_gib": round(max(0.0, node.usable_gib - used_gib), 4),
                "layers": layer_breakdown,
            }
        )

    unassigned_layers = list(range(next_layer, n_layer))
    boundaries: list[BoundaryInfo] = []
    activation_per_token = geometry["n_embd"] * activation_bytes
    activation_per_ubatch = activation_per_token * ubatch
    for left, right in zip(plan_nodes, plan_nodes[1:]):
        if left["layer_end"] is None or right["layer_start"] is None:
            continue
        boundaries.append(
            BoundaryInfo(
                from_node=left["name"],
                to_node=right["name"],
                after_layer=int(left["layer_end"]),
                activation_bytes_per_token=activation_per_token,
                activation_mib_per_token=activation_per_token / float(1024 * 1024),
                activation_bytes_per_ubatch=activation_per_ubatch,
                activation_mib_per_ubatch=activation_per_ubatch / float(1024 * 1024),
            )
        )

    return {
        "version": 1,
        "model": {
            "arch": geometry["arch"],
            "model_paths": geometry["model_paths"],
            "sidecar_manifest": geometry["manifest_path"],
            "layer_count": geometry["n_layer"],
            "leading_dense_block_count": geometry["n_layer_dense_lead"],
            "routed_layer_count": geometry["n_layer_routed"],
            "context_length": geometry["n_ctx"],
            "embedding_length": geometry["n_embd"],
            "expert_count": geometry["expert_count"],
            "key_length": geometry["key_length"],
            "value_length": geometry["value_length"],
            "kv_gib_per_layer": round(geometry["kv_bytes_per_layer"] / GiB, 4),
            "input_embed_gib": round(geometry["input_bytes"] / GiB, 4),
            "output_head_gib": round(geometry["output_bytes"] / GiB, 4),
            "misc_global_gib": round(geometry["misc_bytes"] / GiB, 4),
        },
        "transport": recommend_transport(transport),
        "boundary_activation": {
            "bytes_per_element": activation_bytes,
            "bytes_per_token": activation_per_token,
            "mib_per_token": round(activation_per_token / float(1024 * 1024), 6),
            "ubatch": ubatch,
            "bytes_per_ubatch": activation_per_ubatch,
            "mib_per_ubatch": round(activation_per_ubatch / float(1024 * 1024), 6),
        },
        "nodes": plan_nodes,
        "boundaries": [asdict(boundary) for boundary in boundaries],
        "unassigned_layers": unassigned_layers,
    }


def print_text_plan(plan: dict[str, Any]) -> None:
    model = plan["model"]
    print(f"model arch: {model['arch']}")
    print(
        f"layers: total={model['layer_count']} dense-lead={model['leading_dense_block_count']} routed={model['routed_layer_count']}"
    )
    print(
        f"context={model['context_length']} embd={model['embedding_length']} "
        f"kv/layer={model['kv_gib_per_layer']:.3f} GiB"
    )
    print(
        f"global bytes: input={model['input_embed_gib']:.3f} GiB "
        f"output={model['output_head_gib']:.3f} GiB misc={model['misc_global_gib']:.3f} GiB"
    )
    print(f"transport: {plan['transport']['kind']} ({plan['transport']['note']})")
    print(
        f"boundary activation: {plan['boundary_activation']['mib_per_token']:.4f} MiB/token, "
        f"{plan['boundary_activation']['mib_per_ubatch']:.4f} MiB/ubatch"
    )
    print("")

    for node in plan["nodes"]:
        layer_span = "none"
        if node["layer_start"] is not None:
            layer_span = f"{node['layer_start']}-{node['layer_end']}"
        print(
            f"{node['name']}: mode={node['mode']} layers={layer_span} count={node['layer_count']} "
            f"used={node['used_gib']:.2f}/{node['usable_gib']:.2f} GiB free={node['free_gib']:.2f} GiB"
        )
        if node["fixed_breakdown"]:
            fixed = ", ".join(f"{key}={value:.3f} GiB" for key, value in node["fixed_breakdown"].items())
            print(f"  fixed: {fixed}")
        if node["layers"]:
            first = node["layers"][0]
            last = node["layers"][-1]
            print(
                f"  layer costs: first blk.{first['layer']}={first['total_gib']:.3f} GiB, "
                f"last blk.{last['layer']}={last['total_gib']:.3f} GiB"
            )
    if plan["boundaries"]:
        print("")
        for boundary in plan["boundaries"]:
            print(
                f"boundary {boundary['from_node']} -> {boundary['to_node']}: "
                f"after blk.{boundary['after_layer']}, "
                f"{boundary['activation_mib_per_token']:.4f} MiB/token"
            )
    if plan["unassigned_layers"]:
        print("")
        print(
            f"unassigned layers: {plan['unassigned_layers'][0]}-{plan['unassigned_layers'][-1]} "
            f"({len(plan['unassigned_layers'])} layers)"
        )


def main() -> int:
    args = parse_args()
    nodes = [parse_node_spec(raw) for raw in args.node]
    geometry = load_model_geometry(args.model, args.sidecar, args.ctx, args.kv_bytes)
    plan = plan_split(geometry, nodes, args.activation_bytes, args.ubatch, args.transport)

    payload = json.dumps(plan, indent=2, sort_keys=False)
    if args.out is not None:
        args.out.expanduser().resolve().write_text(payload + "\n", encoding="utf-8")

    if args.json or args.out is not None:
        print(payload)
    else:
        print_text_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
