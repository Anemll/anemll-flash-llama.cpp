#!/usr/bin/env python3
"""Compare routed-MoE graph/lowering trace JSONL dumps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '"format":"dsv4-rmoe-graph-trace-v1"' not in line:
                continue
            row = json.loads(line)
            if row.get("format") != "dsv4-rmoe-graph-trace-v1":
                continue
            out[int(row["token"])] = row
    return out


def node_sig(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        node.get("tensor_name"),
        node.get("op"),
        node.get("src0"),
        node.get("src1"),
        tuple(node.get("shape", [])),
        node.get("dtype"),
        node.get("stage_bucket"),
        node.get("consumer_count"),
    )


def node_order_sig(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        node.get("tensor_name"),
        node.get("op"),
        node.get("stage_bucket"),
    )


def first_counter_diff(a: dict[str, Any], b: dict[str, Any], key: str) -> tuple[str, Any, Any] | None:
    av = a.get(key, {})
    bv = b.get(key, {})
    for name in sorted(set(av) | set(bv)):
        if av.get(name, 0) != bv.get(name, 0):
            return name, av.get(name, 0), bv.get(name, 0)
    return None


def first_node_diff(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, Any, Any] | None:
    an = a.get("nodes", [])
    bn = b.get("nodes", [])
    n = min(len(an), len(bn))
    for idx in range(n):
        if node_sig(an[idx]) != node_sig(bn[idx]):
            return idx, an[idx], bn[idx]
    if len(an) != len(bn):
        return n, an[n] if n < len(an) else None, bn[n] if n < len(bn) else None
    return None


def first_node_order_diff(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, Any, Any] | None:
    an = a.get("nodes", [])
    bn = b.get("nodes", [])
    n = min(len(an), len(bn))
    for idx in range(n):
        if node_order_sig(an[idx]) != node_order_sig(bn[idx]):
            return idx, an[idx], bn[idx]
    if len(an) != len(bn):
        return n, an[n] if n < len(an) else None, bn[n] if n < len(bn) else None
    return None


def summarize_node(node: Any) -> str:
    if node is None:
        return "none"
    keys = ["node_index", "tensor_name", "op", "src0", "src1", "shape", "dtype", "stage_bucket", "consumer_count"]
    return "; ".join(f"{key}={node.get(key)}" for key in keys)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dsv4_compare_rmoe_graph_trace.py BASELINE.jsonl REPLACE.jsonl", file=sys.stderr)
        return 2

    baseline = load(sys.argv[1])
    replace = load(sys.argv[2])
    common = sorted(set(baseline) & set(replace))

    print(f"baseline graph rows: {len(baseline)}")
    print(f"replace graph rows: {len(replace)}")
    print(f"common tokens: {len(common)}")
    print(f"token alignment confirmed: {'yes' if common and len(common) == len(baseline) == len(replace) else 'no'}")

    first_token_counter = None
    first_counter = None
    first_token_stage = None
    first_stage = None
    first_token_node = None
    first_node = None
    first_token_order = None
    first_order = None
    first_token_signature = None

    for token in common:
        a = baseline[token]
        b = replace[token]
        if first_token_signature is None and (
            a.get("node_count_total") != b.get("node_count_total")
            or a.get("signature_hash") != b.get("signature_hash")
            or a.get("relevant_node_count") != b.get("relevant_node_count")
        ):
            first_token_signature = token
        if first_token_counter is None:
            diff = first_counter_diff(a, b, "op_counts")
            if diff is not None:
                first_token_counter = token
                first_counter = diff
        if first_token_stage is None:
            diff = first_counter_diff(a, b, "stage_bucket_counts")
            if diff is not None:
                first_token_stage = token
                first_stage = diff
        if first_token_node is None:
            diff = first_node_diff(a, b)
            if diff is not None:
                first_token_node = token
                first_node = diff
        if first_token_order is None:
            diff = first_node_order_diff(a, b)
            if diff is not None:
                first_token_order = token
                first_order = diff

    print("")
    print(f"first token where graph signature/count differs: {first_token_signature if first_token_signature is not None else 'none'}")
    if first_token_signature is not None:
        a = baseline[first_token_signature]
        b = replace[first_token_signature]
        print(f"  baseline node_count/signature/relevant: {a.get('node_count_total')} / {a.get('signature_hash')} / {a.get('relevant_node_count')}")
        print(f"  replace  node_count/signature/relevant: {b.get('node_count_total')} / {b.get('signature_hash')} / {b.get('relevant_node_count')}")

    print(f"first token where graph counters differ: {first_token_counter if first_token_counter is not None else 'none'}")
    if first_counter is not None:
        name, av, bv = first_counter
        print(f"  first differing counter: {name} baseline={av} replace={bv}")

    print(f"first token where stage bucket counts differ: {first_token_stage if first_token_stage is not None else 'none'}")
    if first_stage is not None:
        name, av, bv = first_stage
        print(f"  first differing stage bucket: {name} baseline={av} replace={bv}")

    print(f"first token where node order differs: {first_token_order if first_token_order is not None else 'none'}")
    if first_order is not None:
        idx, an, bn = first_order
        print(f"  order index: {idx}")
        print(f"  baseline: {summarize_node(an)}")
        print(f"  replace:  {summarize_node(bn)}")

    print(f"first token where producer/consumer node signature differs: {first_token_node if first_token_node is not None else 'none'}")
    if first_node is not None:
        idx, an, bn = first_node
        print(f"  node index: {idx}")
        print(f"  baseline: {summarize_node(an)}")
        print(f"  replace:  {summarize_node(bn)}")

    if common:
        token = first_token_signature or common[0]
        print("")
        print(f"counter summary at token {token}:")
        print(f"  baseline op_counts: {baseline[token].get('op_counts')}")
        print(f"  replace op_counts:  {replace[token].get('op_counts')}")
        print(f"  baseline stage_bucket_counts: {baseline[token].get('stage_bucket_counts')}")
        print(f"  replace stage_bucket_counts:  {replace[token].get('stage_bucket_counts')}")

    print("")
    print("lowering counters note: pair/pswiglu/fglu are emitted by the Metal summary log, not this graph JSONL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
