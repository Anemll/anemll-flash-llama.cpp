#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashmoe_cache_estimator import build_layer_geometry, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one-lookahead Flash-MoE prefetch strategies from a --moe-trace JSONL. "
            "The trace acts as the oracle log, while the simulator compares exact next-call "
            "coverage against simple history-based heuristics such as 'reuse the hot experts "
            "from the last time this layer ran'."
        )
    )
    parser.add_argument("--sidecar", required=True, type=Path, help="sidecar directory or manifest path")
    parser.add_argument("--trace", required=True, type=Path, help="JSONL trace captured with --moe-trace")
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "all"),
        default="prefill",
        help="filter routed calls by phase using n_tokens (>1 = prefill, 1 = decode)",
    )
    parser.add_argument(
        "--strategy",
        choices=("oracle-next-layer", "oracle-next-call", "prev-layer-lastcall", "none"),
        action="append",
        default=[],
        help="prefetch strategy to evaluate; pass multiple times",
    )
    parser.add_argument(
        "--prefetch-experts",
        type=int,
        action="append",
        default=[],
        help="next-layer prefetch budget in unique experts; pass multiple times",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=8,
        help="number of highest-residual-miss layers to print for each strategy/budget",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()
    if not args.strategy:
        args.strategy = ["oracle-next-layer"]
    if not args.prefetch_experts:
        args.prefetch_experts = [8]
    for budget in args.prefetch_experts:
        if budget < 0:
            raise SystemExit("--prefetch-experts must be >= 0")
    return args


def dedupe_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def rank_hot_experts(values: list[int]) -> tuple[list[int], collections.Counter[int]]:
    counts: collections.Counter[int] = collections.Counter()
    first_seen: dict[int, int] = {}
    for index, value in enumerate(values):
        counts[value] += 1
        first_seen.setdefault(value, index)
    ordered = sorted(counts.keys(), key=lambda expert: (-counts[expert], first_seen[expert], expert))
    return ordered, counts


@dataclass
class RoutedCall:
    seq: int
    layer: int
    n_tokens: int
    experts_raw: list[int]
    experts_unique: list[int]
    hot_order: list[int]
    hot_counts: collections.Counter[int]

    @property
    def raw_refs(self) -> int:
        return len(self.experts_raw)


@dataclass
class TransitionMetrics:
    transitions: int = 0
    cold_start_transitions: int = 0
    total_actual_unique: int = 0
    total_actual_refs: int = 0
    total_predicted_unique: int = 0
    total_hits: int = 0
    total_hit_refs: int = 0
    total_full_hit_transitions: int = 0
    total_prefetched_bytes: int = 0
    total_useful_prefetch_bytes: int = 0
    total_residual_bytes: int = 0
    layer_rows: dict[int, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        if self.layer_rows is None:
            self.layer_rows = collections.defaultdict(
                lambda: {
                    "transitions": 0,
                    "cold_start_transitions": 0,
                    "actual_unique": 0,
                    "actual_refs": 0,
                    "predicted_unique": 0,
                    "hits": 0,
                    "hit_refs": 0,
                    "prefetched_bytes": 0,
                    "useful_prefetch_bytes": 0,
                    "residual_bytes": 0,
                }
            )

    @property
    def unique_coverage(self) -> float:
        if self.total_actual_unique == 0:
            return 0.0
        return self.total_hits / float(self.total_actual_unique)

    @property
    def ref_coverage(self) -> float:
        if self.total_actual_refs == 0:
            return 0.0
        return self.total_hit_refs / float(self.total_actual_refs)

    @property
    def precision(self) -> float:
        if self.total_predicted_unique == 0:
            return 0.0
        return self.total_hits / float(self.total_predicted_unique)

    @property
    def full_hit_rate(self) -> float:
        if self.transitions == 0:
            return 0.0
        return self.total_full_hit_transitions / float(self.transitions)

    @property
    def wasted_prefetch_bytes(self) -> int:
        return self.total_prefetched_bytes - self.total_useful_prefetch_bytes


def parse_trace_calls(trace_path: Path, phase: str) -> tuple[list[RoutedCall], list[str]]:
    trace_path = trace_path.expanduser().resolve()
    calls: list[RoutedCall] = []
    warnings: list[str] = []

    with trace_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            n_tokens = max(0, int(record.get("n_tokens", 0)))
            if phase == "prefill" and n_tokens <= 1:
                continue
            if phase == "decode" and n_tokens != 1:
                continue

            experts_raw = [int(value) for value in record.get("experts", [])]
            if not experts_raw:
                continue
            hot_order, hot_counts = rank_hot_experts(experts_raw)
            calls.append(
                RoutedCall(
                    seq=int(record.get("seq", len(calls))),
                    layer=int(record["layer"]),
                    n_tokens=n_tokens,
                    experts_raw=experts_raw,
                    experts_unique=dedupe_preserve(experts_raw),
                    hot_order=hot_order,
                    hot_counts=hot_counts,
                )
            )
            if n_tokens == 0:
                warnings.append(f"trace line {line_no} has n_tokens=0 for layer {record['layer']}")

    return calls, warnings


def top_unique_by_count(call: RoutedCall, budget: int) -> list[int]:
    if budget <= 0:
        return []
    return call.hot_order[:budget]


def predicted_experts_for_strategy(
    strategy: str,
    budget: int,
    next_call: RoutedCall,
    last_hot_by_layer: dict[int, list[int]],
) -> tuple[list[int], bool]:
    if budget <= 0 or strategy == "none":
        return [], False
    if strategy == "oracle-next-call" or strategy == "oracle-next-layer":
        return top_unique_by_count(next_call, budget), False
    if strategy == "prev-layer-lastcall":
        previous = last_hot_by_layer.get(next_call.layer)
        if previous is None:
            return [], True
        return previous[:budget], False
    raise ValueError(f"unsupported strategy {strategy}")


def simulate_prefetch(
    calls: list[RoutedCall],
    layer_geometry: dict[int, dict[str, Any]],
    strategy: str,
    budget: int,
) -> dict[str, Any]:
    metrics = TransitionMetrics()
    last_hot_by_layer: dict[int, list[int]] = {}
    wraps_skipped = 0

    for index in range(len(calls) - 1):
        current = calls[index]
        next_call = calls[index + 1]

        # Treat only forward-moving routed calls as one-lookahead next-layer opportunities.
        if next_call.layer <= current.layer:
            wraps_skipped += 1
            last_hot_by_layer[current.layer] = current.hot_order
            continue

        if next_call.layer not in layer_geometry:
            raise SystemExit(f"trace references layer {next_call.layer}, which is missing from the sidecar manifest")

        predicted, cold_start = predicted_experts_for_strategy(strategy, budget, next_call, last_hot_by_layer)
        actual_set = set(next_call.experts_unique)
        hit_list = [expert for expert in predicted if expert in actual_set]
        hit_set = set(hit_list)
        slot_bytes = int(layer_geometry[next_call.layer]["slot_bytes"])

        metrics.transitions += 1
        metrics.total_actual_unique += len(next_call.experts_unique)
        metrics.total_actual_refs += next_call.raw_refs
        metrics.total_predicted_unique += len(predicted)
        metrics.total_hits += len(hit_list)
        metrics.total_hit_refs += sum(next_call.hot_counts[expert] for expert in hit_set)
        metrics.total_prefetched_bytes += len(predicted) * slot_bytes
        metrics.total_useful_prefetch_bytes += len(hit_list) * slot_bytes
        metrics.total_residual_bytes += (len(next_call.experts_unique) - len(hit_list)) * slot_bytes
        if len(hit_list) == len(next_call.experts_unique):
            metrics.total_full_hit_transitions += 1
        if cold_start:
            metrics.cold_start_transitions += 1

        row = metrics.layer_rows[next_call.layer]
        row["transitions"] += 1
        row["actual_unique"] += len(next_call.experts_unique)
        row["actual_refs"] += next_call.raw_refs
        row["predicted_unique"] += len(predicted)
        row["hits"] += len(hit_list)
        row["hit_refs"] += sum(next_call.hot_counts[expert] for expert in hit_set)
        row["prefetched_bytes"] += len(predicted) * slot_bytes
        row["useful_prefetch_bytes"] += len(hit_list) * slot_bytes
        row["residual_bytes"] += (len(next_call.experts_unique) - len(hit_list)) * slot_bytes
        if cold_start:
            row["cold_start_transitions"] += 1

        last_hot_by_layer[current.layer] = current.hot_order

    layer_rows = []
    for layer, row in sorted(metrics.layer_rows.items(), key=lambda item: (-item[1]["residual_bytes"], item[0])):
        actual_unique = int(row["actual_unique"])
        actual_refs = int(row["actual_refs"])
        predicted_unique = int(row["predicted_unique"])
        hits = int(row["hits"])
        hit_refs = int(row["hit_refs"])
        transitions = int(row["transitions"])
        layer_rows.append(
            {
                "layer": int(layer),
                "transitions": transitions,
                "cold_start_transitions": int(row["cold_start_transitions"]),
                "actual_unique": actual_unique,
                "actual_refs": actual_refs,
                "predicted_unique": predicted_unique,
                "hits": hits,
                "hit_refs": hit_refs,
                "unique_coverage": 0.0 if actual_unique == 0 else hits / float(actual_unique),
                "ref_coverage": 0.0 if actual_refs == 0 else hit_refs / float(actual_refs),
                "precision": 0.0 if predicted_unique == 0 else hits / float(predicted_unique),
                "prefetched_gib": float(row["prefetched_bytes"]) / float(1024 ** 3),
                "useful_gib": float(row["useful_prefetch_bytes"]) / float(1024 ** 3),
                "residual_gib": float(row["residual_bytes"]) / float(1024 ** 3),
            }
        )

    return {
        "strategy": "oracle-next-layer" if strategy == "oracle-next-call" else strategy,
        "prefetch_experts": budget,
        "transitions": metrics.transitions,
        "wraps_skipped": wraps_skipped,
        "cold_start_transitions": metrics.cold_start_transitions,
        "actual_unique": metrics.total_actual_unique,
        "actual_refs": metrics.total_actual_refs,
        "predicted_unique": metrics.total_predicted_unique,
        "hits": metrics.total_hits,
        "hit_refs": metrics.total_hit_refs,
        "unique_coverage": metrics.unique_coverage,
        "ref_coverage": metrics.ref_coverage,
        "precision": metrics.precision,
        "full_hit_rate": metrics.full_hit_rate,
        "prefetched_gib": float(metrics.total_prefetched_bytes) / float(1024 ** 3),
        "useful_gib": float(metrics.total_useful_prefetch_bytes) / float(1024 ** 3),
        "wasted_gib": float(metrics.wasted_prefetch_bytes) / float(1024 ** 3),
        "residual_gib": float(metrics.total_residual_bytes) / float(1024 ** 3),
        "layer_rows": layer_rows,
    }


def print_text(
    trace_path: Path,
    manifest_path: Path,
    phase: str,
    calls: list[RoutedCall],
    warnings: list[str],
    results: list[dict[str, Any]],
    top_layers: int,
) -> None:
    print("flashmoe prefetch oracle:")
    print(f"  trace:   {trace_path.expanduser().resolve()}")
    print(f"  sidecar: {manifest_path}")
    print(f"  phase:   {phase}")
    print(f"  calls:   {len(calls)}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("results:")
    for result in results:
        print(
            f"  {result['strategy']} budget={result['prefetch_experts']:>3} "
            f"transitions={result['transitions']:>6} "
            f"uniq_cov={result['unique_coverage'] * 100.0:5.1f}% "
            f"ref_cov={result['ref_coverage'] * 100.0:5.1f}% "
            f"precision={result['precision'] * 100.0:5.1f}% "
            f"full={result['full_hit_rate'] * 100.0:5.1f}% "
            f"prefetch={result['prefetched_gib']:6.2f} GiB "
            f"useful={result['useful_gib']:6.2f} GiB "
            f"wasted={result['wasted_gib']:6.2f} GiB "
            f"residual={result['residual_gib']:6.2f} GiB "
            f"cold={result['cold_start_transitions']:>5}"
        )
        layer_rows = result["layer_rows"][:top_layers]
        if not layer_rows:
            continue
        print("    top residual layers:")
        for row in layer_rows:
            print(
                f"      layer={row['layer']:>3} transitions={row['transitions']:>5} "
                f"uniq_cov={row['unique_coverage'] * 100.0:5.1f}% "
                f"precision={row['precision'] * 100.0:5.1f}% "
                f"residual={row['residual_gib']:6.2f} GiB"
            )


def main() -> None:
    args = parse_args()
    manifest_path, manifest = load_manifest(args.sidecar)
    layer_geometry = build_layer_geometry(manifest["entries"])
    calls, warnings = parse_trace_calls(args.trace, args.phase)
    if len(calls) < 2:
        raise SystemExit("trace produced fewer than two routed calls after filtering")

    results = []
    for strategy in args.strategy:
        for budget in args.prefetch_experts:
            results.append(simulate_prefetch(calls, layer_geometry, strategy, budget))

    payload = {
        "trace": str(args.trace.expanduser().resolve()),
        "sidecar": str(manifest_path),
        "phase": args.phase,
        "calls": len(calls),
        "warnings": warnings,
        "results": results,
    }

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    print_text(args.trace, manifest_path, args.phase, calls, warnings, results, args.top_layers)


if __name__ == "__main__":
    main()
