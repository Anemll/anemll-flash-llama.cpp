#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flashmoe_cache_estimator import build_layer_geometry, infer_token_count, load_manifest, parse_trace


@dataclass(frozen=True)
class BinSpec:
    name: str
    rate_gib_s: float

    @property
    def rate_bytes_s(self) -> float:
        return self.rate_gib_s * float(1024 ** 3)


@dataclass
class LayerTiming:
    time_s: float
    total_bytes: int
    bytes_by_bin: list[int]
    calls_with_io: int
    parallel_calls: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype multibin Flash-MoE oracle. Replays a trace, approximates slot-bank misses, "
            "assigns each (layer, expert) to a storage bin, and estimates routed source time assuming "
            "per-call reads across bins can overlap."
        )
    )
    parser.add_argument("--sidecar", required=True, type=Path, help="sidecar directory or manifest path")
    parser.add_argument("--trace", required=True, type=Path, help="JSONL trace captured with --moe-trace")
    parser.add_argument(
        "--bin",
        action="append",
        required=True,
        help="bin spec NAME=RATE_GIB_S, for example apple=2.05 or t710=1.95; pass three times for 3-bin experiments",
    )
    parser.add_argument(
        "--mode",
        choices=("lru-misses", "requests"),
        default="lru-misses",
        help="replay raw requests or approximate miss-only traffic with an LRU slot-bank model",
    )
    parser.add_argument(
        "--bank-size",
        type=int,
        default=60,
        help="per-layer slot-bank size used when --mode=lru-misses",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        help="override inferred token count from the trace",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="maximum local-search passes for the oracle assignment",
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=8,
        help="number of highest-savings layers to print for the oracle result",
    )
    parser.add_argument(
        "--assignment-out",
        type=Path,
        help="optional JSON path for the oracle-local per-layer expert->bin assignment",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def parse_bins(values: list[str]) -> list[BinSpec]:
    bins: list[BinSpec] = []
    seen: set[str] = set()
    for raw in values:
        name, sep, rate_text = raw.partition("=")
        name = name.strip()
        if sep != "=" or not name or not rate_text.strip():
            raise SystemExit(f"invalid --bin '{raw}', expected NAME=RATE_GIB_S")
        if name in seen:
            raise SystemExit(f"duplicate --bin name '{name}'")
        try:
            rate_gib_s = float(rate_text)
        except ValueError as exc:
            raise SystemExit(f"invalid rate in --bin '{raw}': {exc}") from exc
        if rate_gib_s <= 0:
            raise SystemExit(f"--bin '{raw}' must use a positive RATE_GIB_S")
        bins.append(BinSpec(name=name, rate_gib_s=rate_gib_s))
        seen.add(name)
    if len(bins) < 2:
        raise SystemExit("pass at least two --bin values")
    return bins


def dedupe_preserve(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_events_by_layer(
    calls_by_layer: dict[int, list[list[int]]],
    layer_geometry: dict[int, dict[str, Any]],
    mode: str,
    bank_size: int,
) -> dict[int, list[list[int]]]:
    if mode == "requests":
        return {
            layer: [dedupe_preserve(call) for call in calls]
            for layer, calls in calls_by_layer.items()
            if layer in layer_geometry
        }

    if bank_size < 0:
        raise SystemExit("--bank-size must be non-negative")

    events_by_layer: dict[int, list[list[int]]] = {}
    for layer, calls in calls_by_layer.items():
        if layer not in layer_geometry:
            continue
        resident: collections.OrderedDict[int, None] = collections.OrderedDict()
        layer_events: list[list[int]] = []
        for requested in calls:
            requested = dedupe_preserve(requested)
            requested_set = set(requested)
            missing = [expert_id for expert_id in requested if expert_id not in resident]
            for expert_id in requested:
                if expert_id in resident:
                    resident.move_to_end(expert_id)
            while len(resident) + len(missing) > bank_size and resident:
                oldest = next(iter(resident))
                if oldest in requested_set:
                    resident.move_to_end(oldest)
                    if all(expert_id in requested_set for expert_id in resident.keys()):
                        break
                    continue
                resident.popitem(last=False)
            for expert_id in missing:
                if len(resident) >= bank_size:
                    evicted = False
                    for candidate in list(resident.keys()):
                        if candidate not in requested_set:
                            resident.pop(candidate)
                            evicted = True
                            break
                    if not evicted:
                        break
                if bank_size > 0:
                    resident[expert_id] = None
            layer_events.append(missing)
        events_by_layer[layer] = layer_events
    return events_by_layer


def build_event_freq(events_by_layer: dict[int, list[list[int]]]) -> dict[int, collections.Counter[int]]:
    freq_by_layer: dict[int, collections.Counter[int]] = {}
    for layer, events in events_by_layer.items():
        counter: collections.Counter[int] = collections.Counter()
        for experts in events:
            counter.update(experts)
        freq_by_layer[layer] = counter
    return freq_by_layer


def frequency_order(counter: collections.Counter[int]) -> list[tuple[int, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def complete_assignment(
    assignment: dict[int, int],
    counter: collections.Counter[int],
    expert_count: int | None,
    bins: list[BinSpec],
) -> dict[int, int]:
    if expert_count is None:
        return assignment
    completed = dict(assignment)
    loads = [0.0 for _ in bins]
    for expert_id, count in counter.items():
        if expert_id not in completed:
            continue
        loads[completed[expert_id]] += float(count) / bins[completed[expert_id]].rate_bytes_s
    for expert_id in range(expert_count):
        if expert_id in completed:
            continue
        best_bin = min(range(len(bins)), key=lambda idx: (loads[idx], idx))
        completed[expert_id] = best_bin
        loads[best_bin] += 1e-9
    return completed


def assign_hot_bands(counter: collections.Counter[int], bins: list[BinSpec], expert_count: int | None) -> dict[int, int]:
    assignment: dict[int, int] = {}
    ordered = frequency_order(counter)
    total = sum(counter.values())
    if total <= 0:
        return complete_assignment(assignment, counter, expert_count, bins)
    weights = [bin_spec.rate_bytes_s for bin_spec in bins]
    weight_total = sum(weights)
    targets = [total * (weight / weight_total) for weight in weights]
    bin_index = 0
    assigned = 0.0
    for expert_id, count in ordered:
        while bin_index < len(bins) - 1 and assigned >= sum(targets[: bin_index + 1]):
            bin_index += 1
        assignment[expert_id] = bin_index
        assigned += count
    return complete_assignment(assignment, counter, expert_count, bins)


def assign_freq_balance(counter: collections.Counter[int], bins: list[BinSpec], expert_count: int | None) -> dict[int, int]:
    assignment: dict[int, int] = {}
    loads = [0.0 for _ in bins]
    for expert_id, count in frequency_order(counter):
        best_bin = min(
            range(len(bins)),
            key=lambda idx: (loads[idx] + float(count) / bins[idx].rate_bytes_s, loads[idx], idx),
        )
        assignment[expert_id] = best_bin
        loads[best_bin] += float(count) / bins[best_bin].rate_bytes_s
    return complete_assignment(assignment, counter, expert_count, bins)


def evaluate_layer(
    events: list[list[int]],
    assignment: dict[int, int],
    slot_bytes: int,
    bins: list[BinSpec],
) -> LayerTiming:
    total_time_s = 0.0
    total_bytes = 0
    bytes_by_bin = [0 for _ in bins]
    calls_with_io = 0
    parallel_calls = 0
    for experts in events:
        if not experts:
            continue
        per_bin_counts = [0 for _ in bins]
        for expert_id in experts:
            per_bin_counts[assignment[expert_id]] += 1
        active_bins = 0
        max_time_s = 0.0
        for idx, count in enumerate(per_bin_counts):
            if count <= 0:
                continue
            active_bins += 1
            bin_bytes = count * slot_bytes
            bytes_by_bin[idx] += bin_bytes
            total_bytes += bin_bytes
            max_time_s = max(max_time_s, bin_bytes / bins[idx].rate_bytes_s)
        calls_with_io += 1
        if active_bins > 1:
            parallel_calls += 1
        total_time_s += max_time_s
    return LayerTiming(
        time_s=total_time_s,
        total_bytes=total_bytes,
        bytes_by_bin=bytes_by_bin,
        calls_with_io=calls_with_io,
        parallel_calls=parallel_calls,
    )


def assign_oracle_local(
    events: list[list[int]],
    slot_bytes: int,
    bins: list[BinSpec],
    counter: collections.Counter[int],
    expert_count: int | None,
    max_passes: int,
) -> dict[int, int]:
    hot_assignment = assign_hot_bands(counter, bins, expert_count)
    freq_assignment = assign_freq_balance(counter, bins, expert_count)
    hot_time = evaluate_layer(events, hot_assignment, slot_bytes, bins).time_s
    freq_time = evaluate_layer(events, freq_assignment, slot_bytes, bins).time_s
    assignment = dict(freq_assignment if freq_time <= hot_time else hot_assignment)

    if not counter:
        return assignment

    current_time = evaluate_layer(events, assignment, slot_bytes, bins).time_s
    ordered = frequency_order(counter)
    for _ in range(max_passes):
        improved = False
        for expert_id, _ in ordered:
            current_bin = assignment[expert_id]
            best_bin = current_bin
            best_time = current_time
            for candidate in range(len(bins)):
                if candidate == current_bin:
                    continue
                assignment[expert_id] = candidate
                candidate_time = evaluate_layer(events, assignment, slot_bytes, bins).time_s
                if candidate_time + 1e-12 < best_time:
                    best_time = candidate_time
                    best_bin = candidate
            assignment[expert_id] = best_bin
            if best_bin != current_bin:
                current_time = best_time
                improved = True
        if not improved:
            break
    return assignment


def assignment_by_strategy(
    strategy: str,
    events_by_layer: dict[int, list[list[int]]],
    freq_by_layer: dict[int, collections.Counter[int]],
    layer_geometry: dict[int, dict[str, Any]],
    bins: list[BinSpec],
    max_passes: int,
) -> dict[int, dict[int, int]]:
    if strategy.startswith("single:"):
        target_name = strategy.split(":", 1)[1]
        target_bin = None
        for idx, bin_spec in enumerate(bins):
            if bin_spec.name == target_name:
                target_bin = idx
                break
        if target_bin is None:
            raise SystemExit(f"unknown single-bin strategy target '{target_name}'")
        assignments: dict[int, dict[int, int]] = {}
        for layer, counter in freq_by_layer.items():
            expert_count = layer_geometry[layer].get("expert_count")
            assignment = {expert_id: target_bin for expert_id in counter.keys()}
            assignments[layer] = complete_assignment(assignment, counter, expert_count, bins)
        return assignments

    if strategy == "hot-bands":
        return {
            layer: assign_hot_bands(counter, bins, layer_geometry[layer].get("expert_count"))
            for layer, counter in freq_by_layer.items()
        }

    if strategy == "freq-balance":
        return {
            layer: assign_freq_balance(counter, bins, layer_geometry[layer].get("expert_count"))
            for layer, counter in freq_by_layer.items()
        }

    if strategy == "oracle-local":
        return {
            layer: assign_oracle_local(
                events_by_layer[layer],
                int(layer_geometry[layer]["slot_bytes"]),
                bins,
                counter,
                layer_geometry[layer].get("expert_count"),
                max_passes=max_passes,
            )
            for layer, counter in freq_by_layer.items()
        }

    raise SystemExit(f"unknown strategy '{strategy}'")


def evaluate_strategy(
    strategy: str,
    assignments: dict[int, dict[int, int]],
    events_by_layer: dict[int, list[list[int]]],
    layer_geometry: dict[int, dict[str, Any]],
    bins: list[BinSpec],
    token_count: int,
) -> dict[str, Any]:
    total_time_s = 0.0
    total_bytes = 0
    total_calls_with_io = 0
    total_parallel_calls = 0
    bytes_by_bin = [0 for _ in bins]
    layer_rows: list[dict[str, Any]] = []

    for layer, events in events_by_layer.items():
        slot_bytes = int(layer_geometry[layer]["slot_bytes"])
        layer_timing = evaluate_layer(events, assignments[layer], slot_bytes, bins)
        total_time_s += layer_timing.time_s
        total_bytes += layer_timing.total_bytes
        total_calls_with_io += layer_timing.calls_with_io
        total_parallel_calls += layer_timing.parallel_calls
        for idx, value in enumerate(layer_timing.bytes_by_bin):
            bytes_by_bin[idx] += value
        layer_rows.append(
            {
                "layer": layer,
                "time_s": layer_timing.time_s,
                "total_gib": layer_timing.total_bytes / float(1024 ** 3),
                "parallel_call_rate": (
                    0.0 if layer_timing.calls_with_io == 0 else layer_timing.parallel_calls / float(layer_timing.calls_with_io)
                ),
                "observed_calls": len(events),
            }
        )

    result = {
        "strategy": strategy,
        "total_time_s": total_time_s,
        "ms_per_token": 0.0 if token_count == 0 else (total_time_s * 1000.0 / float(token_count)),
        "effective_gib_s": 0.0 if total_time_s <= 0.0 else total_bytes / float(1024 ** 3) / total_time_s,
        "total_gib": total_bytes / float(1024 ** 3),
        "calls_with_io": total_calls_with_io,
        "parallel_call_rate": 0.0 if total_calls_with_io == 0 else total_parallel_calls / float(total_calls_with_io),
        "bytes_by_bin_gib": {
            bins[idx].name: value / float(1024 ** 3)
            for idx, value in enumerate(bytes_by_bin)
        },
        "layer_rows": layer_rows,
    }
    return result


def build_assignment_export(
    strategy: str,
    assignments: dict[int, dict[int, int]],
    freq_by_layer: dict[int, collections.Counter[int]],
    layer_geometry: dict[int, dict[str, Any]],
    bins: list[BinSpec],
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer, assignment in sorted(assignments.items()):
        by_bin: dict[str, list[int]] = {bin_spec.name: [] for bin_spec in bins}
        for expert_id, bin_index in sorted(assignment.items()):
            by_bin[bins[bin_index].name].append(expert_id)
        layers[str(layer)] = {
            "expert_count": layer_geometry[layer].get("expert_count"),
            "observed_experts": len(freq_by_layer[layer]),
            "slot_bytes": int(layer_geometry[layer]["slot_bytes"]),
            "by_bin": by_bin,
        }
    return {
        "strategy": strategy,
        "bins": [
            {
                "name": bin_spec.name,
                "rate_gib_s": bin_spec.rate_gib_s,
            }
            for bin_spec in bins
        ],
        "layers": layers,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    bins = parse_bins(args.bin)
    manifest_path, manifest = load_manifest(args.sidecar)
    layer_geometry = build_layer_geometry(manifest["entries"])
    calls_by_layer, _, token_sums_by_layer, warnings = parse_trace(args.trace)
    token_count, token_warnings = infer_token_count(token_sums_by_layer, args.token_count)
    events_by_layer = build_events_by_layer(calls_by_layer, layer_geometry, args.mode, args.bank_size)
    freq_by_layer = build_event_freq(events_by_layer)

    strategies = [f"single:{bin_spec.name}" for bin_spec in bins] + ["hot-bands", "freq-balance", "oracle-local"]
    strategy_assignments: dict[str, dict[int, dict[int, int]]] = {}
    results: list[dict[str, Any]] = []
    for strategy in strategies:
        assignments = assignment_by_strategy(
            strategy,
            events_by_layer,
            freq_by_layer,
            layer_geometry,
            bins,
            max_passes=args.passes,
        )
        strategy_assignments[strategy] = assignments
        results.append(
            evaluate_strategy(
                strategy,
                assignments,
                events_by_layer,
                layer_geometry,
                bins,
                token_count,
            )
        )

    fastest_bin = max(bins, key=lambda bin_spec: bin_spec.rate_bytes_s).name
    baseline_key = f"single:{fastest_bin}"
    baseline = next(result for result in results if result["strategy"] == baseline_key)
    for result in results:
        result["speedup_vs_fastest_single"] = (
            0.0 if result["total_time_s"] <= 0.0 else baseline["total_time_s"] / result["total_time_s"]
        )

    oracle_result = next(result for result in results if result["strategy"] == "oracle-local")
    oracle_layers = {row["layer"]: row for row in oracle_result["layer_rows"]}
    baseline_layers = {row["layer"]: row for row in baseline["layer_rows"]}
    biggest_savings: list[dict[str, Any]] = []
    for layer, oracle_row in oracle_layers.items():
        baseline_row = baseline_layers.get(layer)
        if baseline_row is None:
            continue
        saved_s = baseline_row["time_s"] - oracle_row["time_s"]
        biggest_savings.append(
            {
                "layer": layer,
                "baseline_time_s": baseline_row["time_s"],
                "oracle_time_s": oracle_row["time_s"],
                "saved_s": saved_s,
                "speedup": 0.0 if oracle_row["time_s"] <= 0.0 else baseline_row["time_s"] / oracle_row["time_s"],
                "oracle_parallel_call_rate": oracle_row["parallel_call_rate"],
            }
        )
    biggest_savings.sort(key=lambda row: (-row["saved_s"], row["layer"]))

    payload = {
        "sidecar": str(manifest_path),
        "trace": str(args.trace.expanduser().resolve()),
        "mode": args.mode,
        "bank_size": args.bank_size if args.mode == "lru-misses" else None,
        "token_count": token_count,
        "warnings": warnings + token_warnings,
        "bins": [
            {
                "name": bin_spec.name,
                "rate_gib_s": bin_spec.rate_gib_s,
            }
            for bin_spec in bins
        ],
        "workload": {
            "layers": len(events_by_layer),
            "calls": sum(len(events) for events in events_by_layer.values()),
            "io_calls": sum(1 for events in events_by_layer.values() for experts in events if experts),
            "total_gib": sum(
                len(experts) * int(layer_geometry[layer]["slot_bytes"])
                for layer, events in events_by_layer.items()
                for experts in events
            )
            / float(1024 ** 3),
            "observed_experts_by_layer": {
                str(layer): len(counter)
                for layer, counter in sorted(freq_by_layer.items())
            },
        },
        "results": results,
        "baseline_strategy": baseline_key,
        "top_oracle_savings_layers": biggest_savings[: args.top_layers],
    }

    if args.assignment_out is not None:
        payload["assignment_out"] = str(args.assignment_out.expanduser().resolve())
        payload["oracle_assignment"] = build_assignment_export(
            "oracle-local",
            strategy_assignments["oracle-local"],
            freq_by_layer,
            layer_geometry,
            bins,
        )
    return payload


def print_payload(payload: dict[str, Any]) -> None:
    print("flashmoe bin oracle:")
    print(f"  sidecar: {payload['sidecar']}")
    print(f"  trace:   {payload['trace']}")
    mode = payload["mode"]
    bank_note = "" if payload["bank_size"] is None else f", bank_size={payload['bank_size']}"
    print(f"  workload: mode={mode}{bank_note}, tokens={payload['token_count']}, routed calls={payload['workload']['calls']}")
    print(f"  io volume modeled: {payload['workload']['total_gib']:.2f} GiB")
    if payload["warnings"]:
        for warning in payload["warnings"]:
            print(f"  warning: {warning}")

    print("bins:")
    for bin_spec in payload["bins"]:
        print(f"  {bin_spec['name']}: {bin_spec['rate_gib_s']:.3f} GiB/s")

    print("strategies:")
    for result in payload["results"]:
        split = ", ".join(f"{name}={value:.2f} GiB" for name, value in result["bytes_by_bin_gib"].items())
        print(
            f"  {result['strategy']:<16} "
            f"time={result['total_time_s']:7.2f}s "
            f"ms/tok={result['ms_per_token']:7.2f} "
            f"eff={result['effective_gib_s']:4.2f} GiB/s "
            f"speedup={result['speedup_vs_fastest_single']:4.2f}x "
            f"parallel={result['parallel_call_rate'] * 100.0:5.1f}% "
            f"[{split}]"
        )

    print(f"baseline fastest single-bin: {payload['baseline_strategy']}")
    print("top oracle savings layers:")
    for row in payload["top_oracle_savings_layers"]:
        print(
            f"  layer={row['layer']:>3} "
            f"save={row['saved_s']:6.3f}s "
            f"baseline={row['baseline_time_s']:6.3f}s "
            f"oracle={row['oracle_time_s']:6.3f}s "
            f"speedup={row['speedup']:4.2f}x "
            f"parallel={row['oracle_parallel_call_rate'] * 100.0:5.1f}%"
        )


def maybe_write_assignment(payload: dict[str, Any], out_path: Path | None) -> None:
    if out_path is None:
        return
    export = payload.get("oracle_assignment")
    if export is None:
        return
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print_payload(payload)
    maybe_write_assignment(payload, args.assignment_out)
    if args.assignment_out is not None and not args.json:
        print(f"oracle assignment: {args.assignment_out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
