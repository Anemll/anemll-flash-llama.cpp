#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from flashmoe_sidecar import (  # type: ignore
    FAMILY_ORDER,
    ROUTED_FAMILIES,
    filter_manifest_entries,
    load_manifest,
    parse_family_spec,
    parse_layer_spec,
)

DRIVE_NAMES = ("A", "B", "C")
DEFAULT_PAGE_BYTES = 16 * 1024
DEFAULT_STRATEGY_SPECS = (
    "A-only=1:0:0",
    "A+B-3:1=3:1:0",
    "A+B+C-3:1:1=3:1:1",
    "A+B+C-4:1:1=4:1:1",
    "A+B+C-5:1:1=5:1:1",
)
DARWIN_F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)


@dataclass(frozen=True)
class Entry:
    tensor_name: str
    layer: int
    family: str
    repacked_offset: int
    bytes_per_expert: int
    expert_stride: int
    exact_byte_length: int
    paths: dict[str, str]


@dataclass(frozen=True)
class Strategy:
    name: str
    weights: tuple[int, int, int]


@dataclass(frozen=True)
class Segment:
    drive_idx: int
    page_offset: int
    page_count: int
    page_bytes: int

    @property
    def byte_offset(self) -> int:
        return self.page_offset * self.page_bytes

    @property
    def byte_count(self) -> int:
        return self.page_count * self.page_bytes


@dataclass(frozen=True)
class FamilySample:
    layer: int
    expert: int
    entries: tuple[Entry, ...]


@dataclass
class BenchResult:
    strategy: Strategy
    mode: str
    sample_count: int
    reads: int
    bytes_read: int
    wall_s: float
    latencies_ms: list[float]
    pages_per_drive: tuple[int, int, int]

    @property
    def gib_read(self) -> float:
        return self.bytes_read / float(1024 ** 3)

    @property
    def gib_per_s(self) -> float:
        if self.wall_s <= 0:
            return 0.0
        return self.gib_read / self.wall_s

    @property
    def mib_per_sample(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.bytes_read / float(self.sample_count) / float(1024 ** 2)

    @property
    def reads_per_sample(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.reads / float(self.sample_count)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return percentile(self.latencies_ms, 50.0)

    @property
    def p90_ms(self) -> float:
        return percentile(self.latencies_ms, 90.0)

    @property
    def p99_ms(self) -> float:
        return percentile(self.latencies_ms, 99.0)

    @property
    def cv_pct(self) -> float:
        if len(self.latencies_ms) < 2:
            return 0.0
        mean = self.mean_ms
        if mean <= 0:
            return 0.0
        return statistics.pstdev(self.latencies_ms) * 100.0 / mean


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark weighted multi-drive Flash-MoE sidecar striping using identical sidecar copies on drives A/B/C."
    )
    parser.add_argument("--sidecar-a", required=True, type=Path, help="primary sidecar copy on drive A")
    parser.add_argument("--sidecar-b", required=True, type=Path, help="sidecar copy on drive B")
    parser.add_argument("--sidecar-c", required=True, type=Path, help="sidecar copy on drive C")
    parser.add_argument(
        "--mode",
        choices=("expert", "family"),
        default="expert",
        help="expert: read all selected families for each (layer, expert); family: benchmark one family at a time",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        help="candidate drive weights as NAME=A:B:C or just A:B:C; may be passed multiple times",
    )
    parser.add_argument("--layers", type=str, help="optional layer filter, e.g. '3-10,42'")
    parser.add_argument(
        "--families",
        type=str,
        default="ffn_gate_exps,ffn_up_exps,ffn_down_exps",
        help="family filter; defaults to GLM routed gate/up/down",
    )
    parser.add_argument(
        "--experts",
        type=str,
        help="optional explicit expert ids/ranges, e.g. '0-31,64,127'",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=8,
        help="randomly sample up to N layers after filtering (default: 8)",
    )
    parser.add_argument(
        "--max-experts",
        type=int,
        default=16,
        help="randomly sample up to N experts after filtering (default: 16)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="repeat the sampled workload this many times")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for layer/expert sampling")
    parser.add_argument(
        "--page-bytes",
        type=int,
        default=DEFAULT_PAGE_BYTES,
        help="page granularity for stripe planning (default: 16384)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="best-effort macOS F_NOCACHE on opened files",
    )
    parser.add_argument(
        "--drop-cache-between-runs",
        action="store_true",
        help="best-effort purge(8) before each strategy",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def parse_int_spec(spec: str | None) -> list[int] | None:
    if spec is None or spec.strip() == "":
        return None
    values: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"invalid integer range '{item}'")
            values.update(range(start, end + 1))
        else:
            values.add(int(item))
    return sorted(values)


def parse_strategy(raw: str) -> Strategy:
    if "=" in raw:
        name, payload = raw.split("=", 1)
    else:
        name, payload = raw, raw
    parts = payload.split(":")
    if len(parts) != 3:
        raise SystemExit(f"invalid strategy '{raw}'; expected NAME=A:B:C or A:B:C")
    try:
        weights = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"invalid strategy '{raw}': {exc}") from exc
    if all(weight <= 0 for weight in weights):
        raise SystemExit(f"invalid strategy '{raw}'; at least one drive weight must be > 0")
    if any(weight < 0 for weight in weights):
        raise SystemExit(f"invalid strategy '{raw}'; weights must be non-negative")
    return Strategy(name=name, weights=weights)


def maybe_drop_cache() -> None:
    try:
        os.system("purge >/dev/null 2>&1")
    except Exception:
        pass


def open_fd(path: str, no_cache: bool) -> int:
    fd = os.open(path, os.O_RDONLY)
    if no_cache:
        try:
            fcntl.fcntl(fd, DARWIN_F_NOCACHE, 1)
        except OSError:
            pass
    return fd


def read_exact(fd: int, offset: int, nbytes: int) -> int:
    data = os.pread(fd, nbytes, offset)
    if len(data) != nbytes:
        raise RuntimeError(f"short read: wanted {nbytes}, got {len(data)} at offset {offset}")
    return len(data)


def resolve_sidecar_copy(sidecar: Path) -> tuple[Path, dict[str, Any]]:
    resolved = sidecar.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(
            f"sidecar path does not exist: '{resolved}'\n"
            "expected either a sidecar directory containing manifest.json or a direct manifest.json path"
        )
    manifest_path, manifest = load_manifest(resolved)
    return manifest_path.parent, manifest


def build_entries(
    manifest: dict[str, Any],
    roots: dict[str, Path],
    layer_filter: set[int] | None,
    family_filter: set[str] | None,
) -> list[Entry]:
    entries: list[Entry] = []
    for raw in filter_manifest_entries(manifest["entries"], layer_filter=layer_filter, family_filter=family_filter):
        family = str(raw["tensor_family"])
        if family_filter is not None and family not in family_filter:
            continue
        bytes_per_expert = raw.get("bytes_per_expert")
        if bytes_per_expert is None:
            continue

        repacked_file = str(raw["repacked_file"])
        paths = {drive: str((root / repacked_file).resolve()) for drive, root in roots.items()}
        for drive, path in paths.items():
            candidate = Path(path)
            if not candidate.exists():
                raise SystemExit(f"missing sidecar file for drive {drive}: '{candidate}'")

        entries.append(
            Entry(
                tensor_name=str(raw["tensor_name"]),
                layer=int(raw["layer"]),
                family=family,
                repacked_offset=int(raw["repacked_offset"]),
                bytes_per_expert=int(bytes_per_expert),
                expert_stride=int(raw.get("expert_stride") or bytes_per_expert),
                exact_byte_length=int(raw["exact_byte_length"]),
                paths=paths,
            )
        )

    entries.sort(key=lambda entry: (entry.layer, FAMILY_ORDER.get(entry.family, 99), entry.tensor_name))
    return entries


def choose_subset(values: list[int], limit: int, rng: random.Random) -> list[int]:
    if limit <= 0 or len(values) <= limit:
        return sorted(values)
    return sorted(rng.sample(values, limit))


def allocate_segments(total_bytes: int, page_bytes: int, weights: tuple[int, int, int]) -> list[Segment]:
    if total_bytes % page_bytes != 0:
        raise SystemExit(
            f"bytes_per_expert={total_bytes} is not divisible by page size {page_bytes}; choose a different --page-bytes"
        )

    total_pages = total_bytes // page_bytes
    total_weight = sum(weight for weight in weights if weight > 0)
    if total_weight <= 0:
        raise SystemExit("invalid zero-weight strategy")

    exact_pages = [
        (total_pages * weight / total_weight) if weight > 0 else 0.0
        for weight in weights
    ]
    page_counts = [int(math.floor(value)) for value in exact_pages]
    remainder = total_pages - sum(page_counts)

    order = sorted(
        range(len(weights)),
        key=lambda idx: (exact_pages[idx] - page_counts[idx], weights[idx], -idx),
        reverse=True,
    )
    for idx in order:
        if remainder == 0:
            break
        if weights[idx] <= 0:
            continue
        page_counts[idx] += 1
        remainder -= 1

    segments: list[Segment] = []
    page_cursor = 0
    for drive_idx, page_count in enumerate(page_counts):
        if page_count <= 0:
            continue
        segments.append(
            Segment(
                drive_idx=drive_idx,
                page_offset=page_cursor,
                page_count=page_count,
                page_bytes=page_bytes,
            )
        )
        page_cursor += page_count

    if page_cursor != total_pages:
        raise SystemExit("internal error: page allocation did not cover the expert payload")
    return segments


def build_samples(
    entries: list[Entry],
    families: list[str],
    expert_ids: list[int],
    max_layers: int,
    rng: random.Random,
    mode: str,
) -> list[FamilySample]:
    by_layer: dict[int, dict[str, Entry]] = {}
    for entry in entries:
        by_layer.setdefault(entry.layer, {})[entry.family] = entry

    eligible_layers = [
        layer
        for layer, family_map in by_layer.items()
        if all(family in family_map for family in families)
    ]
    if not eligible_layers:
        raise SystemExit("no layers have all requested families in the chosen sidecar")

    selected_layers = choose_subset(sorted(eligible_layers), max_layers, rng)
    samples: list[FamilySample] = []

    for layer in selected_layers:
        layer_entries = by_layer[layer]
        if mode == "family":
            for family in families:
                for expert in expert_ids:
                    samples.append(FamilySample(layer=layer, expert=expert, entries=(layer_entries[family],)))
        else:
            grouped = tuple(layer_entries[family] for family in families)
            for expert in expert_ids:
                samples.append(FamilySample(layer=layer, expert=expert, entries=grouped))

    return samples


def open_all_fds(entries: list[Entry], no_cache: bool) -> tuple[dict[str, int], list[int]]:
    unique_paths = sorted({path for entry in entries for path in entry.paths.values()})
    fds: dict[str, int] = {}
    raw_fds: list[int] = []
    for path in unique_paths:
        fd = open_fd(path, no_cache=no_cache)
        fds[path] = fd
        raw_fds.append(fd)
    return fds, raw_fds


def bench_strategy(
    strategy: Strategy,
    samples: list[FamilySample],
    fds: dict[str, int],
    page_bytes: int,
    repeats: int,
    seed: int,
) -> BenchResult:
    first_entry = samples[0].entries[0]
    segments = allocate_segments(first_entry.bytes_per_expert, page_bytes, strategy.weights)
    page_counts = [0, 0, 0]
    for segment in segments:
        page_counts[segment.drive_idx] += segment.page_count

    max_workers = max(1, len(segments) * max(len(sample.entries) for sample in samples))
    latencies_ms: list[float] = []
    reads = 0
    bytes_read = 0
    bench_rng = random.Random(seed)
    start = time.perf_counter()

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for repeat_idx in range(repeats):
            ordered_samples = list(samples)
            bench_rng.shuffle(ordered_samples)
            for sample in ordered_samples:
                t0 = time.perf_counter()
                futures: list[cf.Future[int]] = []
                for entry in sample.entries:
                    entry_segments = allocate_segments(entry.bytes_per_expert, page_bytes, strategy.weights)
                    base_offset = entry.repacked_offset + sample.expert * entry.expert_stride
                    for segment in entry_segments:
                        drive = DRIVE_NAMES[segment.drive_idx]
                        fd = fds[entry.paths[drive]]
                        futures.append(
                            executor.submit(
                                read_exact,
                                fd,
                                base_offset + segment.byte_offset,
                                segment.byte_count,
                            )
                        )

                sample_bytes = 0
                for future in futures:
                    sample_bytes += future.result()

                reads += len(futures)
                bytes_read += sample_bytes
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    wall_s = time.perf_counter() - start
    return BenchResult(
        strategy=strategy,
        mode=samples and ("expert" if len(samples[0].entries) > 1 else "family") or "unknown",
        sample_count=len(samples) * repeats,
        reads=reads,
        bytes_read=bytes_read,
        wall_s=wall_s,
        latencies_ms=latencies_ms,
        pages_per_drive=(page_counts[0], page_counts[1], page_counts[2]),
    )


def print_results(
    results: list[BenchResult],
    mode: str,
    families: list[str],
    selected_layers: int,
    selected_experts: int,
    page_bytes: int,
) -> None:
    sample_unit = "expert" if mode == "expert" else "family"
    print(
        f"Mode: {mode} ({sample_unit} samples) | families={','.join(families)} | "
        f"layers={selected_layers} | experts={selected_experts} | page={page_bytes // 1024} KiB"
    )
    print()
    print(
        f"{'Strategy':20} {'Weights':>8} {'Pages A/B/C':>15} {'P50 ms':>9} {'P90 ms':>9} "
        f"{'P99 ms':>9} {'CV %':>7} {'GiB/s':>8} {'Reads/S':>8} {'MiB/Sample':>11}"
    )
    print("-" * 112)
    for result in results:
        weights = ":".join(str(weight) for weight in result.strategy.weights)
        pages = "/".join(str(value) for value in result.pages_per_drive)
        print(
            f"{result.strategy.name:20} {weights:>8} {pages:>15} "
            f"{result.p50_ms:9.2f} {result.p90_ms:9.2f} {result.p99_ms:9.2f} "
            f"{result.cv_pct:7.1f} {result.gib_per_s:8.2f} {result.reads_per_sample:8.2f} "
            f"{result.mib_per_sample:11.2f}"
        )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    strategies = [parse_strategy(spec) for spec in (args.strategy or list(DEFAULT_STRATEGY_SPECS))]
    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)
    expert_filter = parse_int_spec(args.experts)

    if family_filter is None:
        family_filter = set(ROUTED_FAMILIES)
    families = sorted(family_filter, key=lambda family: FAMILY_ORDER.get(family, 99))

    root_a, manifest_a = resolve_sidecar_copy(args.sidecar_a)
    root_b, _manifest_b = resolve_sidecar_copy(args.sidecar_b)
    root_c, _manifest_c = resolve_sidecar_copy(args.sidecar_c)
    roots = {"A": root_a, "B": root_b, "C": root_c}

    entries = build_entries(
        manifest_a,
        roots=roots,
        layer_filter=layer_filter,
        family_filter=family_filter,
    )
    if not entries:
        raise SystemExit("no sidecar entries matched the requested layers/families")

    expert_count = int(manifest_a.get("model", {}).get("expert_count") or 0)
    if expert_count <= 0:
        raise SystemExit("manifest is missing model.expert_count")

    if expert_filter is None:
        expert_ids = choose_subset(list(range(expert_count)), args.max_experts, rng)
    else:
        expert_ids = [expert for expert in expert_filter if 0 <= expert < expert_count]
        if not expert_ids:
            raise SystemExit("no requested experts fall within the model expert range")
    samples = build_samples(
        entries=entries,
        families=families,
        expert_ids=expert_ids,
        max_layers=args.max_layers,
        rng=rng,
        mode=args.mode,
    )
    if not samples:
        raise SystemExit("sampling produced no benchmark cases")

    selected_layers = len({sample.layer for sample in samples})
    selected_experts = len({sample.expert for sample in samples})

    fds, raw_fds = open_all_fds(entries, no_cache=args.no_cache)
    try:
        results: list[BenchResult] = []
        for strategy in strategies:
            if args.drop_cache_between_runs:
                maybe_drop_cache()
            results.append(
                bench_strategy(
                    strategy=strategy,
                    samples=samples,
                    fds=fds,
                    page_bytes=args.page_bytes,
                    repeats=args.repeats,
                    seed=args.seed,
                )
            )
    finally:
        for fd in raw_fds:
            os.close(fd)

    if args.json:
        payload = {
            "mode": args.mode,
            "families": families,
            "selected_layers": selected_layers,
            "selected_experts": selected_experts,
            "page_bytes": args.page_bytes,
            "results": [
                {
                    "strategy": result.strategy.name,
                    "weights": list(result.strategy.weights),
                    "pages_per_drive": list(result.pages_per_drive),
                    "sample_count": result.sample_count,
                    "reads": result.reads,
                    "bytes_read": result.bytes_read,
                    "wall_s": result.wall_s,
                    "gib_per_s": result.gib_per_s,
                    "mib_per_sample": result.mib_per_sample,
                    "reads_per_sample": result.reads_per_sample,
                    "mean_ms": result.mean_ms,
                    "p50_ms": result.p50_ms,
                    "p90_ms": result.p90_ms,
                    "p99_ms": result.p99_ms,
                    "cv_pct": result.cv_pct,
                }
                for result in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print_results(
            results=results,
            mode=args.mode,
            families=families,
            selected_layers=selected_layers,
            selected_experts=selected_experts,
            page_bytes=args.page_bytes,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
