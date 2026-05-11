#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

from gguf import GGML_QUANT_SIZES, GGUFReader  # type: ignore

ROUTED_FAMILIES = {
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_up_exps",  # gemma4: fused gate+up
}

SHARED_FAMILIES = {
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
}

FAMILY_ORDER = {
    "ffn_gate_exps": 0,
    "ffn_up_exps": 1,
    "ffn_down_exps": 2,
    "ffn_gate_up_exps": 3,  # gemma4: fused gate+up
    "ffn_gate_shexp": 4,
    "ffn_up_shexp": 5,
    "ffn_down_shexp": 6,
}

LAYER_TENSOR_RE = re.compile(r"^blk\.(\d+)\.(ffn_[^.]+)\.weight$")
GGUF_SPLIT_RE = re.compile(r"^(?P<prefix>.+-)(?P<idx>\d+)-of-(?P<count>\d+)(?P<suffix>\.gguf)$")


@contextmanager
def atomic_write_path(path: Path, mode: str, encoding: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    kwargs: dict[str, Any] = {}
    if "b" not in mode:
        kwargs["encoding"] = encoding or "utf-8"
    try:
        with tmp.open(mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and verify Flash-MoE GGUF sidecars.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract routed expert tensors into a Flash-MoE sidecar")
    add_common_model_arg(extract)
    add_filter_args(extract)
    extract.add_argument("--out-dir", required=True, type=Path, help="output sidecar directory")
    extract.add_argument("--include-shared", action="store_true", help="include shared expert tensors as well")
    extract.add_argument(
        "--layout",
        choices=("layer-major", "expert-major"),
        default="layer-major",
        help="physical sidecar layout: layer-major copies whole tensor-family runs; expert-major stores selected families contiguously per expert",
    )
    extract.add_argument("--force", action="store_true", help="overwrite manifest and layer files in the output directory")
    extract.set_defaults(func=cmd_extract)

    verify = subparsers.add_parser("verify", help="verify a Flash-MoE sidecar against its source GGUF")
    add_common_model_arg(verify)
    add_filter_args(verify)
    verify.add_argument("--sidecar", required=True, type=Path, help="sidecar directory or manifest path")
    verify.add_argument("--metadata-only", action="store_true", help="only validate metadata and offsets, not raw bytes")
    verify.set_defaults(func=cmd_verify)

    repack = subparsers.add_parser("repack-expert-major", help="repack an existing sidecar so each expert's families are contiguous")
    add_filter_args(repack)
    repack.add_argument("--sidecar", required=True, type=Path, help="source sidecar directory or manifest path")
    repack.add_argument("--out-dir", required=True, type=Path, help="output sidecar directory")
    repack.add_argument("--force", action="store_true", help="overwrite manifest and layer files in the output directory")
    repack.set_defaults(func=cmd_repack_expert_major)

    inspect = subparsers.add_parser("inspect", help="inspect GGUF MoE tensor layout and optional sidecar parity")
    add_common_model_arg(inspect)
    add_filter_args(inspect)
    inspect.add_argument("--sidecar", type=Path, help="optional sidecar directory or manifest path")
    inspect.add_argument("--include-shared", action="store_true", help="include shared expert tensors as well")
    inspect.add_argument("--json", action="store_true", help="emit JSON instead of text")
    inspect.set_defaults(func=cmd_inspect)

    return parser.parse_args()


def add_common_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="canonical GGUF path (single file or first shard of a split model)",
    )


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layers",
        type=str,
        help="optional layer filter, e.g. '1,2,4-7'",
    )
    parser.add_argument(
        "--families",
        type=str,
        help="optional family filter, e.g. 'ffn_gate_exps,ffn_up_exps' or aliases 'routed,shared,all'",
    )


def resolve_model_paths(model_path: Path) -> list[Path]:
    model_path = model_path.expanduser().resolve()
    reader = GGUFReader(str(model_path), "r")
    split_count_field = reader.get_field("split.count")
    split_count = int(split_count_field.contents()) if split_count_field is not None else 1
    if split_count <= 1:
        return [model_path]

    match = GGUF_SPLIT_RE.match(model_path.name)
    if match is None:
        raise SystemExit(
            f"split GGUF detected in metadata, but could not derive shard names from '{model_path.name}'"
        )

    width_idx = len(match.group("idx"))
    width_count = len(match.group("count"))
    prefix = match.group("prefix")
    suffix = match.group("suffix")

    paths: list[Path] = []
    for shard_idx in range(split_count):
        shard_name = f"{prefix}{shard_idx + 1:0{width_idx}d}-of-{split_count:0{width_count}d}{suffix}"
        shard_path = model_path.with_name(shard_name)
        if not shard_path.exists():
            raise SystemExit(f"missing GGUF shard '{shard_path}'")
        paths.append(shard_path.resolve())
    return paths


def load_manifest(sidecar_path: Path) -> tuple[Path, dict[str, Any]]:
    sidecar_path = sidecar_path.expanduser().resolve()
    manifest_path = sidecar_path / "manifest.json" if sidecar_path.is_dir() else sidecar_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("entries"), list):
        raise SystemExit(f"manifest '{manifest_path}' is missing an entries array")
    return manifest_path, manifest


def tensor_family(name: str) -> tuple[int, str] | None:
    match = LAYER_TENSOR_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def include_tensor(name: str, include_shared: bool) -> bool:
    parsed = tensor_family(name)
    if parsed is None:
        return False
    _, family = parsed
    if family in ROUTED_FAMILIES:
        return True
    return include_shared and family in SHARED_FAMILIES


def reader_scalar(reader: GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def parse_layer_spec(spec: str | None) -> set[int] | None:
    if spec is None or spec.strip() == "":
        return None

    layers: set[int] = set()
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"invalid layer range '{item}'")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return layers


def parse_family_spec(spec: str | None) -> set[str] | None:
    if spec is None or spec.strip() == "":
        return None

    families: set[str] = set()
    valid = ROUTED_FAMILIES | SHARED_FAMILIES
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item == "all":
            families.update(valid)
        elif item == "routed":
            families.update(ROUTED_FAMILIES)
        elif item == "shared":
            families.update(SHARED_FAMILIES)
        elif item in valid:
            families.add(item)
        else:
            raise SystemExit(f"unknown family '{item}'")
    return families


def build_tensor_index(
    model_paths: list[Path],
    include_shared: bool | None = None,
    layer_filter: set[int] | None = None,
    family_filter: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    readers = [GGUFReader(str(path), "r") for path in model_paths]
    first_reader = readers[0]
    arch = str(reader_scalar(first_reader, "general.architecture", "unknown"))
    expert_count = reader_scalar(first_reader, f"{arch}.expert_count")
    expert_used_count = reader_scalar(first_reader, f"{arch}.expert_used_count")
    leading_dense = reader_scalar(first_reader, f"{arch}.leading_dense_block_count")

    tensors: dict[str, dict[str, Any]] = {}
    for shard_index, (path, reader) in enumerate(zip(model_paths, readers)):
        for tensor in reader.tensors:
            if include_shared is not None and not include_tensor(tensor.name, include_shared):
                continue
            parsed = tensor_family(tensor.name)
            if parsed is None:
                continue
            layer, family = parsed
            if layer_filter is not None and layer not in layer_filter:
                continue
            if family_filter is not None and family not in family_filter:
                continue
            block_size, _ = GGML_QUANT_SIZES[tensor.tensor_type]
            shape = [int(v) for v in tensor.shape.tolist()]
            bytes_per_expert = None
            if len(shape) >= 3 and shape[2] > 0 and tensor.n_bytes % shape[2] == 0:
                bytes_per_expert = tensor.n_bytes // shape[2]
            elif len(shape) == 1 and expert_count > 0 and tensor.n_bytes % expert_count == 0:
                # 1D per-expert tensors (e.g. ffn_down_exps.scale)
                bytes_per_expert = tensor.n_bytes // expert_count

            tensors[tensor.name] = {
                "layer": layer,
                "tensor_family": family,
                "tensor_name": tensor.name,
                "original_gguf_tensor_name": tensor.name,
                "expert_id": None,
                "quant_type": tensor.tensor_type.name,
                "block_size": int(block_size),
                "exact_byte_length": int(tensor.n_bytes),
                "source_shard": str(path),
                "source_shard_index": shard_index,
                "source_offset": int(tensor.data_offset),
                "shape": shape,
                "bytes_per_expert": int(bytes_per_expert) if bytes_per_expert is not None else None,
                "quant_tier": None,
                "projection_quant_kind": None,
                "alternate_bank_offsets": None,
                "dynamic_quant_policy_tag": None,
            }

    metadata = {
        "arch": arch,
        "expert_count": int(expert_count) if expert_count is not None else None,
        "expert_used_count": int(expert_used_count) if expert_used_count is not None else None,
        "leading_dense_block_count": int(leading_dense) if leading_dense is not None else None,
    }
    return tensors, metadata


def sorted_entries(index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        index.values(),
        key=lambda entry: (
            entry["layer"],
            FAMILY_ORDER.get(entry["tensor_family"], 99),
            entry["tensor_name"],
        ),
    )


def filter_manifest_entries(
    entries: list[dict[str, Any]],
    layer_filter: set[int] | None = None,
    family_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for entry in entries:
        if layer_filter is not None and entry["layer"] not in layer_filter:
            continue
        if family_filter is not None and entry["tensor_family"] not in family_filter:
            continue
        filtered.append(entry)
    return filtered


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    family_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "bytes": 0, "layers": 0})
    family_layers: dict[str, set[int]] = defaultdict(set)
    layer_summary: dict[int, dict[str, Any]] = defaultdict(lambda: {"tensors": 0, "bytes": 0, "families": set()})

    for entry in entries:
        family = entry["tensor_family"]
        layer = int(entry["layer"])
        nbytes = int(entry["exact_byte_length"])

        family_summary[family]["tensors"] += 1
        family_summary[family]["bytes"] += nbytes
        family_layers[family].add(layer)

        layer_summary[layer]["tensors"] += 1
        layer_summary[layer]["bytes"] += nbytes
        layer_summary[layer]["families"].add(family)

    for family, layers in family_layers.items():
        family_summary[family]["layers"] = len(layers)

    by_family = {
        family: {
            "tensors": values["tensors"],
            "bytes": values["bytes"],
            "layers": values["layers"],
        }
        for family, values in sorted(family_summary.items(), key=lambda item: FAMILY_ORDER.get(item[0], 99))
    }
    by_layer = {
        str(layer): {
            "tensors": values["tensors"],
            "bytes": values["bytes"],
            "families": sorted(values["families"], key=lambda family: FAMILY_ORDER.get(family, 99)),
        }
        for layer, values in sorted(layer_summary.items())
    }
    return {
        "tensor_count": len(entries),
        "total_bytes": sum(int(entry["exact_byte_length"]) for entry in entries),
        "by_family": by_family,
        "by_layer": by_layer,
    }


def print_summary(summary: dict[str, Any], title: str) -> None:
    print(title)
    print(f"  tensors: {summary['tensor_count']}")
    print(f"  bytes:   {summary['total_bytes']}")
    print("  families:")
    for family, values in summary["by_family"].items():
        print(
            f"    {family}: tensors={values['tensors']} layers={values['layers']} bytes={values['bytes']}"
        )
    print("  layers:")
    for layer, values in summary["by_layer"].items():
        families = ",".join(values["families"])
        print(f"    blk.{layer}: tensors={values['tensors']} bytes={values['bytes']} families={families}")


def copy_source_range(src: Any, dst: Any, offset: int, nbytes: int, label: str) -> None:
    src.seek(offset)
    remaining = nbytes
    while remaining > 0:
        chunk = src.read(min(8 << 20, remaining))
        if not chunk:
            raise SystemExit(f"unexpected EOF while reading '{label}'")
        dst.write(chunk)
        remaining -= len(chunk)


def copy_exact_bytes(model_paths: list[Path], entries: list[dict[str, Any]], out_dir: Path, layout: str) -> None:
    if layout == "expert-major":
        copy_expert_major_from_source(model_paths, entries, out_dir)
        return

    with ExitStack() as stack:
        source_handles = [stack.enter_context(path.open("rb")) for path in model_paths]
        layer_handles: dict[Path, Any] = {}

        for entry in entries:
            layer_file = out_dir / f"layer_{entry['layer']:03d}.bin"
            handle = layer_handles.get(layer_file)
            if handle is None:
                handle = stack.enter_context(atomic_write_path(layer_file, "wb"))
                layer_handles[layer_file] = handle

            entry["repacked_file"] = layer_file.name
            entry["repacked_offset"] = handle.tell()

            src = source_handles[entry["source_shard_index"]]
            copy_source_range(src, handle, entry["source_offset"], entry["exact_byte_length"], entry["tensor_name"])


def copy_expert_major_from_source(model_paths: list[Path], entries: list[dict[str, Any]], out_dir: Path) -> None:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry.get("bytes_per_expert") is None:
            raise SystemExit(f"entry '{entry['tensor_name']}' cannot use expert-major layout without bytes_per_expert")
        by_layer[int(entry["layer"])].append(entry)

    with ExitStack() as stack:
        source_handles = [stack.enter_context(path.open("rb")) for path in model_paths]

        for layer, layer_entries in sorted(by_layer.items()):
            layer_entries = sorted(layer_entries, key=lambda entry: FAMILY_ORDER.get(entry["tensor_family"], 99))
            layer_file = out_dir / f"layer_{layer:03d}.bin"
            expert_counts = {
                int(entry["exact_byte_length"]) // int(entry["bytes_per_expert"])
                for entry in layer_entries
            }
            if len(expert_counts) != 1:
                raise SystemExit(f"layer {layer} has inconsistent expert counts: {sorted(expert_counts)}")
            expert_count = expert_counts.pop()

            expert_stride = 0
            for entry in layer_entries:
                entry["repacked_file"] = layer_file.name
                entry["repacked_offset"] = expert_stride
                entry["expert_stride"] = sum(int(item["bytes_per_expert"]) for item in layer_entries)
                entry["expert_major"] = True
                expert_stride += int(entry["bytes_per_expert"])

            with atomic_write_path(layer_file, "wb") as handle:
                for expert in range(expert_count):
                    for entry in layer_entries:
                        src = source_handles[entry["source_shard_index"]]
                        bpe = int(entry["bytes_per_expert"])
                        copy_source_range(
                            src,
                            handle,
                            int(entry["source_offset"]) + expert * bpe,
                            bpe,
                            f"{entry['tensor_name']} expert {expert}",
                        )


def cmd_extract(args: argparse.Namespace) -> int:
    model_paths = resolve_model_paths(args.model)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)

    if not args.force and any(out_dir.iterdir()):
        raise SystemExit(f"output directory '{out_dir}' is not empty; use --force to overwrite files")

    tensor_index, metadata = build_tensor_index(
        model_paths,
        include_shared=args.include_shared,
        layer_filter=layer_filter,
        family_filter=family_filter,
    )
    entries = sorted_entries(tensor_index)
    if not entries:
        raise SystemExit("no MoE tensors matched the requested extraction scope")

    copy_exact_bytes(model_paths, entries, out_dir, args.layout)

    manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": "layer_major_expert" if args.layout == "expert-major" else "layer_major_whole_tensor",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_shared": bool(args.include_shared),
        "filters": {
            "layers": sorted(layer_filter) if layer_filter is not None else None,
            "families": sorted(family_filter, key=lambda family: FAMILY_ORDER.get(family, 99)) if family_filter is not None else None,
        },
        "source": {
            "model_files": [str(path) for path in model_paths],
        },
        "model": metadata,
        "entries": entries,
    }
    manifest["model"]["sidecar_layout"] = args.layout

    manifest_path = out_dir / "manifest.json"
    with atomic_write_path(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    per_layer_counts: dict[int, int] = defaultdict(int)
    total_bytes = 0
    for entry in entries:
        per_layer_counts[entry["layer"]] += 1
        total_bytes += entry["exact_byte_length"]

    print(f"wrote {len(entries)} tensors across {len(per_layer_counts)} layers to {out_dir}")
    print(f"total exact bytes copied: {total_bytes}")
    print(f"manifest: {manifest_path}")
    return 0


def compare_exact_bytes(
    source_handle: Any,
    source_offset: int,
    sidecar_handle: Any,
    sidecar_offset: int,
    nbytes: int,
    tensor_name: str,
) -> None:
    source_handle.seek(source_offset)
    sidecar_handle.seek(sidecar_offset)
    remaining = nbytes
    while remaining > 0:
        chunk_size = min(8 << 20, remaining)
        source_chunk = source_handle.read(chunk_size)
        sidecar_chunk = sidecar_handle.read(chunk_size)
        if source_chunk != sidecar_chunk:
            raise SystemExit(f"byte mismatch for tensor '{tensor_name}'")
        remaining -= chunk_size


def compare_expert_major_bytes(
    source_handle: Any,
    source_offset: int,
    sidecar_handle: Any,
    entry: dict[str, Any],
    tensor_name: str,
) -> None:
    bytes_per_expert = int(entry["bytes_per_expert"])
    expert_count = int(entry["expert_count"])
    for expert in range(expert_count):
        compare_exact_bytes(
            source_handle=source_handle,
            source_offset=source_offset + expert * bytes_per_expert,
            sidecar_handle=sidecar_handle,
            sidecar_offset=sidecar_entry_expert_offset(entry, expert),
            nbytes=bytes_per_expert,
            tensor_name=f"{tensor_name} expert {expert}",
        )


def cmd_verify(args: argparse.Namespace) -> int:
    model_paths = resolve_model_paths(args.model)
    manifest_path, manifest = load_manifest(args.sidecar)
    manifest_dir = manifest_path.parent
    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)

    tensor_index, metadata = build_tensor_index(
        model_paths,
        include_shared=None,
        layer_filter=layer_filter,
        family_filter=family_filter,
    )
    manifest_model = manifest.get("model", {})
    if manifest_model.get("arch") not in (None, metadata["arch"]):
        raise SystemExit(
            f"manifest arch '{manifest_model.get('arch')}' does not match source GGUF arch '{metadata['arch']}'"
        )

    entries = filter_manifest_entries(manifest["entries"], layer_filter=layer_filter, family_filter=family_filter)
    with ExitStack() as stack:
        source_handles = [stack.enter_context(path.open("rb")) for path in model_paths]
        sidecar_handles: dict[Path, Any] = {}

        for entry in entries:
            tensor_name = entry.get("original_gguf_tensor_name") or entry["tensor_name"]
            if tensor_name not in tensor_index:
                raise SystemExit(f"tensor '{tensor_name}' not found in source GGUF")

            source = tensor_index[tensor_name]
            for key in ("layer", "tensor_family", "quant_type", "block_size", "exact_byte_length", "source_shard_index", "source_offset"):
                if entry.get(key) != source.get(key):
                    raise SystemExit(
                        f"metadata mismatch for tensor '{tensor_name}' on '{key}': "
                        f"manifest={entry.get(key)!r} source={source.get(key)!r}"
                    )

            if entry.get("shape") != source.get("shape"):
                raise SystemExit(f"shape mismatch for tensor '{tensor_name}'")

            if args.metadata_only:
                continue

            sidecar_path = (manifest_dir / entry["repacked_file"]).resolve()
            sidecar_handle = sidecar_handles.get(sidecar_path)
            if sidecar_handle is None:
                sidecar_handle = stack.enter_context(sidecar_path.open("rb"))
                sidecar_handles[sidecar_path] = sidecar_handle

            source_handle = source_handles[source["source_shard_index"]]
            if entry.get("expert_major") or manifest.get("layout") in ("expert_major", "layer_major_expert"):
                compare_expert_major_bytes(
                    source_handle=source_handle,
                    source_offset=source["source_offset"],
                    sidecar_handle=sidecar_handle,
                    entry=entry,
                    tensor_name=tensor_name,
                )
            else:
                compare_exact_bytes(
                    source_handle=source_handle,
                    source_offset=source["source_offset"],
                    sidecar_handle=sidecar_handle,
                    sidecar_offset=entry["repacked_offset"],
                    nbytes=entry["exact_byte_length"],
                    tensor_name=tensor_name,
                )

    mode = "metadata-only" if args.metadata_only else "metadata+bytes"
    print(f"verified {len(entries)} Flash-MoE sidecar entries against {len(model_paths)} GGUF file(s) using {mode}")
    return 0


def sidecar_entry_expert_offset(entry: dict[str, Any], expert: int) -> int:
    stride = int(entry.get("expert_stride") or entry["bytes_per_expert"])
    return int(entry["repacked_offset"]) + expert * stride


def copy_exact_range(src: Any, dst: Any, offset: int, nbytes: int, label: str) -> None:
    src.seek(offset)
    remaining = nbytes
    while remaining > 0:
        chunk = src.read(min(8 << 20, remaining))
        if not chunk:
            raise SystemExit(f"unexpected EOF while reading '{label}'")
        dst.write(chunk)
        remaining -= len(chunk)


def cmd_repack_expert_major(args: argparse.Namespace) -> int:
    manifest_path, manifest = load_manifest(args.sidecar)
    manifest_dir = manifest_path.parent
    out_dir = args.out_dir.expanduser().resolve()
    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)

    if not args.force and out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output directory '{out_dir}' is not empty; use --force to overwrite files")
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = filter_manifest_entries(manifest["entries"], layer_filter=layer_filter, family_filter=family_filter)
    if not entries:
        raise SystemExit("no sidecar entries matched the requested repack scope")

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_layer[int(entry["layer"])].append(entry)

    new_entries: list[dict[str, Any]] = []
    with ExitStack() as stack:
        source_handles: dict[Path, Any] = {}

        for layer, layer_entries in sorted(by_layer.items()):
            layer_entries = sorted(layer_entries, key=lambda entry: FAMILY_ORDER.get(entry["tensor_family"], 99))
            expert_counts = {int(entry["expert_count"]) for entry in layer_entries}
            if len(expert_counts) != 1:
                raise SystemExit(f"layer {layer} has inconsistent expert counts: {sorted(expert_counts)}")
            expert_count = expert_counts.pop()

            layer_path = out_dir / f"layer_{layer:03d}.bin"
            print(f"repack layer {layer} -> {layer_path} [expert-major]", flush=True)
            family_offsets: dict[str, int] = {}
            expert_stride = 0
            for entry in layer_entries:
                family_offsets[entry["tensor_name"]] = expert_stride
                expert_stride += int(entry["bytes_per_expert"])

            with atomic_write_path(layer_path, "wb") as out:
                for expert in range(expert_count):
                    for entry in layer_entries:
                        source_path = (manifest_dir / entry["repacked_file"]).resolve()
                        src = source_handles.get(source_path)
                        if src is None:
                            src = stack.enter_context(source_path.open("rb"))
                            source_handles[source_path] = src
                        copy_exact_range(
                            src,
                            out,
                            sidecar_entry_expert_offset(entry, expert),
                            int(entry["bytes_per_expert"]),
                            f"{entry['tensor_name']} expert {expert}",
                        )

            for entry in layer_entries:
                item = dict(entry)
                item["repacked_file"] = layer_path.name
                item["repacked_offset"] = family_offsets[entry["tensor_name"]]
                item["expert_stride"] = expert_stride
                item["expert_major"] = True
                new_entries.append(item)

    new_manifest = dict(manifest)
    new_manifest["layout"] = "layer_major_expert"
    new_manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    new_manifest["source"] = dict(manifest.get("source") or {})
    new_manifest["source"]["repacked_from_sidecar"] = str(manifest_path)
    new_manifest["model"] = dict(manifest.get("model") or {})
    new_manifest["model"]["sidecar_layout"] = "expert-major"
    new_manifest["entries"] = new_entries

    manifest_out = out_dir / "manifest.json"
    with atomic_write_path(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(new_manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote expert-major sidecar: {manifest_out}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    model_paths = resolve_model_paths(args.model)
    layer_filter = parse_layer_spec(args.layers)
    family_filter = parse_family_spec(args.families)
    tensor_index, metadata = build_tensor_index(
        model_paths,
        include_shared=args.include_shared,
        layer_filter=layer_filter,
        family_filter=family_filter,
    )
    gguf_entries = sorted_entries(tensor_index)
    gguf_summary = summarize_entries(gguf_entries)
    payload: dict[str, Any] = {
        "model_files": [str(path) for path in model_paths],
        "model": metadata,
        "filters": {
            "layers": sorted(layer_filter) if layer_filter is not None else None,
            "families": sorted(family_filter, key=lambda family: FAMILY_ORDER.get(family, 99)) if family_filter is not None else None,
            "include_shared": bool(args.include_shared),
        },
        "gguf": gguf_summary,
    }

    if args.sidecar is not None:
        manifest_path, manifest = load_manifest(args.sidecar)
        sidecar_entries = filter_manifest_entries(manifest["entries"], layer_filter=layer_filter, family_filter=family_filter)
        sidecar_summary = summarize_entries(sidecar_entries)
        gguf_names = {entry["tensor_name"] for entry in gguf_entries}
        sidecar_names = {
            entry.get("original_gguf_tensor_name") or entry["tensor_name"]
            for entry in sidecar_entries
        }
        payload["sidecar"] = {
            "manifest": str(manifest_path),
            "summary": sidecar_summary,
            "missing_from_sidecar": sorted(gguf_names - sidecar_names),
            "extra_in_sidecar": sorted(sidecar_names - gguf_names),
        }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0

    print(f"model arch: {metadata['arch']}")
    print(f"model files: {len(model_paths)}")
    print_summary(gguf_summary, "gguf summary:")
    if "sidecar" in payload:
        sidecar = payload["sidecar"]
        print_summary(sidecar["summary"], "sidecar summary:")
        print(f"  missing_from_sidecar: {len(sidecar['missing_from_sidecar'])}")
        print(f"  extra_in_sidecar: {len(sidecar['extra_in_sidecar'])}")
    return 0


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
