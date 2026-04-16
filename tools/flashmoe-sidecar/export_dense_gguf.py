#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_ROOT = SCRIPT_DIR.parents[1]
if str(LLAMA_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

import gguf  # type: ignore
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # type: ignore

import flashmoe_sidecar as sidecar_tools

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


ROUTED_TENSOR_MARKERS = (
    ".ffn_gate_up_exps.",
    ".ffn_gate_exps.",
    ".ffn_up_exps.",
    ".ffn_down_exps.",
)


@dataclass
class TensorRecord:
    shard_path: Path
    reader: GGUFReader
    tensor: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a dense/shared-only GGUF by removing routed expert tensors while "
            "keeping the remaining GGUF tensor payload bytes exact."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="canonical GGUF path (single file or first shard of a split model)",
    )
    parser.add_argument(
        "--sidecar",
        required=True,
        type=Path,
        help="Flash-MoE sidecar directory or manifest path",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory that will receive model-dense.gguf and flashmoe-package.json",
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="optional routed-layer filter, mainly for partial exports and smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite model-dense.gguf and flashmoe-package.json if they already exist",
    )
    return parser.parse_args()


def reader_scalar(reader: GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def is_routed_tensor_name(name: str) -> bool:
    if ".scale" in name:
        return False
    return any(marker in name for marker in ROUTED_TENSOR_MARKERS)


def should_remove_tensor(name: str, layer_filter: set[int] | None) -> bool:
    if not is_routed_tensor_name(name):
        return False
    if layer_filter is None:
        return True
    parsed = sidecar_tools.tensor_family(name)
    return parsed is not None and parsed[0] in layer_filter


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name.startswith("split."):
            continue

        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=sub_type)


def load_model_records(
    model_paths: list[Path],
    layer_filter: set[int] | None,
) -> tuple[list[TensorRecord], list[TensorRecord], GGUFReader]:
    kept: list[TensorRecord] = []
    removed: list[TensorRecord] = []

    readers = [GGUFReader(str(path), "r") for path in model_paths]
    first_reader = readers[0]

    for path, reader in zip(model_paths, readers):
        for tensor in reader.tensors:
            record = TensorRecord(shard_path=path, reader=reader, tensor=tensor)
            if should_remove_tensor(tensor.name, layer_filter):
                removed.append(record)
            else:
                kept.append(record)

    return kept, removed, first_reader


def validate_sidecar_manifest(
    manifest: dict[str, Any],
    removed: list[TensorRecord],
    layer_filter: set[int] | None,
) -> dict[str, Any]:
    manifest_entries = sidecar_tools.filter_manifest_entries(
        manifest["entries"],
        layer_filter=layer_filter,
        family_filter=sidecar_tools.ROUTED_FAMILIES,
    )
    manifest_names = {
        entry.get("original_gguf_tensor_name") or entry["tensor_name"]
        for entry in manifest_entries
    }
    removed_names = {record.tensor.name for record in removed}

    missing = sorted(removed_names - manifest_names)
    extra = sorted(manifest_names - removed_names)
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(
            f"sidecar manifest is missing {len(missing)} routed tensors required by the dense export "
            f"(examples: {preview})"
        )
    if extra:
        preview = ", ".join(extra[:5])
        raise SystemExit(
            f"sidecar manifest has {len(extra)} routed tensors that do not match the dense export scope "
            f"(examples: {preview})"
        )

    return sidecar_tools.summarize_entries(manifest_entries)


def summarize_removed(records: list[TensorRecord]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for record in records:
        parsed = sidecar_tools.tensor_family(record.tensor.name)
        layer = parsed[0] if parsed is not None else -1
        family = parsed[1] if parsed is not None else "unknown"
        entries.append(
            {
                "layer": layer,
                "tensor_family": family,
                "tensor_name": record.tensor.name,
                "exact_byte_length": int(record.tensor.n_bytes),
            }
        )
    return sidecar_tools.summarize_entries(entries)


def write_dense_model(
    kept: list[TensorRecord],
    first_reader: GGUFReader,
    output_model: Path,
) -> tuple[int, int]:
    writer = GGUFWriter(output_model, arch=reader_scalar(first_reader, gguf.Keys.General.ARCHITECTURE), endianess=first_reader.endianess)
    copy_metadata(first_reader, writer)

    total_bytes = 0
    for record in kept:
        tensor = record.tensor
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
        total_bytes += int(tensor.n_bytes)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    progress = tqdm(total=total_bytes, unit="byte", unit_scale=True, desc="Writing dense GGUF") if tqdm else None
    try:
        for record in kept:
            writer.write_tensor_data(record.tensor.data, tensor_endianess=record.reader.endianess)
            if progress is not None:
                progress.update(int(record.tensor.n_bytes))
    finally:
        if progress is not None:
            progress.close()
        writer.close()

    return len(kept), total_bytes


def build_package_metadata(
    model_paths: list[Path],
    manifest_path: Path,
    first_reader: GGUFReader,
    kept: list[TensorRecord],
    removed: list[TensorRecord],
    removed_summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    layer_filter: set[int] | None,
) -> dict[str, Any]:
    arch = str(reader_scalar(first_reader, gguf.Keys.General.ARCHITECTURE, "unknown"))
    expert_count = reader_scalar(first_reader, f"{arch}.expert_count")
    expert_used_count = reader_scalar(first_reader, f"{arch}.expert_used_count")
    block_count = reader_scalar(first_reader, f"{arch}.block_count")
    leading_dense = reader_scalar(first_reader, f"{arch}.leading_dense_block_count")

    runtime_hint: dict[str, Any] | None = None
    if arch == "minimax-m2":
        runtime_hint = {
            "moe_mode": "slot-bank",
            "moe_topk": min(4, int(expert_used_count)) if expert_used_count is not None else 4,
            "moe_slot_bank": 64,
            "moe_cache_io_split": 4,
            "moe_prefetch_temporal": True,
            "metal_env": {
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE": "1",
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY": "1",
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT": "65536",
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB": "0",
                "LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES": "1",
            },
        }

    return {
        "schema_version": 1,
        "kind": "flashmoe_dense_package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model_files": [str(path) for path in model_paths],
            "sidecar_manifest": str(manifest_path),
            "layer_filter": sorted(layer_filter) if layer_filter is not None else None,
        },
        "model": {
            "arch": arch,
            "name": reader_scalar(first_reader, gguf.Keys.General.NAME),
            "block_count": int(block_count) if block_count is not None else None,
            "leading_dense_block_count": int(leading_dense) if leading_dense is not None else None,
            "expert_count": int(expert_count) if expert_count is not None else None,
            "expert_used_count": int(expert_used_count) if expert_used_count is not None else None,
            "context_length": reader_scalar(first_reader, f"{arch}.context_length"),
            "embedding_length": reader_scalar(first_reader, f"{arch}.embedding_length"),
            "file_type": reader_scalar(first_reader, gguf.Keys.General.FILE_TYPE),
        },
        "dense_model": {
            "path": "model-dense.gguf",
            "tensor_count": len(kept),
            "total_bytes": sum(int(record.tensor.n_bytes) for record in kept),
        },
        "routed_removed": {
            "tensor_count": len(removed),
            "total_bytes": sum(int(record.tensor.n_bytes) for record in removed),
            "summary": removed_summary,
        },
        "sidecar": {
            "path": str(manifest_path.parent),
            "summary": manifest_summary,
        },
        "runtime_hint": runtime_hint,
    }


def export_dense_package(
    model_path: Path,
    sidecar_path: Path,
    out_dir: Path,
    *,
    force: bool = False,
    layers: str | None = None,
) -> dict[str, Any]:
    model_paths = sidecar_tools.resolve_model_paths(model_path)
    layer_filter = sidecar_tools.parse_layer_spec(layers)
    manifest_path, manifest = sidecar_tools.load_manifest(sidecar_path)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    output_model = out_dir / "model-dense.gguf"
    package_json = out_dir / "flashmoe-package.json"

    if output_model.exists() or package_json.exists():
        if not force:
            raise SystemExit(
                f"output files already exist in '{out_dir}'; use --force to overwrite model-dense.gguf/package json"
            )
        if output_model.exists():
            output_model.unlink()
        if package_json.exists():
            package_json.unlink()

    kept, removed, first_reader = load_model_records(model_paths, layer_filter)
    if not removed:
        raise SystemExit("no routed expert tensors matched the requested export scope")

    manifest_arch = manifest.get("model", {}).get("arch")
    reader_arch = str(reader_scalar(first_reader, gguf.Keys.General.ARCHITECTURE, "unknown"))
    if manifest_arch not in (None, reader_arch):
        raise SystemExit(
            f"sidecar arch '{manifest_arch}' does not match source GGUF arch '{reader_arch}'"
        )

    manifest_summary = validate_sidecar_manifest(manifest, removed, layer_filter)
    removed_summary = summarize_removed(removed)
    kept_count, kept_bytes = write_dense_model(kept, first_reader, output_model)

    package = build_package_metadata(
        model_paths,
        manifest_path,
        first_reader,
        kept,
        removed,
        removed_summary,
        manifest_summary,
        layer_filter,
    )

    with package_json.open("w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote dense-only GGUF: {output_model}")
    print(f"kept tensors: {kept_count}")
    print(f"kept bytes:   {kept_bytes}")
    print(f"removed routed tensors: {len(removed)}")
    print(f"package: {package_json}")
    return package


def main() -> int:
    args = parse_args()
    export_dense_package(
        model_path=args.model,
        sidecar_path=args.sidecar,
        out_dir=args.out_dir,
        force=args.force,
        layers=args.layers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
