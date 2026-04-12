#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLAMA_ROOT = SCRIPT_DIR.parents[1]
if str(LLAMA_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

from gguf import GGUFReader  # type: ignore

import flashmoe_sidecar as sidecar_tools
import export_dense_gguf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Flash-MoE package for MiniMax M2 by extracting a routed sidecar "
            "and exporting the remaining dense/shared GGUF payload."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="canonical GGUF path (single file or first shard of a split model)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="package directory that will receive sidecar/, model-dense.gguf, and flashmoe-package.json",
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="optional routed-layer filter, mainly for partial packaging and smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing generated sidecar/package outputs under --out-dir before rebuilding",
    )
    parser.add_argument(
        "--skip-sidecar",
        action="store_true",
        help="reuse an existing sidecar directory at OUT_DIR/sidecar",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="skip model-dense.gguf export",
    )
    return parser.parse_args()


def reader_scalar(reader: GGUFReader, key: str, default=None):
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def inspect_minimax(model_path: Path) -> dict[str, int | str | None]:
    model_paths = sidecar_tools.resolve_model_paths(model_path)
    reader = GGUFReader(str(model_paths[0]), "r")
    arch = str(reader_scalar(reader, "general.architecture", "unknown"))
    if arch != "minimax-m2":
        raise SystemExit(
            f"expected a minimax-m2 GGUF, but '{model_paths[0]}' reports '{arch}'"
        )

    return {
        "arch": arch,
        "name": reader_scalar(reader, "general.name"),
        "block_count": reader_scalar(reader, "minimax-m2.block_count"),
        "expert_count": reader_scalar(reader, "minimax-m2.expert_count"),
        "expert_used_count": reader_scalar(reader, "minimax-m2.expert_used_count"),
        "embedding_length": reader_scalar(reader, "minimax-m2.embedding_length"),
        "context_length": reader_scalar(reader, "minimax-m2.context_length"),
    }


def main() -> int:
    args = parse_args()

    info = inspect_minimax(args.model)
    out_dir = args.out_dir.expanduser().resolve()
    sidecar_dir = out_dir / "sidecar"

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        if not args.skip_sidecar and sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
        dense_model = out_dir / "model-dense.gguf"
        package_json = out_dir / "flashmoe-package.json"
        if not args.skip_dense and dense_model.exists():
            dense_model.unlink()
        if not args.skip_dense and package_json.exists():
            package_json.unlink()

    if args.layers:
        print("note: --layers is set; this package is intended for partial validation, not full inference.")

    if not args.skip_sidecar:
        extract_args = argparse.Namespace(
            model=args.model,
            out_dir=sidecar_dir,
            include_shared=False,
            force=args.force,
            layers=args.layers,
            families=None,
        )
        sidecar_tools.cmd_extract(extract_args)
    elif not sidecar_dir.exists():
        raise SystemExit(f"missing sidecar directory '{sidecar_dir}'")

    if not args.skip_dense:
        export_dense_gguf.export_dense_package(
            model_path=args.model,
            sidecar_path=sidecar_dir,
            out_dir=out_dir,
            force=args.force,
            layers=args.layers,
        )

    print("MiniMax M2 package summary:")
    print(f"  name:              {info['name']}")
    print(f"  arch:              {info['arch']}")
    print(f"  block_count:       {info['block_count']}")
    print(f"  expert_count:      {info['expert_count']}")
    print(f"  expert_used_count: {info['expert_used_count']}")
    print(f"  embedding_length:  {info['embedding_length']}")
    print(f"  context_length:    {info['context_length']}")
    print(f"  package_dir:       {out_dir}")
    print("")
    print("next step:")
    print(f"  bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
