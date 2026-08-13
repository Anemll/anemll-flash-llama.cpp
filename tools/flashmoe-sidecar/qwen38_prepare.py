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

import export_dense_gguf
import flashmoe_sidecar as sidecar_tools


DEFAULT_MODEL = Path(
    "/Volumes/TB36/Models/Qwen/Qwen3.8-2.4T-A95B-GGUF/UD-Q1_0/"
    "Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf"
)
DEFAULT_OUT_DIR = Path("~/Models/Qwen3.8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Qwen3.8 Flash-MoE package in the standard one-root layout: "
            "a layer-major routed sidecar plus a dense/shared-only GGUF."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"first source GGUF shard (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"package directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--layers", help="optional routed-layer filter for sidecar validation only")
    parser.add_argument("--force", action="store_true", help="replace generated package outputs")
    parser.add_argument("--skip-sidecar", action="store_true", help="reuse OUT_DIR/sidecar")
    parser.add_argument("--skip-dense", action="store_true", help="skip model-dense.gguf export")
    parser.add_argument(
        "--verify-bytes",
        action="store_true",
        help="reread and compare every routed byte after extraction (slow for the full model)",
    )
    return parser.parse_args()


def reader_scalar(reader: GGUFReader, key: str, default=None):
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def inspect_qwen38(model_path: Path) -> dict[str, int | str | None]:
    model_paths = sidecar_tools.resolve_model_paths(model_path)
    reader = GGUFReader(str(model_paths[0]), "r")
    arch = str(reader_scalar(reader, "general.architecture", "unknown"))
    if arch != "qwen35moe":
        raise SystemExit(
            f"expected a Qwen3.8 qwen35moe GGUF, but '{model_paths[0]}' reports '{arch}'"
        )

    return {
        "arch": arch,
        "name": reader_scalar(reader, "general.name"),
        "shard_count": len(model_paths),
        "block_count": reader_scalar(reader, "qwen35moe.block_count"),
        "expert_count": reader_scalar(reader, "qwen35moe.expert_count"),
        "expert_used_count": reader_scalar(reader, "qwen35moe.expert_used_count"),
        "embedding_length": reader_scalar(reader, "qwen35moe.embedding_length"),
        "context_length": reader_scalar(reader, "qwen35moe.context_length"),
    }


def main() -> int:
    args = parse_args()
    model = args.model.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    sidecar_dir = out_dir / "sidecar"
    info = inspect_qwen38(model)

    if args.layers and not args.skip_dense:
        raise SystemExit(
            "--layers is a sidecar canary option; combine it with --skip-dense so routed "
            "tensors from unselected layers are not retained in a misleading partial dense GGUF"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        if not args.skip_sidecar and sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
        if not args.skip_dense:
            for path in (out_dir / "model-dense.gguf", out_dir / "flashmoe-package.json"):
                if path.exists():
                    path.unlink()

    if not args.skip_sidecar:
        sidecar_tools.cmd_extract(
            argparse.Namespace(
                model=model,
                out_dir=sidecar_dir,
                include_shared=False,
                layout="layer-major",
                force=args.force,
                layers=args.layers,
                families=None,
            )
        )
    elif not sidecar_dir.exists():
        raise SystemExit(f"missing sidecar directory '{sidecar_dir}'")

    sidecar_tools.cmd_verify(
        argparse.Namespace(
            model=model,
            sidecar=sidecar_dir,
            metadata_only=not args.verify_bytes,
            layers=args.layers,
            families=None,
        )
    )

    if not args.skip_dense:
        export_dense_gguf.export_dense_package(
            model_path=model,
            sidecar_path=sidecar_dir,
            out_dir=out_dir,
            force=args.force,
            layers=None,
        )

    print("Qwen3.8 Flash-MoE package summary:")
    for key, value in info.items():
        print(f"  {key:20s} {value}")
    print(f"  {'package_dir':20s} {out_dir}")
    print("")
    print("package layout:")
    print(f"  dense model: {out_dir / 'model-dense.gguf'}")
    print(f"  sidecar:     {sidecar_dir}")
    print("")
    print("runtime note:")
    print(
        "  This GGUF uses the new IQ1_XXXS tensor type. Package export is supported; "
        "inference requires the matching CPU/CUDA/Metal quant kernels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
