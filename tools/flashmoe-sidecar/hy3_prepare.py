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


DEFAULT_MODEL = Path("/Volumes/TB36/Models/Hy3/Hy3-GGUF_1b/Hy3-IQ1_M.gguf")
DEFAULT_OUT_DIR = Path("/Volumes/SN8100/hy3-sidecar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Tencent HY V3 Flash-MoE package: expert-major routed "
            "sidecar plus a dense/shared-only GGUF."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"source GGUF (default: {DEFAULT_MODEL})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"package directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--layers", help="optional routed-layer filter for partial smoke packages")
    parser.add_argument("--force", action="store_true", help="replace generated package outputs")
    parser.add_argument("--skip-sidecar", action="store_true", help="reuse OUT_DIR/sidecar")
    parser.add_argument("--skip-dense", action="store_true", help="skip model-dense.gguf export")
    parser.add_argument(
        "--verify-bytes",
        action="store_true",
        help="reread and compare the full routed payload after extraction (slow for the full model)",
    )
    return parser.parse_args()


def reader_scalar(reader: GGUFReader, key: str, default=None):
    field = reader.get_field(key)
    return field.contents() if field is not None else default


def inspect_hy3(model_path: Path) -> dict[str, int | str | None]:
    model_paths = sidecar_tools.resolve_model_paths(model_path)
    reader = GGUFReader(str(model_paths[0]), "r")
    arch = str(reader_scalar(reader, "general.architecture", "unknown"))
    if arch != "hy_v3":
        raise SystemExit(f"expected a hy_v3 GGUF, but '{model_paths[0]}' reports '{arch}'")

    return {
        "arch": arch,
        "name": reader_scalar(reader, "general.name"),
        "block_count": reader_scalar(reader, "hy_v3.block_count"),
        "expert_count": reader_scalar(reader, "hy_v3.expert_count"),
        "expert_used_count": reader_scalar(reader, "hy_v3.expert_used_count"),
        "embedding_length": reader_scalar(reader, "hy_v3.embedding_length"),
        "context_length": reader_scalar(reader, "hy_v3.context_length"),
        "leading_dense_block_count": reader_scalar(reader, "hy_v3.leading_dense_block_count", 1),
    }


def main() -> int:
    args = parse_args()
    model = args.model.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    sidecar_dir = out_dir / "sidecar"
    info = inspect_hy3(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        if not args.skip_sidecar and sidecar_dir.exists():
            shutil.rmtree(sidecar_dir)
        if not args.skip_dense:
            for path in (out_dir / "model-dense.gguf", out_dir / "flashmoe-package.json"):
                if path.exists():
                    path.unlink()

    if args.layers:
        print("note: --layers produces a partial validation package, not a full inference package")

    if not args.skip_sidecar:
        sidecar_tools.cmd_extract(argparse.Namespace(
            model=model,
            out_dir=sidecar_dir,
            include_shared=False,
            layout="expert-major",
            force=args.force,
            layers=args.layers,
            families=None,
        ))
    elif not sidecar_dir.exists():
        raise SystemExit(f"missing sidecar directory '{sidecar_dir}'")

    sidecar_tools.cmd_verify(argparse.Namespace(
        model=model,
        sidecar=sidecar_dir,
        metadata_only=not args.verify_bytes,
        layers=args.layers,
        families=None,
    ))

    if not args.skip_dense:
        export_dense_gguf.export_dense_package(
            model_path=model,
            sidecar_path=sidecar_dir,
            out_dir=out_dir,
            force=args.force,
            layers=args.layers,
        )

    print("HY V3 Flash-MoE package summary:")
    for key, value in info.items():
        print(f"  {key:27s} {value}")
    print(f"  {'package_dir':27s} {out_dir}")
    print("")
    print("smoke test:")
    print(
        f"  ./build/bin/llama-cli -m {out_dir / 'model-dense.gguf'} "
        f"--moe-mode slot-bank --moe-sidecar {sidecar_dir} "
        "--moe-slot-bank 8 --moe-topk 8 --slot8 -fit on "
        "-ub 1 -b 1 -c 128 -ngl 999 --no-warmup -st -p 'Hello' -n 2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
