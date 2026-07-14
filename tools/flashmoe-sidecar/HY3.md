# Tencent HY V3 Flash-MoE Sidecar Guide

This guide exports a Tencent HY V3 GGUF into two runtime components:

- a compact GGUF containing embeddings, attention, the leading dense FFN, routers, and shared experts
- an expert-major Flash-MoE sidecar containing only the routed gate/up/down expert weights

The preparation helper validates the source architecture, extracts and verifies the sidecar metadata, and then writes the dense GGUF and package description.

## Prerequisites

Build `llama-cli` with Metal enabled:

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --target llama-cli -j 8
```

The exporter is run with `uv` so its Python dependencies do not need to be installed into the repository:

```bash
uv run --no-project --with pyyaml --with numpy python --version
```

## Download the tested GGUF

The tested single-file IQ1_M checkpoint is published in
[`AngelSlim/Hy3-GGUF`](https://huggingface.co/AngelSlim/Hy3-GGUF). Download
[`Hy3-IQ1_M.gguf`](https://huggingface.co/AngelSlim/Hy3-GGUF/blob/main/Hy3-IQ1_M.gguf)
from the browser, or use the Hugging Face CLI:

```bash
hf download AngelSlim/Hy3-GGUF Hy3-IQ1_M.gguf \
  --local-dir /Volumes/TB36/Models/Hy3/Hy3-GGUF_1b
```

Direct download URL:

```text
https://huggingface.co/AngelSlim/Hy3-GGUF/resolve/main/Hy3-IQ1_M.gguf?download=true
```

The Hub dry run reports 89,446,312,736 bytes (89.4 GB). The MTP file in the
same repository is not required for the trunk-only Flash-MoE package documented
here.

## Export the local Hy3 checkpoint

The helper defaults to the source and destination used for this package:

- source: `/Volumes/TB36/Models/Hy3/Hy3-GGUF_1b/Hy3-IQ1_M.gguf`
- destination: `/Volumes/SN8100/hy3-sidecar`

From the repository root, run:

```bash
uv run --no-project --with pyyaml --with numpy \
  python ./tools/flashmoe-sidecar/hy3_prepare.py --force
```

`--force` replaces generated files under the destination. It does not modify the source GGUF.

For another Hy3 checkpoint or destination:

```bash
uv run --no-project --with pyyaml --with numpy \
  python ./tools/flashmoe-sidecar/hy3_prepare.py \
  --model /path/to/Hy3.gguf \
  --out-dir /path/to/Hy3-Flash \
  --force
```

The completed package contains:

```text
Hy3-Flash/
├── model-dense.gguf
├── flashmoe-package.json
└── sidecar/
    ├── manifest.json
    ├── layer_001.bin
    ├── ...
    └── layer_079.bin
```

For the checked IQ1_M model, the exported package has:

- 79 routed layers and 237 routed tensors
- 81,961,943,040 exact routed bytes in the sidecar (about 77 GiB on disk)
- a 7,484,352,096-byte dense/shared GGUF (about 7.0 GiB)
- expert-major layout (`layer_major_expert` in the sidecar manifest)

The source checkpoint omits `hy_v3.leading_dense_block_count`; the exporter and runtime correctly infer one leading dense block from layer 0.

## Verify an existing package

Metadata verification is fast and checks the manifest, tensor identities, shapes, offsets, quantization metadata, and file extents:

```bash
uv run --no-project --with pyyaml --with numpy \
  python ./tools/flashmoe-sidecar/hy3_prepare.py \
  --out-dir /Volumes/SN8100/hy3-sidecar \
  --skip-sidecar --skip-dense
```

To reread and byte-compare the entire routed payload, add `--verify-bytes`. This reads roughly 82 GB from both the source and sidecar and is therefore much slower:

```bash
uv run --no-project --with pyyaml --with numpy \
  python ./tools/flashmoe-sidecar/hy3_prepare.py \
  --out-dir /Volumes/SN8100/hy3-sidecar \
  --skip-sidecar --skip-dense --verify-bytes
```

For a small extraction test, use a layer range. A partial package is for validation only and cannot run the full model:

```bash
uv run --no-project --with pyyaml --with numpy \
  python ./tools/flashmoe-sidecar/hy3_prepare.py \
  --out-dir /tmp/hy3-sidecar-smoke \
  --layers 1-2 --force
```

## Run a bounded inference smoke test

Hy3 routes eight experts per token, so use at least eight slots and preserve the native top-K of eight. `-ub 1` is required for this decode-oriented slot-bank configuration, and `-st` makes the CLI exit after the supplied turn.

```bash
./build/bin/llama-cli \
  -m /Volumes/SN8100/hy3-sidecar/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Volumes/SN8100/hy3-sidecar/sidecar \
  --moe-slot-bank 8 \
  --moe-topk 8 \
  -fit on \
  -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup \
  -st -p "Hello" -n 3
```

The validated smoke run produced `Hello!` and reported all 79 routed layers using `pread-slot-bank`.

## Enable the fused slot8 decode kernel

The IQ1_M routed weights satisfy the fused Metal kernel requirements. Add `--slot8` to fuse gate, up, SwiGLU, down, and weighted summation for single-token decode:

```bash
./build/bin/llama-cli \
  -m /Volumes/SN8100/hy3-sidecar/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Volumes/SN8100/hy3-sidecar/sidecar \
  --moe-slot-bank 8 --moe-topk 8 --slot8 \
  -fit on -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup -st -p "Hello" -n 2
```

`--slot8` affects single-token decode. Prompt prefill continues through the regular routed path. To confirm graph eligibility during development:

```bash
LLAMA_FLASH_MOE_SLOT8_DEBUG=1 \
./build/bin/llama-cli \
  -m /Volumes/SN8100/hy3-sidecar/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Volumes/SN8100/hy3-sidecar/sidecar \
  --moe-slot-bank 8 --moe-topk 8 --slot8 \
  -fit on -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup -st -p "Hello" -n 2
```

The debug output should report `slot8 eligible` for routed layers 1 through 79.

## Test reduced top-4 fused decode

`--slot4` selects the same width-generic fused Metal operator with exactly four
active experts. Pair it with `--moe-topk 4`; if the effective top-K does not
match four, the graph safely falls back to the normal slot-bank path.

```bash
./build/bin/llama-cli \
  -m /Volumes/SN8100/hy3-sidecar/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Volumes/SN8100/hy3-sidecar/sidecar \
  --moe-slot-bank 4 --moe-topk 4 --slot4 \
  -fit on -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup -st -p "Hello" -n 2
```

To confirm all routed layers select the four-expert fused graph:

```bash
LLAMA_FLASH_MOE_SLOT4_DEBUG=1 \
./build/bin/llama-cli \
  -m /Volumes/SN8100/hy3-sidecar/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Volumes/SN8100/hy3-sidecar/sidecar \
  --moe-slot-bank 4 --moe-topk 4 --slot4 \
  -fit on -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup -st -p "Hello" -n 2
```

The debug output should report `slot4 eligible` for routed layers 1 through 79.
`--slot4` and `--slot8` are mutually exclusive; when both are supplied on the
CLI, the last enabled flag wins.

## KV-cache memory notes

Hy3 uses a conventional K-and-V cache. To reduce its long-context memory use,
quantize both sides of the cache and enable Flash Attention:

```bash
-fa on -ctk q8_0 -ctv q8_0
```

Using only `-ctk q8_0` leaves Hy3's V cache at its default precision and saves
substantially less memory. DeepSeek V4 uses an MLA cache instead: this branch
allocates only its K-backed cache and omits a separate V cache; see
[`src/llama-kv-cache.cpp`](../../src/llama-kv-cache.cpp#L137). Consequently,
`-ctk` controls nearly all of DeepSeek V4's growing KV-cache allocation.

`--slot4` versus `--slot8` does not change Hy3 KV-cache size. These flags only
change routed-expert execution and the useful expert slot-bank size. For
DeepSeek V4, avoid `--swa-full` when memory efficiency is the goal because it
expands the raw sliding-window cache to the full context allocation.

## Common problems

- **Missing routed tensors while loading:** run the dense GGUF with `--moe-sidecar` pointing to the directory containing `manifest.json`.
- **Slot-bank smaller than route width:** keep `--moe-slot-bank` at 8 or higher when `--moe-topk 8` is used.
- **Reduced top-4 mode:** `--moe-slot-bank 4 --moe-topk 4 --slot4` is sufficient; a larger bank is optional and only changes residency/cache behavior.
- **CLI waits for another prompt:** include `-st` for automated or non-interactive tests.
- **Incomplete export after interruption:** rerun the full preparation command with `--force`; temporary layer files are atomically replaced.
- **Out of memory at a larger context:** reduce `-c` and `-b` first. Keep `-ub 1` and a minimum eight-slot bank for native Hy3 routing.
- **Testing another quantization:** the normal slot-bank path remains available without `--slot4`/`--slot8`; the purpose-built fused Metal fast path is optimized for the checked IQ1_M package.
