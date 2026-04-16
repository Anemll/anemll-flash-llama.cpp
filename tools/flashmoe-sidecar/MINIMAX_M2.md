# MiniMax M2 Flash-MoE Notes

This branch already contains native MiniMax M2 graph support:

- `general.architecture = minimax-m2`
- `src/models/minimax-m2.cpp`
- `LLM_ARCH_MINIMAX_M2` tensor loading and vocab wiring in `src/llama-model.cpp`, `src/llama-arch.cpp`, and `src/llama-vocab.cpp`

For the `MiniMax-M2.7` GGUF checked on `2026-04-12`, the metadata snapshot is:

- name: `Minimax-M2.7`
- block count: `62`
- expert count: `256`
- expert used count: `8`
- embedding length: `3072`
- attention heads: `48`
- KV heads: `8`
- key/value length: `128`
- context length: `196608`
- tokenizer pre: `minimax-m2`

The routed MoE tensors in the provided `UD-IQ2_XXS` package are standard:

- `blk.N.ffn_gate_exps.weight`
- `blk.N.ffn_up_exps.weight`
- `blk.N.ffn_down_exps.weight`

No `leading_dense_block_count` was present in the GGUF metadata during inspection, so dense export for this model means:

- keep every non-routed tensor in the GGUF
- keep any future shared-expert tensors if they exist
- remove only routed expert weight tensors into the Flash-MoE sidecar

## Upstream status

As of `2026-04-12`, public upstream tracking shows:

- feature request: `ggml-org/llama.cpp#16798` is closed
- chat format follow-up: `ggml-org/llama.cpp#16904` is public

The local branch in this workspace definitely contains MiniMax M2 architecture support and Flash-MoE tooling hooks, which is the support level this note is written against.

## Export MiniMax M2.7

This creates:

- `OUT_DIR/sidecar/manifest.json`
- `OUT_DIR/sidecar/layer_XXX.bin`
- `OUT_DIR/model-dense.gguf`
- `OUT_DIR/flashmoe-package.json` with package metadata and runtime defaults for the wrapper

Checked MiniMax package shapes in this workspace:

| Quant | Shards | Dense GGUF | Routed sidecar |
|-------|--------|-----------|----------------|
| UD-IQ2_XXS | 3 | ~2.58 GiB | ~58.3 GiB |
| UD-Q4_K_XL | 4 | (4-bit) | (4-bit routed) |

For the checked `UD-IQ2_XXS` package in this workspace:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS/MiniMax-M2.7-UD-IQ2_XXS-00001-of-00003.gguf \
  --out-dir ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  --force
```

Generic form:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model /path/to/MiniMax-M2.7-00001-of-00003.gguf \
  --out-dir /path/to/MiniMax-M2.7-IQ2_XXS-Flash \
  --force
```

Alternate checked quantisation:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model /Volumes/X2T/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf \
  --out-dir /Volumes/X2T/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  --force
```

For partial validation only, you can scope routed layers:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model /path/to/MiniMax-M2.7-00001-of-00003.gguf \
  --out-dir /tmp/minimax-layer0-check \
  --layers 0 \
  --force
```

Useful export flags:

- `--skip-sidecar` to reuse `OUT_DIR/sidecar` and rebuild only `model-dense.gguf` plus `flashmoe-package.json`
- `--skip-dense` to refresh only `OUT_DIR/sidecar`

The wrapper reads `moe_topk`, `moe_slot_bank`, `moe_cache_io_split`, and `moe_prefetch_temporal` from `flashmoe-package.json` when it is present. Environment variables override those package defaults.

## Inference Examples

The current MiniMax package defaults are speed-oriented, and `run_minimax_m2_flash.sh` reads them from `flashmoe-package.json`:

- `--moe-mode slot-bank`
- `--moe-topk 4`
- `--moe-slot-bank 64`
- `--moe-cache-io-split 4`
- `--moe-prefetch-temporal`
- Metal replay and CPU-visible slot writes enabled by default

Important: the Flash-MoE routed-expert width is controlled by `--moe-topk`, or `MOE_TOPK=...` when using the wrapper. That is different from the sampling flag `--top-k`.

### M5 Max, native MiniMax routing width

MiniMax M2.7 natively uses `8` experts per token. To match the model default on an `M5 Max`, use:

```bash
MOE_TOPK=8 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st \
  -n 128 \
  -p "What is Apple Neural Engine? Answer in 3 sentences."
```

This was smoke-tested locally with:

- `MOE_TOPK=8`
- `MOE_SLOT_BANK=64`
- `CTX=4096`
- `BATCH=64`
- `UBATCH=1`
- `N_GPU_LAYERS=999`

On the checked `M5 Max`, the observed Flash-MoE load reported about:

- `17565 MiB` model memory on `MTL0`
- `992 MiB` context memory on `MTL0`
- about `18560 MiB` total device allocation for the test run

### M1 Max 64 GB

With 64 GB unified memory, a larger slot bank fits comfortably:

```bash
MOE_TOPK=4 MOE_SLOT_BANK=128 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"
```

### Lower-memory profile for a 24 GB-class Apple Silicon machine

For a smaller unified-memory Apple Silicon system, reduce slot-bank reserve first, then context and batch.

Recommended starting point:

```bash
MOE_TOPK=8 \
MOE_SLOT_BANK=32 \
CTX=2048 \
BATCH=32 \
UBATCH=1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st \
  -n 96 \
  -p "Summarize Apple Neural Engine in 3 sentences."
```

Approximate package footprint for this specific `UD-IQ2_XXS-Flash` package, before context and other runtime overhead:

- `MOE_SLOT_BANK=64`: about `17.15 GiB`
- `MOE_SLOT_BANK=32`: about `9.87 GiB`
- `MOE_SLOT_BANK=24`: about `8.04 GiB`
- `MOE_SLOT_BANK=16`: about `6.22 GiB`

If a `24 GB` machine is still tight on memory, reduce in this order:

- lower `MOE_SLOT_BANK` from `32` to `24`, then `16`
- lower `CTX` from `2048` to `1024`
- lower `BATCH` from `32` to `16`
- set `N_GPU_LAYERS=0` only as a last resort, because it will be much slower

### Reproducible comparison test

For side-by-side MiniMax branch comparisons, pin the binary, seed, and temperature. `TEMP=0.2` is a good compromise for long code generations because it avoids the repetition loops that `TEMP=0` can trigger.

```bash
LLAMA_BIN=/absolute/path/to/build/bin/llama-cli \
MOE_TOPK=4 \
SEED=123 \
TEMP=0.2 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"

LLAMA_BIN=/absolute/path/to/build/bin/llama-cli \
MOE_TOPK=8 \
SEED=123 \
TEMP=0.2 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"
```

This keeps the sampling path comparable within the same binary. Across different branches, exact token-for-token identity is still not guaranteed if the runtime changes logits or routing slightly.

Useful overrides:

- `MOE_TOPK=8` to keep the native MiniMax routing width
- `MOE_SLOT_BANK=96` for larger-memory Apple Silicon systems
- `PREFETCH_SIDECAR=/Volumes/OtherSSD/MiniMax-sidecar` to split prefetch traffic onto another drive
- `LLAMA_BIN=/abs/path/to/llama-cli` if you do not want the default `./build/bin/llama-cli`
