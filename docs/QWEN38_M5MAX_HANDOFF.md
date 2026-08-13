# Qwen3.8 M5 Max handoff

Temporary continuation note for branch `Qwen3.8`.

Delete this file in the first M5 Max continuation commit after its contents have been reviewed and the work has resumed.

## Goal

Add end-to-end Flash-MoE inference for the Unsloth Qwen3.8-2.4T-A95B `UD-Q1_0` GGUF, including the fork-local `IQ1_XS`, `IQ1_XXS`, and `IQ1_XXXS` formats on CPU, CUDA, and Metal, then create the standard dense-plus-sidecar package under `~/Models/Qwen3.8`.

Do not create a branch with a `codex/` prefix. Continue on `Qwen3.8`.

## Source and destination

- First source shard: `/Volumes/TB36/Models/Qwen/Qwen3.8-2.4T-A95B-GGUF/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf`
- Physical package destination: `/Volumes/SN8100/Qwen3.8`
- Standard entry point: `~/Models/Qwen3.8 -> /Volumes/SN8100/Qwen3.8`
- The source is immutable. Do not delete source shards to reclaim space.

The full export was started and immediately stopped for this host handoff. Atomic output cleanup worked: `/Volumes/SN8100/Qwen3.8` contained no partial layer or dense file afterward (only `.DS_Store`). Restart the export from the beginning.

## Inspected model

- architecture: `qwen35moe`
- name: `Qwen3.8-2.4T-A95B`
- shards: 10
- blocks: 93 (`blk.0` through `blk.92`)
- experts: 512
- native routed top-k: 10
- embedding length: 8192
- advertised context: 262144
- routed tensors: 279
- exact routed bytes: `360,374,599,680`
- dense/shared tensors: 1423
- exact dense tensor bytes: `36,870,741,504`
- total tensor payload: `397,245,341,184`

Layers 0-91 use `IQ1_XXXS`; layer 92 routed tensors use `Q2_K` and the layer also contains next-token/MTP tensors that must remain in the dense GGUF.

## Upstream authority

Unsloth pins commit `c86ed269986f2dced6325c5c58bda966a2e2ead1` from PR 91:

`IQ1_XS, IQ1_XXS, IQ1_XXXS: three quant types below IQ1_S`

Format contract:

| Type | ID | Block | Grid | Index | bpw |
|---|---:|---:|---:|---:|---:|
| IQ1_XS | 64 | 46 B / 256 | 1024 | 10 bit | 1.4375 |
| IQ1_XXS | 65 | 42 B / 256 | 512 | 9 bit | 1.3125 |
| IQ1_XXXS | 66 | 38 B / 256 | 256 | 8 bit | 1.1875 |

The upstream implementation contains CPU and CUDA MMVQ/MMQ, but no Metal. The current WIP imports/adapts the upstream definitions and adds local Metal matvec work.

## Completed validation

The sidecar parser recognizes IDs 64-66. A real one-family canary copied `blk.0.ffn_gate_exps.weight` into `/tmp/qwen38-sidecar-canary`:

- copied bytes: `1,275,068,416`
- source: all 10 GGUF shards
- verification: metadata plus full byte comparison passed

Python syntax/help and `git diff --check` passed before handoff.

## WIP code state

Implemented or imported:

- GGML/GGUF type IDs, block structs, sizes, traits, names, file types, validation, quant/dequant, and codebook tables
- portable CPU dequant and dot-product wiring
- CUDA dequant, MMVQ, MMQ traits/loaders, dispatch, and template instances adapted to this fork's older CUDA layout
- initial Metal `mul_mv` and `mul_mv_id` shader functions plus host dispatch for all three narrow IQ1 types
- Qwen3.8 one-root packaging helper: `tools/flashmoe-sidecar/qwen38_prepare.py`
- Qwen3.8 package/runtime metadata and sidecar README section

Still required on the M5 Max:

1. Build immediately and fix any compile errors, especially in the newly written Metal shaders and adapted CUDA templates.
2. Add focused quant round-trip and CPU backend tests for IDs 64-66.
3. Compile the Metal library and compare narrow-IQ1 Metal matvec output against CPU reference vectors.
4. Verify generic routed `mul_mv_id` with native top-10. Do not use `--slot4` or `--slot8`; Qwen3.8 routes 10 experts.
5. Check Flash-MoE sidecar slot-bank type allowlists and byte/row-size calculations for IDs 64-66.
6. Run the full export only after the parser/build gates pass, then run smoke inference.
7. CUDA cannot be runtime-tested on the M5 Max; at minimum keep the imported upstream behavior structurally aligned and arrange a CUDA build/test later.

The unrelated untracked `node_modules/`, `package.json`, and `package-lock.json` predated this work and must not be staged.

## Resume commands

Start with a Metal build:

```bash
cmake --build build -j 8
```

Run focused tests after the build:

```bash
./build/bin/test-quantize-fns
./build/bin/test-backend-ops -o MUL_MAT
```

Full package export (authorized by the user, but restart only after the build/parser gates):

```bash
env PYTHONDONTWRITEBYTECODE=1 \
/Users/anemll/SourceRelease/GITHUB/ML_playground/anemll/env-anemll/bin/python3 \
tools/flashmoe-sidecar/qwen38_prepare.py \
  --model /Volumes/TB36/Models/Qwen/Qwen3.8-2.4T-A95B-GGUF/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf \
  --out-dir /Volumes/SN8100/Qwen3.8 \
  --force \
  --verify-bytes
```

Monitor atomic sidecar growth without modifying it:

```bash
du -sh /Volumes/SN8100/Qwen3.8/sidecar
find /Volumes/SN8100/Qwen3.8/sidecar -maxdepth 1 -type f -print | sort | tail
```

Expected package layout:

```text
~/Models/Qwen3.8/
├── model-dense.gguf
├── flashmoe-package.json
└── sidecar/
    ├── manifest.json
    └── layer_000.bin ... layer_092.bin
```

Initial inference shape after runtime validation:

```bash
./build/bin/llama-cli \
  -m ~/Models/Qwen3.8/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/Qwen3.8/sidecar \
  --moe-slot-bank 32 \
  --moe-topk 10 \
  --moe-cache-io-split 4 \
  --moe-prefetch-temporal \
  -fit on -ub 1 -b 1 -ngl 999 \
  -c 128 --no-warmup -st \
  -p "Hello" -n 2
```

Treat that inference command as a starting smoke configuration, not a performance result.
