# HY4 Flash-MoE Dense + Sidecar Guide

This workflow splits the released HY4 preview GGUF into two runtime artifacts:

- `model-dense.gguf`: embeddings, attention/DSA/iHC tensors, routers, shared
  experts, and the leading dense FFN.
- `sidecar/`: only routed `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps` weights, arranged expert-major for streamed slot-bank decode.

The source GGUF is never modified. The extractor copies quantized payload bytes
exactly; it does not dequantize or re-quantize routed weights.

## Source format

The tested source is:

```text
/Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf
```

It is a 214 GiB `hyv4` GGUF with 78 blocks: `blk.0` is dense and `blk.1`
through `blk.77` are 256-expert, native top-8 MoE layers. Its routed tensors
are mixed quantization: 29 gate/up layer pairs use `STQ1_0` (GGML type 43,
42 bytes per 256 weights, 1.3125 bpw) and the remainder use IQ2_XXS,
IQ3_XXS, or IQ4_XS. The complete file averages 2.38 bpw.

`STQ1_0` is sometimes described informally as 1.35-bit. It is not the older
`TQ1_0` format and must not be aliased to it. This branch provides exact
type-43 support for the sidecar tooling and runtime. It uses the AngelSlim
HF release as the format source; Tencent's upstream HY4 repository does not
ship this STQ1_0 quantization.

The package supports ordinary autoregressive decode. It does not add HY4
native-MTP/speculative decoding.

This first runtime port preserves the DSA/indexer weights in the dense GGUF,
but uses the existing full-attention MLA cache because this fork does not yet
have HY4's native sparse-indexer cache. That is equivalent while the visible
causal KV history is at most HY4's `indexer.top_k` of 2048 (including the
recommended short smoke run below). Beyond 2048 keys it is a correctness-safe
full-attention fallback, not an implementation of HY4's advertised 1M-token
DSA behavior.

## Build the runtime

Build from the `HY4-1.25-bit` worktree with Metal enabled:

```bash
cmake -S . -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build --target llama-cli test-quantize-fns test-flashmoe-split-repack -j 8
ctest --test-dir build --output-on-failure -R '^(test-quantize-fns|test-flashmoe-split-repack)$'
```

Those are type/build preflights. The final inference check must run on a
native Metal host rather than a sandbox without a Metal device.

## Inspect before copying

This metadata-only command confirms the expected 231 routed tensors, their
names, type sizes, and expert-major eligibility without materializing the
214 GiB model:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model /Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf \
  --families routed --json
```

The default package destination is:

```text
~/Models/HY4/Hy4-preview-Flash-STQ1_0
```

Use a fresh destination and reserve space for both its dense GGUF and its
routed sidecar. Keep the original source until native decode succeeds. The
helper prints exact sizes after inspecting and extracting, so do not estimate
capacity from the `STQ1_0` name alone.

## One-layer sidecar canary

First make a small, non-runnable sidecar canary. It validates STQ byte sizing,
expert-major offsets, and bounded 8 MiB source copying:

```bash
python3 ./tools/flashmoe-sidecar/hyv4_prepare.py \
  --model /Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf \
  --out-dir /tmp/hyv4-sidecar-smoke \
  --layers 1 --skip-dense
```

The helper metadata-verifies automatically. To byte-compare the copied canary
without rewriting it:

```bash
python3 ./tools/flashmoe-sidecar/hyv4_prepare.py \
  --model /Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf \
  --out-dir /tmp/hyv4-sidecar-smoke \
  --layers 1 --skip-sidecar --skip-dense --verify-bytes
```

`--layers` is intentionally sidecar-only. It requires `--skip-dense` so a
partial package cannot be mistaken for an inferable model.

## Full conversion

For a fresh, empty destination, run:

```bash
python3 ./tools/flashmoe-sidecar/hyv4_prepare.py \
  --model /Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf \
  --out-dir ~/Models/HY4/Hy4-preview-Flash-STQ1_0
```

The package layout is:

```text
Hy4-preview-Flash-STQ1_0/
├── model-dense.gguf
├── flashmoe-package.json
└── sidecar/
    ├── manifest.json
    ├── layer_001.bin
    ├── ...
    └── layer_077.bin
```

`manifest.json` records original tensor names, source offsets, exact byte
counts, quant types, per-entry expert counts, and per-expert strides.
`model-dense.gguf` deliberately omits routed experts and must always be loaded
with its `sidecar/` directory.

`--force` replaces only generated package outputs; use it only for a known
interrupted/disposable destination. It removes and recreates that package's
generated `sidecar/` directory plus its dense GGUF/package manifest; it never
alters the source GGUF.

For a full byte-for-byte post-copy check without rewriting either artifact:

```bash
python3 ./tools/flashmoe-sidecar/hyv4_prepare.py \
  --model /Volumes/TB36/Models/Hy4/Hy4-preview-GGUF/Hy4-preview-STQ1_0.gguf \
  --out-dir ~/Models/HY4/Hy4-preview-Flash-STQ1_0 \
  --skip-sidecar --skip-dense --verify-bytes
```

## SSD-streamed decode smoke test

HY4 routes eight experts per token. Keep `--moe-topk 8` and use at least eight
slots. The expert-major sidecar lets the slot bank pread only selected expert
slices from SSD. Start with `-ub 1`. Do **not** use `--slot4` or `--slot8`:
the existing IQ fused kernels do not support HY4's mixed STQ1_0/IQ weights.

```bash
./build/bin/llama-cli \
  -m ~/Models/HY4/Hy4-preview-Flash-STQ1_0/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/HY4/Hy4-preview-Flash-STQ1_0/sidecar \
  --moe-slot-bank 8 --moe-topk 8 --moe-cache-io-split 4 \
  -fit on -ub 1 -b 1 -c 128 -ngl 999 \
  --no-warmup -st -p 'Hello' -n 2 --perf
```

Successful logs should show sidecar coverage validation, virtualized routed
tensors, and `pread-slot-bank` as the routed source. Use the `--verify-bytes`
conversion command above for a raw source-to-sidecar parity check;
`--moe-cache-io-split 4` keeps sidecar reads page-aligned and bounded. A larger
bank may reduce repeated SSD reads at the cost of unified-memory residency.

With this conservative `-b/-ub 1` smoke configuration, `-ngl 999` offloads
dense/shared work to Metal while the generic sidecar-routed path is allowed to
run on its supported host staging path. It is deliberately not the unfinished
HY4 fused Metal MoE path. Set `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE=1`
only when explicitly testing the experimental Metal slot-decode route.

Keep the smoke context at or below 2048 until native HY4 DSA cache support is
implemented; raising `-c` alone does not make this first port equivalent to the
reference sparse-attention model at long context.

## Problems to avoid

- **Unknown type 43 / STQ1_0:** use a binary and Python tools from this branch,
  not stock llama.cpp or a system GGUF package.
- **Missing routed tensors:** point `--moe-sidecar` at the directory containing
  `manifest.json`; the dense-only HY4 GGUF is intentionally incomplete.
- **Too few slots:** use `--moe-slot-bank 8 --moe-topk 8` or larger. Reducing
  native top-K changes behavior.
- **Fused-kernel error:** remove `--slot4`/`--slot8`; HY4 initially uses the
  generic separate gate/up/down SSD-streamed path.
- **Interrupted copy:** remove only the unfinished generated destination (or
  use `--force` after confirming the target), then retry. Completed layer files
  are atomically published.
