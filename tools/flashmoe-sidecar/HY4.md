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

Build from the `HY4-1.25-bit` checkout with Metal and the Flash-MoE GPU bank
enabled:

```bash
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON \
  -DLLAMA_FLASH_MOE_HY4_DIRECT_IQ2_LUT=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_TESTING=ON
cmake --build build \
  --target llama-cli test-quantize-fns test-flashmoe-split-repack test-flashmoe-slot8-hyv4 \
  -j 12
ctest --test-dir build --output-on-failure -R '^(test-quantize-fns|test-flashmoe-split-repack|test-flashmoe-slot8-hyv4)$'
```

Metal initialization prints the selected compile-time IQ2 implementation on
every startup:

```text
ggml_metal_library_init: HY4 IQ2 LUT: ENABLED (direct constant-memory LUT, compile-time)
```

An OFF build prints `HY4 IQ2 LUT: DISABLED (threadgroup LUT, compile-time)`.
The earlier `Flash-MoE settings:` block also reports
`hy4-iq2-lut = direct-constant (compiled=on)` or
`hy4-iq2-lut = threadgroup (compiled=off)` before model loading.

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
slices from SSD. Start with `-ub 1`. `--slot8` is optional and routes each
layer's eight selected experts through one fused Metal operator (see the
section below); `--slot4` does not apply because HY4 is native top-8.

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

For a 128 GB M5 Max throughput run, use `--moe-slot-bank 96` and add `--slot8`.
The two fast I/O paths are automatic for this macOS Metal SSD-streaming mode.
For the memory-saving configuration, use the same command with eight slots.
Eight is the minimum slot count that preserves HY4's native top-8 routing.

```bash
./build/bin/llama-cli \
  -m ~/Models/HY4/Hy4-preview-Flash-STQ1_0/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/HY4/Hy4-preview-Flash-STQ1_0/sidecar \
  --moe-slot-bank 96 --moe-topk 8 --moe-cache-io-split 4 --slot8 \
  --fp16head \
  -fit on -ub 1 -b 1 -c 2048 -ngl 999 \
  --no-warmup -st --temp 0 --seed 1 \
  -p "Make a game of Tetris in HTML" -n 128 --perf
```

The three fast-path settings have separate jobs:

- `--fp16head` (also spelled `--fp16-head`) converts an untied F32 output head
  to F16 once at load time. The checked HY4 dense GGUF has a 2.77 GiB F32
  `output.weight`; its runtime F16 copy is about 1.38 GiB. This does not rewrite
  the GGUF or sidecar. Omit the option (or use `--no-fp16-head`) to keep F32;
  expect a one-time startup conversion and small logit-rounding differences.
  A successful conversion logs `converted output.weight F32 -> F16`.
- CPU-visible slot writes let `pread()` write directly into the Metal shared
  slot buffer, eliminating the staging-to-bank copy and its expert-upload time.
- Parallel slot reads issue independent SSD miss chunks concurrently. With
  `--moe-cache-io-split 4`, the runtime also batches the split install reads.

Both I/O paths turn on automatically for ordinary macOS Metal SSD `slot-bank`
streaming. The existing environment variables remain explicit overrides for
A/B testing: set
`LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES=0` or
`LLAMA_FLASH_MOE_EXPERIMENTAL_PARALLEL_SLOT_READS=0` to disable the respective
default, and set either to `1` to force it on. Direct slot writes engage only
when the routed buffer is actually Metal shared memory; private Metal buffers
fall back safely to staging/upload. Startup should report
`parallel-reads = on (default)` and `cpu-vis-writes = on (default)`. With I/O
split 4, confirm the final summary says `cpuvis=on preads=on batchrd=on` and
reports zero expert-upload time.

The checked 96-slot run with these paths improved from about 3.8-3.9 to 4.7
generation tokens/s and from 2.0 to 3.2 prompt tokens/s. The temporal-prefetch
heuristic refreshes the current token's experts to bias later slot residency;
it is not a future-router predictor and did not help the checked HY4 trace, so
the fast command omits it.

With this conservative `-b/-ub 1` smoke configuration, `-ngl 999` offloads
dense/shared work to Metal while the generic sidecar-routed path runs the
per-expert `mul_mat_id` decode kernels. Set
`LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE=1` only when explicitly testing
the experimental Metal slot-decode route.

## Fused Metal top-8 (`--slot8`)

Adding `--slot8` to the command above replaces the per-expert gate/up/SwiGLU/
down graph of every routed layer with one `GGML_OP_FLASHMOE_SLOT8_FFN` operator.
On Metal that operator has two implementations:

- **Fused kernels** (default): two dispatches per layer, Phase A
  (`gate + up -> SwiGLU` for all eight experts) and Phase B
  (`down` + routed weighted sum). Phase A is selected from the gate/up type
  (`STQ1_0` or `IQ2_XXS`, gate and up must match) and Phase B independently from
  the down type (`IQ3_XXS` or `IQ4_XS`). All four possible combinations are
  covered (the published package exercises three); the per-row dot products mirror the canonical
  `kernel_mul_mv_*` lane mapping and F32 accumulation order, so the fused output
  is bit-identical to the reference encoder below. IQ2_XXS Phase A reads its
  immutable grid/sign LUTs directly from Metal constant memory; this is the
  compiled default and requires no environment variable. Configure with
  `-DLLAMA_FLASH_MOE_HY4_DIRECT_IQ2_LUT=OFF` to compile the original
  threadgroup-copy LUT path for A/B testing.
- **Reference encoder**: the existing `mul_mv`, `swiglu` and weighted-sum
  kernels orchestrated inside the same operator. It is the correctness oracle
  and the automatic fallback for any other type triplet or for dims that are not
  multiples of 256. Force it with `LLAMA_FLASH_MOE_SLOT8_REFERENCE=1` for A/B
  runs.

Do not trust `slot8 eligible` debug lines alone: they only prove the graph
selected the operator. The Metal backend counts what actually ran and prints it
when it shuts down:

```text
ggml_metal_op_flashmoe_slot8_log_stats: flashmoe_slot8 fused=2849 [iq1=0 stq1_0/iq3_xxs=1073 stq1_0/iq4_xs=0 iq2_xxs/iq3_xxs=1665 iq2_xxs/iq4_xs=111] reference=0
```

For the STQ1_0 package that is 37 tokens x 77 layers, split into 29 STQ1_0/IQ3_XXS
layers, 45 IQ2_XXS/IQ3_XXS layers and 3 IQ2_XXS/IQ4_XS layers; `reference` must
be 0 for a fully fused run. A run that exercises IQ2 layers also reports:

```text
flashmoe_slot8 HY4 IQ2 LUT=direct-constant dispatches=N
```

with `N > 0`. An OFF build reports `LUT=threadgroup` instead. This is a CMake
compile-time selection; `LLAMA_FLASH_MOE_EXPERIMENTAL_HY4_DIRECT_IQ2_LUT` is
not a runtime switch.

`test-flashmoe-slot8-hyv4` (built with `LLAMA_BUILD_TESTS=ON`, needs a Metal
device) checks every combination at a small shape, at the real HY4 routed shape
and at non-8 widths against an exact double-precision evaluation of the
dequantized weights, and compares fused, reference and generic outputs with each
other. Measured on an M5 Max: fused and reference differ by exactly 0, and all
three Metal paths stay within 2.5e-7 of the exact result relative to the output
maximum. `test-flashmoe-slot8-hyv4 --bench 50` also reports per-layer operator
timings at the HY4 shape: the fused kernels take 0.25-0.32 ms of GPU time per
layer versus 0.58-0.63 ms for the reference encoder and 0.26-0.31 ms for the
generic `mul_mat_id` graph.

Expectations for end-to-end decode with the 8-slot bank: the profile is SSD
bound (roughly 80% expert I/O, 20% upload), so `--slot8` does not change the
tokens-per-second figure; it removes about 25 ms per token relative to the
reference encoder and is on par with the generic path. Greedy text can differ
from the generic path after a few tokens: the operator outputs differ only by F32
reordering (about 1e-7 relative), but a near-tie in the router or the sampler
can flip a token, after which the sequences diverge. The reference encoder shows
the same behaviour, and a teacher-forced comparison of the two paths agrees to
5e-5 in the logits on the first decoded token.

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
- **`--slot8` reports `fused=0 ... reference=N`:** some routed layer has a
  type triplet outside the four supported STQ combinations or dims that are not
  multiples of 256, so the operator used its reference encoder. The output is
  still correct, only slower. Omit `--slot8` to use the generic per-expert path.
- **Interrupted copy:** remove only the unfinished generated destination (or
  use `--force` after confirming the target), then retry. Completed layer files
  are atomically published.
