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
timings at the HY4 shape: the fused kernels take approximately 0.25-0.32 ms
per layer versus 0.58-0.63 ms for the reference encoder and 0.26-0.31 ms for
the generic `mul_mat_id` graph. These are marginal wall-time estimates from
chained operations with submission amortized, not hardware GPU timestamps.

End-to-end decode is substantially affected by SSD misses and per-layer GPU
synchronization. Kernel-only timings are not a whole-model speedup, and OS
page-cache changes must not be attributed to fusion. With direct shared-buffer
writes, expert-upload time should be zero.

Earlier end-to-end results predate the slot-reservation correction described
below. Do not attribute unexplained output differences only to F32 rounding:
first verify the expert-to-slot mapping. Small differences between generic and
fused arithmetic can still change near-tied routing decisions; numerical and
token-equivalence tests are both required.

### Slot reservation correctness (2026-09-04)

The runtime now protects **all requested resident experts before reserving any
misses**. Previously, an early miss could evict an expert needed later in the
same top-8 request, assigning two different experts to the same slot. This
affected both generic and fused slot-bank execution, not the packed sidecar
format. The same protection is applied to temporal/oracle prefetch reservations.
No model conversion or re-download is needed; rebuild the runtime.

```bash
cmake -S . -B build -DGGML_METAL=ON -DLLAMA_FLASH_MOE_GPU_BANK=ON \
  -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build --target llama-cli test-flashmoe-slot-reservation \
  test-flashmoe-slot8-hyv4 test-quantize-fns test-flashmoe-split-repack -j 12
ctest --test-dir build --output-on-failure \
  -R '^(test-flashmoe-slot-reservation|test-flashmoe-slot8-hyv4|test-quantize-fns|test-flashmoe-split-repack)$'
```

The reservation regression test uses the production protection/eviction
helpers and covers 8/16/96-slot banks, repeated IDs, overflow, epoch wrap, and
40,000 randomized requests. It can additionally replay an expert trace:

```bash
./build/bin/test-flashmoe-slot-reservation --trace /path/to/recorded.trace.jsonl
```

The checked HY4 `Hello` raw completion produced the same 16 greedy token IDs
with 8 slots, 96 slots, the generic path, the one-op reference path, and an
all-resident replay. The formerly failing 12,012-call trace now passes the
reservation test with no expert-slot aliases. These are bounded checks, not
a claim of full-model quality evaluation.

The final rebuilt runtime also passed a 64-token raw `Hello` comparison:
fused 8-slot, fused 96-slot and reference 96-slot execution produced identical
greedy token IDs and bit-identical captured top-32 logits at every step. Each
path emitted exactly 64 output-logit callbacks, with zero expert-slot aliases.

The batched-read `source_wall` counter now measures elapsed batch time instead
of falling back to the sum of parallel workers' service times. `source` remains
summed service time and can exceed total runtime. Use `install`/`source_wall`
for elapsed I/O comparisons. All-hit replay remains a diagnostic, not a
deployable SSD throughput claim.

The September 4 optimization sweep retained the correctness and profiling
fixes and the shared-FFN overlap described below (now default-on for eligible HY4). Keep 96 slots and
I/O split 4. More read workers,
vectored reads, read-advice hints, alternate fused-kernel row counts and
command-buffer scheduling were tested without a convincing improvement and
are **not deployed switches**. A 104-slot bank consumed another 6.02 GiB but
did not consistently improve TPS. These tests did not purge the OS page cache.

The `--perf` table uses measured I/O wall time where available. The line
`outside routed host` includes both dense and routed GPU execution, waits and
graph scheduling; it must not be interpreted as dense-only kernel time.
The downstream evaluation callback is also no longer invoked twice per output.

### Exact demand-read/shared-FFN overlap (HY4 default)

After the router has selected the **actual eight experts**, this path starts
miss reads in the background while Metal runs the independent shared FFN in
that layer. It joins the reads and publishes slot IDs **before** the routed
FFN executes. This is not speculative next-layer prediction: it issues the same
reads, preserves packed STQ1_0/IQ2_XXS/IQ3_XXS/IQ4_XS bytes, and does not change
top-8, arithmetic, or the final shared+routed addition order.

It is enabled by default for eligible HY4 runs; no environment prefix is needed:

```bash
./build/bin/llama-cli \
  -m ~/Models/HY4/Hy4-preview-Flash-STQ1_0/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/HY4/Hy4-preview-Flash-STQ1_0/sidecar \
  --moe-slot-bank 96 --moe-topk 8 --moe-cache-io-split 4 --slot8 --fp16head \
  -fit on -ub 1 -b 1 -c 2048 -ngl 999 \
  --no-warmup -st --temp 0 --seed 1 \
  -p "Make a game of Tetris in HTML" -n 128 --perf
```

Default is **ON for eligible HY4**. Set
`LLAMA_FLASH_MOE_HY4_SHARED_IO_OVERLAP=0` to disable, `=1` to explicitly enable,
or unset it to restore the default. This is a runtime switch, not a new CMake
option. It requires HY4 single-token `--slot8`
and direct CPU-visible bank writes with batched reads. Unsupported settings
use synchronous installs: generic/non-slot8 execution, resident/oracle modes,
runtime transcoding, staging copies, async upload, temporal/hidden prediction,
and demand striping/distribution/racing are excluded. macOS SSD mode already
defaults to CPU-visible slot writes and parallel reads; explicitly setting
either existing option to `0` can disable eligibility.

The CLI settings show `hy4-shared-io = on (default)` when unset, or
`on (env)` / `off (env)` for an explicit override. The line labels the policy
as HY4-only; other model architectures are unchanged. Actual execution
is separately proven by this line (a requested setting alone is not proof):

```text
HY4 shared-FFN I/O overlap ACTIVE: exact demand reads, join before routed compute
```

The shutdown log reports overlap calls, join wait, and worker time completed
outside the join. Hidden worker time is **not** the net TPS gain: thread launch,
graph scheduling and GPU synchronization still cost time. With overlap on,
`install`/`source_wall` retain worker elapsed time; the `--perf` table's
`Expert I/O + join wait` row counts synchronous reads plus blocking join time,
excluding overlapped worker work. `--perf` also prints precise decode elapsed
milliseconds and generation TPS below the CLI's rounded timing summary.

Matched OFF/ON/ON/OFF runs on the 128 GiB M5 Max, AC power, 96 slots, split 4,
`--slot8 --fp16head`, context 2048 and 128 predictions measured:

| Prompt | OFF TPS | ON TPS | Gain |
|---|---:|---:|---:|
| Tetris HTML | 4.920 | 5.009 | 1.8% |
| SSD versus RAM | 4.676 | 4.777 | 2.2% |

After cancellation/profiling hardening, a reversed ON/OFF/OFF/ON Tetris check
on the final build confirmed **4.916 → 5.025 t/s (+2.2%)**, again with identical
text, expert/slot traces and read volume.

TPS is computed from the mean decode elapsed time for each pair (127 decode
tokens). Both prompts produced identical text and expert/slot traces ON/OFF.
Read volume and call counts were unchanged. These are repeated local runs,
not controlled cold-SSD measurements, and the existing direct-IQ2-LUT CMake
cache was OFF for both arms. Do not extrapolate this small gain into a claimed
6–8 t/s result or a kernel-only speedup.

The runtime drains outstanding work before context/graph reuse or destruction
and invalidates affected slots if direct reads fail. Native interruption/reuse
testing is available with the real local package (not part of default CTest
model downloads):

```bash
cmake --build build --target test-flashmoe-hyv4-lifecycle -j 12
./build/bin/test-flashmoe-hyv4-lifecycle \
  -m ~/Models/HY4/Hy4-preview-Flash-STQ1_0/model-dense.gguf \
  --moe-sidecar ~/Models/HY4/Hy4-preview-Flash-STQ1_0/sidecar
```

This test reports backend-abort coverage separately: Metal may not poll its
abort callback in these tiny graph segments. A skipped polling case is not
proof that backend cancellation was exercised.

Final rebuilt numerical checks: overlap OFF/ON at 96 slots, overlap ON at
8 slots, forced reference at 96 slots, and direct writes disabled at 96 slots
all matched **64 greedy token IDs and all 2,048 captured top-32 logits
bit-for-bit**. The generic fallback with overlap requested stayed synchronous
and matched the first 16 greedy token IDs. Cold-bank exceptions, callback stop,
and repeated exceptions preserved full-vocabulary logits after context reuse.
The four quantization/sidecar/reservation/Metal tests passed. These bounded
checks do not replace a full-model quality evaluation or SSD fault injection.

### Exact iHC post graph simplification (HY4 default)

HY4's independent Hyper-Connections apply a post gate to each residual stream.
The original graph emits a multiply and add for each stream, then concatenates
the results. The broadcast graph replaces those repeated operations with
repeat, multiply and add, preserving the same separate F32 multiplication and
addition and the original output dtype. It does not change the stream
reduction, attention, routing, quantization, or packed expert bytes.

Both optimizations are now enabled by default for HY4 (overlap still requires
the eligible SSD path). Run the same command without either environment prefix:

```bash
./build/bin/llama-cli \
  -m ~/Models/HY4/Hy4-preview-Flash-STQ1_0/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/HY4/Hy4-preview-Flash-STQ1_0/sidecar \
  --moe-slot-bank 96 --moe-topk 8 --moe-cache-io-split 4 --slot8 --fp16head \
  -fit on -ub 1 -b 1 -c 2048 -ngl 999 \
  --no-warmup -st --temp 0 --seed 1 \
  -p "Make a game of Tetris in HTML" -n 128 --perf
```

The broadcast switch is runtime-only and defaults **ON for HY4**. Set
`LLAMA_FLASH_MOE_HY4_HC_POST_BROADCAST=0` to use the original stream-loop graph;
unset it to restore the default, or set it to `1` to explicitly enable.
No new CMake option or model conversion is required. After rebuilding,
startup settings show the default or explicit override:

```text
hy4-hc-post = broadcast (default) (HY4 only)
hy4-hc-post = broadcast (env) (HY4 only)
hy4-hc-post = stream-loop (env) (HY4 only)
```

The settings line is visible at the CLI's normal log level. With `--verbose`,
actual HY4 graph selection additionally reports
`HY4 iHC post broadcast graph enabled`. Other architectures are unchanged.
With `-fit on`, that first graph can be the fit dry run, whose logs are
debug-level; `-lv 3` alone may not show this one-time additional message.

For a baseline with both optimizations disabled, prefix the same command with
`LLAMA_FLASH_MOE_HY4_SHARED_IO_OVERLAP=0 LLAMA_FLASH_MOE_HY4_HC_POST_BROADCAST=0`.
Existing `=1` commands still work unchanged. This default change does not alter
your slot count, I/O split, `--fp16head`, or IQ2-LUT compile-time selection.

Default-on validation also covered 100 slots with I/O split 8: unset and
explicit-both-`1` matched 32 token IDs and all 1,024 captured top-32 logit values
exactly. Independent `0` overrides and both disabled matched an eight-token
check, with overlap activation and startup labels verified in every case.

Native CPU/Metal fixtures compare the production helper against the original
graph, including F32/F16, 1/2/4/8 streams, multi-token input, padded strides,
and the full 6144-by-4 shape. All 68 cases (34 Metal) matched bit-for-bit.
The full-shape graph has 12 nodes instead of 30, including views and casts;
node counts are not GPU dispatch counts. Run the checks with:

```bash
cmake -S . -B build -DGGML_METAL=ON -DLLAMA_FLASH_MOE_GPU_BANK=ON \
  -DLLAMA_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build --target llama-cli test-hyv4-hc-post -j 12
./build/bin/test-hyv4-hc-post --require-metal --bench 100
```

On the AC-powered 128 GiB M5 Max, an OFF/ON/ON/OFF Tetris comparison measured
**5.042 → 5.168 generation t/s (+2.5%)** and 3.066 → 3.103 prompt t/s.
Both arms used shared-I/O overlap ON, 96 slots, split 4, `--slot8 --fp16head`,
context 2048, single-token batches, and the existing direct-IQ2-LUT build
cache OFF. All four runs had identical text, complete expert/slot traces,
168.80 GiB of application-level reads, zero uploads and zero slot aliases.
Separately, 64 greedy token IDs and all 2,048 captured top-32 logit values
matched the validated original graph exactly. These are bounded, repeated
local tests, not controlled cold-SSD measurements or a full quality evaluation.
After removing the rejected kernel experiments and rebuilding, a reversed
ON/OFF pair confirmed **5.018 → 5.147 t/s (+2.6%)**. The five targeted regression
tests passed, and the 64-token/logit comparison was again exact. Eight-slot
interruption/reuse checks with broadcasting enabled also retained bit-identical
full-vocabulary logits; backend-abort polling remained unexercised as described
above.

The parallel dense audit tested 16 Q5_K/Q6_K row/threadgroup layouts using
actual dense GGUF tensors. All 162 full outputs were identical, but there was
no convincing performance gain; the original dense kernels remain unchanged.
Routed testing includes **STQ1_0** (Sherry's 1.25-bit payload, 1.3125 stored bpw),
IQ2_XXS, IQ3_XXS and IQ4_XS, not only the IQ2 path.
Multiple-SIMD-group routed layouts also passed numerical checks, but the best
microbenchmark candidate did not establish a full-model gain. Those layouts
were removed too; neither tiling experiment is a deployed runtime switch.

The interruption review confirmed that exact host routing still requires a
GPU boundary at each routed layer. A second wait at the shared-I/O join may
be avoidable, but safely removing it requires preserving in-flight Metal
command-buffer lifetime and error checks. That scheduler change is **not
implemented**. Existing graph reuse is supported; do not assume that every
decoded token necessarily rebuilds the graph.

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
