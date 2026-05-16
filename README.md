# anemll-flash-llama.cpp

Flash-MoE inference for large GGUF Mixture-of-Experts models on Apple Silicon, using `llama.cpp`.

> This repo stays close to upstream `ggml-org/llama.cpp`, but adds sidecar-backed routed expert loading, streamed slot-bank execution, trace capture, oracle replay, and bank-modeling tools for Flash-MoE workflows.
>
> Flash-MoE-specific guide: [tools/flashmoe-sidecar/README.md](./tools/flashmoe-sidecar/README.md)
>
> MiniMax M2.7 note: [tools/flashmoe-sidecar/MINIMAX_M2.md](./tools/flashmoe-sidecar/MINIMAX_M2.md)
>
> Flash-MoE bank-modeling workflow: [docs/moe-bank-modeling-workflow.md](./docs/moe-bank-modeling-workflow.md)
>
> Native slot-bank model-porting guide: [docs/native-slot-bank-porting.md](./docs/native-slot-bank-porting.md)

## Quick jump — sidecar extract & run

Per-model extract + run recipes live in [tools/flashmoe-sidecar/README.md](./tools/flashmoe-sidecar/README.md):

| Model | Arch | Status | Extract | Run |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | qwen3moe | Stable (anchor) | [Extract](./tools/flashmoe-sidecar/README.md#extract-a-qwen35-sidecar) | [Run](./tools/flashmoe-sidecar/README.md#run-with-the-sidecar) |
| Qwen3.5-397B-A17B | qwen35moe (linear-attn) | Stable (slot-bank required) | [Extract](./tools/flashmoe-sidecar/README.md#extract-a-qwen35-sidecar) | [Run](./tools/flashmoe-sidecar/README.md#run-with-the-sidecar) |
| Gemma4-26B-A4B | gemma4 | Stable | [Extract](./tools/flashmoe-sidecar/README.md#extract-a-gemma4-26b-a4b-sidecar) | [Run](./tools/flashmoe-sidecar/README.md#run-gemma4-26b-a4b-with-the-sidecar) |
| Kimi K2 / K2.5 | deepseek2 (MLA) | Experimental | [Extract](./tools/flashmoe-sidecar/README.md#extract-only-selected-layers) | [Run](./tools/flashmoe-sidecar/README.md#estimate-persistent-bank-cost-and-coverage) |
| MiniMax-M2.7 | minimax-m2 | Experimental (dense-package path) | [Export](./tools/flashmoe-sidecar/README.md#export-a-minimax-m27-flash-package) | [Run](./tools/flashmoe-sidecar/README.md#run-minimax-m27-with-flash-moe) |
| **GLM-5.1 (256×22B)** | **glm-dsa (MLA + DSA indexer)** | **Experimental** | [**Extract**](./tools/flashmoe-sidecar/README.md#extract-a-glm-51-sidecar) | [**Run**](./tools/flashmoe-sidecar/README.md#run-glm-51-with-the-sidecar) |

Build instructions are below; once built, jump to the extract + run recipe for your model.

## Current Model Support

- **`Qwen3.5` GGUF MoE** is the current anchor path for bring-up and regression work. Stable.
- **`Gemma4-26B-A4B`** GGUF MoE: stable. Native `n_expert_used = 8`. Sensitive to slot-bank size on smaller-memory machines — start with `--moe-slot-bank 8` or `16`.
- **`Kimi-K2` and `Kimi-K2.5`** GGUF support is experimental: sidecar extraction, slot-bank runtime, trace capture, and bank modeling work, but quality and performance are still being tuned. Kimi currently requires `-ub 1` for correct output. The `-ngl 99` dense GPU path produces degraded output on some runs due to a Metal compute issue being investigated.
- **`MiniMax-M2.7` (`minimax-m2`, 256 experts, native K=8)** is experimental but working in dense-package + slot-bank form on this branch. The checked `UD-IQ2_XXS` package was smoke-tested on M5 Max; use `MOE_TOPK=8` to match native routing width, or keep the wrapper default `MOE_TOPK=4` for lower expert I/O.
- **`GLM-5.1` (`glm-dsa`, 256×22B MoE)** is experimental: 79 layers, MLA attention with `q_lora_rank=2048`, DeepSeek-V3-style sigmoid routing (K=8 of 256), shared expert, and a per-layer DSA sparse-attention indexer. Slot-bank streaming works; the DSA indexer adds per-layer dense matmul overhead K2.5 doesn't have, and IQ1_M / IQ2_XXS `mul_mat_id` Metal kernels are not yet on the hot path. Realistic decode on M5 Max 128 GB is currently ~3.9 tok/s with `--moe-topk 4 --moe-prefetch-temporal --moe-slot-bank 64`. Shares the DeepSeek2 GPU-bank fallback path.
- The recommended build uses `-DLLAMA_FLASH_MOE_GPU_BANK=ON` (the default). In slot-bank mode, routed experts are **not** loaded into GPU memory — they stream from SSD. Use `-ngl 99` to offload dense/shared weights to GPU; the fitter clamps dense/shared placement against the routed slot-bank budget.
- For Kimi/DeepSeek2/GLM-5.1, GPU-bank placement of routed experts is enabled by default in GPU-bank builds. Set `LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1` to force the host-backed slot-bank path if you hit hangs or memory pressure.

## Purpose

Flash-MoE is not mainly about SSD reads.
It is about changing the boundary between dense execution, expert storage, expert selection, and expert consumption.

The dense path should stay inside the backend that is already good at it.
The sparse path should be reshaped around a stable bank or slot-bank plus ids-as-data.

By "banking" we mean this: instead of treating every routed expert as a fresh tensor object that has to be materialized on demand, the runtime keeps a small resident execution surface per layer, made of stable slots. A routed `expert_id` is first resolved to a resident `slot_id`; if the expert is already there, execution is just an indexed hit, and if not, the runtime loads the expert bytes and commits them into a victim slot before use. The important part is that the execution shape stays stable while expert identity changes, which is much cheaper than rebuilding a tiny K-expert bank every token.

## The Big Lesson

A stable bank with changing ids is fast.
Per-token K-expert materialization is slow.

The expensive part is not dynamic routing by itself. The expensive part is forcing the runtime to keep rebuilding selected experts in the hot path. Once that boundary is moved, a streamed slot-bank becomes the first realistic external-expert shape. Miss path and hit path become different problems. Oracle ceilings become mandatory, because they tell you whether the next bottleneck is consume, commit, or prefetch.

## Reusable Workflow

This is the workflow to reuse across backends:

1. Measure the normal resident baseline.
2. Build a resident packed-bank path.
3. Prove that changing ids is cheap on that good path.
4. Add one intentionally bad K-materialization diagnostic.
5. Build a real streamed slot-bank.
6. Measure locality before designing cache policy.
7. Build oracle all-hit replay and one-step oracle prefetch replay.
8. Decide whether the next work belongs in the hit path or the miss path.
9. Only then move more of the boundary into native consume if backend-level slot-bank work stalls.

## What Matters Most

Stable bank or slot plus ids is the first shape to try.
Per-token K-expert materialization is useful as a diagnostic, but usually the wrong product shape.
Hit path and miss path are different bottlenecks.
Early slot-bank numbers are rungs, not ceilings.
And native code only matters if the execution boundary really moves.

## Flash-MoE Additions In This Fork

- `--moe-sidecar` to attach a routed-expert sidecar manifest or bank directory
- `--moe-mode stock|resident|resident-bank|slot-bank|oracle-all-hit|oracle-prefetch`
- `--moe-slot-bank` for streamed resident slot capacity per routed MoE layer
- `--moe-topk` for an experimental reduction-only routed expert override
- `--moe-prefetch-temporal` for runtime one-step temporal prefetch on top of `slot-bank`
- `--moe-shared-only` for a shared-expert-only diagnostic path that bypasses routed experts at graph build time
- `--moe-trace-harness` for long non-interactive raw trace runs from `llama-cli`
- `--moe-trace` and `--moe-verify-sidecar` for replay and validation workflows
- sidecar extract / inspect / verify tooling under [`tools/flashmoe-sidecar/`](./tools/flashmoe-sidecar/)

## Build

```bash
git clone https://github.com/Anemll/anemll-flash-llama.cpp.git
cd anemll-flash-llama.cpp

# Metal (Apple Silicon) — recommended
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON

cmake --build build --config Release -j"$(sysctl -n hw.ncpu)" \
  --target llama-cli llama-bench llama-perplexity

# CUDA (NVIDIA)
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON

cmake --build build --config Release -j"$(nproc)" \
  --target llama-cli llama-bench

# CPU-only fallback
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)" \
  --target llama-cli llama-bench
```

To disable GPU-bank placement at compile time (also controllable at runtime via `LLAMA_FLASH_MOE_DISABLE_GPU_BANK=1`):

```bash
cmake -S . -B build \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_FLASH_MOE_GPU_BANK=OFF
```

## Choosing `--moe-slot-bank` Size

The slot-bank holds N recently-used experts per layer in memory. Larger banks mean higher hit rates but more memory.

**Qwen3.5-35B-A3B** (256 experts/layer, ~920 KB per expert slot):

| Bank | Memory | Good for |
|-----:|-------:|----------|
| 8 | 0.3 GiB | 8 GB machines — leaves room for dense weights |
| 16 | 0.6 GiB | 8-16 GB — good balance for small machines |
| 32 | 1.1 GiB | 16-32 GB |
| 64 | 2.2 GiB | 32-64 GB |
| 128 | 4.5 GiB | 64+ GB — diminishing returns above this |

Rule of thumb: bank should be 5-15% of available RAM. On machines where the GGUF fits entirely in memory, just use `--moe-mode stock` (no sidecar needed).

**Quick reference by machine:**

| Machine | RAM | Recommended | Notes |
|---------|----:|-------------|-------|
| M1/M2 8 GB | 8 GB | `--moe-slot-bank 16 -ngl 0` | Dense on CPU, experts streamed |
| M1/M2 16 GB | 16 GB | `--moe-slot-bank 32 -ngl 99` | Dense on GPU if it fits |
| M3/M4 Pro 36 GB | 36 GB | `--moe-slot-bank 64 -ngl 99` | 35B fits resident — use stock |
| M4/M5 Max 128 GB | 128 GB | `--moe-slot-bank 128 -ngl 99` | 35B: use stock. 397B: slot-bank |

## How It Works: Dense + Experts Split

A Mixture-of-Experts model has two kinds of weights:

1. **Dense weights** — attention, norms, embedding, LM head, shared experts. These are small, used on every token, and stay resident in memory. In a GGUF file these are the non-`_exps` tensors.

2. **Routed expert weights** — gate, up, and down projections for each of the N experts per layer. These are large (often 90%+ of total model size) but only K of N are activated per token. In a GGUF these are `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`.

In **stock mode**, `llama.cpp` loads everything into one memory space — dense and experts together. This works when the model fits in RAM.

In **slot-bank mode**, the runtime splits execution:

- The **GGUF file** provides dense weights (loaded normally via mmap)
- A **sidecar directory** provides routed expert weights as separate per-layer binary files, streamed from SSD on demand via `pread()`

```
GGUF file (dense)              Sidecar directory (experts)
┌─────────────────────┐        ┌─────────────────────────┐
│ attention Q/K/V/O   │        │ layer_000.bin  (3 tensors: gate + up + down)
│ norms, embedding    │        │ layer_001.bin
│ LM head             │        │ ...
│ shared experts      │        │ layer_059.bin
│ routing gates       │        │ manifest.json
└─────────────────────┘        └─────────────────────────┘
       ~5-30 GB                      ~9 GB (35B) / ~217 GB (K2.5)
    always in memory               streamed from SSD per token
```

The sidecar is created by extracting routed expert tensors from the GGUF into separate files. The `manifest.json` maps tensor names to byte offsets within each layer file, preserving the original quantization format. At runtime, only the K active experts per layer are read from disk — not all N.

This split is what makes it possible to run models whose expert weights exceed available RAM. The dense weights stay resident while the slot-bank runtime manages a small cache of recently-used experts, evicting and loading as routing decisions change.

## Quick Start: Sidecar Extraction

### Qwen3.5-35B-A3B (single GGUF, 9 GB sidecar)

```bash
# 1. Download model
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  --include "Qwen3.5-35B-A3B-UD-IQ2_M.gguf" \
  --local-dir ~/Models

# 2. Inspect routed expert tensors
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf
# → 40 layers, 120 tensors, 9.0 GB (ffn_gate_exps + ffn_up_exps + ffn_down_exps)

# 3. Extract sidecar (all routed experts)
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir ~/Models/flash/qwen35 \
  --force
# → wrote 120 tensors across 40 layers (9.0 GB)
# → manifest: ~/Models/flash/qwen35/manifest.json

# 4. Verify byte-level integrity
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --sidecar ~/Models/flash/qwen35
# → verified 120 Flash-MoE sidecar entries against 1 GGUF file(s)

# 5. Run (adjust --moe-slot-bank and -ngl to your machine, see table above)
#    128 GB: --moe-slot-bank 128 -ngl 99
#    8 GB:   --moe-slot-bank 16  -ngl 0
build/bin/llama-cli \
  -m ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/qwen35 \
  --moe-slot-bank 16 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace-harness --no-warmup \
  -ub 1 -b 64 -ngl 99 -c 256 --seed 0 --temp 0 \
  -p "What is Apple Neural Engine" -n 120
```

`-ub 1`: required for correct MoE prefill on GPU — multi-token ubatch produces incorrect output.
`-ngl 99`: offload dense weights to GPU (use when dense weights fit in VRAM).
`-ngl 0`: keep everything on CPU (8 GB machines, or when GPU memory is tight).
`--moe-slot-bank N`: see the sizing table above — 16 for 8 GB, 128 for 128 GB.

### Kimi-K2.5 (5-shard GGUF, 217 GB sidecar)

```bash
# 1. Download model (5 shards, ~250 GB total)
huggingface-cli download moonshotai/Kimi-K2.5-GGUF \
  --include "Kimi-K2.5-UD-TQ1_0-*.gguf" \
  --local-dir ~/Models/Kimi

# 2. Inspect — pass the first shard, tool auto-discovers all 5
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf
# → 60 layers, 180 tensors, 217 GB (deepseek2 arch, 256 experts/layer)

# 3. Extract sidecar (all routed experts, ~20 min)
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --out-dir ~/Models/flash/Kimi-K2.5-sidecar \
  --force
# → wrote 180 tensors across 60 layers (217 GB)

# 4. Verify
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --sidecar ~/Models/flash/Kimi-K2.5-sidecar
# → verified 180 Flash-MoE sidecar entries against 5 GGUF file(s)

# 5. Run (adjust --moe-slot-bank and -ngl to your machine, see table above)
#    128 GB: --moe-slot-bank 128 -ngl 99
#    8 GB:   --moe-slot-bank 16  -ngl 0
build/bin/llama-cli \
  -m ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace-harness --no-warmup \
  -ub 1 -b 64 -ngl 0 -c 256 --seed 0 --temp 0 \
  -p "Compare Apple Neural Engine vs Google's TPU" -n 100
```

In slot-bank mode, routed expert weights are **not** loaded into GPU memory at startup.
Dense and shared weights are offloaded to GPU via `-ngl 99` as usual.
Routed experts are streamed from SSD on demand by the slot-bank runtime —
only the K active experts per token are read, and recently-used experts
are kept in a host-memory cache (sized by `--moe-slot-bank`).

### MiniMax-M2.7 UD-IQ2_XXS (3-shard GGUF, 2.58 GiB dense + 58.3 GiB routed)

MiniMax M2.7 currently uses the package workflow in this fork: extract routed experts into `sidecar/`, export dense/shared tensors into `model-dense.gguf`, then run the pair together.

```bash
# 1. Build the Flash-MoE package
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS/MiniMax-M2.7-UD-IQ2_XXS-00001-of-00003.gguf \
  --out-dir ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  --force

# 2. Run on M5 Max with native MiniMax routing width (K=8)
MOE_TOPK=8 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 128 \
  -p "What is Apple Neural Engine? Answer in 3 sentences."

# 3. M1 Max 64 GB (MOE_TOPK=4 for lower expert I/O, large slot bank)
MOE_TOPK=4 MOE_SLOT_BANK=128 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"

# 4. Smaller-memory starting point for a 24 GB-class Apple Silicon machine
MOE_TOPK=8 MOE_SLOT_BANK=32 CTX=2048 BATCH=32 UBATCH=1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 96 \
  -p "Summarize Apple Neural Engine in 3 sentences."

# 5. Reproducible branch-to-branch comparison runs
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

Use `MOE_TOPK`, not sampler `--top-k`, to control MiniMax routed-expert width in Flash-MoE. The wrapper default `MOE_TOPK=4` is speed-oriented; use `MOE_TOPK=8` for model-default routing width. For branch-to-branch comparisons, pin `LLAMA_BIN`, `SEED`, `TEMP`, prompt, and `-n`. See [tools/flashmoe-sidecar/MINIMAX_M2.md](./tools/flashmoe-sidecar/MINIMAX_M2.md) for the detailed export/inference notes and slot-bank sizing guidance.

### Troubleshooting: Trace and Log Output

For debugging slot-bank issues, enable backend tracing, a routing trace file, and verbose logging:

```bash
env \
  LLAMA_FLASH_MOE_BACKEND_TRACE=1 \
  build/bin/llama-cli \
  --color off --simple-io \
  -m ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace /tmp/flashmoe.trace.jsonl \
  --log-file /tmp/flashmoe.log \
  --log-prefix --log-timestamps -v \
  --no-warmup \
  -ub 1 -b 64 -ngl 99 -c 4096 --seed 0 --temp 0 \
  -p "What is Apple Neural Engine?" -n 128
```

Key flags:
- `LLAMA_FLASH_MOE_BACKEND_TRACE=1` — logs backend-level slot install/eviction events
- `LLAMA_FLASH_MOE_DEEP_LOG=1` — even more verbose internal logging
- `--moe-trace /tmp/flashmoe.trace.jsonl` — writes per-token routing decisions (layer, expert ids, slot hits/misses)
- `--log-file /tmp/flashmoe.log` — redirects all log output to a file
- `--log-prefix --log-timestamps -v` — adds timestamps and verbose detail to log lines

Output files:
- `/tmp/flashmoe.trace.jsonl` — one JSON line per token with routing decisions per layer
- `/tmp/flashmoe.log` — full runtime log with timestamps

### Sidecar Directory Layout

After extraction:

```
~/Models/flash/qwen35/
  manifest.json          # tensor map: names, shapes, quant types, offsets
  layer_000.bin          # all routed expert tensors for layer 0
  layer_001.bin          # ...
  ...
  layer_039.bin          # layer 39

~/Models/flash/Kimi-K2.5-sidecar/
  manifest.json
  layer_001.bin          # Kimi starts routed MoE at layer 1
  ...
  layer_060.bin
```

### Optional: Partial Extraction

```bash
# Extract only layers 0-4 (quick test)
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir /tmp/qwen35-partial \
  --layers 0-4 --force

# Extract only gate/up experts (skip down_proj)
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir /tmp/qwen35-gateup \
  --families ffn_gate_exps,ffn_up_exps --force

# Include shared experts alongside routed
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --out-dir ~/Models/flash/Kimi-K2.5-with-shared \
  --include-shared --force
```

## Benchmarks — M5 Max 128 GB

Current Flash-MoE `slot-bank` examples use `-ub 1`. Historical benchmark tables below are kept as recorded experiments; if a benchmark heading or table reflects a different ubatch value, treat that as measurement context rather than the current recommended setting.

**Qwen3.5-35B-A3B-UD-IQ2_M** (256 experts/layer, k→4 via `--moe-topk`, 10.60 GiB, 2.63 BPW)

### Slot-Bank Streaming (sidecar, `-ngl 99 -ub 1 -b 64 -c 256`, 120 tokens)

Recommended config:

```bash
llama-cli -m Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/qwen35 \
  --moe-slot-bank 16 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace-harness --no-warmup \
  -ub 1 -b 64 -ngl 99 -c 256 --seed 0 --temp 0 \
  -p “What is Apple Neural Engine” -n 120
```

| Bank | Decode tok/s | Prompt tok/s | Hit rate | Misses/call | I/O GiB | pread ops |
|-----:|------------:|-----------:|--------:|:-----------:|--------:|----------:|
| 16 | 46.9 | 79.2 | 58.8% | 1.68 | 7.21 | 24,642 |
| 32 | 51.9 | 75.5 | 72.8% | 1.11 | 4.77 | 16,299 |
| 64 | **53.2** | 61.5 | 81.8% | 0.74 | 3.18 | 10,869 |
| **128** | **53.0** | **75.5** | **84.5%** | **0.64** | **2.72** | **9,297** |
| 256 | **55.2** | 76.0 | 84.5% | 0.63 | 2.72 | 9,291 |

bank=64-128 is the sweet spot: 53 t/s decode at 82-85% hit rate. bank=256 squeezes out 2 more t/s but doubles resident memory. Temporal prefetch: near-100% hit after warm-up.

### Top-k Reduction (bank=128, `-ngl 99 -ub 1`)

| topk | Decode tok/s | Prompt tok/s | Hit % | I/O GiB | pread ops |
|-----:|-----------:|-----------:|------:|--------:|----------:|
| 2 | 52.0 | 59.5 | 77.9% | 1.94 | 6,630 |
| **4** | **52.9** | **71.5** | **84.5%** | **2.72** | **9,297** |
| 8 (default) | 42.8 | 31.2 | 86.3% | 4.79 | 16,389 |

`--moe-topk 4` is the sweet spot — 24% faster than default k=8, with better prompt throughput. k=2 saves I/O but slightly lower hit rate.

### Ubatch Sweep (bank=128, `-ngl 99`)

| ub | Decode tok/s | Prompt tok/s | Notes |
|---:|------------:|------------:|-------|
| 1 | 55.0 | 45.2 | Best decode, slow prompt |
| **2** | **56.7** | **61.2** | **Best decode** |
| **4** | **54.8** | **74.3** | **Best overall balance** |
| 8 | 54.1 | 75.3 | |
| 16 | 52.4 | 75.6 | |
| 32 | 50.8 | 73.7 | |

`-ub 1` is required for correct MoE prefill on GPU. Multi-token ubatch (`-ub 2` and above) may produce incorrect output with MoE models due to a prefill batching issue.

### Runtime Breakdown (bank=128, `-ngl 99 -ub 1 --moe-topk 4`)

```
topk resolve:    0.17 ms   (routing + softmax + top-k selection)
slot resolve:    0.96 ms   (LRU lookup + victim selection)
pread install: 265.55 ms   (9,297 pread ops, 2.72 GiB total)
  source I/O:  149.01 ms   (pread from sidecar layer files)
  GPU upload:  113.73 ms   (copy to Metal buffer)
  slot-write:    0.11 ms
  trace:         0.09 ms
```

The GPU upload cost (114 ms) is new vs the old CPU-only path — but it's worth it because GPU compute on the dense path is 3-4x faster.

### Baseline vs Slot-Bank

| Mode | Decode tok/s | Prefill tok/s | Notes |
|------|------------:|-------------:|-------|
| `stock` (all resident, `-ngl 99`) | **109.0** | 3,070 | Ceiling — all experts in GPU |
| **`slot-bank 128`** (sidecar, `-ngl 99`) | **53.0** | 75.5 | GPU-bank + pread streaming |

Slot-bank reaches 49% of stock. The gap is pread I/O + GPU upload for the 16% of experts that miss the bank each token.

Slot-bank decode is 49% of stock — the gap is pread I/O + GPU upload for the 16% of experts that miss the bank each token.

### Prompt Processing Scaling (stock, `-ngl 99`)

| Batch | tok/s |
|------:|------:|
| 1 | 97 |
| 128 | 1,955 |
| 256 | 2,567 |
| 512 | 2,997 |
| 1024 | 3,021 |
| 2048 | 3,000 |

### Decode Scaling (stock, `-ngl 99`, llama-bench)

| Tokens | tok/s |
|-------:|------:|
| 32 | 108.3 |
| 64 | 108.9 |
| 128 | 109.1 |
| 256 | 109.3 |
| 512 | 105.0 |

### Perplexity — WikiText-2

| Quant | PPL | BPW | Size |
|-------|----:|----:|-----:|
| IQ2_M | ~9.97 | 2.63 | 10.60 GiB |

Context 4352, stride 512, 4 chunks. IQ2_M is a streaming/speed test quant — quality targets require Q4_K_M or higher.

### Generation Quality

Trace-harness mode produces coherent, detailed text (tested: RISC vs CISC architecture explanation, Apple Neural Engine description). `llama-simple` produces degenerate output due to sampler chain differences.

### Sidecar Tooling

| Tool | Test | Result |
|------|------|--------|
| `flashmoe_sidecar.py inspect` | 40 layers, 120 expert tensors | PASS |
| `flashmoe_sidecar.py extract` | 40 layers → 120 tensors (9.0 GB) | PASS |
| `flashmoe_sidecar.py verify` | metadata + byte-level check | PASS (9/9) |

## Benchmarks — Kimi-K2.5 (M5 Max 128 GB)

**Kimi-K2.5-UD-TQ1_0** (256 experts/layer, 60 layers, 217 GB sidecar — real SSD streaming)

### Slot-Bank Streaming (current Kimi safe fallback, `-ngl 0 -ub 1 -b 64 -c 256`, 100 tokens)

```bash
llama-cli -m Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace-harness --no-warmup \
  -ub 1 -b 64 -ngl 0 -c 256 --seed 0 --temp 0 \
  -p "What is Apple Neural Engine" -n 100
```

| Bank | Decode tok/s | Hit rate | Misses/call | I/O GiB | pread ops |
|-----:|------------:|--------:|:-----------:|--------:|----------:|
| 16 | 3.2 | 55.6% | 1.82 | 107.0 | 33,435 |
| 32 | 3.2 | 64.1% | 1.47 | 86.2 | 27,012 |
| **64** | **3.3** | **73.7%** | **1.08** | **62.8** | **19,755** |
| 128 | 3.2 | 76.4% | 0.97 | 56.2 | 17,781 |
| 256 | 3.2 | 76.4% | 0.97 | 56.2 | 17,781 |

bank=64 is the best balance for Kimi K2.5. Hit rate saturates at bank=128 (76.4%).

Temporal prefetch: 100% hit (10 cold misses / 6,120 calls). Prefetch I/O: 0.11 GiB — near-zero overhead.

### Top-k Reduction (bank=64, prefetch-temporal)

| topk | Decode tok/s | Hit % | I/O GiB | pread ops |
|-----:|-----------:|------:|--------:|----------:|
| 2 | **4.7** | 67.6% | 38.3 | 12,192 |
| **4** | **3.3** | **73.7%** | **62.8** | **19,755** |
| 8 (default) | 2.2 | 67.2% | 157.9 | 49,266 |

`--moe-topk 4` halves I/O vs default k=8 while maintaining generation quality. `--moe-topk 2` is 43% faster but may degrade output for complex tasks.

### Generation Quality

Coherent, factually accurate output at k=4:

> Apple Neural Engine (ANE) is a specialized hardware component that is designed to accelerate machine learning tasks on Apple devices. It is integrated into Apple's A-series and M-series processors, which power iPhones, iPads, and Macs. The ANE is specifically designed to handle machine learning tasks such as image recognition, natural language processing, and augmented reality.

### Runtime Breakdown (bank=64, topk=4)

```
Model:    Kimi-K2.5, 60 layers, 256 experts/layer, sidecar 217 GB
Experts:  3.28 MB each (TQ1_0)
Per-token: ~k×60 = 240 expert loads on miss, ~3.3 MB × 240 = ~790 MB/tok worst case

topk resolve:      1.89 ms
slot resolve:      2.40 ms
pread install: 13,852 ms total (19,755 ops, 62.8 GiB)
  per pread:       0.70 ms avg (3.28 MB per read, ~4.7 GB/s effective)
prefetch:         22.1 ms total (100% hit after warm-up)
```

SSD throughput: ~4.7 GB/s effective (Apple Fabric, 3.28 MB sequential pread). Bottleneck is pure I/O — the 217 GB sidecar exceeds the 128 GB page cache, so most experts are cold reads from NVMe.

Kimi currently uses `-ngl 0` as the safe fallback. The `-ngl 99` dense GPU path is faster (4.5 t/s) but produces degraded output on some runs due to a Metal compute issue with the DeepSeek2 architecture. Use `-ngl 99` for speed testing, `-ngl 0` for quality.

## Testing Qwen3.5-397B-A17B (M5 Max 128 GB)

The 397B model uses the same slot-bank workflow as the 35B — just with larger sidecars and more expert layers (60 vs 40).

### 1. Download

```bash
# Dense GGUF (IQ2_M quantization)
huggingface-cli download unsloth/Qwen3.5-397B-A17B-GGUF \
  --include "Qwen3.5-397B-A17B-UD-IQ2_M-*.gguf" \
  --local-dir ~/Models/Qwen3.5-397B

# Or use a different quantization — check unsloth for available options
```

### 2. Extract sidecar

```bash
python3 tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Qwen3.5-397B/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00005.gguf \
  --out-dir ~/Models/flash/qwen397b \
  --force

python3 tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model ~/Models/Qwen3.5-397B/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00005.gguf \
  --sidecar ~/Models/flash/qwen397b
```

### 3. Run

```bash
build/bin/llama-cli \
  -m ~/Models/Qwen3.5-397B/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00005.gguf \
  --moe-mode slot-bank --moe-sidecar ~/Models/flash/qwen397b \
  --moe-slot-bank 128 --moe-topk 4 --moe-prefetch-temporal \
  --moe-trace-harness --no-warmup \
  -ub 1 -b 64 -ngl 99 -c 256 --seed 0 --temp 0 \
  -p "What is Apple Neural Engine?" -n 200
```

Key differences from 35B:
- `--moe-slot-bank 128` — larger bank for 512 experts/layer (vs 256 for 35B)
- `-ngl 99` — dense weights (~30 GB) fit in GPU; routed experts (~200+ GB) stream from SSD
- `-ub 1` — required for correct MoE prefill on GPU
- Sidecar is multi-shard — pass the first shard, the tool auto-discovers the rest

## Upstream Base

This is still `llama.cpp`.
The goal is a fork you can rebase from upstream while keeping the Flash-MoE patch surface small and explicit.

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, such as [ModelScope](https://modelscope.cn/), by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. For example, you may opt to downloading model checkpoints from ModelScope or other model sharing communities by setting the environment variable, e.g. `MODEL_ENDPOINT=https://www.modelscope.cn/`.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Check the issue tracker and project notes before sending larger changes.
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
