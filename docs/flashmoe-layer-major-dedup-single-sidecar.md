# Flash-MoE Layer-Major Dedup (SSD-streamed)

This note describes the current `--moe-prefill-layer-major` path in this fork.

It focuses on the SSD-streamed prompt-prefill MoE path used by routed MoE
layers (MiniMax-M2, GLM-5.1, Kimi K2.5, Qwen 3.5, Gemma4, DeepSeek V4) when
the full routed expert weights do not fit in memory and have to be loaded on
demand from the Flash-MoE sidecar instead of staying resident:

- `src/llama-context.cpp`
- `common/arg.cpp`

## Index

- [Short Description](#short-description)
- [Confirmed Models](#confirmed-models)
- [Example Results](#example-results)
- [Why This Path Exists](#why-this-path-exists)
- [High-Level Shape](#high-level-shape)
- [Dedup Diagram](#dedup-diagram)
- [Core Planner](#core-planner)
- [Read Path](#read-path)
- [What `--moe-prefill-banks` Means Now](#what---moe-prefill-banks-means-now)
- [How Micro-Batch Reuse Actually Works](#how-micro-batch-reuse-actually-works)
- [Micro-Batch Reuse Diagram](#micro-batch-reuse-diagram)
- [Reuse Timeline](#reuse-timeline)
- [Compute Step](#compute-step)
- [Reuse Model](#reuse-model)
- [Perf Counters To Watch](#perf-counters-to-watch)
- [Tuning Rules](#tuning-rules)
- [Example Benchmark Command](#example-benchmark-command)
- [Summary](#summary)
- [Decode Read Locality (`--moe-sort-decode-expert-ids`)](#decode-read-locality---moe-sort-decode-expert-ids)
- [GLM-5.1 Default Test](#glm-51-default-test)
- [Additional Model Smoke Tests](#additional-model-smoke-tests) (Kimi K2.5, Qwen 3.5, Gemma4)
- [DeepSeek V4 and DeepSeek V4 Flash](#deepseek-v4-native-hf-export-and-inference)
- [DeepSeek V4 / Flash Variants](#supported-deepseek-v4-variants)
- [DeepSeek V4 / Flash Conversion](#convert-original-hf-weights)
- [DeepSeek V4 / Flash Quantization Preservation](#quantization-preservation)
- [DeepSeek V4 / Flash Inference Smoke Test](#simple-inference-smoke-test)
- [DeepSeek V4 / Flash Decode Probe](#decode-performance-probe)
- [DeepSeek V4 / Flash Top-K Prompt Sweep](#layer-major-prefill-and-top-k-sweep)
- [DeepSeek V4 / Flash Server Mode](#deepseek-v4-server-mode)
- [Server Mode](#server-mode)
- [Largest Verified Safe Q8 Server Context](#largest-verified-safe-q8-server-context)
- [Testing the Server with MiniMax-M2.7 and DroidAI (M5 Max 128)](#testing-the-server-with-minimax-m27-and-droidai-m5-max-128)
- [Testing Flash-MoE with `@badlogicgames` pi-agent (badlogic)](#testing-flash-moe-with-badlogicgames-pi-agent-badlogic)

## Short Description

Layer-major dedup runs prefill one layer at a time. For each layer, it groups
repeated expert selections across the current prefill chunk, streams each
needed expert from the sidecar once, and reuses it for all tokens that need it
instead of reloading it repeatedly.

The bigger the prefill chunk, the more expert reuse we usually get, so SSD
traffic and prefill cost both drop. That is the main reason this path exists
for large routed MoE models that cannot keep all expert weights resident on the
target system.

## Confirmed Models

Tested with `--moe-prefill-layer-major` on M5 Max:

- MiniMax-M2.7 — `minimax-m2`
- Gemma4 26B-A4B — `gemma4`
- GLM-5.1 — `glm-dsa`
- Qwen 3.5 35B-A3B — `qwen3moe`
- Kimi K2.5 — `deepseek2`
- DeepSeek V4 — `deepseek4` through the native HF-to-SSD exporter

## Example Results

Hardware: `M5 Max`, runtime: `Flash-MoE`, storage: `SSD`.

### 4K prefill across models


| Model            | Quant        | Prefill tokens | Prompt      | Decode     |
| ---------------- | ------------ | -------------- | ----------- | ---------- |
| Gemma4 26B-A4B   | `UD-Q3_K_XL` | `5009`         | `718.3 t/s` | `21.4 t/s` |
| Qwen 3.5 35B-A3B | `IQ2`        | `4661`         | `535.5 t/s` | `52.2 t/s` |
| MiniMax-M2.7     | `UD-Q4_K_XL` | `4K`           | `227.3 t/s` | `16.6 t/s` |
| Kimi K2.5        | `UD-TQ1`     | `4053`         | `109.4 t/s` | `4.4 t/s`  |
| GLM-5.1          | `IQ1`        | `4070`         | `94.5 t/s`  | `4.4 t/s`  |


### 1K prefill (selected)


| Model          | Quant        | Prefill tokens | Prompt      | Decode     |
| -------------- | ------------ | -------------- | ----------- | ---------- |
| Gemma4 26B-A4B | `UD-Q3_K_XL` | `1260`         | `630.6 t/s` | `36.8 t/s` |
| MiniMax-M2.7   | `UD-Q4_K_XL` | `1K`           | `78.1 t/s`  | `16.7 t/s` |
| Kimi K2.5      | `UD-TQ1`     | `1008`         | `53.0 t/s`  | `6.3 t/s`  |


### MiniMax-M2.7 prefill sweep (`UD-Q4_K_XL`)


| Prefill tokens | Prompt      | Decode     |
| -------------- | ----------- | ---------- |
| `1K`           | `78.1 t/s`  | `16.7 t/s` |
| `4K`           | `227.3 t/s` | `16.6 t/s` |
| `16K`          | `240.0 t/s` | `15.6 t/s` |
| `22K`          | `187.1 t/s` | `13.3 t/s` |


## Why This Path Exists

The regular `-b/-ub` path in `llama.cpp` is ubatch-major:

1. take one prompt chunk
2. split it into ubatches
3. run each ubatch through the full layer stack

That is a poor shape for routed MoE prefill, because repeated experts inside the same layer are harder to reuse.

`--moe-prefill-layer-major` changes the control surface for prompt prefill:

- `--moe-prefill-batch` chooses the prompt slab size seen by the dedicated prefill runtime
- `--moe-prefill-micro-batch` sets the maximum routed-ref chunk size processed for one expert per `ggml_backend_graph_compute()` call
- `--moe-prefill-banks` sets the number of unique experts loaded into one host-side read-ahead batch
- `--moe-prefill-io-split` chooses how many page-aligned reads are used per expert-family payload

The goal is simple: within one layer and one prompt slab, dedup the routed experts, load each unique expert once, and reuse it for every token/ref that needs it.

## High-Level Shape

For one prompt slab of `N = min(remaining_tokens, --moe-prefill-batch)`:

1. Run the dense/router part of the layer and capture routed top-k expert ids plus weights.
2. Build a per-layer prefill plan over the whole slab.
3. Dedup expert ids across all `N * topk` routed refs.
4. Sort unique experts by sidecar order for better read locality.
5. Load unique experts from the sidecar in host-side read batches of `--moe-prefill-banks`.
6. While computing the current staged batch, asynchronously read the next staged batch.
7. For each unique expert, gather only the token rows that reference it.
8. Run `gate`, `up`, `swiglu`, `down`.
9. Scatter the weighted result back into the layer output rows.
10. Move to the next layer.

The important reuse is:

- not across all layers
- not across the whole prompt permanently
- but within one layer and one prompt slab

That is where the large `prefill dedup saved` and `prefill reuse factor` numbers come from.

## Dedup Diagram

```text
Prompt slab
N tokens
    |
    v
Dense/router pass
capture top-k experts + weights
    |
    v
Flatten routed refs
N * topk entries
    |
    v
Count refs per expert
    |
    v
Build unique_experts[]
    |
    v
Sort unique experts by sidecar offset
    |
    v
Pack refs per expert
- expert_ref_offsets
- expert_ref_tokens
- expert_ref_weights
    |
    v
Stage unique experts in batches
batch size = --moe-prefill-banks
    |
    v
For each unique expert
gather rows -> compute -> scatter
```

The main point of the planner is that all repeated references to the same expert are collapsed into one `unique_experts[]` entry plus one contiguous ref range. That is the dedup step.

## Core Planner

The planner lives in `build_prefill_selected_plan()` in `src/llama-context.cpp`.

Inputs:

- routed expert ids for every `(token, k)` pair
- routed weights for every `(token, k)` pair
- `topk`
- `n_tokens`

Outputs:

- `unique_experts`
- `expert_ref_offsets`
- `expert_ref_tokens`
- `expert_ref_weights`
- `max_refs_per_expert`

The plan does three useful things:

1. Counts how many refs each expert receives.
2. Produces one deduped expert list for the whole layer slab.
3. Packs all token refs for each expert into a contiguous range.

It also sorts `unique_experts` by sidecar offset, so reads walk the sidecar more sequentially instead of bouncing around by expert id.

## Read Path

The dedicated executor is `compute_prefill_moe_tensor()` in `src/llama-context.cpp`.

Each expert has three routed families:

- `gate`
- `up`
- `down`

So one unique expert miss loads three payloads from the sidecar.

If `--moe-prefill-io-split N` is greater than `1`, each family payload is split into `N` page-aligned `pread` tasks. That improves queue depth on the SSD.

## What `--moe-prefill-banks` Means Now

In the current implementation, `--moe-prefill-banks` is not “full scratch banks per layer.”

It is the host-side prefill read-ahead batch depth.

Concretely:

- the executor takes up to `--moe-prefill-banks` unique experts
- reads their raw bytes into a current host-side batch
- launches an async read for the next batch while computing the current one

In code, `prefill_bank_depth` is used as:

- the size of one staged batch in `load_prefill_batch()`
- the batch size for async read-ahead

It is **not** used as:

- the number of experts simultaneously installed into `gate_w`, `up_w`, and `down_w`
- the number of experts simultaneously active in the temp ggml graph

Those temp tensors are still loaded one expert at a time inside the compute loop.

In practice:

- `--moe-prefill-banks 1` means no read-ahead batching
- `--moe-prefill-banks 4` means stage 4 experts, then read the next 4 while computing the current 4
- larger values increase queue depth, not permanent expert residency

## How Micro-Batch Reuse Actually Works

This is the part that is easy to miss in the source:

- `--moe-prefill-banks` batches **host-side expert loads**
- `--moe-prefill-micro-batch` limits **ref_count chunks for one already-loaded expert**

So the reuse boundary is:

- load one unique expert once
- keep its `gate/up/down` weights resident in the temporary tensors
- process that expert's contiguous ref range in chunks of `--moe-prefill-micro-batch`
- only after all ref-chunks for that expert are done do we move to the next expert

That means the same loaded expert is reused across multiple graph invocations when:

- one expert has many refs in the current layer slab
- `max_refs_per_expert > --moe-prefill-micro-batch`

Example:

- one expert receives `407` routed refs in the current layer slab
- `--moe-prefill-micro-batch 32`
- that expert is loaded once
- then it is reused across `ceil(407 / 32) = 13` compute chunks

What does **not** happen today:

- the executor does not reload that expert for every micro-batch
- the executor does not keep using that expert after it finishes its full ref range
- the executor does not currently persist those loaded expert weights across future prompt slabs

So the two reuse mechanisms are different:

- dedup reuse: one expert load serves many routed refs
- ref-chunk reuse: one already-loaded expert serves many ref-chunks

## Micro-Batch Reuse Diagram

```text
Unique expert E
    |
    v
Copy gate/up/down into gate_w/up_w/down_w once
    |
    v
Ref range for E
example: 407 refs
    |
    v
Chunk refs by --moe-prefill-micro-batch
example: 32 refs per compute chunk

chunk 0   gather rows -> ggml_backend_graph_compute -> scatter
chunk 1   gather rows -> ggml_backend_graph_compute -> scatter
chunk 2   gather rows -> ggml_backend_graph_compute -> scatter
...
chunk 12  gather rows -> ggml_backend_graph_compute -> scatter

Same staged expert weights are reused for every chunk above.
Only after the full ref range is done do we advance to the next expert.
```

The important detail is that `gate_w`, `up_w`, and `down_w` are loaded once for expert `E`, then reused for every ref-chunk of `E`. The code does not reload them between those chunks.

## Reuse Timeline

```text
Prefill read-ahead level:

batch 0 experts staged     [E0 E1 E2 E3]
batch 1 experts staged     [E4 E5 E6 E7]    read asynchronously while batch 0 computes

These are host-side loaded byte buffers, not 4 experts simultaneously copied into ggml weight tensors.

Current expert compute level:

load E0 once
  -> chunk 0 of E0 refs
  -> chunk 1 of E0 refs
  -> chunk 2 of E0 refs
  -> ...
  -> final chunk of E0 refs

then move to E1
  -> chunk 0 of E1 refs
  -> chunk 1 of E1 refs
  -> ...

then move to E2, E3, ...
```

Another way to say it:

- `prefill-banks` controls how many future experts are loaded into the host-side queue ahead
- `prefill-micro-batch` controls how many routed refs of the **current** expert are processed per graph call

The executor overlaps these two levels:

- async read-ahead for the next staged expert batch
- repeated compute reuse for the current expert across its ref-chunks

## Compute Step

For each unique expert:

1. Look up its contiguous ref range from the plan.
2. Gather only the input token rows that reference that expert.
3. Process that expert's ref range in chunks of `--moe-prefill-micro-batch`.
4. Run the temporary MoE graph:
  - `gate = ggml_mul_mat(gate_w, cur_in)`
  - `up = ggml_mul_mat(up_w, cur_in)`
  - `act = ggml_swiglu_split(gate, up)`
  - `out = ggml_mul_mat(down_w, act)`
5. Scatter weighted expert output back into the destination rows.

This is why `--moe-prefill-micro-batch` is a per-expert ref-chunk compute knob, not the dedup window.

The dedup window is the full `--moe-prefill-batch`.

## Reuse Model

In the current implementation, the real win is dedup reuse, not persistent bank replay.

That means:

- if one expert is referenced 50 times in the same layer slab, it is loaded once
- its refs are processed in one grouped pass
- the measured reuse factor is roughly `routed_refs / unique_experts`

It does **not** currently mean:

- persistent reuse across all layers
- persistent reuse across all prompt slabs

So it is normal for `Bank replay` to stay near zero in current prefill profiles. The path is winning by grouping refs per unique expert inside a layer slab, not by holding a long-lived expert cache across the whole prompt.

## Perf Counters To Watch

The most useful counters for tuning are:

- `prefill dedup saved`
  - how many routed refs were eliminated by deduping unique experts
- `prefill reuse factor`
  - average refs served per unique expert load
- `prefill read bandwidth`
  - effective sidecar read throughput
- `prefill source overlap`
  - summed task time divided by wall time
- `prefill pacing`
  - whether the current run is data-bound or compute-bound

Interpretation:

- high `dedup saved` and high `reuse factor` mean the layer-major grouping is working
- if `bottleneck=data`, the SSD/read path is limiting prefill
- if `bottleneck=compute`, increasing I/O parallelism will help less than changing compute chunking

## Tuning Rules

The practical knobs are:

- `--moe-prefill-batch`
  - larger gives a wider dedup window
- `--moe-prefill-micro-batch`
  - smaller reduces per-expert working-set pressure
- `--moe-prefill-banks`
  - increases host-side expert read-ahead batch depth
- `--moe-prefill-io-split`
  - increases per-expert read parallelism on one SSD

A good starting point is:

```bash
--moe-prefill-layer-major \
--moe-prefill-batch 2048 \
--moe-prefill-micro-batch auto \
--moe-prefill-banks 4 \
--moe-prefill-io-split 8
```

If the run is still data-bound:

- try a larger `--moe-prefill-banks`
- try a larger `--moe-prefill-io-split`

If the run becomes compute-bound:

- keep the I/O knobs
- adjust `--moe-prefill-micro-batch`

## Example Benchmark Command

This is one concrete command for the `16K` coding prompt:

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
MOE_TOPK=4 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=32768 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_16k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

For the result table above, the decode length was `1024` tokens.

## Summary

The current layer-major path is:

- layer-major at the prompt-prefill MoE stage
- deduped across the full prefill slab
- read-ahead batched by `--moe-prefill-banks`
- per-expert compute chunked by `--moe-prefill-micro-batch`

The main speedup comes from:

- loading each unique expert once per layer slab
- grouping all token refs for that expert together
- overlapping staged reads for the next expert batch with compute on the current batch

That is the simplified algorithm behind the benchmark numbers.

## Decode Read Locality (`--moe-sort-decode-expert-ids`)

Prefill already reads experts in near-sequential sidecar order — the
planner sorts `unique_experts[]` by sidecar offset before staging reads
(see [Core Planner](#core-planner)). Decode does not get that for free:
at `n_tokens == 1` there is no dedup, no planner, and the routed top-K
selected by the router arrives in arbitrary id order.

`--moe-sort-decode-expert-ids` (default disabled) fixes the decode side by
reordering the routed top-K expert ids in ascending order before the routed
MLP runs. Because expert payloads are laid out on disk by ascending expert
id per layer, iterating the top-K in sorted id order turns the per-token
sidecar `pread()` fan-out into a near-sequential SSD access pattern, which
improves read locality and throughput.

Implementation:

- gated by `n_tokens == 1 && n_expert_used > 1` and only on layers backed
  by the Flash-MoE slot-bank (sidecar) path
  (`src/llama-graph.cpp`, around the `ffn_moe_topk_id_sorted` node)
- inserted as a `ggml_map_custom1` sort on the selected top-K id tensor
  just before the weight gather, so both the gather and the routed matmuls
  iterate experts in the reordered order
- disabled by default; opt in per-run with `--moe-sort-decode-expert-ids`,
  the env `LLAMA_ARG_MOE_SORT_DECODE_EXPERT_IDS=1`, or negate explicitly
  with `--no-moe-sort-decode-expert-ids`
- prefill is unaffected — the layer-major prefill path already walks the
  sidecar in offset order

## GLM-5.1 Default Test

This is a simple GLM-5.1 prefill smoke test using the current default
`--moe-prefill-batch = 8192`.

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
SIDECAR_PATH=~/Models/GLM/GLM-5.1-sidecar \
MOE_TOPK=4 \
MOE_SLOT_BANK=96 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=6000 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/GLM/GLM-5.1-IQ1-Dense \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

Expected startup line:

```text
prefill-batch    = 8192 (default)
```

This is a good default M5 Max starting point for GLM-5.1:

- `--moe-topk 4`
- `--moe-slot-bank 96`
- layer-major prefill enabled
- explicit micro-batch / I/O-split / prefill-banks tuning

## Additional Model Smoke Tests

These are the tested layer-major prefill smoke tests for the other
architectures currently enabled on this branch.

### Kimi K2.5

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
SIDECAR_PATH=~/Models/Kimi/Kimi-K2.5-UD-TQ1/sidecar \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=8192 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/Kimi/Kimi-K2.5-UD-TQ1/dense \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

### Qwen 3.5

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
SIDECAR_PATH=~/Models/Q35A3_IQ2_GUFF/sidecar \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=8192 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/Q35A3_IQ2_GUFF/dense \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

### Gemma4

Build the Flash package once (dense GGUF + routed sidecar under one directory):

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/gemma4/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf \
  --out-dir ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense/sidecar \
  --force

python3 ./tools/flashmoe-sidecar/export_dense_gguf.py \
  --model ~/Models/gemma4/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf \
  --sidecar ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense/sidecar \
  --out-dir ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense \
  --force
```

Resulting layout (Q3_K_XL): `model-dense.gguf` 2.4 GB, `sidecar/` 9.6 GB,
`flashmoe-package.json`.

#### 1K prefill

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=4096 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_1k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

#### 4K prefill

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=8192 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

## DeepSeek V4 Native HF Export and Inference

DeepSeek V4 can be exported directly from the original Hugging Face
checkpoint into the Flash-MoE SSD package layout:

- dense GGUF: `dense/model-dense.gguf`
- routed expert sidecar: `sidecar/manifest.json` plus layer bank files
- local DS4 prompt encoder: `encoding/encoding_dsv4.py`

This path does not require a native full-model GGUF input. It reads the
original HF `safetensors` package and writes the dense model plus routed expert
sidecar that the slot-bank runtime expects. Quantization is preserved as in the
checkpoint: native **FP8** for the dense stack and native **FP4** for routed MoE
experts, repacked into GGUF / sidecar formats **without requantization**.

### Supported DeepSeek V4 Variants

The native HF exporter supports both full DeepSeek V4 and DeepSeek V4 Flash.
The exporter reads geometry from the HF tensors, so it can tolerate stale local
`config.json` values as long as the tensor set is complete and internally
consistent.

| Variant | Expected tensor geometry | Native top-k | Example output |
| --- | --- | --- | --- |
| DeepSeek V4 | 61 layers, 384 routed experts, hidden size 7168 | 6 | `$HOME/Models/DeepSeek-V4-SSD` |
| DeepSeek V4 Flash | 43 layers, 256 routed experts, hidden size 4096 | 6 | `$HOME/Models/DeepSeek-V4-Flash-SSD` |

If a local DeepSeek V4 Flash checkout has an old full-model-looking config
(`hidden_size=7168`, `num_hidden_layers=61`, `n_routed_experts=384`), refresh
`config.json` from Hugging Face or rely on the exporter warning and
tensor-derived overrides. The official Flash config is `4096 / 43 / 256`.

### Convert Original HF Weights

Example source and output paths:

- source HF checkpoint: `$HOME/Models/DeepSeek-V4`
- output Flash-MoE SSD package: `$HOME/Models/DeepSeek-V4-SSD`

```bash
python3 ./tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py \
  --hf "$HOME/Models/DeepSeek-V4" \
  --out "$HOME/Models/DeepSeek-V4-SSD" \
  --preserve-quant \
  --experts fp4 \
  --dense fp8 \
  --allow-low-space
```

Use `--force` only when intentionally replacing an existing dense/sidecar
export under the same output directory:

```bash
python3 ./tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py \
  --hf "$HOME/Models/DeepSeek-V4" \
  --out "$HOME/Models/DeepSeek-V4-SSD" \
  --preserve-quant \
  --experts fp4 \
  --dense fp8 \
  --allow-low-space \
  --force
```

For DeepSeek V4 Flash from original HF files:

```bash
python3 ./tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py \
  --hf "$HOME/Models/DeepSeek-V4-Flash" \
  --out "$HOME/Models/DeepSeek-V4-Flash-SSD" \
  --preserve-quant \
  --experts fp4 \
  --dense fp8 \
  --allow-low-space
```

If a previous Flash export used stale geometry, re-export dense and refresh the
manifest while reusing the large existing `sidecar/layer_*.bin` files:

```bash
python3 ./tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py \
  --hf "$HOME/Models/DeepSeek-V4-Flash" \
  --out "$HOME/Models/DeepSeek-V4-Flash-SSD" \
  --preserve-quant \
  --experts fp4 \
  --dense fp8 \
  --skip-sidecar \
  --allow-low-space \
  --force
```

After export, the package should be self-contained for prompt formatting:

```bash
ls "$HOME/Models/DeepSeek-V4-SSD/encoding/encoding_dsv4.py"
ls "$HOME/Models/DeepSeek-V4-Flash-SSD/encoding/encoding_dsv4.py"
```

### Quantization Preservation

The exporter is intentionally a native-quantization preserving path, not a
requantizer:

- Dense FP8 tensors from HF use `F8_E4M3` weights plus UE8M0-style scale
tensors. The exporter packs these into GGUF `F8_E4M3_B128`, preserving the
original FP8 payload and per-128-element scale structure.
- Routed experts are stored in HF as native FP4: `I8` packed weights plus
`F8_E8M0` scales. The exporter packs them into sidecar `MXFP4` expert blocks
with one scale byte plus 16 data bytes per 32 values.
- Existing HF `BF16` and `F32` dense tensors remain `BF16` and `F32`.
- `attn.wo_a.weight` is the current exception: HF stores it as FP8, but the
DeepSeek V4 reference conversion materializes this low-rank output projection
as `BF16`, and this exporter follows that layout for runtime compatibility.
- MTP tensors are not exported into this dense+sidecar runtime package.

The resulting package metadata records this as:

- dense quantization: `native_fp8_e4m3_ue8m0`
- expert quantization: `native_fp4_mxfp4`

### Simple Inference Smoke Test

Build the DS4 chat prompt from the exported package, not from the original HF
checkout. Avoid naming the shell variable `PROMPT` in zsh because `PROMPT` is
also the interactive prompt/PS1 variable.

```bash
DSV4_PACKAGE="$HOME/Models/DeepSeek-V4-SSD"
# For DeepSeek V4 Flash:
# DSV4_PACKAGE="$HOME/Models/DeepSeek-V4-Flash-SSD"

DSV4_INPUT=$(PYTHONPATH="$DSV4_PACKAGE/encoding" python3 - <<'PY'
from encoding_dsv4 import encode_messages

print(encode_messages(
    [{"role": "user", "content": "What is Apple Neural Engine?"}],
    thinking_mode="chat",
), end="")
PY
)
```

Small correctness smoke test:

```bash
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES=1 \
./build/bin/llama-cli \
  -m "$DSV4_PACKAGE/dense/model-dense.gguf" \
  --moe-mode slot-bank \
  --moe-sidecar "$DSV4_PACKAGE/sidecar" \
  --moe-slot-bank 16 \
  --moe-topk 6 \
  -ngl 999 \
  --moe-cache-io-split 8 \
  -c 1024 -b 32 -ub 1 \
  --no-warmup \
  --temp 0.0 \
  --no-display-prompt \
  --simple-io \
  --moe-trace-harness \
  -p "$DSV4_INPUT" \
  -n 128 --perf -st \
  --moe-prefetch-temporal
```

This checks output quality but is not a speed benchmark. `--moe-slot-bank 16`
is too small for full DeepSeek V4 and will thrash the routed expert cache.

### Decode Performance Probe

For a decode-side performance probe on M5 Max, start with a larger slot bank,
parallel sidecar reads, batched install reads, and top-k reduced to isolate the
runtime bottleneck:

```bash
DSV4_PACKAGE="$HOME/Models/DeepSeek-V4-SSD"
# For DeepSeek V4 Flash:
# DSV4_PACKAGE="$HOME/Models/DeepSeek-V4-Flash-SSD"

DSV4_INPUT=$(PYTHONPATH="$DSV4_PACKAGE/encoding" python3 - <<'PY'
from encoding_dsv4 import encode_messages

print(encode_messages(
    [{"role": "user", "content": "What is Apple Neural Engine?"}],
    thinking_mode="chat",
), end="")
PY
)

LLAMA_FLASH_MOE_EXPERIMENTAL_PARALLEL_SLOT_READS=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_BATCHED_INSTALL_READS=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES=1 \
./build/bin/llama-cli \
  -m "$DSV4_PACKAGE/dense/model-dense.gguf" \
  --moe-mode slot-bank \
  --moe-sidecar "$DSV4_PACKAGE/sidecar" \
  --moe-slot-bank 36 \
  --moe-topk 2 \
  -ngl 999 \
  --moe-cache-io-split 8 \
  -c 1024 -b 32 -ub 1 \
  --no-warmup \
  --temp 0.0 \
  --no-display-prompt \
  --simple-io \
  --moe-trace-harness \
  -p "$DSV4_INPUT" \
  -n 500 --perf -st \
  --moe-prefetch-temporal \
  --moe-predict-prev-token
```

What to watch in `--perf` output:

- `gpu-bank = on (compiled=on, env-disable=off)`
- `preads=on batchrd=on`
- `slot-bank cached expert hit rate`
- early-layer summaries such as `layer=0`, `layer=1`, `layer=2`
- `Expert I/O source` versus dense time

If `MTL0 free` is only a few GiB, do not increase `--moe-slot-bank` further.
If `layer=0/1/2` hit rates remain low, the run is limited by the DeepSeek V4
early-layer expert working set versus available slot-bank memory rather than
by broad CPU graph fallback.

### Layer-Major Prefill and Top-K Sweep

Use `run_dsv4_flash_profile.sh` for a single layer-major prefill run. The
wrapper is manifest-driven, so the same script works for full DeepSeek V4 and
DeepSeek V4 Flash:

```bash
DISPLAY_PROMPT=0 \
SIMPLE_IO=1 \
RAW_COMPLETION=1 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=1 \
CTX=90000 \
SEED=123 \
TEMP=0 \
MOE_SLOT_BANK=32 \
LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS=1 \
bash ./tools/flashmoe-sidecar/run_dsv4_flash_profile.sh \
  "$HOME/Models/DeepSeek-V4-Flash-SSD" \
  --temp 1.0 --top-p 1.0 \
  -f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
  --moe-predict-top1-prev \
  --moe-topk 4 \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 500 -st
```

For a small benchmark matrix across routed top-k and prompt size, use:

```bash
tools/flashmoe-sidecar/run_dsv4_topk_prompt_sweep.sh \
  "$HOME/Models/DeepSeek-V4-Flash-SSD"
```

Defaults are:

- `TOPKS="6 4 2"`
- `PROMPT_LABELS="128 1k 4k"`
- `MOE_SLOT_BANK=32`
- `CTX=90000`
- `N_PREDICT=500`

Each variation writes a log and a row in
`flashmoe-results/dsv4-topk-prompt-sweep/<timestamp>/summary.csv`. The script
also prints a final table containing prefill t/s and decode t/s for every
top-k / prompt combination.

#### M5 Max sweep summaries (`run_dsv4_topk_prompt_sweep.sh`)

Same script and default matrix (`TOPKS="6 4 2"`, `PROMPT_LABELS="128 1k 4k"`,
`MOE_SLOT_BANK=32`, `N_PREDICT=500` unless overridden). Layer-major prefill
and sidecar dedup are on for these runs.

**DeepSeek V4 Flash** — package: `$HOME/Models/DeepSeek-V4-Flash-SSD`

| status | topk | prompt | prefill tok/s | decode tok/s | prefill tok | decode tok | prefill uniq | dedup % | decode hit % | decode GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 6 | 128 | 25.5 | 5.5 | 140 | 499 | 5754 | 84.3 | 62.1 | 606.73 |
| ok | 6 | 1k | 90.0 | 5.2 | 1089 | 499 | 9854 | 96.5 | 62.6 | 600.11 |
| ok | 6 | 4k | 156.8 | 3.7 | 4335 | 499 | 10815 | 99.0 | 41.7 | 934.36 |
| ok | 4 | 128 | 31.8 | 7.1 | 140 | 499 | 4503 | 81.6 | 65.5 | 368.83 |
| ok | 4 | 1k | 98.2 | 7.5 | 1089 | 499 | 8868 | 95.3 | 67.4 | 348.28 |
| ok | 4 | 4k | 179.5 | 4.8 | 4335 | 499 | 10538 | 98.6 | 43.1 | 608.20 |
| ok | 2 | 128 | 46.9 | 10.8 | 140 | 499 | 2968 | 75.7 | 72.0 | 149.68 |
| ok | 2 | 1k | 121.5 | 10.6 | 1089 | 499 | 6919 | 92.6 | 68.3 | 169.17 |
| ok | 2 | 4k | 192.5 | 7.5 | 4335 | 499 | 10002 | 97.3 | 46.4 | 286.43 |

**DeepSeek V4 (full)** — package: `$HOME/Models/DeepSeek-V4-SSD`

| status | topk | prompt | prefill tok/s | decode tok/s | prefill tok | decode tok | prefill uniq | dedup % | decode hit % | decode GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ok | 6 | 128 | 4.5 | 1.3 | 140 | 499 | 10208 | 80.4 | 54.8 | 2698.19 |
| ok | 6 | 1k | 16.7 | 1.2 | 1089 | 499 | 19471 | 95.1 | 48.3 | 3083.96 |
| ok | 6 | 4k | 39.7 | 1.3 | 4335 | 499 | 21571 | 98.6 | 54.5 | 2716.36 |
| ok | 4 | 128 | 6.1 | 2.3 | 140 | 499 | 7594 | 78.1 | 70.1 | 1189.71 |
| ok | 4 | 1k | 19.5 | 1.9 | 1089 | 499 | 17105 | 93.6 | 59.8 | 1599.93 |
| ok | 4 | 4k | 44.3 | 1.7 | 4335 | 499 | 20734 | 98.0 | 54.3 | 1817.58 |
| ok | 2 | 128 | 9.5 | 3.1 | 140 | 499 | 4752 | 72.6 | 63.1 | 733.93 |
| ok | 2 | 1k | 25.3 | 3.2 | 1089 | 499 | 12693 | 90.5 | 64.6 | 703.92 |
| ok | 2 | 4k | 49.6 | 2.4 | 4335 | 499 | 18689 | 96.5 | 44.8 | 1097.61 |

On this matrix, Flash runs about **4–6×** faster prefill tok/s and **3–4×**
faster decode tok/s than full V4 at the same top-k and prompt class, with
decode-phase expert traffic (`decode GiB`) often **~4× lower** for Flash. On
both variants, prefill dedup % tightens toward **99%** on the 4k prompt; decode
hit % peaks around **topk 4** for Flash and **topk 4** on the 128/1k prompts for
full V4, while 4k decode hit % drops on both models as the working set grows
versus `MOE_SLOT_BANK=32`.

### DeepSeek V4 Server Mode

`run_flashmoe_server.sh` can serve the native HF-to-SSD DeepSeek V4 packages
through the OpenAI-compatible `/v1` API. For `deepseek4` packages, the server
wrapper defaults to routed **top-k 4** unless `MOE_TOPK` is set explicitly.
The model's native routing top-k is 6, but top-k 4 is the current practical
server default on M5 Max because it reduces SSD traffic and improves decode
cache residency. Use `MOE_TOPK=6` when you want native routing fidelity, and
`MOE_TOPK=2` only for low-I/O diagnostics.

Full DeepSeek V4 server, matching a client model id of `deepseek-v4-ssd` and a
base URL of `http://127.0.0.1:8092/v1`:

```bash
HOST=127.0.0.1 \
PORT=8092 \
MODEL_ALIAS=deepseek-v4-ssd \
MOE_TOPK=4 \
MOE_SLOT_BANK=32 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=1 \
CTX=90000 \
SEED=123 \
TEMP=0 \
FLASHMOE_SERVER_PERF=1 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  "$HOME/Models/DeepSeek-V4-SSD" \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  --jinja \
  --chat-template-file "$PWD/models/templates/deepseek-ai-DeepSeek-V3.1.jinja"
```

For DeepSeek V4 Flash, use the Flash package and a distinct alias if it will
run alongside the full model:

```bash
HOST=127.0.0.1 \
PORT=8093 \
MODEL_ALIAS=deepseek-v4-flash-ssd \
MOE_TOPK=4 \
MOE_SLOT_BANK=32 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=1 \
CTX=90000 \
SEED=123 \
TEMP=0 \
FLASHMOE_SERVER_PERF=1 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  "$HOME/Models/DeepSeek-V4-Flash-SSD" \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  --jinja \
  --chat-template-file "$PWD/models/templates/deepseek-ai-DeepSeek-V3.1.jinja"
```

The `MODEL_ALIAS` string is what `/v1/models` advertises, and clients must send
the same string in the JSON request's `model` field. For the full V4 command
above, a generic chat-completions client should use:

```json
{
  "model": "deepseek-v4-ssd",
  "baseUrl": "http://127.0.0.1:8092/v1",
  "apiKey": "not-needed",
  "provider": "generic-chat-completion-api"
}
```

Quick endpoint checks:

```bash
curl -s http://127.0.0.1:8092/v1/models | jq

curl -s http://127.0.0.1:8092/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-ssd",
    "messages": [{"role": "user", "content": "What is Apple Neural Engine?"}],
    "temperature": 0,
    "max_tokens": 128
  }' | jq
```

The `/v1/chat/completions` route needs a chat template because the server is
formatting messages internally. The local `DeepSeek-V3.1` Jinja template uses
the same DeepSeek special tokens and is the current closest built-in template.
For exact DSv4 prompt parity with `encoding/encoding_dsv4.py`, add a dedicated
DSv4 Jinja template or embed one in the exported GGUF.

## Server Mode

`llama-server` on this branch can use the same dedicated
`--moe-prefill-layer-major` path as `llama-cli`. The server helper workflow is:

- `tools/flashmoe-sidecar/run_flashmoe_server.sh`
Starts `llama-server` with Flash-MoE sidecar defaults, slot-bank sizing,
Metal replay env vars, and the layer-major prefill flags you pass through.
- `tools/flashmoe-sidecar/flashmoe_server_turn_test.py`
Sends a large first-turn `/completion` request with `cache_prompt=true`, then
sends a follow-up turn on the same `id_slot`.
- `tools/flashmoe-sidecar/flashmoe_server_smoke.sh`
Convenience wrapper that launches the server, waits for readiness, runs the
turn test, shuts the server down, and prints the key prefill/decode summaries.

### Gemma4 Server Benchmark

This is the benchmark workflow we used for a big-prefill Gemma4 server test:

```bash
HOST=127.0.0.1 \
PORT=8091 \
PROMPT_LABEL=16k \
N_PREDICT=64 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=25000 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/flashmoe_server_smoke.sh \
  ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4
```

Observed Gemma4 server result:

- turn 1: `20008` prompt tokens at `638.8 t/s`, decode `64` tokens at `13.5 t/s`
- turn 2: `74` new prompt tokens with `cache_n=20008`, decode `64` tokens at `14.1 t/s`
- shutdown summary: `src=prefill-layer-major`, dedup `saved=2320548 (99.6%)`,
reuse `246.72x`

### Server Smoke Matrix

The following models were tested end-to-end in server mode with
`--moe-prefill-layer-major` enabled:


| Model            | Prompt              | Slot Bank | Turn 1                                | Turn 2                                               | Prefill Dedup      | Notes                                                                                       |
| ---------------- | ------------------- | --------- | ------------------------------------- | ---------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------- |
| Gemma4 26B-A4B   | `16k` (`20008` tok) | `64`      | prompt `638.8 t/s`, decode `13.5 t/s` | `74` new tokens, `cache_n=20008`, prompt `115.3 t/s` | `99.6%`, `246.72x` | big-prefill benchmark                                                                       |
| MiniMax-M2.7     | `4k` (`4097` tok)   | `128`     | prompt `211.9 t/s`, decode `6.2 t/s`  | `73` new tokens, `cache_n=4097`, prompt `12.9 t/s`   | `97.8%`, `46.11x`  | server prompt reuse works                                                                   |
| GLM-5.1          | `4k` (`4070` tok)   | `96`      | prompt `83.1 t/s`, decode `3.6 t/s`   | `72` new tokens, `cache_n=4070`, prompt `8.9 t/s`    | `97.7%`, `44.01x`  | server prompt reuse works                                                                   |
| Qwen 3.5 35B-A3B | `4k` (`4661` tok)   | `64`      | prompt `539.8 t/s`, decode `46.8 t/s` | full reprocess, `cache_n=0`, prompt `534.8 t/s`      | `98.7%`, `77.07x`  | server log reports forced full prompt re-processing due to SWA / hybrid memory cache limits |
| Kimi K2.5        | `1k` (`1008` tok)   | `64`      | prompt `52.7 t/s`, decode `5.5 t/s`   | `73` new tokens, `cache_n=1007`, prompt `10.1 t/s`   | `90.0%`, `10.01x`  | keep `UBATCH=1` for stable output                                                           |


### Notes for Server Turn Tests

- The turn test uses `/completion`, not `/v1/chat/completions`, so the prompts
are raw completion prompts on purpose.
- The second request reuses the same `id_slot` and sends `cache_prompt=true`.
- For models without SWA / hybrid-memory cache restrictions, the second turn
should show a small `prompt_n` and a large `cache_n`.
- Qwen 3.5 is the current exception in this tested set. The server logs:
`forcing full prompt re-processing due to lack of cache data`, so the second
turn is still valid server inference but not a cache-reuse benchmark.

## Largest Verified Safe Q8 Server Context

After a naive long-context server test froze the machine, we switched to a
guarded workflow:

- quantized KV cache: `-ctk q8_0 -ctv q8_0`
- no warmup: `--no-warmup`
- no context checkpoints: `--ctx-checkpoints 0 --checkpoint-every-n-tokens -1`
- small starting MoE settings: low `MOE_SLOT_BANK`, low `UBATCH`,
`--moe-prefill-banks 1`
- hard free-memory abort: stop the probe if system free memory reaches `20%`

These numbers are the largest **startup-safe** server contexts we verified on
this M5 Max using that guard. They are not all full prompt-prefill runs; they
are the largest guarded server contexts that reached `server ready` without
crossing the abort threshold.


| Model            | Q8 server context | Result                     | Memory guard notes                                                                     |
| ---------------- | ----------------- | -------------------------- | -------------------------------------------------------------------------------------- |
| MiniMax-M2.7     | `196608`          | safe                       | stayed around `38-39%` free after ready                                                |
| Qwen 3.5 35B-A3B | `262144`          | safe                       | remained very comfortable, around `85%` free at ready                                  |
| Kimi K2.5        | `262144`          | safe                       | reached ready around `49%` free, use `UBATCH=1`                                        |
| GLM-5.1          | `188160`          | safe                       | reached ready at `21%` free; `202752` tripped the guard                                |
| Gemma4 26B-A4B   | n/a in full Q8 KV | unsupported in this config | quantized `V` cache requires Flash Attention, and this server reserve path disables it |


Notes:

- MiniMax cannot reach `250k` because its training context is `196608`.
- Qwen and Kimi can load at `262144`, so `250k` is below the verified
startup-safe ceiling on this machine.
- GLM is the tightest of the supported Q8 runs in this set. `188160` is the
largest verified safe value so far; `202752` was not safe.
- Gemma4 still works well for normal layer-major server inference, but not with
full quantized Q8 KV in the current server reserve configuration.

Example guarded startup-only probe:

```bash
HOST=127.0.0.1 \
PORT=8101 \
STARTUP_ONLY=1 \
READY_STABILIZE_SEC=3 \
FREE_MEM_ABORT_PERCENT=20 \
MEMORY_CHECK_SEC=2 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=1024 \
UBATCH=16 \
CTX=196608 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/flashmoe_server_smoke.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -ctk q8_0 \
  -ctv q8_0 \
  --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 \
  --no-warmup \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 2048 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

## Testing the Server with MiniMax-M2.7 and DroidAI (M5 Max 128)

End-to-end recipe for wiring a local Flash-MoE server into the Factory AI /
DroidAI desktop client. The critical piece is `MODEL_ALIAS=minimax-m2`: it
becomes the model id returned by `/v1/models`, and the client's `model` field
must match it byte-for-byte.

### Build `llama-server`

The server binary is not built by default. From the repo root, configure and
compile the `llama-server` target with Metal enabled on Apple Silicon:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_FLASH_MOE_GPU_BANK=ON

cmake --build build --target llama-server -j
```

The resulting binary is `./build/bin/llama-server`. If you build the server
for the first time, also build `llama-cli` once so smoke scripts that expect
both binaries keep working:

```bash
cmake --build build --target llama-cli llama-server -j
```

The `run_flashmoe_server.sh` wrapper picks up the binary automatically at
`./build/bin/llama-server`. Override the path with `LLAMA_SERVER_BIN=...`
if you built into a different directory.

### Launch the server

```bash
HOST=0.0.0.0 \
PORT=8080 \
MODEL_ALIAS=minimax-m2 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=4096 \
UBATCH=16 \
CTX=96000 \
SEED=123 \
TEMP=0.0 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -ctk q8_0 \
  -ctv q8_0 \
  --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 \
  --no-warmup \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 4096 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4
```

Notes on the server flags:

- `HOST=0.0.0.0` lets the client reach the server on `127.0.0.1` and from other
machines on the LAN. Use `HOST=127.0.0.1` if you want loopback only.
- `MODEL_ALIAS=minimax-m2` makes the server advertise `minimax-m2` as the
model id; the DroidAI `model` field below must be the same string.
- `CTX=96000` with `-ctk q8_0 -ctv q8_0` keeps KV cache memory in check for
long coding / tool-use sessions.
- `--moe-prefill-layer-major` with `--moe-prefill-micro-batch auto` and
`--moe-prefill-banks 4` is the current recommended layer-major prefill path
for MiniMax on M5 Max 128. Drop to `--moe-prefill-banks 1` only for the
guarded large-context startup probe or on memory-constrained machines.

High memory use to improve decoding cache hit rate, with prefill and M5 flash-attn
optimizations enabled. On M5 Max 128GB this combines a 128-slot expert bank with
the Metal4 non-vec flash-attention path (`NONVEC_M4_ENABLE=1`) and Morton walk
order for better threadgroup locality during long-context prefill:

```bash
env \
  HOST=0.0.0.0 \
  PORT=8080 \
  MODEL_ALIAS=minimax-m2 \
  MOE_TOPK=4 \
  MOE_SLOT_BANK=128 \
  MOE_CACHE_IO_SPLIT=8 \
  BATCH=4096 \
  UBATCH=16 \
  CTX=96000 \
  SEED=123 \
  TEMP=0.1 \
  GGML_METAL_FLASH_ATTN_NONVEC_M4_ENABLE=1 \
  GGML_METAL_FLASH_ATTN_NONVEC_NSG=4 \
  GGML_METAL_FLASH_ATTN_NONVEC_WALK=morton \
  LLAMA_FLASH_MOE_PERF_PREFILL_INLINE_PROGRESS=1 \
  LLAMA_FLASH_MOE_PERF_PREFILL_LAYER_STATS=0 \
  FLASHMOE_SERVER_PERF=1 \
  ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
    ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
    --ctx-checkpoints 0 \
    --checkpoint-every-n-tokens -1 \
    --no-warmup \
    --moe-predict-top1-prev \
    --moe-prefill-layer-major \
    --moe-prefill-batch 16384 \
    --moe-prefill-micro-batch auto \
    --moe-prefill-io-split 8 \
    --moe-prefill-banks 4
```

Flash-attention tunables used above (M5-only; silently ignored on M3/M4):

- `GGML_METAL_FLASH_ATTN_NONVEC_M4_ENABLE=1` — route the non-vec (prefill)
  flash-attention QK matmul through the Metal 4 tensor-ops path. Requires
  `dk=dv=128` and matching K/V dtype (f16 or f32).
- `GGML_METAL_FLASH_ATTN_NONVEC_NSG=4` — simdgroups per threadgroup for the
  non-vec kernel (valid: 1, 2, 4, 8). 4 is the default for `dk < 512`; pinning
  it here keeps the shader variant stable across runs.
- `GGML_METAL_FLASH_ATTN_NONVEC_WALK=morton` — Morton-order threadgroup
  dispatch (alternatives: `legacy`, `regular`). Morton improves L2 locality for
  long-context prefill where many Q blocks share KV tiles.

Related knobs not set above but useful to know:

- `GGML_METAL_FLASH_ATTN_NONVEC_2PASS_ENABLE=1` (default on) — enables the
  split-K 2-pass reduction. Kicks in automatically when `kv ≥ 16384`; override
  `nwg` with `GGML_METAL_FLASH_ATTN_NONVEC_2PASS_NWG=1|2|4|8`.
- `GGML_METAL_FLASH_ATTN_VEC_M4_ENABLE=1` — Metal4 path for the vec (decode)
  kernel. Decode is already fast on M5; enable only when benchmarking.
- `GGML_METAL_FLASH_ATTN_DEBUG=1` — stderr one-shot log of every unique
  flash-attn shape/path combination encountered. Use for verifying the Metal4
  path is actually being taken.

### Adding the server to `~/.factory/settings.json` or `config.json`

The **Droid CLI** stores BYOK custom models in **`~/.factory/settings.json`**
(camelCase `customModels`); see [Factory: Settings](https://docs.factory.ai/cli/configuration/settings).
Some **desktop** installs or older docs use **`~/.factory/config.json`**
instead—edit whichever path your Factory / Droid build actually updates (both
may exist; do not duplicate conflicting `customModels` across files without
knowing how your version merges them). If the JSON shape differs from below,
follow [Factory BYOK](https://docs.factory.ai/cli/byok/overview) for your file.

Append a `customModels` object (in `settings.json` when using the CLI layout)
for each local server. The server and client must agree on two things: the
`model` field must match `MODEL_ALIAS`, and `baseUrl` must match `HOST:PORT/v1`.

**DeepSeek V4 (full, native HF → SSD package):** set
`MODEL_ALIAS=deepseek-v4-ssd` on `run_flashmoe_server.sh` and use
`"model": "deepseek-v4-ssd"` in Factory / Droid (same string, no spaces). If
you omit `MODEL_ALIAS`, `/v1/models` will advertise the GGUF name
`DeepSeek-V4-SSD` instead; you can use that as `"model"` instead, but the
hyphenated alias avoids case/spacing surprises in clients.

Example fragment—merge the new object into your **existing** `customModels`
array (pick an unused `index` and a unique `id`):

```json
    {
      "model": "deepseek-v4-ssd",
      "id": "custom:DeepSeek-V4-(m5m-local)-0",
      "index": 2,
      "baseUrl": "http://127.0.0.1:8092/v1",
      "apiKey": "not-needed",
      "displayName": "DeepSeek V4 (m5m local)",
      "noImageSupport": true,
      "provider": "generic-chat-completion-api"
    }
```

Use port **8092** only if that is where you run the DeepSeek server; it must
match `PORT` on `llama-server`. MiniMax can stay on **8080**.

Minimal second-server launch (loopback, layer-major prefill; tune `MOE_*` /
`CTX` like your benchmarks):

```bash
HOST=127.0.0.1 \
PORT=8092 \
MODEL_ALIAS=deepseek-v4-ssd \
MOE_TOPK=4 \
MOE_SLOT_BANK=32 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=1 \
CTX=90000 \
SEED=123 \
TEMP=0 \
FLASHMOE_SERVER_PERF=1 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  "$HOME/Models/DeepSeek-V4-SSD" \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch auto \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  --jinja \
  --chat-template-file "$PWD/models/templates/deepseek-ai-DeepSeek-V3.1.jinja"
```

Full-file shape (MiniMax + DeepSeek) for a fresh `customModels` block:

```json
{
  "enabledPlugins": {
    "core@factory-plugins": true
  },
  "logoAnimation": "off",
  "customModels": [
    {
      "model": "minimax-m2",
      "id": "custom:MiniMax-M2.7-(m5m-local)-0",
      "index": 1,
      "baseUrl": "http://127.0.0.1:8080/v1",
      "apiKey": "not-needed",
      "displayName": "MiniMax-M2.7 (m5m local)",
      "noImageSupport": true,
      "provider": "generic-chat-completion-api"
    },
    {
      "model": "deepseek-v4-ssd",
      "id": "custom:DeepSeek-V4-(m5m-local)-0",
      "index": 2,
      "baseUrl": "http://127.0.0.1:8092/v1",
      "apiKey": "not-needed",
      "displayName": "DeepSeek V4 (m5m local)",
      "noImageSupport": true,
      "provider": "generic-chat-completion-api"
    }
  ]
}
```

Field-by-field:

- `"model": "minimax-m2"` must equal `MODEL_ALIAS` on the MiniMax server. If the
alias changes, this string must change with it.
- `"model": "deepseek-v4-ssd"` must equal `MODEL_ALIAS=deepseek-v4-ssd` on the
DeepSeek V4 SSD server (or match the unaliased GGUF id if you choose not to set
`MODEL_ALIAS`).
- `"id"` is the internal Droid handle: `custom:` prefix plus a unique suffix
per entry (for example `custom:DeepSeek-V4-(m5m-local)-0`). The `(m5m-local)`
tag is a naming convention for this machine versus `(m3u-local)` or hosted
endpoints.
- `"index"` controls display order in the Droid model picker; pick any unused
integer per entry.
- `"baseUrl"` must match the server (`http://127.0.0.1:8080/v1` for MiniMax in
the example above, `http://127.0.0.1:8092/v1` for DeepSeek if `PORT=8092`).
Keep the trailing `/v1`.
- `"apiKey": "not-needed"` — the local server does not validate the key,
but the field must be present for the provider plugin.
- `"provider": "generic-chat-completion-api"` selects the OpenAI-compatible
transport, which our `/v1/chat/completions` endpoint implements.
- `"noImageSupport": true` is required for these text-only local models.

To register additional hosts, copy an object, bump `index`, give a new unique
`id`, and point `baseUrl` at that host's `HOST:PORT`.

## Testing Flash-MoE with `@badlogicgames` pi-agent (badlogic)

`[pi` / `pi-coding-agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)`
can use the local Flash-MoE server as a custom OpenAI-compatible provider. The
provider entry lives in `~/.pi/agent/models.json`.

```json
{
  "providers": {
    "flashmoe-local": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "dummy",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "minimax-m2",
          "name": "MiniMax-M2.7 Flash (Local)",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 96000,
          "maxTokens": 32000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Notes:

- `id` must equal `MODEL_ALIAS` on the server. If the server advertises
`minimax-m2`, the `pi` model id must also be `minimax-m2`.
- Keep `baseUrl` on `127.0.0.1` when `pi` runs on the same machine, even if
the server binds to `HOST=0.0.0.0` for LAN access.
- `supportsDeveloperRole=false` and `supportsReasoningEffort=false` keep `pi`
on the simplest OpenAI-compatible path for the current server behavior.
- `supportsUsageInStreaming=true` is required for the `pi` footer to show
correct context / cache usage. With it set to `false`, `pi` does not send
`stream_options.include_usage`, the server omits the trailing usage chunk,
and `pi` reports `cacheRead`/`cacheWrite`/totals as `0`. The llama.cpp
server emits `prompt_tokens_details.cached_tokens` in the final stream
chunk when this option is enabled.
- After editing `~/.pi/agent/models.json`, open `/model` inside `pi` to reload
the file and select `MiniMax-M2.7 Flash (Local)`.

### Quick sanity check

Before opening DroidAI or `pi`, confirm the alias and endpoint are reachable:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq
curl -s http://127.0.0.1:8092/v1/models | jq   # DeepSeek on 8092
```

The response must include the model id you configured (`"id": "minimax-m2"` or
`"id": "deepseek-v4-ssd"` when using the recommended alias). If it does not,
the server `MODEL_ALIAS` and the Droid `model` field are out of sync and the
client will fail to route requests.
