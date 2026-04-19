# Flash-MoE Layer-Major Dedup (SSD-streamed)

This note describes the current `--moe-prefill-layer-major` path in this fork.

It focuses on the SSD-streamed prompt-prefill MoE path used by routed MoE
layers (MiniMax-M2, GLM-5.1, Kimi K2.5, Qwen 3.5, Gemma4) when the full routed
expert weights do not fit in memory and have to be loaded on demand from the
Flash-MoE sidecar instead of staying resident:

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
- [GLM-5.1 Default Test](#glm-51-default-test)
- [Additional Model Smoke Tests](#additional-model-smoke-tests) (Kimi K2.5, Qwen 3.5, Gemma4)
- [Server Mode](#server-mode)
- [Largest Verified Safe Q8 Server Context](#largest-verified-safe-q8-server-context)
- [Testing the Server with MiniMax-M2.7 and DroidAI (M5 Max 128)](#testing-the-server-with-minimax-m27-and-droidai-m5-max-128)

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

## Example Results

Hardware: `M5 Max`, runtime: `Flash-MoE`, storage: `SSD`.

### 4K prefill across models

| Model             | Quant                | Prefill tokens | Prompt      | Decode     |
| ----------------- | -------------------- | -------------: | ----------: | ---------: |
| Gemma4 26B-A4B    | `UD-Q3_K_XL`         |         `5009` | `718.3 t/s` | `21.4 t/s` |
| Qwen 3.5 35B-A3B  | `IQ2`                |         `4661` | `535.5 t/s` | `52.2 t/s` |
| MiniMax-M2.7      | `UD-Q4_K_XL`         |           `4K` | `227.3 t/s` | `16.6 t/s` |
| Kimi K2.5         | `UD-TQ1`             |         `4053` | `109.4 t/s` |  `4.4 t/s` |
| GLM-5.1           | `IQ1`                |         `4070` |  `94.5 t/s` |  `4.4 t/s` |

### 1K prefill (selected)

| Model             | Quant                | Prefill tokens | Prompt      | Decode     |
| ----------------- | -------------------- | -------------: | ----------: | ---------: |
| Gemma4 26B-A4B    | `UD-Q3_K_XL`         |         `1260` | `630.6 t/s` | `36.8 t/s` |
| MiniMax-M2.7      | `UD-Q4_K_XL`         |           `1K` |  `78.1 t/s` | `16.7 t/s` |
| Kimi K2.5         | `UD-TQ1`             |         `1008` |  `53.0 t/s` |  `6.3 t/s` |

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
--moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1024 -st
```

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
  --moe-prefill-micro-batch 32 \
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

| Model | Prompt | Slot Bank | Turn 1 | Turn 2 | Prefill Dedup | Notes |
|-------|-------:|----------:|-------:|-------:|--------------:|-------|
| Gemma4 26B-A4B | `16k` (`20008` tok) | `64` | prompt `638.8 t/s`, decode `13.5 t/s` | `74` new tokens, `cache_n=20008`, prompt `115.3 t/s` | `99.6%`, `246.72x` | big-prefill benchmark |
| MiniMax-M2.7 | `4k` (`4097` tok) | `128` | prompt `211.9 t/s`, decode `6.2 t/s` | `73` new tokens, `cache_n=4097`, prompt `12.9 t/s` | `97.8%`, `46.11x` | server prompt reuse works |
| GLM-5.1 | `4k` (`4070` tok) | `96` | prompt `83.1 t/s`, decode `3.6 t/s` | `72` new tokens, `cache_n=4070`, prompt `8.9 t/s` | `97.7%`, `44.01x` | server prompt reuse works |
| Qwen 3.5 35B-A3B | `4k` (`4661` tok) | `64` | prompt `539.8 t/s`, decode `46.8 t/s` | full reprocess, `cache_n=0`, prompt `534.8 t/s` | `98.7%`, `77.07x` | server log reports forced full prompt re-processing due to SWA / hybrid memory cache limits |
| Kimi K2.5 | `1k` (`1008` tok) | `64` | prompt `52.7 t/s`, decode `5.5 t/s` | `73` new tokens, `cache_n=1007`, prompt `10.1 t/s` | `90.0%`, `10.01x` | keep `UBATCH=1` for stable output |

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

| Model | Q8 server context | Result | Memory guard notes |
|-------|-------------------:|--------|--------------------|
| MiniMax-M2.7 | `196608` | safe | stayed around `38-39%` free after ready |
| Qwen 3.5 35B-A3B | `262144` | safe | remained very comfortable, around `85%` free at ready |
| Kimi K2.5 | `262144` | safe | reached ready around `49%` free, use `UBATCH=1` |
| GLM-5.1 | `188160` | safe | reached ready at `21%` free; `202752` tripped the guard |
| Gemma4 26B-A4B | n/a in full Q8 KV | unsupported in this config | quantized `V` cache requires Flash Attention, and this server reserve path disables it |

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
  --moe-prefill-micro-batch 32 \
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
  --moe-prefill-batch 16192 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

Notes on the server flags:

- `HOST=0.0.0.0` lets the client reach the server on `127.0.0.1` and from other
  machines on the LAN. Use `HOST=127.0.0.1` if you want loopback only.
- `MODEL_ALIAS=minimax-m2` makes the server advertise `minimax-m2` as the
  model id; the DroidAI `model` field below must be the same string.
- `CTX=96000` with `-ctk q8_0 -ctv q8_0` keeps KV cache memory in check for
  long coding / tool-use sessions.
- `--moe-prefill-layer-major` plus `--moe-prefill-banks 1` is the verified
  single-bank layer-major prefill path for MiniMax on M5 Max 128. Raise
  `--moe-prefill-banks` only if you have extra memory headroom.

High memory use to improve decoding cache hit rate:

```bash
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
  --moe-prefill-batch 16192 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

### Adding the server to `~/.factory/settings.json`

Factory AI / DroidAI reads custom OpenAI-compatible models from
`~/.factory/settings.json`. Add one `customModels` entry that points at the
local server. The server and client must agree on two things: the `model`
field must match `MODEL_ALIAS`, and `baseUrl` must match `HOST:PORT/v1`.

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
    }
  ]
}
```

Field-by-field:

- `"model": "minimax-m2"` must equal `MODEL_ALIAS` on the server. If the
  alias changes, this string must change with it.
- `"id": "custom:MiniMax-M2.7-(m5m-local)-0"` is the internal DroidAI handle.
  The `custom:` prefix plus a unique suffix per entry are the hard
  requirements. The `(m5m-local)` tag is a naming convention we use to mark
  this as the local M5 Max server and to distinguish it from a separate
  `(m3u-local)` entry on another host or from a hosted MiniMax endpoint.
- `"index"` controls display order in the DroidAI model picker; pick any
  free integer.
- `"baseUrl": "http://127.0.0.1:8080/v1"` points at the loopback server
  started above. For LAN access or a different port, match `HOST` and
  `PORT` exactly. Keep the trailing `/v1`.
- `"apiKey": "not-needed"` — the local server does not validate the key,
  but the field must be present for the provider plugin.
- `"provider": "generic-chat-completion-api"` selects the OpenAI-compatible
  transport, which our `/v1/chat/completions` endpoint implements.
- `"noImageSupport": true` is required: MiniMax-M2.7 is text-only.

To register additional local servers (for example a second machine on the
LAN), copy the object, bump `index`, give it a new unique `id` like
`custom:MiniMax-M2.7-(m3u-local)-0`, and point `baseUrl` at that host.

### Quick sanity check

Before opening DroidAI, confirm the alias and endpoint are reachable:

```bash
curl -s http://127.0.0.1:8080/v1/models | jq
```

The response must include `"id": "minimax-m2"`. If it does not, the server
`MODEL_ALIAS` and the DroidAI `model` field are out of sync and the client
will fail to route requests.
