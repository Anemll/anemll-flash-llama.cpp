# Flash-MoE Layer-Major Dedup

This note describes the current `--moe-prefill-layer-major` path in this fork.

It focuses on the dedicated prompt-prefill MoE path used by routed MoE layers (MiniMax-M2, GLM-5.1, Kimi K2.5, Qwen 3.5, Gemma4) in:

- `src/llama-context.cpp`
- `common/arg.cpp`

## Short Description

Layer-major dedup runs prefill one layer at a time. For each layer, it groups repeated expert selections across the current prefill chunk, loads each expert once, and reuses it for all tokens that need it instead of reloading it repeatedly.

The bigger the prefill chunk, the more expert reuse we usually get, so prefill becomes cheaper. That should help both GPU and ANE prefill at large batch sizes.

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

