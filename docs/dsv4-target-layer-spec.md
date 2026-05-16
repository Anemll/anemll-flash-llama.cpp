# DSV4 Target Decode-Layer Executor Spec

## Target

Define the production DSV4 decode-layer executor as an op-subtraction replacement in the main ggml graph.

When `LLAMA_FLASH_MOE_DSV4_LAYER_EXECUTOR=1` is enabled for a target decode layer, the existing per-layer subgraph is not emitted for that layer. One `GGML_OP_DSV4_DECODE_LAYER` is emitted instead.

Production shape:

```text
existing target-layer subgraph emitted: no
executor op emitted: yes, exactly one per active target layer
parallel branch: no
fallback branch in the same graph: no
cache side effects: owned by executor for the target layer
```

The executor replaces:

```text
QKV setup
compressor/update
KV finalizer / cache metadata handoff
attention core / output / AOHC
HC pre/post boundaries
routed-MoE / shared branch / final FFN
layer output anchor
```

## Baseline trace facts

Paired trace facts from Turns #32/#33:

```text
ours tok/s: ~21.5-21.9
DS4 tok/s: ~36.4-36.5

ours dispatch/proxy rows per token: ~831.73
DS4 boundary proxy rows per token: ~450.00

ours coarse FFN rows: 8724
DS4 routed-MoE boundary rows: 1419
FFN ratio: 6.15x
```

Current repo counter baseline:

```text
current metal_dispatch n400: 1399039
decode tokens: 399
approx metal_dispatch counter units per token: 3506
```

`metal_dispatch` counter units and paired-trace boundary proxy rows are not identical units. Both must move downward for the executor path to be considered structurally successful.

## DS4 per-layer kernel sequence

| stage | DS4 routine / kernel name | expected inputs | expected outputs | current repo equivalent graph region | target executor ownership |
| --- | --- | --- | --- | --- | --- |
| qkv setup | DS4 decode Q/KV projection sequence | layer input, attention norm output, Q/KV weights, token position | Q rope tail, KV rope source | `q_lora`, `q_lora_norm`, `Qnorm`, `Qcur`, `KVnorm`, `KVrope` | owned |
| compressor/update | DS4 compressor/update decode path | projection source, compressor KV/gate/APE/norm weights, previous compressor state | updated KV state, score state, pooled compressed row | `dsv4_build_compressor_decode`, `dsv4_build_compressor_decode_chunk` | owned |
| KV/cache store/finalizer | `ds4_metal_kv_fp8_store_raw_tensor` plus cache row handoff | Q/KV rope, quantized KV row, cache indices, cache position | SWA raw cache row, compressed cache row metadata | `ggml_dsv4_fp8_kv_quantize`, `mctx_swa->cpy_k`, `dsv4_store_cache_rows` | owned |
| indexed/mixed attention | `ds4_metal_attention_indexed_mixed_batch_heads_tensor` | Q, raw KV cache, compressed KV cache, index top-k, masks | mixed attention output | `build_attn_mha`, indexed/mixed attention graph, masks | owned |
| indexed/mixed attention kernel | `kernel_dsv4_indexed_mixed_attention_heads8` | heads8 attention inputs, cache rows, masks | attention heads output | Metal mixed/indexed attention region | owned |
| indexed/mixed attention rb4 kernel | `kernel_dsv4_indexed_mixed_attention_heads8_rb4` | ratio-4 indexed attention inputs | ratio-4 indexed output | ratio-4 indexer attention path | owned |
| attention output high | `ds4_metal_attention_output_q8_batch_tensor` | attention core output, high projection q8 weights | attention high/output tensor | attention output q8/high projection graph | owned |
| attention output low | `ds4_metal_attention_output_low_q8_tensor` | attention low projection inputs, low q8 weights | low/output contribution | `attn_low`, attention output low branch | owned |
| HC pre/post | DS4 HC norm/post routines | HC residual, HC weights, RMSNorm weights, attention/FFN boundary tensors | HC pre norm, HC post output, layer boundary tensors | `dsv4_hc_pre`, HC post / expand, `hc_ffn_post` | owned |
| routed-MoE one-tensor | `ds4_metal_routed_moe_one_tensor` | FFN input, top-k ids/weights, expert/shared weights | final FFN output | routed-MoE generic graph and one-tensor shadow | owned |
| result/head handoff | DS4 result/head handoff | final layer output, final norm inputs | result_hc/result_norm/head input | `result_hc`, `result_norm`, logits input path | owned for final layer handoff metadata |

## Current repo per-layer graph sequence

Current decode-layer graph sequence:

```text
1. layer input / residual
2. attention HC pre and attention RMSNorm
3. Q/KV projections and RoPE
4. raw SWA KV quant/store
5. optional compressor/update and compressed cache store
6. raw, compressed, indexed, or mixed attention
7. attention output projection
8. attention HC post / residual handoff
9. FFN HC pre and FFN RMSNorm
10. router/top-k
11. generic expert gate/up
12. generic packed CLAMP -> SILU -> CLAMP -> MUL
13. generic expert down and routed weighted sum
14. generic shared branch
15. final FFN add
16. FFN HC post
17. layer output anchor
```

Local replacement attempts showed that changing only one subregion of this sequence changes lowering/order enough to break transcript-exact decode. The executor target must own the whole sequence for the selected layer.

## Target dispatch budget

Target:

```text
target dispatches/token: <= 450
approx active decode layers: 30
target dispatches/layer: ~15
net graph goal per active layer: subtract ~28 ops, add 1 executor op
```

Current rough baseline:

```text
current metal_dispatch n400: 1399039
decode tokens: 399
approx dispatches/token: ~3506 metal_dispatch counter units
```

Acceptance expectation:

```text
both paired-trace boundary proxies and metal_dispatch counter units must drop
dispatch reduction without transcript stability is rejected
transcript stability without dispatch reduction is support-only
```

## Executor input contract

Required inputs for one target decode layer:

```text
layer input / residual
attention norm inputs and weights
Q/K/V projection weights
compressor state inputs
compressor KV/gate/APE/norm weights
SWA cache views
compressed cache views
cache row indices, row count, row width, cache position
attention masks / indexed attention metadata
HC state and HC weights
FFN norm input and weights
routed-MoE top-k ids and weights
expert gate/up/down weights
shared gate/up/down weights
layer id and token/decode index metadata
```

Input stability requirement:

```text
metadata visible before executor lowering: yes
cache row metadata equal to generic path: yes
quant/cache byte rows available for exact validation: yes
```

## Executor output contract

Required outputs:

```text
layer output tensor
updated HC state candidate
updated raw SWA KV/cache rows
updated compressed KV/cache rows
updated compressor state rows
result/head handoff metadata when target layer is final layer
optional validation-only stage summaries
```

The production output feeds downstream graph consumers directly. There is no parallel generic fallback branch in the same graph.

## Cache/dependency ownership

The executor owns target-layer cache/dependency semantics:

```text
raw SWA KV row store
compressed cache row store
compressor state update
SET_ROWS/cache write dependency metadata
cache position and row index validation
attention mask/index metadata consumption
HC pre/post dependency ordering
layer output dependency anchor
```

Cache validation gates:

```text
quant/cache byte rows: byte-exact required
cache metadata: equal required
no candidate cache mutation in shadow validation
no unbounded all-layer consume before layer-0 gates pass
```

## Op-subtraction requirements

For an active target layer:

```text
emit existing per-layer subgraph: no
emit GGML_OP_DSV4_DECODE_LAYER: yes
emit generic fallback branch: no
emit side branch for production: no
expected net graph change: subtract ~28 ops, add 1 op
```

The executor must preserve deterministic decode by owning graph ordering and lowering consistently across the full layer boundary.

## Falsification criteria

T87 validation gate:

```text
transcript gate:
  10 fixed prompts x 400 tokens
  transcript match rate required by active policy: 100%
  PPL n=1000 suite: future validation suite, not a T87 gate until constructed
```

T87 all-layer executor result:

```text
>30 tok/s:
  ship as first accepted executor performance path

25-30 tok/s:
  executor works but kernels are too granular
  allow one consolidation pass only

<25 tok/s with dispatches/token near DS4 ~450:
  gap is kernel-internal
  stop graph-shape work and switch to Instruments-driven per-kernel profiling

<25 tok/s with dispatches/token still high:
  spec was not aggressive enough
  revise dispatch budget/spec before more executor iteration

layer-0 cutover drifts beyond tolerance:
  executor reduction/order is wrong
  fix executor implementation
  do not widen tolerance to accept it
```
