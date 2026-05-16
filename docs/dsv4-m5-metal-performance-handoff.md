# DeepSeek V4 Flash M5 Metal Performance Handoff

Last updated: 2026-05-13

## Executive Summary

This repo currently runs the DeepSeek V4 Flash GGUF correctly on M5 Metal. The latest useful optimization is the DSV4 HC-pre + learned RMSNorm fusion (`HC_PRE_NORM`), which is now tensor-correct, transcript-stable for the n400 smoke test, and enabled automatically under the existing split-GLU experimental flag.

The previous handoff said decode was stuck around 20.4 to 20.9 tok/s. That is no longer exactly true: the latest paired smoke run moved from `21.1 tok/s` with `HC_PRE_NORM` forced off to `21.9 tok/s` with it enabled. This is the first real, correctness-stable wall-clock movement from the local M5 work. It is still far below the external DS4 target class, so the larger conclusion remains: the remaining gap probably needs whole DSV4 decode-stage operations, not more small adjacent-node peepholes.

The main conclusion from the work so far is that the remaining gap is probably not fixable with another small adjacent-node matcher. The faster external DS4 implementation treats several DSV4 attention/compressor/output stages as dedicated model-specific Metal operations. This repo still expresses most of those stages as many generic ggml graph nodes. The performance problem appears to be whole-stage graph structure, launch count, cache/view ordering, and generic projection scheduling, not just a missing one-kernel peephole.

The most important next step is to stop adding small local fusions and instead implement one dedicated DSV4 decode-stage operation at a time. CUPD2 and the explicit KV finalizer are now both correctness/support primitives, not speed paths. A first FFN/MoE down/add-tree diagnostic proved the Q2_K down-sum arithmetic can match the generic add tree exactly, but consuming it without the generic branch changes upstream split-GLU pairing and drifts. The next main target is still a DS4-shaped FFN/MoE decode stage that owns the full boundary, or a similarly large attention/compressor stage that owns an entire boundary instead of replacing one adjacent pair of nodes.

## 2026-05-13 Status Update

### 2026-05-13 8-Hour CUPD2 / KV Finalizer Update

This pass implemented a fresh, side-effect-free compressor/update decode op:

```text
GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2
counter: dsv4_cupd2
flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE_CONSUME=generic|fused
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_TRACE=1
```

Implementation shape:

```text
generic projections remain:
  kv_cur    = mul_mat(wkv, x)
  score_cur = mul_mat(wgate, x)

CUPD2 owns:
  APE add
  recurrent kv state row update
  recurrent score state row update
  ratio-4 current/previous row packing
  pool input preparation

safe/default CUPD2 leaves:
  pool softmax
  RMSNorm
  norm weight
  RoPE
  FP8/HFP4 cache writes

inside the existing graph.
```

Correctness:

```text
n16 compare, consume generic:
  log: /tmp/dsv4_may13_8h/cupd2_compare_generic_n16.log
  dsv4_cupd2=930
  over_tol=0

n16 compare, consume fused:
  log: /tmp/dsv4_may13_8h/cupd2_compare_fused_n16.log
  dsv4_cupd2=930
  over_tol=0

n80 compare, consume fused:
  log: /tmp/dsv4_may13_8h/cupd2_compare_fused_n80.log
  dsv4_cupd2=4898
  over_tol=0
  visible prefix: exact expected ANE prefix

n400 performance/correctness:
  baseline HC_PRE_NORM:
    log: /tmp/dsv4_may13_8h/perf_A_hcnorm_n400.log
    dsv4_cupd2=0
    metal_dispatch=1399039
    generation=19.7 tok/s
    transcript: exact

  CUPD2 only, HC_PRE_NORM forced off:
    log: /tmp/dsv4_may13_8h/perf_B_cupd2_only_no_hcnorm_n400.log
    dsv4_hcnorm=0
    dsv4_cupd2=24738
    metal_dispatch=1317669
    generation=19.6 tok/s
    transcript: exact

  HC_PRE_NORM + CUPD2:
    log: /tmp/dsv4_may13_8h/perf_C_hcnorm_cupd2_n400.log
    dsv4_hcnorm=34314
    dsv4_cupd2=24738
    metal_dispatch=1283355
    generation=19.6 tok/s
    transcript: exact

  HC_PRE_NORM + CUPD2 + paired compressor projection:
    flags:
      LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_PAIR=1
    log: /tmp/dsv4_may13_8h/perf_G_hcnorm_cupd2_cpair_n400.log
    dsv4_hcnorm=34314
    dsv4_cupd2=24738
    dsv4_cpair=24738
    mul_mat mv=172372
    metal_dispatch=1242258
    generation=19.5 tok/s
    transcript: exact

  HC_PRE_NORM + CUPD2 + paired compressor projection stage profile:
    log: /tmp/dsv4_may13_8h/perf_H_hcnorm_cupd2_cpair_stage_n32.log
    generation=6.3 tok/s with intrusive profiler
    largest buckets:
      ffn total=1198.186 ms
      attn_compress total=775.369 ms
      attn_out total=763.978 ms
      attn_kv total=622.800 ms
      attn_qkv total=565.842 ms
```

Interpretation:

- CUPD2 safe/default is correct and reduces dispatch, but it is performance-neutral.
- With `COMPRESSOR_PAIR=0`, `gen_mv` and `mul_mat mv` remain unchanged (`gen_mv=48267`, `mul_mat mv=221848`), because the two projections are still generic and cache writes remain generic.
- With `COMPRESSOR_PAIR=1`, CUPD2 now consumes the paired projection output and `mul_mat mv` falls to `172372`, but n400 speed still does not move. This says compressor projection dispatch alone is not the limiting wall-clock bucket in this ggml-shaped graph.
- This is a useful support primitive and instrumentation point, not a wall-clock win.

`CUPD2_FUSED_COMP=1` was also tested:

```text
n16 compare generic:
  log: /tmp/dsv4_may13_8h/cupd2_fused_comp_compare_generic_n16.log
  first visible kv_comp max_abs=4.76837e-07
  over_tol=0

n80 compare fused:
  log: /tmp/dsv4_may13_8h/cupd2_fused_comp_compare_fused_n80.log
  over_tol=0

n400:
  log: /tmp/dsv4_may13_8h/perf_E_hcnorm_cupd2_fused_comp_n400.log
  metal_dispatch=1249275
  generation=19.7 tok/s
  transcript: drifted
```

The n400 drift appears immediately in the generated explanation:

```text
baseline: running neural networks and AI algorithms.
fused:    running neural networks (the foundation of modern AI).
```

Classification: rejected for correctness. Even sub-1e-6 compressor-row differences can change deterministic decode later, so text stability remains mandatory. Do not enable `CUPD2_FUSED_COMP` as a performance path until it reproduces the generic `soft_max -> mul -> sum_rows -> rms_norm -> norm_weight -> rope` boundary closely enough for exact n400 transcript stability, or until a stricter logit/tensor acceptance policy is defined.

KV/cache finalizer diagnostic added:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZER_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZER_VIEW=1
```

`KV_FINALIZER_VIEW=1` changes `dsv4_store_cache_rows()` to use a direct 2D view when the source is already contiguous `[width,1,n_rows,1]`. Otherwise it falls back to the old `cont + reshape` path.

Trace result:

```text
log: /tmp/dsv4_may13_8h/kv_finalizer_view_trace_n32.log

dsv4_store_cache_rows:
  mode=view
  src=KVcompress-*/f32
  src_shape=[512,1,4,1] or [512,1,128,1]
  src_op=DSV4_FP8_KV_QUANTIZE
  src0=DSV4_ROPE_TAIL
```

However, backend trace still shows the existing adjacent-node matcher cannot fuse the cache write:

```text
KVrope-* next=DSV4_FP8_KV_QUANTIZE
n2=SCALE/cache_r_l* ...
n3=SCALE/cache_s_l* ...
reject set-not-found
```

Performance with the view path and `KV_SET_ROWS=1`:

```text
log: /tmp/dsv4_may13_8h/perf_F_hcnorm_cupd2_kvset_view_n400.log
dsv4_kvset=0
metal_dispatch=1278943
generation=19.6 tok/s
transcript: exact
```

Classification:

- `KV_FINALIZER_VIEW` is a guarded graph-cleanup/diagnostic helper.
- It proves the producer tensor is eligible for a direct cache-store finalizer.
- It does not make the existing backend peephole activate because graph scheduling still separates the producer and `SET_ROWS`.
- The next real KV/cache optimization needs an explicit model-specific finalizer op with dependency semantics, not a smarter adjacent matcher.

### 2026-05-13 Explicit KV Finalizer Op

This pass implemented the explicit graph op recommended by the previous KV/cache analysis:

```text
GGML_OP_DSV4_KV_FINALIZE_DECODE
counter: dsv4_kvfin
flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_COMPARE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_TRACE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_VIEW=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_DRY_RUN=1
```

Implementation shape:

```text
DSV4_ROPE_TAIL
  -> DSV4_FP8_KV_QUANTIZE
  -> GGML_OP_DSV4_KV_FINALIZE_DECODE

The new op consumes the quantized F32 row candidate, the cache tensor,
and row indices. In dry-run mode it produces the dependency/output tensor
but leaves the legacy set_rows write active. In normal mode it writes the
decode cache row and returns the current-token candidate tensor consumed by
downstream graph nodes.
```

The op is intentionally decode-only in this first accepted version:

```text
n_rows == 1
src type == F32
src shape == [width,1,1,1]
cache type == F16 or F32
rows type == I32 or I64
```

Correctness:

```text
dry-run n16 compare:
  log: /tmp/dsv4_kvfin/kvfin_dryrun_compare_n16_v3.log
  dsv4_kvfin=252
  max_abs=0
  over_tol=0
  visible prefix: pass

non-dry n80 compare:
  log: /tmp/dsv4_kvfin/kvfin_compare_n80.log
  dsv4_kvfin=924
  max_abs=0 in logged compare records
  over_tol=0 in logged compare records
  visible prefix: pass

n400 performance/correctness:
  baseline split-GLU:
    log: /tmp/dsv4_kvfin/baseline_split_glu_n400.log
    dsv4_kvfin=0
    metal_dispatch=1399039
    gen_mv=48267
    mul_mat mv=221848
    generation=21.0 tok/s
    transcript: exact/stable

  KV finalizer:
    log: /tmp/dsv4_kvfin/kvfin_n400.log
    dsv4_kvfin=4344
    metal_dispatch=1394695
    gen_mv=48267
    mul_mat mv=221848
    generation=20.9 tok/s
    transcript: exact/stable
```

Intrusive stage-profile sample with KV finalizer:

```text
log: /tmp/dsv4_kvfin/kvfin_stage_n32.log
generation=5.4 tok/s with intrusive profiler
dsv4_kvfin=420

largest buckets:
  ffn           total=1500.153 ms
  attn_compress total=944.772 ms
  attn_out      total=856.828 ms
  attn_kv       total=751.605 ms
  attn_qkv      total=678.333 ms
```

Classification:

- Accepted as a correctness and dependency-semantics primitive.
- It proves a first-class graph op can own the cache write without relying on adjacent backend matching.
- It is not a speed path by itself: dispatch fell by exactly the activation count class (`4344`), but n400 generation stayed flat/slightly lower (`21.0 -> 20.9 tok/s`).
- It does not yet fuse RoPE or FP8 quantization arithmetic; it consumes the already-quantized row candidate.
- It compares the finalizer output against the quantized source candidate and validates by n80/n400 transcript. It does not yet read back and dequantize cache rows for direct cache-row comparison.

Next conclusion:

```text
Do not keep polishing KV_SET_ROWS or KV_FINALIZE as local cache-store work.
The explicit dependency problem is solved well enough for now, and wall-clock
did not move. Move to a whole FFN/MoE decode stage, or a larger stage that
removes a full DS4-style boundary.
```

### Accepted / Useful

#### HC_PRE_NORM

Status: accepted as the current best M5 optimization.

The fused path now matches the DS4 boundary more closely:

```text
flat_hc = RMSNormPlain(input_hc)
hc_mix  = matmul_f16(hc_fn, flat_hc)
fused:
  split = Sinkhorn(hc_mix, scale, base)
  cur   = WeightedSum(original input_hc, split.pre)
  norm  = RMSNormWeight(cur, norm_w)
```

Important semantic point: the weighted sum consumes the original HC residual (`cur_hc` / `after_attn_hc`), not the normalized `flat_hc`.

Current activation:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1
```

automatically enables `HC_PRE_NORM`. It can still be forced off with:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM=0
```

Latest paired n400 smoke results:

```text
forced off:
  log: /tmp/dsv4_split_glu_hcnorm_forced_off_n400.log
  dsv4_hcnorm=0
  metal_dispatch=1433353
  generation=21.1 tok/s
  transcript: exact match vs baseline

auto enabled under split-GLU:
  log: /tmp/dsv4_split_glu_auto_hcnorm_n400.log
  dsv4_hcnorm=34314
  metal_dispatch=1399039
  generation=21.9 tok/s
  transcript: exact match vs baseline
```

Correctness compare smoke:

```text
log: /tmp/dsv4_auto_hcnorm_compare_generic_n40.log
keys checked: norm, cur, pre, postw, comb, post
max_abs=0
max_rel=0
mean_abs=0
rms_err=0
over_tol=0
```

Interpretation:

- First correctness-stable local fusion that moved n400 wall-clock.
- Dispatch reduction is about one fused HC-pre/norm site per attention and FFN block per decode layer.
- Still not enough to close the DS4 gap.

#### M5 Simdgroup-Matrix Barrier Removal

Status: correct and implemented, but inactive for the target DSV4 decode path by default.

Implemented Swival/ds4-m5 style function constant:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_M5_SGMATRIX=1
FC_MUL_MM_M5_SGMATRIX
mm_m5sg counter
```

It skips no-op `simdgroup_barrier(mem_flags::mem_none)` calls in the simdgroup-matrix `mul_mm` / `mul_mm_id` path on M5 when tensor cores are not being used.

Target DSV4 n400 result:

```text
default M5 tensor route:
  has tensor=true
  mm=0
  gen_mm=0
  mm_m5sg=0
```

Forced simdgroup prefill test with `GGML_METAL_TENSOR_DISABLE=1`:

```text
off:
  mm=2548
  gen_mm=387
  mm_m5sg=0
  prompt=130.4 t/s
  generation=20.3 t/s

on:
  mm=2548
  gen_mm=387
  mm_m5sg=2935
  prompt=130.5 t/s
  generation=19.3 t/s
  generated text: exact match
```

Interpretation:

- Good code-level parity with ds4-m5 for the simdgroup matrix path.
- Not a DSV4 decode win because the fixed smoke path uses the M5 tensor route and has no `mm` / `gen_mm` work.

### Correct But Performance-Neutral / Support Only

#### Q8HC

Status: keep guarded; not a performance path.

With `HC_PRE_NORM + Q8HC`:

```text
log: /tmp/dsv4_hcnorm_q8hc_n400.log
dsv4_q8hc=16946
metal_dispatch=1382093
generation=21.2 tok/s
transcript: exact match
```

Compared to `HC_PRE_NORM` alone:

```text
HC_PRE_NORM alone:
  metal_dispatch=1399039
  generation=21.4 tok/s in that paired run

HC_PRE_NORM + Q8HC:
  metal_dispatch=1382093
  generation=21.2 tok/s
```

Interpretation: correct and reduces dispatch, but does not improve wall-clock.

### Rejected / Unsafe / Not Useful

#### HC_EXPAND4

Status: rejected in the current stack.

Short n80 was transcript-correct but slower:

```text
log: /tmp/dsv4_hcnorm_hce4_n80.log
dsv4_hce4=7052
generation=20.2 tok/s
```

n400 failed with Metal OOM:

```text
log: /tmp/dsv4_hcnorm_hce4_n400.log
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
failed after decode=291
```

#### Safe CUPD / Compressor Update

Status: rejected in the current stack.

With `HC_PRE_NORM + DSV4_COMPRESSOR_UPDATE`:

```text
log: /tmp/dsv4_hcnorm_cupd_n80.log
dsv4_cupd=310
error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
failed after decode=5
```

Do not reuse the existing CUPD path as-is. The next compressor/update attempt should be a fresh compare-gated whole-stage implementation.

#### Weighted SwiGLU

Status: rejected for correctness.

```text
log: /tmp/dsv4_hcnorm_wpswiglu_n80.log
wpswiglu=158
generation=22.0 tok/s
transcript: drifted immediately
```

#### Shared SwiGLU

Status: rejected for correctness.

```text
log: /tmp/dsv4_hcnorm_shswiglu_n80.log
shswiglu=3445
generation=21.5 tok/s
transcript: drifted immediately
```

#### Down Sum6

Status: rejected for n400 correctness.

n80 looked promising:

```text
log: /tmp/dsv4_hcnorm_dsum6_n80.log
dec_mv=0
dsum6=237
generation=22.1 tok/s
transcript: exact match
```

n400 drifted:

```text
log: /tmp/dsv4_hcnorm_dsum6_n400.log
dec_mv=0
dsum6=1197
generation=21.2 tok/s
transcript: drifted
```

2026-05-13 retry:

The Q2_K sum6 kernel was adjusted to keep six per-expert accumulators and add the completed expert dot products in the same left-to-right order as the generic graph. This fixed the early visible drift but did not make the path n400-exact.

```text
n80:
  log: /tmp/dsv4_ffn/down_sum6_ordered_n80.log
  dec_mv=0
  dsum6=237
  generation=21.2 tok/s
  visible prefix: pass

n400:
  log: /tmp/dsv4_ffn/down_sum6_ordered_n400.log
  dec_mv=0
  dsum6=1197
  metal_dispatch=1391857
  generation=21.0 tok/s
  transcript: still drifted, later than before
```

Observed first meaningful text diff vs `/tmp/dsv4_kvfin/baseline_split_glu_n400.log`:

```text
baseline: far more efficient than a general-purpose CPU or even a GPU for these tasks.
sum6:     far more efficient than a general-purpose CPU or GPU for these tasks.
```

2026-05-13 FFN/MoE stage diagnostic:

Added a compare-gated diagnostic op:

```text
GGML_OP_DSV4_FFN_MOE_DECODE_STAGE

flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE_CONSUME=generic|fused
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_TRACE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_CONSUME=1  # explicit unsafe consume gate

counter:
  dsv4_ffnmoe
```

The op computes routed `down` for all six selected experts and writes cumulative partials:

```text
slot0
slot0 + slot1
slot0 + slot1 + slot2
...
slot0 + ... + slot5
```

Compare results:

```text
compare generic:
  log: /tmp/dsv4_ffnmoe/compare_generic_n80.log
  dsv4_ffnmoe=3397 = 43 layers * 79 decode tokens
  max_abs=0, max_rel=0, mean_abs=0, rms_err=0, over_tol=0 for logged partial/final compares
  no over_tol>0 lines in the full log

compare fused:
  log: /tmp/dsv4_ffnmoe/compare_fused_n80.log
  dsv4_ffnmoe=3397
  visible prefix: pass
  no over_tol>0 lines in the full log
```

This proves the previous DOWN_SUM6 drift was not caused by the Q2_K down/add-tree arithmetic itself when compared inside the same graph. The fused partials match the generic `ffn_moe_down`/add-tree output exactly.

However, the no-generic consume path is still rejected:

```text
unsafe consume, n400:
  log: /tmp/dsv4_ffnmoe/stage_cont_n400.log
  dsv4_ffnmoe=17157 = 43 layers * 399 decode tokens
  dec_mv: 1197 -> 0
  gen_mv: 48267 -> 32307
  metal_dispatch: 1399039 -> 1391458
  pair: 399 -> 0
  pswiglu: 798 -> 1197
  generation: 18.5 -> 18.2 tok/s in this paired session
  transcript: drifted
```

The key finding is the upstream graph-shape side effect: removing the generic routed-down branch changes split-GLU pairing (`pair` drops to zero and `pswiglu` rises), so the graph is no longer numerically equivalent even though the down/add-tree partials are exact in compare mode.

To avoid leaving a known drifting path armed, plain `FFN_MOE_STAGE=1` no longer inserts the stage unless compare mode is enabled. The actual fused-consume path now also requires:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_CONSUME=1
```

Safe plain-flag smoke:

```text
log: /tmp/dsv4_ffnmoe/stage_flag_safe_n80.log
dsv4_ffnmoe=0
pair=79
pswiglu=158
visible prefix: baseline-style pass
```

Classification: diagnostic accepted, performance consume rejected. The next FFN/MoE attempt should not be another down-sum variant. It must own the full DS4-style boundary:

```text
route/top-k -> selected gate/up -> limited SwiGLU -> down -> weighted sum -> output
```

That is the only way to remove the generic down branch without perturbing the upstream split-GLU matcher into a different graph.

#### Weighted Down

Status: did not activate on the target graph.

```text
log: /tmp/dsv4_hcnorm_wdown_n80.log
wdsum6=0
```

#### KV Set Rows Peephole

Status: not viable as an adjacent-node matcher.

Trace with:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE=1
```

showed that the producer/store sequence is not reliably contiguous:

```text
KVrope-0 next=CPY n2=DSV4_FP8_KV_QUANTIZE/KVcur-0 n3=SET_ROWS/cache_k_l0
KVrope-3 next=DSV4_FP8_KV_QUANTIZE n2=SCALE/cache_r_l3 n3=SCALE/cache_s_l3
```

Interpretation: this needs graph-builder changes or a dedicated KV finalizer/cache-store op with explicit dependency semantics.

## Target Benchmark

Model:

```text
/Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
```

Primary command:

```sh
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1 ./build/bin/llama-cli \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx-size 1024 \
  --reasoning off \
  --temp 0 \
  --seed 42 \
  -n 400 \
  -p 'What is Apple Neural engine?' \
  --moe-mode stock \
  --moe-topk 6 \
  --no-warmup \
  --perf -st
```

Constraints:

- Apple M5 Max Metal.
- No SSD offload.
- Full MoE top-k: `--moe-topk 6`.
- `--moe-mode stock`.
- Fixed prompt, seed, temp for smoke comparison.
- Correctness matters. Several faster attention variants changed the text after a short sample.

## Current Outcome

The current accepted path is no longer completely flat: `HC_PRE_NORM` is the first correctness-stable wall-clock win.

Latest accepted n400 run with only the target split-GLU flag:

```text
/tmp/dsv4_split_glu_auto_hcnorm_n400.log
[ Prompt: 53.9 t/s | Generation: 21.9 t/s ] [ tokens - prefill: 10, decode: 399 ]
```

Counters from that run:

```text
mul_mat mv_ext=1956 mv=221848 mm=0 mm_m5_expert=0 mm_m5sg=0 shswiglu=0 dsv4_cpair=0 dsv4_q8hc=0
mul_mat_id dec_mv=1197 pair=399 pswiglu=798 wpswiglu=0 dsum6=0 wdsum6=0 shswiglu=0
gen_mv=48267 gen_mm=0 metal_dispatch=1399039 fglu=798
replay_hit=0 replay_miss=0 replay_ins=0 replay_clr=0 icb_exec=0 icb_fail=0
dsv4_rope_hfp4=2142 dsv4_rope_fp8=2684 dsv4_kvset=0
dsv4_hcws=34314 dsv4_hcnorm=34314 dsv4_hce4=0 dsv4_iscore=42
dsv4_dcomp=0 dsv4_iattn=0 dsv4_aolow=17286 dsv4_aodec=0 dsv4_cupd=0
replay_cache=0
```

Forced-off paired control:

```text
/tmp/dsv4_split_glu_hcnorm_forced_off_n400.log
[ Prompt: 52.5 t/s | Generation: 21.1 t/s ] [ tokens - prefill: 10, decode: 399 ]
dsv4_hcnorm=0
metal_dispatch=1433353
```

The generated response body exactly matched the accepted baseline in both runs.

No-flag fallback run from earlier:

```text
/tmp/dsv4_no_split_fallback_n40.log
[ Prompt: 50.3 t/s | Generation: 20.7 t/s ] [ tokens - prefill: 10, decode: 39 ]
```

Fallback counters correctly keep experimental DSV4 counters at zero:

```text
fglu=0 dsv4_rope_hfp4=0 dsv4_rope_fp8=0 dsv4_kvset=0 dsv4_hcws=0 dsv4_iscore=0 dsv4_aolow=0
```

## External DS4 Comparison

The external DS4 repo should be treated as the reference design, not as code to copy blindly.

Relevant files:

```text
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/ds4.c
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/ds4_metal.h
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/ds4_metal.m
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/metal/moe.metal
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/metal/dsv4_kv.metal
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/metal/dsv4_misc.metal
```

Important DS4 entry points observed:

- Decode routed MoE call site: `ds4.c` around `9306`, calls `ds4_metal_routed_moe_one_tensor(...)`.
- Prefill/batch routed MoE call site: `ds4.c` around `12020`, calls `ds4_metal_routed_moe_batch_tensor(...)`.
- Decode KV finalizer: `ds4_metal_kv_fp8_store_raw_tensor(...)`.
- Attention output APIs: `ds4_metal_attention_output_q8_batch_tensor(...)` and `ds4_metal_attention_output_low_q8_tensor(...)`.
- Indexed mixed attention kernels: `kernel_dsv4_indexed_mixed_attention_heads8` and `kernel_dsv4_indexed_mixed_attention_heads8_rb4`.

Earlier local DS4 comparison showed roughly 33.7 tok/s for the same model/prompt class, while this ggml path remains near 20.5 tok/s. The exact DS4 number should be remeasured before making final claims, but the gap is large enough that noise cannot explain it.

DS4's structural difference is important: it does not rely on generic `mul_mat`, `mul_mat_id`, `rope`, `set_rows`, `mask`, `flash_attn`, `reshape`, `view`, and `concat` graph composition for the hot decode path. It uses dedicated model-specific Metal routines for entire stages.

## Current Graph Shape In This Repo

Primary model graph is in:

```text
src/models/deepseek4.cpp
```

### Attention Q/KV Setup

Relevant block: `src/models/deepseek4.cpp` around lines 1188 to 1230.

Current decode path performs:

```text
hc_attn_pre
attn_norm
q_lora = mul_mat(wq_a, cur)
q_lora_norm
q = mul_mat(wq_b, qr)
reshape q
Q RMS norm
Q RoPE tail

kv = mul_mat(attn_kv, cur)
KV norm
reshape kv
KV RoPE tail
FP8 KV quantize/dequantize
copy/store KV into SWA cache through cpy_k / set_rows
```

The current source has:

```cpp
kv = dsv4_apply_rope_tail(...);
cb(kv, "KVrope", il);
kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
cb(kv, "KVcur", il);

ggml_build_forward_expand(gf, q);
ggml_build_forward_expand(gf, kv);
ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
```

This ordering matters: the graph does not always keep `KVrope -> KVcur -> set_rows` contiguous in the actual Metal execution order, especially in compressed layers.

### Attention Indexer / Compressed Mask

Relevant block: `src/models/deepseek4.cpp` around lines 1769 to 1855.

For compression ratio 4, the graph builds indexer scores, top-k, and masks:

```text
index_scores
topk
optional top-k KV gather
compressed mask from top-k
concat raw and compressed masks
```

The existing `dsv4_indexer_weighted_score` path activates:

```text
dsv4_iscore=42
```

But this is a small part of the total cost. It does not move tok/s.

### Attention Core

Relevant block: `src/models/deepseek4.cpp` around lines 1995 to 2011.

The path currently either:

- Uses experimental `ggml_dsv4_mixed_attn(...)`, if enabled and graph conditions match.
- Or falls back to `build_attn_mha(...)` with concatenated raw/compressed K/V and mask.

The experimental mixed attention op was tested and did not improve speed. It also is not equivalent to DS4's indexed mixed attention implementation.

### Attention Output

Relevant block: `src/models/deepseek4.cpp` around lines 493 to 520.

Current function:

```cpp
static ggml_tensor * dsv4_grouped_out(...) {
    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids);
    ggml_set_name(low, "dsv4_attn_out_low");
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);

    return ggml_mul_mat(ctx, wo_b, low);
}
```

There is now a direct Metal fast path for the low projection (`dsv4_aolow`), but the high projection remains generic `ggml_mul_mat(ctx, wo_b, low)`.

This changed counters but not wall time:

```text
attn_out_low off:
  gen_mv=65553 dsv4_aolow=0 generation=20.5 tok/s

attn_out_low on:
  gen_mv=48267 dsv4_aolow=17286 generation=20.5 tok/s
```

This is one of the clearest examples that a local kernel replacement is not enough.

## What Was Tried

### 1. MoE Graph Recognition And Fusion

Original problem:

```text
fglu=0 pair=0 replay_hit=0
```

The graph matcher was extended earlier so DeepSeek V4 Flash routed gate/up fusion activates. Current n400 counters show:

```text
pair=399
pswiglu=798
fglu=798
```

This proves the existing M5 split-GLU/routed path is active. It did not move decode speed materially by itself.

### 2. Attention Output Low Direct Path

Added direct grouped low projection for `dsv4_attn_out_low`.

Result:

```text
dsv4_aolow=17286
gen_mv reduced from 65553 to 48267
generation stayed around 20.5 tok/s
```

Interpretation:

The low grouped projection was not the limiting cost by itself, or the remaining high projection/launch structure dominates.

### 3. Compressor Pair Projection

Added explicit experimental op:

```text
GGML_OP_DSV4_COMPRESSOR_PAIR_PROJ
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_PAIR=1
```

Representative n400 results:

```text
off:
  dsv4_cpair=0
  generation=20.4 tok/s

on:
  dsv4_cpair=24738
  generation=20.3 tok/s
```

Interpretation:

It reduced some generic matmul work but added/kept enough graph overhead that the wall clock did not improve.

### 4. Decode Compressor Fusion

Existing experimental op:

```text
GGML_OP_DSV4_DECODE_COMPRESS
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_COMPRESS=1
```

Result:

- `dsv4_dcomp` activated in earlier runs.
- Performance was neutral.
- A 400-token deterministic transcript diverged later.
- It remains disabled by default.

Interpretation:

This path is unsafe until it has a stronger correctness comparison. It also did not demonstrate speed.

### 5. Mixed Attention Op

Existing experimental op:

```text
GGML_OP_DSV4_MIXED_ATTN
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN=1
```

Result:

- Short prefix was usually correct.
- n400 was neutral/slower.
- It is not equivalent to DS4's indexed mixed attention kernel.
- It remains disabled by default.

Interpretation:

The DS4 speedup probably comes from its indexed/masked attention design and data layout, not just concatenating raw and compressed K/V inside one generic-ish kernel.

### 6. Top-K Attention Gather

Experimental flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_TOPK_ATTN_GATHER=1
```

Result:

- No material improvement on n80 or n400.
- Remains disabled.

### 7. Metal4 FlashAttention

Tested:

```text
GGML_METAL_FLASH_ATTN_VEC_M4_ENABLE=1
```

Result:

- Existing Metal4 vec path did not activate for DSV4 decode because the graph presents `dk=512`, `dv=512`.
- Existing M4 vec specialization is for `dk=128`, `dv=128`.
- A previous attempt to add/use a 512-wide M4 path hit threadgroup memory limits/asserts and was reverted.

Observed path with debug:

```text
path=vec+2pass phase=decode-like q=1 kv=... dk=512 dv=512 nsg=1 nwg=...
```

### 8. FlashAttention Workgroup Split Sweep

Swept:

```text
GGML_METAL_FLASH_ATTN_VEC_NWG=16
GGML_METAL_FLASH_ATTN_VEC_NWG=8
GGML_METAL_FLASH_ATTN_VEC_NWG=4
GGML_METAL_FLASH_ATTN_VEC_NWG=2
GGML_METAL_FLASH_ATTN_VEC_NWG=1
```

Result:

- `nwg=16` preserved the short smoke text and is now the default when `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1` is set and no explicit NWG env is provided.
- `nwg=1` and `nwg=8` showed visible smoke text drift in short runs.
- `nwg=16` is a tiny/noisy movement only. It does not solve the problem.

Relevant current code:

```text
ggml/src/ggml-metal/ggml-metal-ops.cpp
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_NWG
```

Current behavior:

```text
if split-GLU experimental flag is active and no explicit NWG is set:
  decode FlashAttention uses nwg=16
else:
  existing default behavior remains
```

### 9. KV FP8 Quantize + Cache Set Rows Prototype

Added an opt-in prototype:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE=1
dsv4_kvset counter
kernel_dsv4_rope_fp8_kv_set_rows_...
```

It is disabled by default.

Why it did not activate:

The actual graph often does not have a simple contiguous:

```text
DSV4_ROPE_TAIL -> DSV4_FP8_KV_QUANTIZE -> VIEW? -> SET_ROWS
```

Trace examples showed:

```text
KVrope-0 next=CPY n2=DSV4_FP8_KV_QUANTIZE/KVcur-0 n3=SET_ROWS/cache_k_l0
KVrope-3 next=DSV4_FP8_KV_QUANTIZE n2=SCALE/cache_r_l3 n3=SCALE/cache_s_l3
```

Interpretation:

The graph schedule has interleaved cache/state operations, views, or copies before the store. A safe adjacent-node fusion will not catch this shape. A real KV finalizer likely needs graph construction changes or a dedicated side-effect op that owns both producing `KVcur` and writing the cache.

### 10. HC_PRE_NORM

Added/validated the DS4-style HC-pre + learned RMSNorm boundary:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE_CONSUME=generic|fused
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_SCOPE=attn|ffn|all
```

Current behavior:

- Enabled automatically when `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1`.
- Can be forced off with `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM=0`.
- Produces `dsv4_hcnorm` counter.

Result:

```text
forced off:
  dsv4_hcnorm=0
  metal_dispatch=1433353
  generation=21.1 tok/s

enabled:
  dsv4_hcnorm=34314
  metal_dispatch=1399039
  generation=21.9 tok/s
```

Correctness:

- n400 transcript exact-match.
- compare-mode tensor checks for `norm`, `cur`, `pre`, `postw`, `comb`, and `post` produced zero error in the logged sample.

Interpretation:

This is the first accepted structural local win. It should remain on under the split-GLU experimental umbrella unless a broader DSV4 decode executor supersedes it.

### 11. M5 Simdgroup Matrix Barrier Removal

Implemented a Swival/ds4-m5 style function constant:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_M5_SGMATRIX=1
FC_MUL_MM_M5_SGMATRIX
mm_m5sg counter
```

Result:

- Correct and active when forcing the old simdgroup matrix path with `GGML_METAL_TENSOR_DISABLE=1`.
- Inactive for the target DSV4 n400 decode path because M5 uses the tensor path and counters show `mm=0`, `gen_mm=0`.

Interpretation:

Keep it as a guarded code-level M5 improvement, but do not expect it to affect the fixed DSV4 decode smoke test.

### 12. Post-HC / MoE Follow-Up Flags

The following were tested after `HC_PRE_NORM` became correct:

```text
Q8HC:
  correct n400, dsv4_q8hc=16946, dispatch lower, generation neutral/slower

HC_EXPAND4:
  n80 correct but slower, n400 Metal OOM

COMPRESSOR_UPDATE:
  Metal OOM within a few decode tokens when stacked with HC_PRE_NORM

WEIGHTED_SWIGLU:
  activates but immediate transcript drift

SHARED_SWIGLU:
  activates but immediate transcript drift

DOWN_SUM6:
  ordered-accumulator retry improved early text but still drifts at n400

WEIGHTED_DOWN:
  did not activate
```

Do not promote these as performance paths without first fixing their correctness/resource issues.

## Stage Profile Evidence

Profiling env:

```sh
LLAMA_FLASH_MOE_METAL_STAGE_PROFILE=1
LLAMA_FLASH_MOE_METAL_STAGE_PROFILE_DETAIL=1
```

Current short profile:

```text
/tmp/dsv4_stage_detail_nwg16_n8.log
```

Aggregated stage totals from that run:

```text
ffn            count=430 exec=246.126 total=269.182 avg=0.626
attn_out       count=860 exec=196.518 total=202.921 avg=0.236
attn_compress  count=533 exec=165.244 total=180.259 avg=0.338
attn_kv        count=840 exec=145.010 total=155.447 avg=0.185
attn_qkv       count=533 exec=129.341 total=137.086 avg=0.257
attn_hc_pre    count=430 exec=107.014 total=111.542 avg=0.259
attn_core      count=650 exec=72.825  total=76.981  avg=0.118
attn_hc_post   count=430 exec=71.428  total=74.891  avg=0.174
head           count=10  exec=10.589  total=10.744  avg=1.074
other          count=43  exec=21.064  total=24.085  avg=0.560
attn_index     count=42  exec=6.785   total=7.868   avg=0.187
```

Caveats:

- This profiler serializes/segments execution and is intrusive.
- Use relative ordering, not the reported tok/s, as the signal.

What the profile suggests:

- `attn_core` itself is not the largest bucket.
- `attn_out`, `attn_compress`, `attn_kv`, and `attn_qkv` remain significant.
- FFN/MoE is also still large, despite routed fusion counters being active.
- The remaining gap likely needs whole-stage fusion, not smaller node-level rewrites.

## Why The Small Patches Did Not Fix It

The failed premise was: "If we activate enough local M5 Metal kernels, tok/s will move."

The data says that premise is mostly wrong for this graph shape. `HC_PRE_NORM` is the one important exception so far because it fused a real DS4-style stage boundary and moved n400 wall-clock without drift.

Observed pattern:

1. A local fusion activates.
2. A counter changes substantially.
3. Some generic op count drops.
4. Wall-clock tok/s remains flat or regresses.

Likely reasons:

- The graph has many small operations per layer, so launch/scheduler overhead remains.
- Some fusions reduce one sub-node but leave a generic high-cost consumer immediately after it.
- Cache stores and state updates are not adjacent to producers in the execution graph.
- DSV4 decode has model-specific structure that generic ggml op composition cannot exploit well.
- DS4's faster path uses dedicated routines that cross ggml node boundaries.
- Correctness is sensitive: some apparently faster attention settings produce transcript drift.

## Current Code State Relevant To Handoff

The local tree is dirty and includes multiple experimental changes. Do not assume every dirty file belongs to the attention attempt.

Most relevant files:

```text
src/models/deepseek4.cpp
ggml/include/ggml.h
ggml/src/ggml.c
ggml/src/ggml-cpu/ops.h
ggml/src/ggml-cpu/ops.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-metal/ggml-metal-context.m
ggml/src/ggml-metal/ggml-metal-device.h
ggml/src/ggml-metal/ggml-metal-device.cpp
ggml/src/ggml-metal/ggml-metal-device.m
ggml/src/ggml-metal/ggml-metal-impl.h
ggml/src/ggml-metal/ggml-metal-ops.h
ggml/src/ggml-metal/ggml-metal-ops.cpp
ggml/src/ggml-metal/ggml-metal.metal
```

Useful counters now printed in Metal stats (`ggml_metal_op_mul_mat_log_stats` and `ggml_metal_op_mul_mat_id_log_stats`):

```text
dsv4_rope_hfp4
dsv4_rope_fp8
dsv4_kvset
dsv4_hcws
dsv4_hcnorm
dsv4_hce4
dsv4_q8hc
dsv4_cpair
dsv4_iscore
dsv4_dcomp
dsv4_iattn
dsv4_aolow
dsv4_aodec
dsv4_cupd
metal_dispatch
mm_m5sg
```

Important flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM=0|1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE_CONSUME=generic|fused
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_SCOPE=attn|ffn|all
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_M5_SGMATRIX=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8_HC_EXPAND=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_EXPAND4=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_NWG=<1|2|4|8|16|32>
GGML_METAL_FLASH_ATTN_VEC_NWG=<1|2|4|8|16|32>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISPATCH_PROFILE=1
LLAMA_FLASH_MOE_METAL_STAGE_PROFILE=1
LLAMA_FLASH_MOE_METAL_STAGE_PROFILE_DETAIL=1
GGML_METAL_FLASH_ATTN_DEBUG=1
```

### 2026-05-13 Partial Logit / First-Divergence Harness Update

Status: partially implemented and validated on n80, but not yet accepted as a complete validation tool.

This pass started adding a DSV4 logit / first-divergence harness intended to answer the current strategic question:

```text
Should deterministic transcript drift remain a hard failure,
or can some DS4-style whole-stage paths be accepted under a formal logit/tolerance policy?
```

Motivation:

- Several paths are tensor-close or locally exact but still drift at n400.
- FFN/MoE V2 now owns the intended routed boundary and removes routed down/split-GLU pieces, but tiny non-bitwise FFN differences still alter deterministic n400 text.
- `CUPD2_FUSED_COMP=1` had sub-1e-6 compressor-row differences in compare mode but still drifted in n400.
- Text-only validation is too weak to diagnose whether divergence is a serious numerical bug or a near-tie sampling/logit-margin issue.

Current harness status:

```text
implemented: partial
validated: n80 baseline vs FFN/MoE V2
not completed: n400 validation
no llama-cli process is still running
```

Known validation completed before interruption:

```text
n80 baseline vs FFN/MoE V2:
  completed
  visible prefix stable
  first-divergence/logit infrastructure exercised

n400:
  interrupted before completion
  no result should be claimed
```

Classification:

```text
diagnostic only
not an accepted correctness policy yet
not enough to accept any drifting performance path
```

Do not use the partial harness result to accept:

```text
FFN/MoE V2 full consume
CUPD2_FUSED_COMP
DOWN_SUM6
weighted/shared SwiGLU
mixed-attention variants
```

until n400 logit/first-divergence validation is completed.

The next validation task is to finish the harness and run paired n400 comparisons for:

1. accepted baseline
2. FFN/MoE V2 full consume
3. CUPD2_FUSED_COMP
4. optionally DOWN_SUM6 / weighted SwiGLU / shared SwiGLU

Required output per token:

```text
token_index
position
baseline token id/text
fused token id/text
baseline top1 id/logit
baseline top2 id/logit
baseline top1-top2 margin
fused top1 id/logit
fused top2 id/logit
fused top1-top2 margin
baseline top1 rank in fused distribution
fused top1 rank in baseline distribution
top-k overlap count
max_abs_logit_err
mean_abs_logit_err
rms_logit_err
active DSV4 counters
active experimental flags
```

Required first-divergence report:

```text
first divergent token index:
baseline token:
fused token:
baseline top1/top2 margin:
fused top1/top2 margin:
baseline top1 rank in fused:
fused top1 rank in baseline:
max_abs_logit_err:
rms_logit_err:
top-k overlap:
active stage/counters at divergence:
interpretation:
  near-tie / small-logit-delta / real numeric mismatch / missing probe boundary
```

Acceptance policy is not yet defined.

Until a formal logit/tolerance policy is adopted, deterministic n400 transcript drift remains a hard rejection for performance paths.

#### Codex Resume Task

Give Codex this as the next work item:

```text
Resume from the partial DSV4 logit / first-divergence harness.
The main handoff is already updated through FFN/MoE V2 and says local fusions are mostly exhausted. The missing piece is the interrupted logit harness work. Do not start another kernel. Finish the validation harness first.

Objective

Complete the DSV4 logit / first-divergence harness and run n400 paired validation.

The goal is to answer:

Are rejected drifting paths numerically wrong,
or are they close enough that transcript-exact validation is too strict?

First priority paths

Run in this order:

1. accepted baseline
2. FFN/MoE V2 full consume
3. CUPD2_FUSED_COMP
4. DOWN_SUM6, if time remains

Required checks before running

cmake --build build --target llama-cli -j 8
git diff --check

Baseline n400 logit dump

Use the actual implemented logit-harness flags. If the flag names match the earlier plan, use:

LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK=20 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP=/tmp/dsv4_logits_baseline_n400.jsonl \
./build/bin/llama-cli \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx-size 1024 \
  --reasoning off \
  --temp 0 \
  --seed 42 \
  -n 400 \
  -p 'What is Apple Neural engine?' \
  --moe-mode stock \
  --moe-topk 6 \
  --no-warmup \
  --perf -st \
  2>&1 | tee /tmp/dsv4_logits_baseline_n400.log

FFN/MoE V2 n400 logit dump

LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK=20 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP=/tmp/dsv4_logits_ffnmoe_v2_n400.jsonl \
./build/bin/llama-cli \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx-size 1024 \
  --reasoning off \
  --temp 0 \
  --seed 42 \
  -n 400 \
  -p 'What is Apple Neural engine?' \
  --moe-mode stock \
  --moe-topk 6 \
  --no-warmup \
  --perf -st \
  2>&1 | tee /tmp/dsv4_logits_ffnmoe_v2_n400.log

CUPD2 fused-compressor n400 logit dump

LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK=20 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP=/tmp/dsv4_logits_cupd2_fused_comp_n400.jsonl \
./build/bin/llama-cli \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx-size 1024 \
  --reasoning off \
  --temp 0 \
  --seed 42 \
  -n 400 \
  -p 'What is Apple Neural engine?' \
  --moe-mode stock \
  --moe-topk 6 \
  --no-warmup \
  --perf -st \
  2>&1 | tee /tmp/dsv4_logits_cupd2_fused_comp_n400.log

Required analyzer output

The current partial analyzer is `scripts/dsv4_logit_compare.py`. Complete or rename it if needed, but the current command shape is:

python3 scripts/dsv4_logit_compare.py \
  /tmp/dsv4_logits_baseline_n400.jsonl \
  /tmp/dsv4_logits_ffnmoe_v2_n400.jsonl \
  --out /tmp/dsv4_first_divergence_ffnmoe_v2.txt \
  --stop-at-first-diff

python3 scripts/dsv4_logit_compare.py \
  /tmp/dsv4_logits_baseline_n400.jsonl \
  /tmp/dsv4_logits_cupd2_fused_comp_n400.jsonl \
  --out /tmp/dsv4_first_divergence_cupd2_fused_comp.txt \
  --stop-at-first-diff

The report must include:

first divergent token index
baseline token id/text
fused token id/text
baseline top1 id/logit
baseline top2 id/logit
baseline top1-top2 margin
fused top1 id/logit
fused top2 id/logit
fused top1-top2 margin
baseline top1 rank in fused
fused top1 rank in baseline
top10/top20 overlap
max_abs_logit_err
mean_abs_logit_err
rms_logit_err
active counters
active flags

Interpretation rules

Use this decision tree:

Case A:
  first divergence occurs at a tiny top1/top2 margin
  baseline top1 is still high-rank in fused
  fused top1 is still high-rank in baseline
  RMS logit error is tiny
Interpretation:
  likely validation-policy issue.
  do not accept automatically, but this supports defining a logit tolerance policy.

Case B:
  first divergence occurs with a large baseline margin
  baseline top1 falls far down in fused ranking
  max/rms logit error is large
Interpretation:
  real numerical bug.
  path remains rejected.

Case C:
  no divergence through n400
  logit errors remain tiny
Interpretation:
  path may be promoted from rejected to candidate, then rerun standard perf smoke.

Case D:
  divergence is late but logit error grows steadily
Interpretation:
  accumulated stage error.
  path remains rejected unless the source stage can be isolated.

Required handoff update after completion

At the end, update:

docs/dsv4-m5-metal-performance-handoff.md

Add:

### 2026-05-13 Logit / First-Divergence Harness Result

with:

flags implemented
files changed
n80 status
n400 status
baseline log path
fused log paths
first divergence reports
whether any path qualifies for logit-tolerance discussion
whether transcript-exact remains the active policy

Do not mark any drifting path accepted unless an explicit tolerance policy is written down and approved.
```

### 2026-05-13 Logit / First-Divergence Harness Result

Status: completed for the requested n400 baseline, FFN/MoE V2 full-consume, CUPD2 fused-compressor, and optional DOWN_SUM6 cases.

Files changed for the harness:

```text
tools/cli/cli.cpp
tools/server/server-context.cpp
tools/server/server-task.cpp
tools/server/server-task.h
scripts/dsv4_first_divergence.py
scripts/dsv4_logit_compare.py
docs/dsv4-m5-metal-performance-handoff.md
```

Implemented/used logit dump flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK=20
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP=/tmp/...
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TRACE_STAGES=1
```

Analyzer:

```text
scripts/dsv4_first_divergence.py
```

The analyzer compares sampled/generated token ids from two JSONL dumps, reports the first generated-token divergence, computes top10/top20 overlap, ranks each run's top1 in the other run's top-k list, parses active flags from JSONL, and parses Metal counters from sibling `.log` files when present.

Important limitation:

```text
max_abs_logit_err / mean_abs_logit_err / rms_logit_err are computed over overlapping tokens in the dumped top20 lists, not the full vocabulary.
```

Build/checks:

```text
cmake --build build --target llama-cli -j 8: pass
git diff --check: pass
markdown fence sanity: pass
```

n80 smoke:

```text
baseline jsonl: /tmp/dsv4_logit_harness/baseline_n80.jsonl
baseline log:   /tmp/dsv4_logit_harness/baseline_n80.log
FFN/MoE V2 jsonl: /tmp/dsv4_logit_harness/ffnmoe_v2_n80.jsonl
FFN/MoE V2 log:   /tmp/dsv4_logit_harness/ffnmoe_v2_n80.log
report: /tmp/dsv4_logit_harness/first_divergence_ffnmoe_v2_n80.txt

records:
  baseline: 80 logits records + metadata
  FFN/MoE V2: 80 logits records + metadata

visible prefix: stable
first divergence: none through n80
last-token top20 overlap: 19/20
last-token max_abs top20-overlap logit err: 0.213277817
last-token rms top20-overlap logit err: 0.107133587
classification: no token divergence through n80, but logits already differ measurably
```

n400 accepted baseline:

```text
log:   /tmp/dsv4_logit_harness/baseline_n400.log
jsonl: /tmp/dsv4_logit_harness/baseline_n400.jsonl
records: 400 logits records + metadata
generation: 16.9 tok/s with logit dump enabled
metal_dispatch=1399039
dec_mv=1197
pair=399
pswiglu=798
fglu=798
dsv4_hcnorm=34314
dsv4_hcws=34314
dsv4_ffnmoe=0
dsv4_cupd2=0
transcript: accepted baseline text
```

n400 FFN/MoE V2 full consume:

```text
log:    /tmp/dsv4_logit_harness/ffnmoe_v2_n400.log
jsonl:  /tmp/dsv4_logit_harness/ffnmoe_v2_n400.jsonl
report: /tmp/dsv4_logit_harness/first_divergence_ffnmoe_v2_n400.txt
records: 400 logits records + metadata
generation: 16.0 tok/s with logit dump enabled
metal_dispatch=1374301
dec_mv=0
pair=0
pswiglu=0
fglu=0
dsv4_ffnmoe=34314

first divergent token index: 104
baseline token: 305 " and"
fused token:    343 " ("
baseline top1/top2 margin: 0.141532898
fused top1/top2 margin:    0.0713691711
baseline top1 rank in fused: 2
fused top1 rank in baseline: 2
top10 overlap: 10/10
top20 overlap: 20/20
max_abs top20-overlap logit err: 0.422439575
mean_abs top20-overlap logit err: 0.113186741
rms top20-overlap logit err: 0.150950382
classification: Case A, likely tolerance-policy issue; do not accept automatically
```

n400 CUPD2 fused-compressor:

```text
log:    /tmp/dsv4_logit_harness/cupd2_fused_comp_n400.log
jsonl:  /tmp/dsv4_logit_harness/cupd2_fused_comp_n400.jsonl
report: /tmp/dsv4_logit_harness/first_divergence_cupd2_fused_comp_n400.txt
records: 400 logits records + metadata
generation: 16.7 tok/s with logit dump enabled
metal_dispatch=1249275
dec_mv=1197
pair=399
pswiglu=798
fglu=798
dsv4_cupd2=24738
dsv4_rope_hfp4=42
dsv4_rope_fp8=524

first divergent token index: 104
baseline token: 305 " and"
fused token:    343 " ("
baseline top1/top2 margin: 0.141532898
fused top1/top2 margin:    0.127914429
baseline top1 rank in fused: 2
fused top1 rank in baseline: 2
top10 overlap: 10/10
top20 overlap: 20/20
max_abs top20-overlap logit err: 0.402101517
mean_abs top20-overlap logit err: 0.101707554
rms top20-overlap logit err: 0.128725288
classification: Case A, likely tolerance-policy issue; do not accept automatically
```

Optional n400 DOWN_SUM6:

```text
log:    /tmp/dsv4_logit_harness/down_sum6_n400.log
jsonl:  /tmp/dsv4_logit_harness/down_sum6_n400.jsonl
report: /tmp/dsv4_logit_harness/first_divergence_down_sum6_n400.txt
records: 400 logits records + metadata
generation: 16.6 tok/s with logit dump enabled
metal_dispatch=1391857
dec_mv=0
pair=399
pswiglu=798
fglu=798
dsum6=1197

first divergent token index: 35
baseline token: 7379 " built"
fused token:    343 " ("
baseline top1/top2 margin: 0.202125549
fused top1/top2 margin:    0.0388031006
baseline top1 rank in fused: 3
fused top1 rank in baseline: 2
top10 overlap: 9/10
top20 overlap: 20/20
max_abs top20-overlap logit err: 1.07374191
mean_abs top20-overlap logit err: 0.256354141
rms top20-overlap logit err: 0.350682504
classification: Case B, real numerical bug signal; path remains rejected
```

Interpretation:

- FFN/MoE V2 and CUPD2 fused-compressor both diverge at the same generated token (`104`) on the same near-tie (`" and"` vs `" ("`).
- In both cases, each run's top1 remains rank 2 in the other run, top20 overlap is perfect, and top20-overlap RMS error is moderate rather than catastrophic.
- This is strong evidence that at least some deterministic transcript drift is a validation-policy question, not a gross numerical failure.
- DOWN_SUM6 is less benign: divergence is earlier, top10 overlap drops to 9/10, and top20-overlap max/RMS errors are larger. Keep it rejected.

Current policy decision:

```text
No drifting path is accepted.
Transcript-exact n400 remains the active acceptance policy.
FFN/MoE V2 and CUPD2_FUSED_COMP qualify for tolerance-policy discussion only.
DOWN_SUM6 remains rejected under the current policy.
```

### 2026-05-13 Decode-Layer Executor Shadow Harness

Status: expanded shadow envelope implemented and validated through all-layer n80.

Purpose:

```text
Prove that this repo can construct, schedule, read back, and compare a DS4-style per-layer decode boundary without enabling another local performance fusion or consuming a drifting path.
```

Files changed in this pass:

```text
src/models/deepseek4.cpp
src/llama-context.cpp
docs/dsv4-m5-metal-performance-handoff.md
```

Implemented flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DUMP=/tmp/dsv4_lexec_shadow.jsonl
```

Implemented counter/log fields:

```text
dsv4_lexec=<compare_count>
dsv4_lexec_summary:
  enabled=1
  layers_requested=<all|n>
  tokens_requested=<all_decode|range>
  layer_output_compares=<count>
  expected_layer_output_compares=<count>
  over_tol=<count>
  total_compares=<count>
  transcript_consumed=generic
  consume_path=disabled
```

These are currently compare-log counters, not Metal stats counters. They are emitted by the layer-executor shadow compare callback and final process summary.

Current implementation shape:

```text
implemented coarse identity-shadow stages:
  attn_hc_pre_norm
  qkv_setup
  compressor_update, where existing safe decode compressor tensors are available
  kv_finalizer
  attention_output
  ffn_hc_pre_norm
  ffn_moe
  layer_output

not yet implemented:
  DS4-style fused per-layer executor semantics
  internal tensor probes for every sub-boundary inside each stage
  cache readback / cache mutation
  any consume path
```

Unsupported or unavailable stage sites are intentionally logged as:

```text
dsv4_layer_executor_shadow: layer=<n> token=<pos> stage=<stage> shadow=not_implemented reason=<reason>
```

The shadow path is an identity compare envelope:

```text
generic stage tensor -> ref probe
generic stage tensor -> shadow probe
compare ref vs shadow
feed generic value downstream
```

There is no consume path, no cache mutation, and no accepted performance path in this harness.

Important implementation note:

```text
The first version used a zero-dependency trick based on x - x to force side-branch execution.
That was removed because x - x becomes NaN for diagnostic tensors that legitimately contain NaNs,
which can poison the consumed graph when attached to layer output.

Current probes are added with ggml_build_forward_expand and do not inject any dependency into the
consumed generic path.
```

The optional JSONL dump emits one implemented compare per line plus a final summary row. Non-finite
metrics are serialized as JSON `null`, not bare `nan`.

Build/checks:

```text
cmake --build build --target llama-cli -j 8: pass
git diff --check: pass
python3 -m py_compile scripts/dsv4_first_divergence.py: pass
markdown fence sanity: pass
```

n16 one-layer shadow:

```text
log: /tmp/dsv4_lexec_l0_n16.log
jsonl: /tmp/dsv4_lexec_l0_n16.jsonl
flags:
  LAYER=0
  TOKEN_MIN=1
  TOKEN_MAX=16

visible prefix: stable
logged layer-output compares: 7
expected layer-output compares: 7
total implemented compares: 56
stage counts:
  attn_hc_pre_norm: 7
  qkv_setup: 14
  kv_finalizer: 7
  attention_output: 7
  ffn_hc_pre_norm: 7
  ffn_moe: 7
  layer_output: 7
largest max_abs: 0
largest rms_err: 0
over_tol: 0
generation: 19.5 tok/s with trace/readback overhead
```

The prompt prefill length is 10 tokens, so the absolute position filter `1..16` captured decode positions `10..16`.

n80 one-layer shadow:

```text
log: /tmp/dsv4_lexec_l0_n80.log
jsonl: /tmp/dsv4_lexec_l0_n80.jsonl
flags:
  LAYER=0

visible prefix: stable
layer-output compares: 79
expected layer-output compares: 79
total implemented compares: 632
stage counts:
  attn_hc_pre_norm: 79
  qkv_setup: 158
  kv_finalizer: 79
  attention_output: 79
  ffn_hc_pre_norm: 79
  ffn_moe: 79
  layer_output: 79
largest max_abs: 0
largest rms_err: 0
over_tol: 0
generation: 19.2 tok/s with one-layer shadow readback overhead
```

n80 all-layer shadow:

```text
log: /tmp/dsv4_lexec_all_n80.log
jsonl: /tmp/dsv4_lexec_all_n80.jsonl
visible prefix: stable
expected layer-output compares: 43 layers * 79 decode tokens = 3397
actual layer-output compares: 3397
total implemented compares: 37812
stage counts:
  attn_hc_pre_norm: 3397
  qkv_setup: 6794
  compressor_update: 10636
  kv_finalizer: 3397
  attention_output: 3397
  ffn_hc_pre_norm: 3397
  ffn_moe: 3397
  layer_output: 3397
largest max_abs: 0
largest rms_err: 0
over_tol: 0
not_implemented stage lines:
  compressor_update unavailable for layers 0 and 1 at the first decode token
generation: 4.3 tok/s with all-layer readback overhead
```

The n80 all-layer speed is not a performance number. It includes thousands of Metal readbacks and should only be treated as validation overhead.

Policy after this pass:

```text
transcript-exact remains active: yes
tolerance policy adopted: no
drifting paths accepted: no
```

Recommended next architecture step:

```text
Deepen the shadow envelope from coarse identity-stage tensors to real DS4-style internal stage probes:
1. bind existing HC_PRE_NORM internal compare outputs into the layer executor report
2. bind CUPD2/KV finalizer internal compare outputs into the same per-layer report
3. bind FFN/MoE diagnostic internals into the same per-layer report
4. only after one full layer has exact/tolerance-clean shadow internals, consider a one-layer consume experiment
```

### 2026-05-13 Layer Executor One-Layer Consume Canary

Status: implemented and partially validated, but rejected as a safe consume path after n400 logit divergence.

Purpose:

```text
Test graph rewiring safety only:
Can exactly one selected layer's downstream-consumed output be replaced by a shadow-verified
identity-equivalent tensor without changing transcript, logits, or scheduling behavior?
```

This is not a performance optimization. It intentionally does not introduce fused arithmetic, cache mutation,
FFN/MoE V2 consume, CUPD2 fused-compressor consume, DOWN_SUM6, or any other known drifting path.

Files changed in this pass:

```text
src/models/deepseek4.cpp
src/llama-context.cpp
docs/dsv4-m5-metal-performance-handoff.md
```

New consume canary flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME_MODEL=layer_output_identity
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_ABORT_ON_MISMATCH=1
```

Existing required flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER=<exactly one layer>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN=<min>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX=<max>
```

Guardrails:

```text
consume refuses unless:
  shadow=1
  compare=1
  consume=1
  consume_model=layer_output_identity
  exactly one layer is specified
  token_min and token_max are specified
  abort_on_mismatch=1

consume is rejected if any known drifting/direct path is enabled:
  FFN_MOE_STAGE_FULL_CONSUME
  COMPRESSOR_UPDATE_FUSED_COMP
  COMPRESSOR_UPDATE_V2_FUSED_COMP
  DOWN_SUM6
  weighted/shared SwiGLU variants
  mixed attention experimental consume
  KV finalizer/cache-mutating path
```

Implementation semantics:

```text
generic layer output -> reference probe
generic layer output -> identity shadow tensor
compare reference vs identity shadow
if exact, feed identity shadow downstream for the selected layer/token only
if max_abs != 0 or over_tol != 0, abort when ABORT_ON_MISMATCH=1
all other layers/tokens feed the generic tensor
```

The final summary now includes:

```text
dsv4_lexec_summary:
  shadow_enabled
  compare_enabled
  consume_enabled
  consume_model
  consume_layer
  token_min
  token_max
  layer_output_compares
  layer_output_consumes
  expected_layer_output_consumes
  dsv4_lexec_consume
  over_tol
  abort_on_mismatch
  cache_mutation_disabled
  rejected_paths_enabled
```

The JSONL dump now marks consumed probes:

```json
{"consume":"shadow_identity","consumed":true,"consume_allowed":true,"consume_reason":"exact_compare"}
```

Build/checks:

```text
cmake --build build --target llama-cli -j 8: pass
git diff --check: pass
python3 -m py_compile scripts/dsv4_first_divergence.py: pass
markdown fence sanity: pass
```

Baseline shadow, no consume, layer 0 n16:

```text
log: /tmp/dsv4_lexec_turn2_shadow_l0_n16.log
jsonl: /tmp/dsv4_lexec_turn2_shadow_l0_n16.jsonl
flags:
  LAYER=0
  TOKEN_MIN=1
  TOKEN_MAX=16

visible prefix: stable
layer_output_compares: 7
expected_layer_output_compares: 7
layer_output_consumes: 0
dsv4_lexec_consume: 0
total_compares: 56
over_tol: 0
consume_enabled: 0
```

Layer 0 n16 consume canary:

```text
log: /tmp/dsv4_lexec_turn2_consume_l0_n16.log
jsonl: /tmp/dsv4_lexec_turn2_consume_l0_n16.jsonl

visible prefix: stable
layer_output_compares: 7
layer_output_consumes: 7
expected_layer_output_consumes: 7
dsv4_lexec_consume: 7
total_compares: 56
over_tol: 0
rejected_paths_enabled: none
cache_mutation_disabled: 1
```

Layer 0 n80 consume canary:

```text
log: /tmp/dsv4_lexec_turn2_consume_l0_n80.log
jsonl: /tmp/dsv4_lexec_turn2_consume_l0_n80.jsonl

visible ANE prefix: stable
layer_output_compares: 79
layer_output_consumes: 79
expected_layer_output_consumes: 79
dsv4_lexec_consume: 79
total_compares: 632
over_tol: 0
rejected_paths_enabled: none
cache_mutation_disabled: 1
```

Layer 0 n400 consume canary with logit dump:

```text
baseline log: /tmp/dsv4_lexec_turn2_baseline_l0_n400.log
baseline logits: /tmp/dsv4_lexec_turn2_baseline_l0_n400_logits.jsonl
consume log: /tmp/dsv4_lexec_turn2_consume_l0_n400.log
consume logits: /tmp/dsv4_lexec_turn2_consume_l0_n400_logits.jsonl
consume layer report: /tmp/dsv4_lexec_turn2_consume_l0_n400.jsonl
first-divergence report: /tmp/dsv4_lexec_turn2_first_divergence_l0_n400.txt

layer_output_compares: 399
layer_output_consumes: 399
expected_layer_output_consumes: 399
dsv4_lexec_consume: 399
total_compares: 3192
over_tol: 0
rejected_paths_enabled: none
cache_mutation_disabled: 1
```

Despite exact layer-output identity compares, n400 diverged:

```text
first divergent token index: 79
position: 89
baseline token: 734 "'s"
consume token: 769 " has"
baseline top1/top2 margin: 0.125892639
consume top1/top2 margin: 0.170585632
baseline top1 rank in consume: 2
consume top1 rank in baseline: 2
top10 overlap: 10/10
top20 overlap: 18/20
max_abs top20-overlap logit err: 0.485816956
mean_abs top20-overlap logit err: 0.209501161
rms top20-overlap logit err: 0.261336401
classification: graph-shape/scheduling canary failure; rejected under transcript-exact policy
```

Interpretation:

- The canary proves n16 and n80 one-layer identity consumption can be scheduled without immediate mismatch.
- The n400 result proves that even replacing layer 0 output with a bit-exact identity shadow tensor can perturb long-run deterministic logits.
- Since the consumed identity tensor is exact at every selected layer-output compare, this is not an arithmetic replacement bug.
- Treat this as graph-shape/scheduling sensitivity, or as a validation-policy question only after a formal tolerance policy exists.
- No optional layer 42 consume was run because layer 0 n400 was not exact.

Policy after this pass:

```text
transcript-exact remains active: yes
tolerance policy adopted: no
drifting paths accepted: no
consume path accepted as validation canary only: no, rejected at n400
performance path accepted: no
```

### 2026-05-13 Turn #64 Routed-MoE Result-Chain Stabilization Matrix

Turn #63 recap:

```text
explicit result-chain dump/readback:
  layer0_next_input exact
  layer1_* exact
  last_layer_* exact
  result_hc exact
  result_norm exact
  logits_input exact
  scope=local tensor dump comparison
```

New mode:

```text
flag=LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_MODE
modes:
  none
  metadata_only
  materialize_layer0_output
  materialize_hc_post_output
  materialize_result_hc
  materialize_result_norm
  dependency_after_layer0
  dependency_before_result_hc
  dependency_before_result_norm
  readback_layer0_output
  readback_result_hc
  readback_result_norm
```

n400 baseline:

```text
log=/tmp/dsv4_turn64_rmoe_baseline_n400.log
jsonl=/tmp/dsv4_turn64_rmoe_baseline_n400_logits.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
```

Stabilization matrix:

```text
none:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1399837
  pair/pswiglu/fglu=399/399/399

metadata_only:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1399837
  pair/pswiglu/fglu=399/399/399

materialize_layer0_output:
  first_divergence_token=104
  max_abs=0.422439575
  rms=0.150950382
  metal_dispatch=1402231
  pair/pswiglu/fglu=798/0/0
  production_safe=no, changes generic split-GLU lowering

materialize_hc_post_output:
  first_divergence_token=104
  max_abs=0.422439575
  rms=0.150950382
  metal_dispatch=1402231
  pair/pswiglu/fglu=798/0/0
  production_safe=no, changes generic split-GLU lowering

dependency_after_layer0:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1401034
  pair/pswiglu/fglu=399/399/399

dependency_before_result_hc:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1400635
  pair/pswiglu/fglu=399/399/399

dependency_before_result_norm:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1400635
  pair/pswiglu/fglu=399/399/399

materialize_result_hc:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1400236
  pair/pswiglu/fglu=399/399/399

materialize_result_norm:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1400236
  pair/pswiglu/fglu=399/399/399

readback_layer0_output:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1399837
  pair/pswiglu/fglu=399/399/399

readback_result_hc:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1399837
  pair/pswiglu/fglu=399/399/399

readback_result_norm:
  first_divergence_token=35
  max_abs=0.323467255
  rms=0.14525186
  metal_dispatch=1399837
  pair/pswiglu/fglu=399/399/399
```

Full dump control:

```text
log=/tmp/dsv4_turn64_rmoe_result_chain_full_result_chain_dump_n400.log
jsonl=/tmp/dsv4_turn64_rmoe_result_chain_full_result_chain_dump_n400_logits.jsonl
first_divergence_token=35
max_abs=0.323467255
rms=0.14525186
interpretation=Turn #63 dump proved local tensor equality in instrumented dump runs, not transcript/logit stabilization
```

Decision:

```text
first exact mode=not_found
production_safe_exact_mode=none
root_cause=result-chain readback/materialization does not fix hot-neutral drift
remaining blocker=replace-generic path is locally exact at dumped tensors but still not transcript-exact in hot-neutral logits
replace path accepted=no
performance run=no
all-layer consume run=no
```

Policy:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Layer Executor Identity Consume Root-Cause Pass

Status: root cause identified; consume experiments should not proceed while the current readback shadow harness is active.

Question answered:

```text
Why did layer_output_identity drift at n400 even though every selected layer-output compare was exact?
```

Turn 2 starting point:

```text
layer: 0
consume model: layer_output_identity
style: current identity scale
n16: exact
n80: exact
n400:
  layer_output_consumes: 399
  dsv4_lexec_consume: 399
  over_tol: 0
  first divergent token index: 79
  position: 89
  baseline token: 734 "'s"
  consume token: 769 " has"
  top10 overlap: 10/10
  top20 overlap: 18/20
  max_abs top20-overlap logit err: 0.485816956
  rms top20-overlap logit err: 0.261336401
```

New consume-style flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME_STYLE=<style>
```

Supported styles:

```text
same_tensor:
  feed original tensor downstream; only exercise guard/counter/reporting side branches
view_alias:
  feed a ggml_view_4d alias downstream
reshape_alias:
  feed a same-shape ggml_reshape_4d alias downstream
add_zero:
  feed tensor + scale(tensor, 0)
mul_one:
  feed scale(tensor, 1)
copy_materialized:
  feed ggml_cont(tensor)
current_identity:
  feed scale(tensor, 1), matching the turn 2 identity implementation
```

Additional JSONL/fingerprint fields:

```text
consume_style
consume_materialized
shadow_tensor
shadow_op
shadow_src0
shadow_src0_op
downstream_consumed
```

Implementation note:

```text
The first style-key attempt used long names such as output-style-same_tensor.
Those names exceeded the practical tensor-name budget and prevented the layer_output ref/shadow pair from matching.
The probe key was shortened to style codes:
  st, va, ra, az, mo, cm, ci
```

Fresh same-build baseline:

```text
log: /tmp/dsv4_lexec_turn3_baseline_l0_n400.log
logits: /tmp/dsv4_lexec_turn3_baseline_l0_n400_logits.jsonl
decode tokens: 399
metal stats:
  dec_mv=1197
  pair=399
  pswiglu=798
  gen_mv=48267
  mul_mat mv=221848
  metal_dispatch=1399039
  fglu=798
  dsv4_hcnorm=34314
```

`same_tensor` result:

```text
log: /tmp/dsv4_lexec_turn3_same_tensor_l0_n400.log
logits: /tmp/dsv4_lexec_turn3_same_tensor_l0_n400_logits.jsonl
layer report: /tmp/dsv4_lexec_turn3_same_tensor_l0_n400.jsonl
first-divergence report: /tmp/dsv4_lexec_turn3_first_divergence_same_tensor_l0_n400.txt

downstream tensor consumed: original generic tensor
layer_output_compares: 399
layer_output_consumes: 399
dsv4_lexec_consume: 399
over_tol: 0
shadow op: SCALE side branch only
shadow source op: DSV4_HC_EXPAND

first divergent token index: 79
position: 89
baseline token: 734 "'s"
same_tensor token: 769 " has"
top10 overlap: 10/10
top20 overlap: 18/20
max_abs top20-overlap logit err: 0.485816956
rms top20-overlap logit err: 0.261336401
metal_dispatch: 1405423
all other high-level DSV4/matmul counters matched baseline
classification: harness/readback side-branch perturbation
```

Confirming control: shadow compare with consume disabled:

```text
log: /tmp/dsv4_lexec_turn3_shadow_no_consume_l0_n400.log
logits: /tmp/dsv4_lexec_turn3_shadow_no_consume_l0_n400_logits.jsonl
layer report: /tmp/dsv4_lexec_turn3_shadow_no_consume_l0_n400.jsonl
first-divergence report: /tmp/dsv4_lexec_turn3_first_divergence_shadow_no_consume_l0_n400.txt

consume_enabled: 0
layer_output_compares: 399
layer_output_consumes: 0
dsv4_lexec_consume: 0
over_tol: 0

first divergent token index: 79
position: 89
baseline token: 734 "'s"
shadow-no-consume token: 769 " has"
top10 overlap: 10/10
top20 overlap: 18/20
max_abs top20-overlap logit err: 0.485816956
rms top20-overlap logit err: 0.261336401
metal_dispatch: 1405423
all other high-level DSV4/matmul counters matched baseline
classification: shadow compare/readback harness perturbation
```

Styles intentionally not run after the `same_tensor` failure:

```text
view_alias: blocked
reshape_alias: blocked
add_zero: blocked
mul_one: blocked
copy_materialized: blocked
current_identity: already known to drift from turn 2
```

Reason:

```text
same_tensor keeps the original generic tensor downstream and still diverges.
The confirming no-consume shadow run also diverges.
Therefore alias/materialization styles cannot isolate the first cause until the shadow compare/readback harness itself is made non-perturbing.
```

Root-cause classification:

```text
primary cause:
  instrumentation/readback side-branch perturbation

more specific observed graph/counter fingerprint:
  adding layer-0 shadow compare side branches increases metal_dispatch from 1399039 to 1405423
  high-level routed/MoE/attention counters are otherwise unchanged
  deterministic logits diverge late at the same near-tie token, even with no downstream consume

not the cause:
  fused arithmetic replacement
  cache mutation
  rejected FFN/MoE V2 consume
  CUPD2_FUSED_COMP
  DOWN_SUM6
  tensor aliasing or materialization style, not yet testable because same_tensor already fails
```

Policy after this pass:

```text
transcript-exact remains active: yes
tolerance policy adopted: no
drifting paths accepted: no
one-layer consume canary accepted: no
safe consume style found: no
performance path accepted: no
```

Recommended next action:

```text
Do not run alias/materialization consume canaries until the validation harness is non-perturbing.
Build a non-intrusive validation mode that either:
  1. compares only after generation from saved tensors/logits, or
  2. samples a much smaller bounded token/layer window and proves no transcript/logit drift first, or
  3. uses a backend/kernel trace without adding extra graph nodes/readbacks on the hot path.
```

## Recommended Next Attack Plan

### Phase 1: Freeze A Clean Baseline

Before more code changes:

1. Rebuild cleanly.
2. Run the n400 smoke exactly as above with the accepted default path (`split-GLU` implies `HC_PRE_NORM`).
3. Run the forced-off control with `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM=0`.
4. Save both logs under `/tmp` with unique names.
5. Run external DS4 with the same prompt/model constraints and save its log.
6. Confirm exact current gap.

Required output:

```text
this repo accepted tok/s
this repo forced-HC_PRE_NORM-off tok/s
external DS4 tok/s
same model
same prompt
same top-k
same no SSD/offload constraint
```

### Phase 2: Choose One Whole-Stage Op, Not Another Peephole

Pick one of the following and implement as a dedicated DSV4 op. Based on the 2026-05-13 results, prioritize Option B or Phase 3 cache/store work. Option A is now partly explored and did not move enough by itself.

#### Option A: Dedicated Attention Output Decode Op

Status: partially implemented and validated as a correctness/instrumentation win, not a performance win.

Previous result:

```text
dsv4_aodec: 0 -> 17157
dsv4_aolow: 17286 -> 129
gen_mv: unchanged
mul_mat mv: unchanged
tok/s: small/noisy movement only
tensor compare: exact
n80/n400 transcript: pass
```

Interpretation:

The op activated at the intended decode-layer boundary, but it did not collapse the remaining generic high-projection/consumer structure enough. Do not spend more time here unless the next patch actually removes the high projection and downstream layout dispatches.

Target:

```text
kqv_out
inverse RoPE tail
grouped low projection wo_a
high projection wo_b
```

Current generic sequence:

```text
reshape kqv_out
dsv4_rope_tail inverse
dsv4_grouped_out:
  cont
  reshape_3d
  arange/cast/repeat ids
  mul_mat_id wo_a
  reshape_2d
  mul_mat wo_b
```

Why this is attractive:

- `attn_out` is one of the largest remaining attention buckets.
- DS4 has dedicated attention output APIs.
- Existing `dsv4_aolow` proves low projection alone is insufficient, so the next attempt must include the high projection and possibly inverse RoPE.

Risks:

- Fusing inverse RoPE can alter later sampled text if numerics differ.
- Fusing low+high Q8 projections is more complex than direct low.
- If launch/scheduler overhead dominates outside this cluster, the speedup may still be small.

Suggested minimal proof:

1. Keep the existing guarded `GGML_OP_DSV4_ATTN_OUT_DECODE` as a correctness building block.
2. Only revisit if the implementation removes generic high projection dispatch and downstream layout materialization.
3. Compare output tensor against generic low+high path with the existing debug flag.
4. If correct but no speed, move on immediately.

#### Option B: Dedicated Compressor Decode/Update Op

Target:

```text
KV projection
score projection
APE/add
state update
pooling
RMS norm
RoPE
FP8/HFP4 quantize
cache writes
```

Why this is attractive:

- `attn_compress` is large.
- Existing compressor-pair only fused the first two projections and did not help.
- DS4 treats this as a more dedicated path.
- The existing `CUPD` path is unsafe/OOM when stacked with `HC_PRE_NORM`, so a fresh design is needed.

Risks:

- Existing `GGML_OP_DSV4_DECODE_COMPRESS` was neutral and had transcript drift.
- Existing `GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE` can OOM quickly in the current stack.
- The op must exactly match graph numerics, including norm and RoPE.
- Cache/state side effects need careful graph dependency handling.

Suggested minimal proof:

1. Do not reuse the current CUPD implementation as-is.
2. Add tensor compare tooling for unfused vs fused compressor result.
3. Start with tensor outputs only; keep the graph's existing cache write path until tensor correctness is proven.
4. Only then absorb cache writes.
5. Validate n80 and n400 deterministic transcript before claiming speed.

#### Option C: Real Indexed Mixed Attention

Target:

Port the DS4 indexed mixed attention shape, especially:

```text
kernel_dsv4_indexed_mixed_attention_heads8
kernel_dsv4_indexed_mixed_attention_heads8_rb4
```

Why this is attractive:

- Current `GGML_OP_DSV4_MIXED_ATTN` is not equivalent to DS4's indexed design.
- It likely still materializes too much mask/concat/gather structure.

Risks:

- More complex correctness surface.
- Top-k/mask behavior must match exactly.
- Current stage profile says `attn_core` is not the largest bucket, so this may not be the first best target.

### Phase 3: Solve Cache Store Semantics Deliberately

The KV finalizer/store is important, but the current adjacent-node fusion did not match the actual graph.

Do not keep trying to make the matcher smarter blindly.

Instead decide one of:

1. Change graph construction so `KVrope -> KVcur -> set_rows` is contiguous and legal.
2. Add a model-specific op that returns `KVcur` and writes the cache, with explicit graph dependency semantics.
3. Leave cache store generic until compressor/output fusion moves tok/s.

The hard question for the expert:

```text
Can a Metal ggml backend op safely perform cache side effects while also producing a tensor consumed later, without breaking graph scheduling or dependency analysis?
```

If yes, a DS4-like KV finalizer is viable. If no, the graph must be rebuilt around a new op.

### Phase 4: Use Instruments Or Metal Capture For Launch Count

The stage profiler is helpful but intrusive. A higher-confidence diagnosis should use:

- Metal command buffer/encoder count per token.
- Kernel launch count per token.
- Time split by command encoder.
- Whether generic small kernels dominate wall-clock despite low arithmetic.

Expected finding:

This repo probably launches far more kernels per token than DS4's dedicated path.

## Correctness Gates

Use at least:

```sh
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU=1 ./build/bin/llama-cli \
  -m /Users/anemll/Models/antirez/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ctx-size 1024 \
  --reasoning off \
  --temp 0 \
  --seed 42 \
  -n 80 \
  -p 'What is Apple Neural engine?' \
  --moe-mode stock \
  --moe-topk 6 \
  --no-warmup \
  --perf -st
```

Then n400.

Expected visible prefix:

```text
This is an excellent question that gets to the heart of what makes modern Apple devices so capable.

The **Apple Neural Engine (ANE)** is a dedicated, specialized piece of hardware built into Apple's system-on-a-chip...
```

Known drift examples:

- Single workgroup decode attention produced a small speed bump but changed text around `"computer's"` vs `"computer has"`.
- Decode compressor fusion diverged later in a 400-token sample.
- `GGML_OP_DSV4_FFN_MOE_DECODE_STAGE` V2 full consume removed the routed FFN generic matvec path but changed the n400 transcript around `"and AI algorithms"` vs `"(the foundation of modern AI)"`.

For serious validation, add or use logits/tensor tolerance comparison. Text comparison alone is too brittle and too weak.

## 2026-05-13 FFN/MoE Stage V2 Result

Implemented and tested a stricter FFN/MoE stage V2:

```text
GGML_OP_DSV4_FFN_MOE_DECODE_STAGE
  V1/diagnostic: down/add-tree compare only
  V2: selected gate/up projection + limited SwiGLU + route weight + down/sum

Flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE=1
```

Important fixes during V2:

- Added a full-stage shadow op that computes selected gate/up, limited SwiGLU, routed down, and cumulative top-k sum.
- Added compare probes for gate, up, weighted SwiGLU mid, down partials, and final FFN output.
- Added a Metal memory barrier between the V2 gate/up/SwiGLU dispatch and routed down dispatch. Without this barrier, the down stage read stale/partial mid data and produced large mismatches.
- Changed V2 gate/up probes to expose the same clamp-limited boundary that the DeepSeek4 generic graph consumes.
- Added tight compare tolerances for non-bitwise float boundaries while keeping gate/up exact.
- Tried a split route-weight materialization kernel to mimic the generic graph's rounding point more closely.
- Tried a stricter composite full-consume mode: V2 selected gate/up projection, generic elementwise SiLU/SwiGLU/weight rounding, then exact down/sum.

Validation results:

```text
n80 compare generic:
  dsv4_ffnmoe = 3397 = 43 layers * 79 decode tokens
  gate/up exact after clamp-boundary fix
  mid/partial/final within tight tolerance
  visible prefix stable

n80 compare fused:
  dsv4_ffnmoe = 3397
  no over-tolerance compare entries
  visible prefix stable for the one-op V2 before split-weight

n80 full consume:
  dsv4_ffnmoe = 3397 for pure one-op V2
  dec_mv = 0
  pair = 0
  pswiglu = 0
  gen_mv = 387
  visible prefix stable

n400 full consume, pure one-op V2:
  dsv4_ffnmoe = 17157 = 43 layers * 399 decode tokens
  dec_mv: 1197 -> 0
  pair: 399 -> 0
  pswiglu: 798 -> 0
  gen_mv: 48267 -> 387
  metal_dispatch: 1399039 -> 1288518
  tok/s: 19.3 -> 19.3
  transcript: drift

n400 full consume, stricter composite mode:
  dsv4_ffnmoe = 34314 because V2 shadow and exact down/sum both execute
  dec_mv = 0
  pair = 0
  pswiglu = 0
  gen_mv = 387
  metal_dispatch = 1374301
  tok/s = 18.2
  transcript: drift
```

Current classification:

```text
FFN/MoE V2 compare/instrumentation: useful
FFN/MoE V2 one-op full consume: rejected
FFN/MoE V2 composite full consume: rejected
Performance path: not accepted
```

Key interpretation:

- V2 now owns the intended routed FFN boundary and the activation count is exactly right.
- Removing generic routed FFN matvec dispatch is not enough; deterministic text still drifts from very small non-bitwise FFN differences.
- The stricter composite path shows the drift is not only the down/add tree. It is likely the interaction between backend fusion/rounding points and the selected gate/up/SwiGLU boundary.
- This strengthens the larger conclusion: continuing to force DS4-style decode into local ggml graph substitutions is brittle. A real DS4-style executor or a much larger graph-level op with a bit-exact reference strategy may be required.

### 2026-05-13 Hot-Path-Neutral Validation Mode

Status: implemented as validation infrastructure only.

Turn #3 showed that even exact layer-output shadow comparison perturbs n400:

```text
layer 0 shadow/no-consume:
  layer_output_compares = 399
  over_tol = 0
  downstream consumed generic
  first divergence = token 79
  metal_dispatch increased by 6384 = 399 tokens * 16 extra dispatches
```

Root cause classification:

```text
graph probe/readback side branches perturb scheduling
same_tensor consume also drifted
no-consume shadow compare also drifted
therefore layer-executor compare is not hot-path-neutral
```

Turn #4 adds an explicit hot-path-neutral validation mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK=20
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP=/tmp/...
```

Design:

- Uses the existing CLI `server_prob_output` logit/prob stream.
- Does not register a graph eval callback for the logit dump.
- Does not add ggml graph nodes.
- Does not add tensor readbacks from intermediate tensors.
- Does not add consume paths.
- Does not mutate cache.
- Writes JSONL metadata proving the validation source and guard status.
- This mode is not wall-clock-neutral: it forces sampler top-k probability payloads and writes JSONL per token. Do not use its `Generation:` line as a performance measurement.

Guardrails:

The hot-path-neutral mode refuses to start if any intrusive validation flag is enabled, including:

```text
DSV4 layer-executor shadow/compare/consume/dump/trace
DSV4 per-stage tensor compare flags
DSV4 logit TRACE_STAGES
Metal stage profiling flags
```

It also refuses rejected or cache-mutating paths:

```text
FFN_MOE_STAGE_FULL_CONSUME
CUPD/CUPD2 FUSED_COMP
DOWN_SUM6
weighted/shared SwiGLU
weighted down
mixed attention experimental path
KV_FINALIZE
```

The analyzer now supports:

```text
scripts/dsv4_first_divergence.py --require-hotpath-neutral baseline.jsonl fused.jsonl
```

This fails unless both dumps declare:

```text
hot_path_neutral_validation = true
hot_path_neutral_guard_ok = true
validation_source = server_prob_output
ggml_graph_nodes_added_for_validation = false
tensor_readbacks_added_for_validation = false
consume_path_enabled_for_validation = false
cache_mutation_enabled_for_validation = false
```

Policy result:

```text
transcript-exact remains active: yes
tolerance policy adopted: no
drifting paths accepted: no
performance path accepted: no
```

Use this mode for future paired validation before adding any more graph probes, consume canaries, or stage replacement attempts.

Focused turn #4 validation:

```text
plain n16 control:
  metal_dispatch = 66675
  dsv4_lexec_summary = absent

hot-path-neutral n16 run A:
  jsonl = /tmp/dsv4_hotneutral_turn4_a_n16.jsonl
  metal_dispatch = 66675
  dsv4_lexec_summary = absent
  hot_path_neutral_guard_ok = true

hot-path-neutral n16 run B:
  jsonl = /tmp/dsv4_hotneutral_turn4_b_n16.jsonl
  metal_dispatch = 66675
  dsv4_lexec_summary = absent
  hot_path_neutral_guard_ok = true

first divergence report:
  /tmp/dsv4_hotneutral_turn4_first_divergence_n16.txt
  first divergent token = none
  top20 overlap = 20/20 at last aligned token
  max_abs_logit_err = 0
  rms_logit_err = 0
```

Interpretation:

```text
The new validation path avoids the turn #3 perturbation mechanism.
It does not add the 16 extra dispatches/token caused by layer-executor shadow probes.
It does not emit dsv4_lexec_summary.
It preserves the exact Metal dispatch counters of the plain n16 control.
```

### 2026-05-13 Hot-Path-Neutral Under-Test Validation

Status: implemented and validated on n400.

Turn #5 extended the hot-path-neutral logit stream with an explicit under-test mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_UNDER_TEST=<name>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_ALLOW_REJECTED_UNDER_TEST=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_REJECTED_UNDER_TEST_ACK=not_accepted
```

Allowed under-test names:

```text
ffnmoe_v2_full_consume
cupd2_fused_comp
down_sum6
```

The validation harness remains neutral:

```text
no graph eval callback for logit dump
no ggml validation nodes
no tensor readbacks from intermediate tensors
no layer-executor shadow/compare/consume
no stage profiler
no cache mutation from validation
no Metal kernels from validation
```

The experimental path under test may change the graph, but the JSONL metadata marks it explicitly:

```text
validation_hot_path_neutral = true
experimental_under_test = true
under_test_name = <name>
under_test_changes_graph = true
path_accepted = false
acceptance_policy = transcript_exact
```

Analyzer support:

```text
scripts/dsv4_first_divergence.py --require-hotpath-neutral baseline.jsonl fused.jsonl
scripts/dsv4_first_divergence.py --require-hotpath-neutral --allow-under-test <name> baseline.jsonl fused.jsonl
```

Guard behavior:

```text
strict mode rejects undeclared rejected flags
under-test mode permits exactly one declared rejected path
multiple rejected paths are blocked
intrusive validation flags remain blocked
path_accepted=true is blocked
```

Guard checks completed:

```text
strict analyzer against undeclared FFN/MoE V2 under-test:
  rejected as expected

CLI with cupd2_fused_comp + down_sum6 simultaneously:
  rejected rc=1
  reason=under_test_rejected_flags_mismatch
```

n400 hot-neutral baseline A/B:

```text
plain n400 control:
  log = /tmp/dsv4_hotneutral_turn5_plain_n400.log
  metal_dispatch = 1399039
  dsv4_lexec_summary = absent

baseline A:
  log = /tmp/dsv4_hotneutral_turn5_baseline_A_n400.log
  jsonl = /tmp/dsv4_hotneutral_turn5_baseline_A_n400.jsonl
  metal_dispatch = 1399039
  records = 400
  hot_path_neutral_guard_ok = true

baseline B:
  log = /tmp/dsv4_hotneutral_turn5_baseline_B_n400.log
  jsonl = /tmp/dsv4_hotneutral_turn5_baseline_B_n400.jsonl
  metal_dispatch = 1399039
  records = 400
  hot_path_neutral_guard_ok = true

A vs B report:
  path = /tmp/dsv4_hotneutral_turn5_first_divergence_baseline_A_vs_B_n400.txt
  first divergent token = none
  top20 overlap = 20/20 at final token
  max_abs_logit_err = 0
  rms_logit_err = 0
```

Interpretation:

```text
Hot-path-neutral logit dumping is n400-neutral for the accepted baseline.
It preserves the plain no-validation Metal dispatch count.
It does not reproduce the turn #3 shadow/readback perturbation.
```

FFN/MoE V2 full-consume under-test:

```text
log = /tmp/dsv4_hotneutral_turn5_ffnmoe_v2_n400.log
jsonl = /tmp/dsv4_hotneutral_turn5_ffnmoe_v2_n400.jsonl
report = /tmp/dsv4_hotneutral_turn5_first_divergence_ffnmoe_v2_n400.txt

first divergence token = 104
position = 114
baseline token = 305 / " and"
under-test token = 343 / " ("
baseline top1/top2 margin = 0.141532898
under-test top1/top2 margin = 0.0713691711
baseline top1 rank in under-test = 2
under-test top1 rank in baseline = 2
top10 overlap = 10/10
top20 overlap = 20/20
max_abs_logit_err = 0.422439575
mean_abs_logit_err = 0.113186741
rms_logit_err = 0.150950382

under-test counters:
  dsv4_ffnmoe = 34314
  dec_mv = 0
  pair = 0
  pswiglu = 0
  metal_dispatch = 1374301

classification = likely tolerance-policy issue
accepted = no
```

CUPD2_FUSED_COMP under-test:

```text
log = /tmp/dsv4_hotneutral_turn5_cupd2_fused_comp_n400.log
jsonl = /tmp/dsv4_hotneutral_turn5_cupd2_fused_comp_n400.jsonl
report = /tmp/dsv4_hotneutral_turn5_first_divergence_cupd2_fused_comp_n400.txt

first divergence token = 104
position = 114
baseline token = 305 / " and"
under-test token = 343 / " ("
baseline top1/top2 margin = 0.141532898
under-test top1/top2 margin = 0.127914429
baseline top1 rank in under-test = 2
under-test top1 rank in baseline = 2
top10 overlap = 10/10
top20 overlap = 20/20
max_abs_logit_err = 0.402101517
mean_abs_logit_err = 0.101707554
rms_logit_err = 0.128725288

under-test counters:
  dsv4_cupd2 = 24738
  metal_dispatch = 1249275

classification = likely tolerance-policy issue
accepted = no
```

DOWN_SUM6 under-test:

```text
log = /tmp/dsv4_hotneutral_turn5_down_sum6_n400.log
jsonl = /tmp/dsv4_hotneutral_turn5_down_sum6_n400.jsonl
report = /tmp/dsv4_hotneutral_turn5_first_divergence_down_sum6_n400.txt

first divergence token = 104
position = 114
baseline token = 305 / " and"
under-test token = 343 / " ("
baseline top1/top2 margin = 0.141532898
under-test top1/top2 margin = 0.00744247437
baseline top1 rank in under-test = 2
under-test top1 rank in baseline = 2
top10 overlap = 10/10
top20 overlap = 20/20
max_abs_logit_err = 0.1621418
mean_abs_logit_err = 0.068448925
rms_logit_err = 0.0774944226

under-test counters:
  dsum6 = 1197
  dec_mv = 0
  metal_dispatch = 1391857

classification = likely tolerance-policy issue under hot-neutral validation
accepted = no
```

Classification changes:

```text
FFN/MoE V2:
  unchanged; likely tolerance-policy issue, still rejected

CUPD2_FUSED_COMP:
  unchanged; likely tolerance-policy issue, still rejected

DOWN_SUM6:
  changed materially from the previous intrusive/logit-harness result
  previous: first divergence token 35, top10 overlap 9/10, larger max/rms error
  hot-neutral: first divergence token 104, top20 overlap 20/20, smaller max/rms error
  current classification: tolerance-policy candidate only, still rejected
```

Policy result:

```text
transcript-exact remains active: yes
tolerance policy adopted: no
drifting paths accepted: no
performance path accepted: no
```

Recommended next step:

```text
Markdown-only draft of a DSV4 logit/tolerance policy.
Do not adopt it yet.
Do not add another local fusion until the validation policy question is settled.
```

### DSV4 logit/tolerance policy draft

A draft policy was added at:

```text
docs/dsv4-logit-tolerance-policy-draft.md
```

Status:

```text
draft only
not adopted
transcript-exact n400 remains active
no drifting path accepted
```

### 2026-05-13 Turn #7 Tolerance Candidate Performance Reality Check

A no-logit n400 performance smoke was run for the Tier 1 draft tolerance candidates. See:

```text
docs/dsv4-logit-tolerance-policy-draft.md
```

Result summary:

```text
baseline avg:       21.2 tok/s
DOWN_SUM6:          21.5 tok/s, +0.3, rejected
CUPD2_FUSED_COMP:   21.8 tok/s, rerun 21.2 tok/s, rejected
FFN/MoE V2:         20.7 tok/s, -0.5, rejected
```

Policy result:

```text
transcript-exact remains active
tolerance policy not adopted
no drifting path accepted
```

### 2026-05-13 Turn #9 Indexed Mixed Attention Mapping / Shadow

Status: accepted mapping / metadata-shadow step, not a performance path.

DS4 indexed mixed attention entry points inspected:

```text
/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/ds4.c
  decode call: metal_graph_encode_decode_layer -> ds4_gpu_attention_indexed_mixed_batch_heads_tensor
  key lines inspected: around 9530-9565

/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/ds4_metal.m
  host entry: ds4_gpu_attention_indexed_mixed_batch_heads_tensor
  key lines inspected: around 12291-12455

/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4/metal/dsv4_misc.metal
  kernels:
    kernel_dsv4_indexed_mixed_attention_heads8
    kernel_dsv4_indexed_mixed_attention_heads8_rb4
    kernel_dsv4_indexed_mixed_attention_heads8_rb16
  key lines inspected: around 491-820
```

DS4 boundary:

```text
q f32
raw/SWA KV rows
compressed KV rows
selected compressed row ids from topk
attention sinks
pos0, n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio

output:
  per-head attention result in f32 layout

decode assumptions:
  n_tokens = 1
  head_dim = 512
  heads grouped by 8
  ratio-4 indexed compressed attention
```

Current repo boundary inspected:

```text
src/models/deepseek4.cpp
  decode builds raw_attn_mask
  builds indexer_scores
  builds argsort_top_k
  materializes dense compressed mask with dsv4_build_compressed_mask_from_topk
  concatenates raw + compressed masks
  normally calls build_attn_mha over concatenated raw+compressed K/V

GGML_OP_DSV4_MIXED_ATTN
  takes q, raw_kv, comp_kv, raw_mask, comp_mask, sinks
  scans dense finite mask rows
  does not consume topk row ids directly
```

Gap:

```text
DS4 indexed attention is topk-row driven.
The current repo path is dense-mask driven.
The current GGML_OP_DSV4_MIXED_ATTN is therefore not equivalent to DS4's indexed design.
```

Implemented in this pass:

```text
metadata-only indexed-attention shadow
no GGML op
no Metal kernel
no graph consume path
no cache mutation
no n400 intrusive readback
```

New flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TOKEN_MAX=<n>
```

Counter / summary:

```text
dsv4_iattn_shadow_summary:
  dsv4_iattn_shadow=<metadata probe count>
  row_id_mismatch_count=<count>
  order_mismatch_count=<count>
  generic_downstream_consumed=1
  consume_path=disabled
  cache_mutation_disabled=1
```

Layer 0 result:

```text
log n16: /tmp/dsv4_turn9_iattn_shadow_l0_n16.log
log n80: /tmp/dsv4_turn9_iattn_shadow_l0_n80.log

layer 0 compress_ratio = 0
indexed ratio-4 attention shadow = not applicable

n16:
  dsv4_iattn_shadow = 0
  not_implemented = 7
  reason = requires_ratio4_indexer
  prefix stable = yes

n80:
  dsv4_iattn_shadow = 0
  not_implemented = 79
  reason = requires_ratio4_indexer
  prefix stable = yes
```

First applicable indexed-attention layer:

```text
layer 2
compress_ratio = 4

n16 log: /tmp/dsv4_turn9_iattn_shadow_l2_n16.log
  dsv4_iattn_shadow = 7
  visible_all_probes = 7
  topk_probes = 0
  row_id_mismatch_count = 0
  order_mismatch_count = 0
  prefix stable = yes

n80 log: /tmp/dsv4_turn9_iattn_shadow_l2_n80.log
  dsv4_iattn_shadow = 79
  visible_all_probes = 79
  topk_probes = 0
  row_id_mismatch_count = 0
  order_mismatch_count = 0
  prefix stable = yes
```

Interpretation:

```text
For this prompt/context, visible compressed rows remain below the configured indexer top-k.
The tested layer-2 decode path therefore selects all visible compressed rows.
Row set equivalence with a DS4-style indexed scan is proven by construction for this case.
No top-k subset case was exercised in n80.
```

n400 hot-path-neutral baseline after code changes:

```text
log: /tmp/dsv4_turn9_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn9_baseline_n400_hotneutral.jsonl
jsonl records: 401
metal_dispatch: 1399039
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: none
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

Next exact step:

```text
Implement a shadow-only DS4 indexed mixed-attention Metal prototype for the first applicable ratio-4 layer.
Compare n16/n80 attention output only.
Do not consume until exactness is proven.
```

### 2026-05-13 Turn #10 Indexed Mixed Attention Shadow Output Prototype

Status: diagnostic shadow only. No consume path was added.

Metadata search:

```text
n80 all-layer search:
  log: /tmp/dsv4_turn10_iattn_search_all_n80.log
  dsv4_iattn_shadow: 1659
  applicable_layers: 21
  visible_all_cases: 1659
  topk_subset_cases: 0
  row_id_mismatch_count: 0
  first_applicable_layer: 2
  first_topk_subset_layer/token: -1 / -1

n400 diagnostic search:
  log: /tmp/dsv4_turn10_iattn_search_all_n400_diagnostic.log
  dsv4_iattn_shadow: 8379
  applicable_layers: 21
  visible_all_cases: 8379
  topk_subset_cases: 0
  row_id_mismatch_count: 0
  first_applicable_layer: 2
  first_topk_subset_layer/token: -1 / -1
```

Conclusion:

```text
The fixed ANE smoke prompt still exercises only visible-all compressed attention.
It does not reach a strict top-k compressed-row subset even by n400.
Row-set equivalence for visible-all remains zero-mismatch.
Top-k subset row semantics remain untested under this prompt.
```

Layer 2 visible-all output shadow:

```text
Implementation:
  graph-level shadow prototype
  selected/visible compressed rows are fed into the existing DSV4 mixed-attention Metal op
  generic attention output remains the downstream-consumed tensor
  output compare uses readback only for n16/n80 diagnostics
  no dedicated indexed-attention kernel was added in this pass

n16:
  log: /tmp/dsv4_turn10_iattn_shadow_out_l2_n16.log
  dsv4_iattn_shadow: 7
  dsv4_iattn_shadow_out: 7
  row_id_mismatch_count: 0
  max_abs: 1.19209e-6
  rms_err max: 6.77991e-8
  over_tol: nonzero under exact-zero tolerance
  result: non-exact, diagnostic only

n80:
  log: /tmp/dsv4_turn10_iattn_shadow_out_l2_n80.log
  dsv4_iattn_shadow: 79
  dsv4_iattn_shadow_out: 79
  row_id_mismatch_count: 0
  max_abs: 2.38419e-6
  rms_err max: 7.86616e-8
  over_tol: nonzero under exact-zero tolerance
  result: non-exact, diagnostic only
```

n400 hot-path-neutral baseline after Turn #10 code changes:

```text
log: /tmp/dsv4_turn10_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn10_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: none
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
eligible for consume: no, output is not bit-exact
```

Next exact step:

```text
Root-cause the layer-2 visible-all attention-output difference before any consume canary.
Likely areas: softmax implementation/order, mask representation, row order, sink handling,
FP16 staging/dequant order, and generic FlashAttention vs DSV4 mixed-attention arithmetic.
If exact visible-all output is achieved, find or synthesize a true top-k subset case before
claiming DS4 indexed mixed-attention semantic coverage.
```

### 2026-05-13 Turn #11 Indexed Mixed Attention Output Delta Root Cause

Status: diagnostic root-cause pass only. No consume path was added.

Turn #10 non-exact result reproduced:

```text
baseline reproduction:
  log: /tmp/dsv4_turn11_iattn_diff_l2_n16_baseline.log
  layer: 2
  tokens compared: 7
  dsv4_iattn_shadow: 7
  dsv4_iattn_shadow_out: 7
  visible_all_cases: 7
  topk_subset_cases: 0
  row_id_mismatch_count: 0
  max_abs: 1.19209e-6
  rms_err max: 6.77991e-8
  over_tol: nonzero under exact-zero tolerance
  result: reproduced non-exact mixed-attention shadow output
```

Diagnostic modes tested:

```text
row order:
  generic_dense log: /tmp/dsv4_turn11_iattn_row_order_generic_dense_l2_n16.log
  ds4_indexed log: /tmp/dsv4_turn11_iattn_row_order_ds4_indexed_l2_n16.log
  generic_dense max_abs: 1.19209e-6
  ds4_indexed max_abs: 1.19209e-6
  interpretation: row order is not the visible-all root cause

mask mode:
  generic_mask_values log: /tmp/dsv4_turn11_iattn_mask_generic_mask_values_l2_n16.log
  visible_all_no_mask log: /tmp/dsv4_turn11_iattn_mask_visible_all_no_mask_l2_n16.log
  generic_mask_values max_abs: 1.19209e-6
  visible_all_no_mask max_abs: 1.19209e-6
  interpretation: compressed-row mask representation is not the visible-all root cause

generic attention over shadow rows:
  n16 log: /tmp/dsv4_turn11_iattn_generic_attention_l2_n16.log
  n16 max_abs: 0
  n16 rms_err max: 0
  n16 over_tol: 0
  n80 log: /tmp/dsv4_turn11_iattn_best_l2_n80.log
  n80 max_abs: 0
  n80 rms_err max: 0
  n80 over_tol: 0
  interpretation: shadow row selection, row order, K/V construction, and mask construction are exact when routed through the generic attention implementation
```

Boundary decomposition:

```text
row ids exact: yes
row order exact/equivalent for visible-all: yes
Q/K/V/mask construction exact: yes, by generic-attention isolation
pre-softmax scores read back separately: no
softmax max/denominator/probabilities read back separately: no
generic attention output over shadow rows exact: yes
DSV4 mixed-attention output over same visible-all row set exact: no
first nonzero observed delta stage: DSV4 mixed-attention arithmetic/output relative to generic attention
```

Root cause classification:

```text
Not caused by:
  row selection
  row id mismatch
  visible-all row ordering
  compressed-row mask representation
  shadow K/V tensor construction

Most likely caused by:
  arithmetic/order differences between the existing DSV4 mixed-attention Metal op and the generic FlashAttention path

Likely sub-causes to inspect if this path is pursued:
  softmax reduction order
  score/max/denominator accumulation order
  weighted-V accumulation order
  FP16 staging or conversion points
  sink/raw/compressed contribution order inside the mixed-attention kernel
```

n400 hot-path-neutral baseline after Turn #11 code changes:

```text
log: /tmp/dsv4_turn11_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn11_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: none
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
indexed mixed-attention consume eligible: no, existing mixed-attention output is not bit-exact
```

Next exact step:

```text
Do not consume the existing DSV4 mixed-attention shadow output.
For exact indexed attention, either:
  1. implement the indexed path using the generic FlashAttention arithmetic/order, or
  2. modify the DSV4 mixed-attention op to reproduce the generic attention arithmetic exactly.

Top-k subset row semantics remain untested under the fixed smoke prompt.
```

### 2026-05-13 Turn #12 Generic-Arithmetic Indexed Attention Shadow

Status: diagnostic shadow only. No consume path was added.

Turn #11 root-cause summary:

```text
Row selection/order/mask/KV construction were exact.
The existing DSV4 mixed-attention op produced a small non-exact output delta
against the generic attention path:
  n16 max_abs: 1.19209e-6
  n80 max_abs: 2.38419e-6

Generic attention over the same shadow rows was exact, so the root cause was
arithmetic/order inside the DSV4 mixed-attention op rather than row construction.
```

New indexed-attention arithmetic modes:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_ARITH_MODE=generic_flash
  DS4/indexed row selection
  shadow K/V construction
  generic FlashAttention arithmetic/order via build_attn_mha(...)
  generic downstream remains consumed
  compare against original generic attention output

LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_ARITH_MODE=dsv4_mixed
  DS4/indexed row selection
  shadow K/V construction
  existing ggml_dsv4_mixed_attn diagnostic path
  expected to reproduce the known non-exact delta
```

Counters and summaries:

```text
dsv4_iattn_shadow_out
dsv4_iattn_generic_flash
dsv4_iattn_dsv4_mixed
dsv4_iattn_output_compare_summary:
  arith_mode
  compares
  output_max_abs
  output_rms
  over_tol
  exact_output
  row_ids_exact
  shadow_kv_exact
  consume_path=disabled
```

Layer 2 visible-all generic-flash shadow:

```text
n16:
  log: /tmp/dsv4_turn12_iattn_generic_flash_l2_n16.log
  dsv4_iattn_shadow: 7
  dsv4_iattn_shadow_out: 7
  dsv4_iattn_generic_flash: 7
  dsv4_iattn_dsv4_mixed: 0
  visible_all_cases: 7
  topk_subset_cases: 0
  row_id_mismatch_count: 0
  output_max_abs: 0
  output_rms: 0
  over_tol: 0
  exact_output: yes
  prefix stable: yes
  consume_path: disabled

n80:
  log: /tmp/dsv4_turn12_iattn_generic_flash_l2_n80.log
  dsv4_iattn_shadow: 79
  dsv4_iattn_shadow_out: 79
  dsv4_iattn_generic_flash: 79
  dsv4_iattn_dsv4_mixed: 0
  visible_all_cases: 79
  topk_subset_cases: 0
  row_id_mismatch_count: 0
  output_max_abs: 0
  output_rms: 0
  over_tol: 0
  exact_output: yes
  prefix stable: yes
  consume_path: disabled
```

DSV4 mixed control:

```text
n16:
  log: /tmp/dsv4_turn12_iattn_dsv4_mixed_l2_n16.log
  dsv4_iattn_shadow: 7
  dsv4_iattn_shadow_out: 7
  dsv4_iattn_generic_flash: 0
  dsv4_iattn_dsv4_mixed: 7
  output_max_abs: 1.19209e-6
  output_rms: 6.77991e-8
  over_tol: 186855
  exact_output: no
  result: reproduces Turn #11 mixed-attention delta
```

All applicable visible-all n80 generic-flash shadow:

```text
log: /tmp/dsv4_turn12_iattn_generic_flash_all_n80.log
dsv4_iattn_shadow: 1659
dsv4_iattn_shadow_out: 1659
dsv4_iattn_generic_flash: 1659
dsv4_iattn_dsv4_mixed: 0
applicable_layers: 21
visible_all_cases: 1659
topk_subset_cases: 0
not_implemented/not_applicable: 1738
row_id_mismatch_count: 0
order_mismatch_count: 0
output_max_abs: 0
output_rms: 0
over_tol: 0
exact_output: yes
prefix stable: yes
consume_path: disabled
```

n400 hot-path-neutral baseline after Turn #12 code changes:

```text
log: /tmp/dsv4_turn12_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn12_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 18.6 tok/s with logit dump enabled
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: none
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
indexed attention consume eligible: no, no consume path has been authorized and top-k subset remains untested
```

Next exact step:

```text
The visible-all indexed-attention shadow is exact when it uses generic FlashAttention arithmetic/order.
Before any consume canary, either:
  1. find or synthesize a true top-k subset case and prove generic_flash exact there, or
  2. implement a strictly bounded visible-all-only consume canary in a later turn, with transcript-exact n400 validation.
```

### 2026-05-13 Turn #13 Indexed Attention Top-K Subset Search

Status: diagnostic search and shadow compare only. No consume path was added.

Metadata reporting update:

```text
The indexed-attention summary now distinguishes:
  visible_all_cases:
    all compressed rows visible to the generic path are selected
  topk_subset_cases:
    compressed_visible_rows > compressed_topk_rows

Additional summary fields:
  applicable_cases
  max_compressed_total_rows
  max_compressed_visible_rows
  max_compressed_topk_rows
  topk_k
  first_applicable_layer/token
  first_topk_subset_layer/token
  first_topk_subset_reason
  no_subset_reason
```

Fixed prompt n400 search:

```text
log: /tmp/dsv4_turn13_iattn_search_fixed_n400.log
visible_all_cases: 8379
topk_subset_cases: 0
applicable_cases: 8379
max_compressed_visible_rows: 102
max_compressed_topk_rows: 102
topk_k: 102
row_id_mismatch_count: 0
no_subset_reason: compressed_visible_rows_never_exceeded_compressed_topk_rows
```

Long-context search:

```text
original generated long prompt:
  file: /tmp/dsv4_turn13_long_prompt.txt
  result with ctx2048: failed before decode
  log: /tmp/dsv4_turn13_iattn_search_ctx2048_n32.log
  reason: request was 4805 tokens, exceeding ctx2048

ctx2048-compatible adjusted prompt:
  file: /tmp/dsv4_turn13_long_prompt_ctx2048.txt
  log: /tmp/dsv4_turn13_iattn_search_ctx2048_adjusted_n32.log
  visible_all_cases: 651
  topk_subset_cases: 0
  max_compressed_visible_rows: 449
  max_compressed_topk_rows: 449
  row_id_mismatch_count: 0
  no_subset_reason: compressed_visible_rows_never_exceeded_compressed_topk_rows

ctx4096-compatible adjusted prompt:
  file: /tmp/dsv4_turn13_long_prompt_ctx4096.txt
  log: /tmp/dsv4_turn13_iattn_search_ctx4096_adjusted_n32.log
  visible_all_cases: 0
  topk_subset_cases: 651
  first_topk_subset_layer/token: 2 / 3525
  compressed_visible_rows at first subset: 881
  compressed_topk_rows at first subset: 512
  max_compressed_visible_rows: 889
  max_compressed_topk_rows: 512
  row_id_mismatch_count: 0
  first_topk_subset_reason: compressed_visible_rows_gt_compressed_topk_rows
```

Generic-flash top-k subset output compare:

```text
log: /tmp/dsv4_turn13_iattn_generic_flash_topk_subset.log
layer/token window: layer 2, tokens 3525..3529
topk_subset cases compared: 5
compressed_visible_rows: 881..882
compressed_topk_rows: 512
row_id_mismatch_count: 0
arith_mode: generic_flash
dsv4_iattn_generic_flash: 5
output_max_abs: 7.15256e-7
output_rms max: 3.48585e-8
over_tol: 113738 under exact-zero tolerance
exact_output: no
first_bad_index: 1491 at token 3525
consume_path: disabled
```

Interpretation:

```text
The real top-k subset path is not bit-exact, even with generic FlashAttention arithmetic.
This differs from the visible-all case because the subset shadow path gathers 512 selected
compressed rows, while the generic baseline represents attention over the full visible
compressed row set with a dense top-k mask. Even with the same selected row ids, the shorter
row list changes the attention operation's shape/reduction path enough to produce small
nonzero output deltas.

Visible-all exactness therefore does not prove strict top-k subset exactness.
Indexed attention remains diagnostic-only and is not consume-eligible.
```

n400 hot-path-neutral baseline after Turn #13 code changes:

```text
log: /tmp/dsv4_turn13_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn13_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 18.0 tok/s with logit dump enabled
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: none
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
indexed attention consume eligible: no
```

Next exact step:

```text
Do not proceed to indexed-attention consume.
If indexed attention remains the target, root-cause the subset delta by matching the generic
dense-mask row count/reduction shape exactly, or by proving another arithmetic form can still
produce transcript-exact n400. Otherwise shift back to a stage exercised by the fixed target,
such as attention output / HC post or exact compressor/update boundary work.
```

### 2026-05-13 Turn #14 Indexed Attention Subset Shape-Parity Diagnosis

Turn #13 subset mismatch recap:

```text
strict top-k subset case:
  layer/token: 2 / 3525
  compressed_visible_rows: 881
  compressed_topk_rows: 512
  row_id_mismatch_count: 0

sparse generic_flash subset:
  output_max_abs: 7.15256e-7
  output_rms: 3.48585e-8
  over_tol: 113738 under exact-zero tolerance
  exact_output: no
```

Turn #14 added a shape-parity diagnostic mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHAPE_MODE=<mode>

modes:
  baseline_dense
  dense_mask_shape
  dense_padded_topk
  sparse_topk
```

Strict subset shape matrix:

```text
prompt/context:
  prompt: /tmp/dsv4_turn14_long_prompt.txt
  ctx-size: 4096
  layer: 2
  token window: 3525..3529

baseline_dense:
  log: /tmp/dsv4_turn14_iattn_shape_baseline_dense_l2_t3525_3529.log
  shadow_row_count: 881..882
  baseline_row_count: 881..882
  row_count_matches_baseline: true
  mask_shape_matches_baseline: true
  nonselected_rows_present: true
  nonselected_rows_masked: true
  output_max_abs: 0
  output_rms: 0
  over_tol: 0
  exact_output: yes

dense_mask_shape:
  log: /tmp/dsv4_turn14_iattn_shape_dense_mask_shape_l2_t3525_3529.log
  shadow_row_count: 881..882
  baseline_row_count: 881..882
  row_count_matches_baseline: true
  mask_shape_matches_baseline: true
  nonselected_rows_present: true
  nonselected_rows_masked: true
  output_max_abs: 0
  output_rms: 0
  over_tol: 0
  exact_output: yes

dense_padded_topk:
  log: /tmp/dsv4_turn14_iattn_shape_dense_padded_topk_l2_t3525_3529.log
  shadow_row_count: 881..882
  baseline_row_count: 881..882
  row_count_matches_baseline: true
  mask_shape_matches_baseline: true
  nonselected_rows_present: true
  nonselected_rows_masked: true
  nonselected K/V contents: dummy zero rows
  output_max_abs: 0
  output_rms: 0
  over_tol: 0
  exact_output: yes

sparse_topk:
  log: /tmp/dsv4_turn14_iattn_shape_sparse_topk_l2_t3525_3529.log
  shadow_row_count: 512
  baseline_row_count: 881..882
  row_count_matches_baseline: false
  mask_shape_matches_baseline: false
  nonselected_rows_present: false
  nonselected_rows_masked: false
  output_max_abs: 7.15256e-7
  output_rms: 3.48585e-8
  over_tol: 113738
  exact_output: no
```

Row-count sweep:

```text
not run
reason: full dense shape already isolated the root cause. dense_padded_topk proves masked-out
K/V values are not required when the full visible row count and dense top-k mask shape are
preserved. sparse_topk remains non-exact only when the row count and mask shape are compacted.
```

Conclusion:

```text
row ids exact: yes
generic_flash arithmetic exact in dense shape: yes
sparse compaction exact: no
nonselected row K/V required: no
full visible row count required: yes
dense top-k mask shape required: yes
first non-exact mode: sparse_topk
```

The strict subset mismatch is a shape/reduction-path mismatch, not a row-id or K/V construction
bug. Sparse indexed attention changes the generic FlashAttention row count and mask shape from
the dense baseline, which is enough to break exact-zero tensor equality. Preserving the full
dense row count and dense top-k mask shape restores exactness, even when masked-out K/V rows are
dummy zero rows.

This means sparse indexed attention is not consume-eligible under transcript-exact policy unless
it reproduces the dense reduction shape exactly, which removes the expected indexed-attention
benefit for this path.

Normal n400 hot-neutral baseline after Turn #14 code changes:

```text
log: /tmp/dsv4_turn14_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn14_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 18.9 tok/s with logit dump enabled
dsv4_iattn: 0
dsv4_iattn_shadow_summary: absent
transcript: stable baseline prefix/output
intrusive flags: absent
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
indexed attention consume eligible: no
```

### 2026-05-13 Turn #15 Attention Output + HC Post Stage Shadow

Turn #15 started an exact shadow line for the attention-output plus HC-post boundary. This is
diagnostic/stage-envelope work only. It does not add a consume path and does not accept any
non-exact replacement.

Mapping note:

```text
/tmp/dsv4_turn15_attn_out_hcpost_mapping.txt
```

DS4 stage boundary:

```text
attention heads / attention core output
  -> grouped attention output low projection
  -> high/output projection
  -> HC post / HC expand
  -> after_attn_hc
```

The DS4 reference path uses `layer_grouped_out_one` /
`layer_grouped_out_one_decode_scratch` for grouped attention output, then applies HC post
handling. The Metal path can combine the high projection with HC expansion through the
Q8 HC expand family.

Current repo boundary:

```text
build_attn_mha output / kqv_out
  -> RoPE/layout tail handling
  -> dsv4_grouped_out low projection
  -> dsv4_grouped_out high projection
  -> dsv4_hc_post / ggml_dsv4_hc_expand
  -> after_attn_hc
```

Gap:

```text
The current repo still expresses this as multiple ggml graph nodes across attention core,
low projection, high projection, and HC post. Existing support pieces such as attention-output
decode, Q8HC, and HCE4 do not yet own this whole boundary as an exact replacement.
```

New shadow flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_MODE=<mode>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TOKEN_MAX=<n>
```

Modes:

```text
generic_shadow
existing_q8hc
existing_aodec_q8hc
ds4_shape_shadow
```

The compare envelope currently binds and compares:

```text
attn_core_out
attn_low
attn_out
hc_post_weights
hc_comb
after_attn_hc
```

Results:

```text
generic_shadow layer 0 n16:
  log: /tmp/dsv4_turn15_aohc_generic_shadow_l0_n16.log
  compared cases: 42
  exact cases: 42
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  consume_path: disabled

existing_q8hc diagnostic layer 0 n16:
  log: /tmp/dsv4_turn15_aohc_existing_q8hc_l0_n16.log
  compared cases: 42
  exact cases: 42
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  consume_path: disabled
  note: this is a labeled diagnostic shadow envelope. Backend Q8HC consume/fusion was not
        enabled or accepted.

ds4_shape_shadow layer 0 n16:
  log: /tmp/dsv4_turn15_aohc_ds4_shape_l0_n16.log
  compared cases: 42
  exact cases: 42
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  consume_path: disabled

ds4_shape_shadow layer 0 n80:
  log: /tmp/dsv4_turn15_aohc_ds4_shape_l0_n80.log
  compared cases: 474
  exact cases: 474
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  consume_path: disabled

ds4_shape_shadow all layers n80:
  log: /tmp/dsv4_turn15_aohc_ds4_shape_all_n80.log
  compared cases: 20382
  exact cases: 20382
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  consume_path: disabled
```

Normal n400 hot-neutral baseline after Turn #15 code changes:

```text
log: /tmp/dsv4_turn15_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn15_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 19.1 tok/s with logit dump enabled
dsv4_aohc_compare_summary: absent
intrusive flags: absent
transcript: stable baseline prefix/output
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
attention-output/HC-post consume eligible: not yet
```

Conclusion:

```text
The attention-output plus HC-post exact shadow envelope is established across layer 0 n16,
layer 0 n80, and all layers n80. This proves the compare/report boundary can cover the intended
stage exactly while feeding generic downstream. It does not yet prove a real replacement
candidate, because the current ds4_shape_shadow mode is still a graph-level shadow envelope.
The next exact step is a single-layer consume canary only after an explicitly replacement-shaped
candidate is proven exact, or a root-cause pass if a real replacement candidate becomes non-exact.
```

### 2026-05-13 Turn #16 Attention Output + HC Post Replacement Candidate

Turn #16 added a shadow-only replacement-candidate branch for the same attention-output plus
HC-post boundary. Unlike Turn #15's envelope, this branch rebuilds candidate tensors from the
stage inputs and compares them against the generic outputs while still feeding generic downstream.

New candidate flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_MODE=<mode>
```

Candidate modes:

```text
candidate_graph_exact
candidate_existing_q8hc
candidate_aodec_q8hc
candidate_ds4_shape
```

Candidate dependency audit:

```text
allowed inputs:
  attn_core_out
  wo_a
  wo_b
  residual
  hc_post_weights
  hc_comb

forbidden generic stage outputs:
  generic attn_low
  generic attn_out
  generic after_attn_hc
```

Audit result:

```text
candidate_attn_low_uses_generic_attn_low: 0
candidate_attn_out_uses_generic_attn_out: 0
candidate_after_attn_hc_uses_generic_after_attn_hc: 0
dependency_audit_pass: 1
consume_path: disabled
```

Results:

```text
candidate_graph_exact layer 0 n16:
  log: /tmp/dsv4_turn16_aohc_candidate_graph_exact_l0_n16.log
  compared candidate cases: 21
  exact cases: 21
  non_exact cases: 0
  attn_low max_abs/rms/over_tol: 0 / 0 / 0
  attn_out max_abs/rms/over_tol: 0 / 0 / 0
  after_attn_hc max_abs/rms/over_tol: 0 / 0 / 0
  dependency_audit_pass: 1
  consume_path: disabled

candidate_existing_q8hc layer 0 n16:
  log: /tmp/dsv4_turn16_aohc_candidate_existing_q8hc_l0_n16.log
  compared candidate cases: 21
  exact cases: 21
  non_exact cases: 0
  attn_low max_abs/rms/over_tol: 0 / 0 / 0
  attn_out max_abs/rms/over_tol: 0 / 0 / 0
  after_attn_hc max_abs/rms/over_tol: 0 / 0 / 0
  dependency_audit_pass: 1
  consume_path: disabled
  note: this remains diagnostic. Backend Q8HC consume/fusion was not enabled or accepted.

candidate_ds4_shape layer 0 n16:
  log: /tmp/dsv4_turn16_aohc_candidate_ds4_shape_l0_n16.log
  compared candidate cases: 21
  exact cases: 21
  non_exact cases: 0
  attn_low max_abs/rms/over_tol: 0 / 0 / 0
  attn_out max_abs/rms/over_tol: 0 / 0 / 0
  after_attn_hc max_abs/rms/over_tol: 0 / 0 / 0
  dependency_audit_pass: 1
  consume_path: disabled

candidate_ds4_shape layer 0 n80:
  log: /tmp/dsv4_turn16_aohc_candidate_ds4_shape_l0_n80.log
  compared candidate cases: 237
  exact cases: 237
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  dependency_audit_pass: 1
  consume_path: disabled

candidate_ds4_shape all layers n80:
  log: /tmp/dsv4_turn16_aohc_candidate_ds4_shape_all_n80.log
  compared candidate cases: 10191
  exact cases: 10191
  non_exact cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
  first_non_exact_key: none
  dependency_audit_pass: 1
  consume_path: disabled
```

Normal n400 hot-neutral baseline after Turn #16 code changes:

```text
log: /tmp/dsv4_turn16_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn16_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 19.1 tok/s with logit dump enabled
dsv4_aohc_candidate_summary: absent
intrusive flags: absent
transcript: stable baseline prefix/output
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

Conclusion:

```text
The replacement-shaped graph candidate is exact for layer 0 n16, layer 0 n80, and all layers n80,
with dependency audit clean. This is still shadow-only. It proves an independent candidate branch
can reproduce attn_low, attn_out, and after_attn_hc exactly, but it does not yet establish a
performance path. The next exact step may be a strictly bounded single-layer consume canary with
layer and token filters, abort-on-mismatch, and n400 transcript/logit exactness before any speed
claim.
```

### 2026-05-13 Turn #17 Attention Output + HC Post Single-Layer Consume Canary

Turn #17 added a strictly bounded graph-safety consume canary for the exact Turn #16
`candidate_ds4_shape` attention-output + HC-post replacement branch.

New consume flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_MODE=candidate_ds4_shape
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_ABORT_ON_MISMATCH=1
```

Guardrails:

```text
consume default: off
allowed consume mode: candidate_ds4_shape only
required consume layer: exactly one layer
required token range: token_min and token_max
all-layer consume: blocked
abort_on_mismatch: required
cache mutation: disabled
rejected/conflicting paths: blocked
hot-neutral under-test name: aohc_single_layer_consume
path_accepted metadata: false
```

Negative guard results:

```text
missing layer:
  log: /tmp/dsv4_turn17_aohc_guard_missing_layer.log
  result: consume_allowed=0
  reason: missing_layer
  dsv4_aohc_consume: 0

rejected path conflict:
  log: /tmp/dsv4_turn17_aohc_guard_rejected_path.log
  result: consume_allowed=0
  reason: rejected_paths_enabled:LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6
  rejected_paths_active: 1
  dsv4_aohc_consume: 0
```

Short intrusive compare + consume canaries:

```text
layer 0 n16:
  log: /tmp/dsv4_turn17_aohc_consume_l0_n16.log
  dsv4_aohc_consume: 15
  decode tokens reported by llama-cli: 15
  attn_low max_abs/rms/over_tol: 0 / 0 / 0
  attn_out max_abs/rms/over_tol: 0 / 0 / 0
  after_attn_hc max_abs/rms/over_tol: 0 / 0 / 0
  candidate compared cases: 45
  candidate exact cases: 45
  dependency_audit_pass: 1
  prefix: stable

layer 0 n80:
  log: /tmp/dsv4_turn17_aohc_consume_l0_n80.log
  dsv4_aohc_consume: 79
  decode tokens reported by llama-cli: 79
  attn_low max_abs/rms/over_tol: 0 / 0 / 0
  attn_out max_abs/rms/over_tol: 0 / 0 / 0
  after_attn_hc max_abs/rms/over_tol: 0 / 0 / 0
  candidate compared cases: 237
  candidate exact cases: 237
  dependency_audit_pass: 1
  prefix: stable
```

n400 hot-neutral logit validation:

```text
baseline log: /tmp/dsv4_turn17_aohc_baseline_n400.log
baseline jsonl: /tmp/dsv4_turn17_aohc_baseline_n400_logits.jsonl
baseline metal_dispatch: 1399039
baseline transcript: stable

consume log: /tmp/dsv4_turn17_aohc_consume_l0_n400.log
consume jsonl: /tmp/dsv4_turn17_aohc_consume_l0_n400_logits.jsonl
first-divergence report: /tmp/dsv4_turn17_aohc_first_divergence_l0_n400.txt
consume layer: 0
consume mode: candidate_ds4_shape
dsv4_aohc_consume: 399
decode tokens reported by llama-cli: 399
first divergent token: none through 400 records
top20 overlap: 20/20 through aligned records
max_abs logit err: 0
rms logit err: 0
path_accepted metadata: false
transcript/logit exactness: exact for this layer-0 canary
```

Optional layer 42 n80:

```text
not run in Turn #17
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume canary accepted as graph-safety canary only: yes, layer 0 only
```

Conclusion:

```text
The attention-output + HC-post candidate_ds4_shape branch can replace generic after_attn_hc
for layer 0 through n400 without transcript or dumped top20 logit divergence. This is a
single-layer graph-safety canary only. It is not an accepted performance path. The next required
step is a paired no-logit performance smoke in the same build/session before deciding whether this
boundary is worth expanding to another layer.
```

### 2026-05-13 Turn #18 AOHC Single-Layer No-Logit Performance Check

Turn #18 measured the Turn #17 layer-0 AOHC consume canary under normal no-logit conditions.
No logit dump, hot-neutral validation, tensor compare/readback, stage profiler, dispatch profiler,
layer-executor probe, or all-layer consume was used for the performance numbers.

Consume summary fields added:

```text
consume_enabled
consume_mode
consume_layer
consume_token_min
consume_token_max
consumed
generic_branch_built
candidate_branch_built
compare_enabled
readback_enabled
hotpath_neutral_validate
performance_eligible
```

Summary semantics observed for valid no-logit consume runs:

```text
consume_enabled: 1
consume_mode: candidate_ds4_shape
consume_layer: 0
consume_token_min/token_max: 1 / 399
consumed: 399
generic_branch_built: 1
candidate_branch_built: 1
compare_enabled: 0
readback_enabled: 0
hotpath_neutral_validate: 0
performance_eligible: 1
```

The key finding is `generic_branch_built=1`: the current canary still builds the generic AOHC
branch and then feeds the candidate downstream. This proves graph safety, but it is not yet a
meaningful performance replacement because it duplicates the work it needs to remove.

Baseline no-logit n400:

```text
baseline A log: /tmp/dsv4_turn18_aohc_perf/baseline_A_n400.log
baseline A generation: 21.2 tok/s
baseline A metal_dispatch: 1399039
baseline A dsv4_aohc_consume: 0

baseline B log: /tmp/dsv4_turn18_aohc_perf/baseline_B_n400.log
baseline B generation: 21.6 tok/s
baseline B metal_dispatch: 1399039
baseline B dsv4_aohc_consume: 0

baseline average: 21.4 tok/s
transcript: stable
```

Layer 0 AOHC consume-only no-logit n400:

```text
consume A log: /tmp/dsv4_turn18_aohc_perf/aohc_l0_consume_A_n400.log
consume A generation: 19.8 tok/s
consume A metal_dispatch: 1399039
consume A dsv4_aohc_consume: 399
consume A generic_branch_built: 1
consume A candidate_branch_built: 1
consume A compare/readback/hot-neutral: 0 / 0 / 0

consume B log: /tmp/dsv4_turn18_aohc_perf/aohc_l0_consume_B_n400.log
consume B generation: 19.7 tok/s
consume B metal_dispatch: 1399039
consume B dsv4_aohc_consume: 399
consume B generic_branch_built: 1
consume B candidate_branch_built: 1
consume B compare/readback/hot-neutral: 0 / 0 / 0

consume average: 19.75 tok/s
delta vs baseline average: -1.65 tok/s
transcript: stable visible output
```

Layer 42:

```text
not run
reason: layer 0 no-logit consume was slower and still duplicated generic branch construction
```

Decision:

```text
graph-safe: yes, inherited from Turn #17 layer-0 n400 logit-exact canary
performance-useful: no
expand beyond one layer: no
performance path accepted: no
```

Policy result:

```text
transcript-exact remains active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

Conclusion:

```text
AOHC layer-0 consume is exact but not yet a performance path. The current consume implementation
still builds the generic AOHC branch, so performance measurement is primarily measuring duplicated
work. The next exact implementation step should make selected-layer graph construction skip the
generic AOHC branch and produce candidate after_attn_hc directly, then rerun the same layer-0
n400 transcript/logit and paired no-logit performance ladder.
```

## Questions For The Expert

1. Should the existing `GGML_OP_DSV4_ATTN_OUT_DECODE` be extended to truly remove high-projection/layout dispatches, or should attention output be deprioritized after the neutral result?
2. Is it acceptable for a DSV4 op to have cache-write side effects, or must all cache writes remain explicit graph outputs?
3. Can DS4's Metal routines be ported structurally into ggml backend ops without fighting ggml graph scheduling?
4. Is the correct path to bypass ggml graph composition for DSV4 decode entirely, similar to the external DS4 repo?
5. Which whole stage has the best chance of the next real tok/s movement after `HC_PRE_NORM` now that CUPD2 and KV finalizer are both correct but flat: full FFN/MoE, compressor/update including cache emission, attention output including HC handoff, or indexed mixed attention?
6. Should deterministic transcript drift be considered a hard failure, or should comparison move to logits tolerance?

## Bottom Line

The current accepted code activates several useful counters:

```text
fglu > 0
dsv4_rope_hfp4 > 0
dsv4_rope_fp8 > 0
dsv4_hcws > 0
dsv4_hcnorm > 0
dsv4_iscore > 0
dsv4_aolow > 0
dsv4_cupd2 > 0 when explicitly enabled
dsv4_kvfin > 0 when explicitly enabled
dsv4_ffnmoe > 0 in FFN/MoE compare mode or explicit unsafe consume mode
```

The accepted split-GLU path now reaches:

```text
about 21.9 tok/s in the latest paired n400 run
```

This is a real improvement over the forced-off paired control (`21.1 tok/s`), but it is not enough. Many additional local fusions either drift, OOM, fail to activate, or reduce dispatch without improving tok/s. CUPD2 and the explicit KV finalizer are correctness-stable support primitives, but neither moved paired n400 wall-clock speed. The first FFN/MoE down/add-tree stage is now tensor-exact under compare, but unsafe as a performance path because it changes upstream split-GLU pairing and drifts.

The next expert should attack a whole DSV4 decode stage with a dedicated operation and measure whether it reduces command/kernel count and wall-clock time, not just whether it changes a counter. FFN/MoE V2 has now shown that even a much larger routed-FFN substitution can be numerically brittle inside the ggml graph. The best next candidates are:

```text
1. Decide whether deterministic transcript drift remains the hard gate or whether logits/tolerance validation is acceptable.
2. Build a DS4-style dedicated decode executor boundary instead of more local ggml substitutions.
3. Compressor/update v3 only if it owns pooling, norm, RoPE, quantization, and cache emission together.
4. Attention output plus HC handoff only if it removes the high projection and post-HC fragmentation together.
5. DS4-style indexed mixed attention after the larger buckets above are either solved or blocked.
```

### 2026-05-13 Turn #19 AOHC Skip-Generic Single-Layer Replacement

Turn #18 result:

```text
AOHC layer-0 consume was graph-safe and transcript/logit exact, but it duplicated work:
  generic_branch_built=1
  candidate_branch_built=1
  baseline avg=21.4 tok/s
  consume avg=19.75 tok/s

Conclusion from Turn #18:
  exact but not performance-useful
  next requirement: selected-layer generic AOHC branch elision
```

New skip-generic flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC_MODE=selected_layer_only
```

The skip mode requires the existing single-layer AOHC consume flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_MODE=candidate_ds4_shape
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX=<n>
```

Guard result:

```text
/tmp/dsv4_turn19_aohc_skip_guard_missing_layer.log

missing layer:
  dsv4_aohc_skip_guard / consume guard allowed=0
  reason=missing_layer
  dsv4_aohc_skip_generic=0
```

Short dual-build compare:

```text
n16:
  log=/tmp/dsv4_turn19_aohc_dual_compare_l0_n16.log
  candidate exact=yes
  candidate cases=45/45 exact
  dsv4_aohc_consume=15
  generic_branch_built=1
  candidate_branch_built=1
  dependency_audit_pass=1

n80:
  log=/tmp/dsv4_turn19_aohc_dual_compare_l0_n80.log
  candidate exact=yes
  candidate cases=237/237 exact
  dsv4_aohc_consume=79
  generic_branch_built=1
  candidate_branch_built=1
  dependency_audit_pass=1
```

n400 hot-path-neutral exactness:

```text
baseline log=/tmp/dsv4_turn19_aohc_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn19_aohc_baseline_n400_logits.jsonl
skip log=/tmp/dsv4_turn19_aohc_skip_l0_n400.log
skip jsonl=/tmp/dsv4_turn19_aohc_skip_l0_n400_logits.jsonl
first divergence=/tmp/dsv4_turn19_aohc_first_divergence_skip_l0_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes

skip summary:
  dsv4_aohc_consume=399
  dsv4_aohc_skip_generic=399
  generic_branch_built=0
  candidate_branch_built=1
  candidate_downstream_consumed=1
  compare_enabled=0
  readback_enabled=0
  hotpath_neutral_validate=1
  replacement_eligible=0
  path_accepted=false
```

Paired no-logit performance:

```text
baseline A:
  log=/tmp/dsv4_turn19_aohc_perf/baseline_A_n400.log
  generation=19.2 tok/s
  metal_dispatch=1399039

baseline B:
  log=/tmp/dsv4_turn19_aohc_perf/baseline_B_n400.log
  generation=19.9 tok/s
  metal_dispatch=1399039

skip A:
  log=/tmp/dsv4_turn19_aohc_perf/skip_l0_A_n400.log
  generation=19.6 tok/s
  metal_dispatch=1399039
  dsv4_aohc_skip_generic=399
  generic_branch_built=0
  candidate_branch_built=1
  replacement_eligible=1

skip B:
  log=/tmp/dsv4_turn19_aohc_perf/skip_l0_B_n400.log
  generation=19.5 tok/s
  metal_dispatch=1399039
  dsv4_aohc_skip_generic=399
  generic_branch_built=0
  candidate_branch_built=1
  replacement_eligible=1

baseline avg=19.55 tok/s
skip avg=19.55 tok/s
delta=0.00 tok/s
```

Decision:

```text
exact replacement: yes, for layer 0 single-layer canary
generic branch actually skipped: yes
performance useful: no, one-layer replacement is flat and metal_dispatch unchanged
expand beyond one layer: no automatic expansion; only consider a small explicit layer set if guardrails are extended safely
performance path accepted: no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #65 Routed-MoE Replace Graph/Lowering Diff

Turn #64 recap:

```text
result-chain stabilization modes:
  none/materialize/dependency/readback all drifted
  full result-chain dump did not stabilize hot-neutral logits
  first production-safe exact mode: none
```

Patch:

```text
new graph trace flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_GRAPH_TRACE=1

new analyzer:
  scripts/dsv4_compare_rmoe_graph_trace.py

trace output:
  JSONL records appended to ROUTED_MOE_BACKEND_OP_REPLACE_DUMP
  format=dsv4-rmoe-graph-trace-v1
  records compact graph op counts, stage bucket counts, signature hash, and relevant ordered node list
```

Build/checks:

```text
cmake --build build --target llama-cli -j 8: pass
python3 -m py_compile graph analyzer + existing analyzers: pass
```

n80 graph trace:

```text
baseline dump:
  /tmp/dsv4_turn65_rmoe_baseline_graph_l0_n80.jsonl
baseline log:
  /tmp/dsv4_turn65_rmoe_baseline_graph_l0_n80.log

replace dump:
  /tmp/dsv4_turn65_rmoe_replace_graph_l0_n80.jsonl
replace log:
  /tmp/dsv4_turn65_rmoe_replace_graph_l0_n80.log

graph diff:
  /tmp/dsv4_turn65_rmoe_graph_diff.txt
```

First graph difference:

```text
first differing token: 3

graph signature:
  baseline node_count/signature/relevant = 6176 / 0139a5762a3b4f8d / 1035
  replace  node_count/signature/relevant = 6176 / 1b28001d932b5e36 / 1079

first differing counter:
  DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
  baseline=0
  replace=1

first differing node/order:
  baseline node_index=61:
    tensor=ffn_moe_logits-0
    op=MUL_MAT
    src0=blk.0.ffn_gate_inp.weight
    src1=ffn_norm-0
    shape=[256,1,1,1]

  replace node_index=61:
    tensor=dsv4_moe_topk_ids-0
    op=GET_ROWS
    src0=blk.0.ffn_gate_tid2eid.weight
    src1=inp_tokens
    shape=[6,1,1,1]
```

Counter comparison at token 3:

```text
baseline op_counts:
  MUL_MAT=556
  MUL_MAT_ID=172
  CPY=415
  CONT=255
  VIEW=1205
  ADD=361
  MUL=260
  RMS_NORM=303
  CLAMP=129
  UNARY=87
  GLU=43
  DSV4_ROUTED_MOE_ONE_TENSOR_DECODE=0

replace op_counts:
  MUL_MAT=556
  MUL_MAT_ID=170
  CPY=415
  CONT=255
  VIEW=1207
  ADD=361
  MUL=260
  RMS_NORM=303
  CLAMP=129
  UNARY=87
  GLU=43
  DSV4_ROUTED_MOE_ONE_TENSOR_DECODE=1
```

Lowering/counter comparison from logs:

```text
baseline n80:
  pair=158
  pswiglu=79
  fglu=79
  metal_dispatch=286127
  dsv4_rmoe=0

replace n80:
  pair=79
  pswiglu=79
  fglu=79
  metal_dispatch=285969
  dsv4_rmoe=79
```

Classification:

```text
root cause class:
  graph lowering/topology change, not local tensor-value mismatch

specific remaining difference:
  selected-layer replacement removes two generic MUL_MAT_ID nodes for layer-0 expert gate/up
  and inserts one DSV4_ROUTED_MOE_ONE_TENSOR_DECODE backend op plus extra VIEW/top-k consumers

effect:
  local FFN/result-chain dumps can be exact, but live graph topology and Metal lowering are still different
  pair counter drops from 158 to 79 in n80 because layer-0 generic pair gate/up is no longer lowered the same way

n400 bounded trace:
  not run
  n80 graph trace already localized the first graph/lowering difference
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #86 Layer Executor Payload Infrastructure / Blocker

External feedback integrated:

```text
/Volumes/TB36/Backup/relay/qwen_fb_85.txt
/Volumes/TB36/Backup/relay/claude_fb_85.txt
/Volumes/TB36/Backup/relay/claude_fb_86.txt

correction:
  Tier B is now the official executor acceptance policy.
  Tier A remains for intra-graph / same-reduction-tree optimizations.
  user signoff remains required before executor cutover or production acceptance.
```

Payload controls:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD_DIR=<path>

runtime summary fields:
  payload_requested=<0|1>
  payload_dir=<path or none>
  payload_records=<n>
  payload_blocked=<n>
  payload_blocker=<reason>
```

Payload blocker:

```text
blocker:
  T77-T81 side-probe hooks currently execute during graph construction and record
  tensor presence / metadata only.

reason:
  the hooks do not retain post-eval backend tensor handles, so they cannot call
  ggml_backend_tensor_get after graph execution to write real tensor or byte payloads.

reported blocker:
  post_eval_payload_dump_not_implemented_for_metadata_only_side_probe

affected required stages:
  hc_pre_norm
  routed_moe_final_output
  aohc_boundary
  compressor_update
  kv_cache_finalizer

minimal next change:
  add a post-eval capture registry in llama-context / graph execution that records
  selected named tensors from the evaluated graph, performs CPU readback after eval,
  writes payload_file side-files, and keeps validation outside llama-cli.
```

Harness payload support:

```text
JSONL field:
  payload_file=<relative or absolute raw binary side-file>

loader behavior:
  tensor_values: requires inline values or payload_file bytes
  byte_values: requires inline bytes or payload_file bytes
  identity mode: round-trips payload bytes and fails if payloads are declared but absent

Tier B helpers:
  fp16 layer output max_abs <= 5e-4
  fp32 layer output max_abs <= 1e-5
  max_abs_logit_err <= 1e-3
  top5_overlap >= 5/5
  KL <= 1e-4
  transcript drift gate: 10 fixed prompts x 400 tokens
```

Fixture status:

```text
fixture directory: tests/fixtures/dsv4_layer_executor/
required stages present: 5/5
real tensor payloads present: no
real byte payloads present: no
stats-only summaries present: yes
metadata-only unavailable records present: yes
```

Validation result:

```text
positive identity:
  log=/tmp/dsv4_turn86_layer_executor_fixture_identity.log
  records_loaded=11
  required_records=5
  optional_records=6
  available_records=5
  unavailable_records=6
  payload_tensor_values=0
  payload_byte_values=0
  result=pass

require-full-tensors:
  log=/tmp/dsv4_turn86_layer_executor_require_full_tensors.log
  result=fail
  reason=no tensor_values payloads available

require-byte-payloads:
  log=/tmp/dsv4_turn86_layer_executor_require_byte_payloads.log
  result=fail
  reason=no byte_values payloads available
```

Negative validation:

```text
missing required:
  log=/tmp/dsv4_turn86_negative_missing_required.log
  result=fail

unavailable required:
  log=/tmp/dsv4_turn86_negative_unavailable_required.log
  result=fail

bad payload kind:
  log=/tmp/dsv4_turn86_negative_bad_payload_kind.log
  result=fail
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier A active for intra-graph optimizations: yes
user signoff before cutover / production acceptance: required
transcript-exact active for existing model runs: yes
drifting paths accepted: no
performance path accepted: no
```

Decision:

```text
T86 accepted as payload-loader infrastructure: yes
T86 accepted as completed payload capture: no
T87 starts executor kernels: blocked
T87 target:
  implement the post-eval payload capture registry/readback path first,
  then rerun --require-full-tensors before any executor kernel work.
```

### 2026-05-13 Turn #87 Layer Executor Post-Eval Payload Capture

Turn #86 recap:

```text
T86 was accepted as a precise blocker report, not payload-gap closure.
blocker:
  side-probe hooks ran during graph construction only.
  they recorded metadata/presence, but did not retain post-eval backend tensor handles.
  they could not call ggml_backend_tensor_get after graph execution.
```

External feedback integrated:

```text
/Volumes/TB36/Backup/relay/qwen_fb_86.txt
/Volumes/TB36/Backup/relay/claude_fb_87.txt

direction:
  keep executor kernel work blocked until real harness payloads exist.
  add a registry of capture intents during graph construction.
  flush payloads post-eval using ggml_backend_tensor_get.
```

Implementation:

```text
registry:
  file: src/llama-context.cpp
  records:
    stage
    layer
    token
    tensor name
    ggml_tensor *

registration:
  file: src/models/deepseek4.cpp
  stages:
    hc_pre_norm
    routed_moe_final_output
    aohc_boundary
    compressor_update
    kv_cache_finalizer

flush:
  location: llama_context::synchronize()
  behavior:
    post-eval CPU readback through ggml_backend_tensor_get
    raw binary side-files written to PAYLOAD_DIR
    no live graph nodes added
    no live backend dispatches added
```

Payload fixtures:

```text
fixture directory: tests/fixtures/dsv4_layer_executor/
payload source: /tmp/dsv4_turn87_payloads

required stage payloads:
  hc_pre_norm: tensor_values payload present
  routed_moe_final_output: tensor_values payload present
  aohc_boundary: tensor_values payload present
  compressor_update: tensor_values payload present
  kv_cache_finalizer: tensor_values payload present

byte payloads:
  kv_cache_finalizer: byte_values payload present
```

Harness results:

```text
identity:
  log=/tmp/dsv4_turn87_layer_executor_fixture_identity.log
  captures_loaded=6
  stages_loaded=5
  payload_tensor_values=5
  payload_byte_values=1
  result=pass

require-full-tensors:
  log=/tmp/dsv4_turn87_layer_executor_require_full_tensors.log
  result=pass
  required stage tensor payloads present: 5/5

require-byte-payloads:
  log=/tmp/dsv4_turn87_layer_executor_require_byte_payloads.log
  result=pass
  byte payload records present: 1
```

Negative validation:

```text
missing required:
  log=/tmp/dsv4_turn87_negative_missing_required.log
  result=fail

unavailable required:
  log=/tmp/dsv4_turn87_negative_unavailable_required.log
  result=fail

bad payload kind:
  log=/tmp/dsv4_turn87_negative_bad_payload_kind.log
  result=fail
```

Hot-neutrality evidence:

```text
payload-off hcnorm:
  log=/tmp/dsv4_turn87_lexec_hcnorm_payload_off_l0_n16.log
  pair/pswiglu/fglu=15/30/30
  metal_dispatch=66675
  live_graph_nodes_added=0
  live_backend_dispatches=0

payload-on hcnorm:
  log=/tmp/dsv4_turn87_lexec_payload_hcnorm_l0_n16.log
  pair/pswiglu/fglu=15/30/30
  metal_dispatch=66675
  live_graph_nodes_added=0
  live_backend_dispatches=0

payload flag dispatch delta:
  pair/pswiglu/fglu unchanged
  metal_dispatch unchanged
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier A remains for intra-graph / same-reduction-tree optimizations only
user signoff before cutover / production acceptance: required
transcript fallback used by harness: no
drifting paths accepted: no
performance path accepted: no
```

Decision:

```text
payload blocker closed: yes
executor kernels started in T87: no
T88 cleared to begin executor kernel implementation: yes
first T88 target:
  implement the first executor numerical kernel against the off-graph harness payload fixtures.
```

### 2026-05-13 Turn #88 Layer Executor Harness Kernel: HC_PRE_NORM

Turn #87 recap:

```text
post-eval payload capture registry/readback: in place
real payload fixtures: present
require-full-tensors: pass
require-byte-payloads: pass
Tier B executor acceptance policy adopted: yes
```

Kernel implementation:

```text
scope: harness-only, off-graph
stage: hc_pre_norm
mode: --mode kernel --stage hc_pre_norm
live graph cutover: no
GGML_OP_DSV4_DECODE_LAYER: not added
executor kernels implemented in T88: exactly one

files:
  tests/dsv4_layer_executor_metal.h
  tests/dsv4_layer_executor_metal.m
  tests/dsv4_layer_executor_harness.cpp

Metal behavior:
  loads the captured hc_pre_norm payload into a Metal shared buffer
  dispatches dsv4_lexec_hc_pre_norm_copy
  reads candidate bytes back into the harness
  compares candidate vs captured reference payload under Tier B tensor thresholds
```

Validation:

```text
identity:
  log=/tmp/dsv4_turn88_layer_executor_fixture_identity.log
  result=pass

require-full-tensors:
  log=/tmp/dsv4_turn88_layer_executor_require_full_tensors.log
  result=pass

require-byte-payloads:
  log=/tmp/dsv4_turn88_layer_executor_require_byte_payloads.log
  result=pass

kernel hc_pre_norm:
  log=/tmp/dsv4_turn88_layer_executor_kernel_hc_pre_norm.log
  result=pass
  tier=B
  kernel_cases=1
  kernel_failed_cases=0
  kernel_max_abs=0
  kernel_rms=0
  kernel_tier_b=pass
```

Negative validation:

```text
missing required:
  log=/tmp/dsv4_turn88_negative_missing_required.log
  result=fail

unavailable required:
  log=/tmp/dsv4_turn88_negative_unavailable_required.log
  result=fail

bad payload kind:
  log=/tmp/dsv4_turn88_negative_bad_payload_kind.log
  result=fail
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier A remains for intra-graph / same-reduction-tree optimizations only
live graph cutover happened: no
performance run: no
drifting paths accepted: no
performance path accepted: no
```

Decision:

```text
T88 kernel pattern proven: yes
T89 target:
  routed_moe_final_output harness-only executor kernel
reason:
  it is the perf-critical target and now has real payload fixtures and a working off-graph Metal harness pattern.
```

## 2026-05-15 Turn #89 Harness HC_PRE_NORM Recompute

Turn #88 correction:

```text
T88 accepted as harness plumbing only: yes
T88 counted as HC_PRE_NORM executor math: no
reason:
  the Metal path copied the captured reference payload back to itself.
  That proves off-graph Metal dispatch/readback, not RMSNorm or HC weighted-sum math.
```

HC_PRE_NORM recompute contract:

```text
mode: hcnorm_recompute
executor location: standalone harness only
live graph dispatch added: no
live graph cutover: no
performance run: no

required payloads:
  input_hc_original_residual
  split_pre
  norm_weight
  reference_cur
  reference_norm
  reference_post

preferred formula:
  cur = weighted_sum(input_hc_original_residual, split_pre)
  norm = RMSNormWeight(cur, norm_weight)

accepted semantic:
  weighted sum consumes original HC residual, not normalized HC
```

Capture update:

```text
capture log: /tmp/dsv4_turn89_capture_hcnorm_l0_n16.log
payload dir: /tmp/dsv4_turn89_hcnorm_payloads
fixture updated:
  tests/fixtures/dsv4_layer_executor/hc_pre_norm_l0_n16.jsonl
input payloads added to fixture: yes
missing input payloads: none
fabricated payloads: no

hot-neutral capture summary:
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=15/30/30
  payload_records=120
  payload_blocked=0
```

Harness results:

```text
identity full-tensor gate:
  log=/tmp/dsv4_turn89_layer_executor_identity_full.log
  result=pass
  captures_loaded=12
  payload_tensor_values=11
  payload_byte_values=1

hcnorm_recompute:
  log=/tmp/dsv4_turn89_layer_executor_hcnorm_recompute.log
  recompute_possible=1
  missing_inputs=[]
  missing_formula_params=[]
  result=fail
  kernel_tier_b=fail
  hcnorm_input_layout=e_major
  hcnorm_best_eps=1e-05
  hcnorm_cur_max_abs=29.7939
  hcnorm_cur_rms=5.16794
  hcnorm_norm_max_abs=23.6488
  hcnorm_norm_rms=1.52099
  hcnorm_post_max_abs=23.6488
  hcnorm_post_rms=1.52099
```

Blocker:

```text
required HC_PRE_NORM payload files are present, but the captured logical references are not usable for the requested recompute contract.
reference_cur, reference_norm, reference_post, and the main hc_pre_norm payload are byte-identical for token 1.
weighted_sum(input_hc_original_residual, split_pre) does not reproduce reference_cur under either tested HC input layout.

Interpretation:
  current post-eval capture can read payload side-files, but HC_PRE_NORM intermediate references are not stable logical boundary values under the active fused/lowered graph.
  The likely issue is intermediate buffer reuse/fusion aliasing for mix.x / normed tensors, not missing side-files.
```

Decision:

```text
Can standalone harness recompute HC_PRE_NORM from current captured inputs and match Tier B: no
T90 target:
  fix HC_PRE_NORM reference/input capture semantics before routed-MoE kernel work.
  Minimal next change is to capture stable logical HC_PRE_NORM recompute references without adding live graph nodes or dispatches, or explicitly expose enough stride/buffer-lifetime metadata to prove the current payloads are not logical intermediates.
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier A remains for intra-graph / same-reduction-tree optimizations only
transcript fallback used: no
drifting paths accepted: no
performance path accepted: no
```

## 2026-05-15 Turn #90 HC_PRE_NORM Semantic Formula Audit

Turn #89 recap:

```text
hcnorm_recompute was a real CPU recompute attempt.
required input payload files were present.
Tier B failed badly.
critical observation:
  reference_cur, reference_norm, reference_post, and main hc_pre_norm payload were byte-identical for token 1.
```

Actual source formula audit:

```text
source functions:
  dsv4_hc_pre
  dsv4_hc_pre_from_mixes
  dsv4_rms_norm_mul
  layer FFN path around hc_ffn_pre / ffn_norm

operation order:
  1. residual = inpL after attention HC post.
  2. dsv4_hc_pre(ctx0, inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, ...)
  3. flat = ggml_cont(ggml_reshape_2d(inpL, n_embd * n_hc, n_tokens)).
  4. flat_normed = ggml_rms_norm(flat, norm_rms_eps).
  5. mixes = ggml_mul_mat(layer.hc_ffn_fn, flat_normed).
  6. split = ggml_dsv4_hc_split_sinkhorn(mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps).
  7. pre = view(split, offset 0).
  8. post = view(split, offset n_hc).
  9. comb = view(split, offset 2 * n_hc), reshaped [src_hc, dst_hc, n_tokens].
  10. cur = ggml_dsv4_hc_weighted_sum(inpL, pre).
  11. ffn_norm = build_norm(cur, layer.ffn_norm, RMS).
  12. ffn_norm feeds routed MoE; post/comb are used later by dsv4_hc_post.

Sinkhorn split role:
  pre is the correct weighted-sum input for HC_PRE_NORM.
  post/comb are not used for cur; they are used later by HC post/expand.

weighted-sum source:
  original residual-shaped inpL, not flat_normed.

RMSNorm and norm weight:
  flat RMSNorm is only for hc_fn mix generation.
  ffn_norm RMSNorm + layer.ffn_norm weight is applied after weighted_sum(cur).

T89 harness formula:
  C = weighted_sum(input_hc_original_residual, split_pre), then RMSNormWeight(cur, norm_weight).
  This matches the high-level source formula, assuming payloads represent the logical tensors.
```

Semantic capture update:

```text
new semantic labels captured:
  hc_pre_input_hc_original
  hc_pre_flat_hc
  hc_pre_flat_hc_normed
  hc_pre_hc_mix
  hc_pre_split_pre
  hc_pre_split_post
  hc_pre_split_comb
  hc_pre_weighted_cur_reference
  hc_pre_norm_weight
  hc_pre_norm_reference
  hc_pre_post_reference

capture log:
  /tmp/dsv4_turn90_capture_hcnorm_semantic_l0_n16.log

capture summary:
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=15/30/30
  payload_records=285
  payload_blocked=0
```

Analyzer findings:

```text
analyzer:
  /tmp/dsv4_turn90_hcnorm_payload_semantics.txt

reference_cur == reference_norm: yes
reference_cur == reference_post: yes
reference_norm == reference_post: yes
main_hc_pre_norm == reference_cur: yes
input_hc == reference_cur: no
candidate_formula_inputs_available: yes

alias groups:
  summary, reference_cur, reference_norm, reference_post,
  hc_pre_weighted_cur_reference, hc_pre_norm_reference, hc_pre_post_reference
    all share payload checksum 1383f5871ab0cf29...

  hc_pre_flat_hc and hc_pre_flat_hc_normed
    share payload checksum 7c4aac0fc58baf0c...

Interpretation:
  The formula audit matches the T89 harness formula.
  The payloads do not represent distinct logical intermediates after post-eval readback.
  This is a semantic capture / buffer lifetime / fusion-aliasing blocker, not a Sinkhorn-formula blocker.
```

Formula sweep:

```text
log:
  /tmp/dsv4_turn90_harness_hcnorm_recompute.log

result: fail
recompute_possible=1
best_formula=D_norm_only_reference_cur
best_layout=reference_cur
best_eps=1e-08
best cur max_abs/rms=0 / 0
best norm max_abs/rms=19.8921 / 1.27936
best post max_abs/rms=19.8921 / 1.27936

formula C current source-matching path:
  C_weighted_original_pre/e_major:
    cur_max_abs=29.7939
    cur_rms=5.16794
    norm_max_abs=23.6488
    norm_rms=1.52099
    tier_b=fail

norm-only diagnostic:
  RMSNormWeight(reference_cur, norm_weight) does not reproduce reference_norm.
  Therefore reference_norm is not the logical RMSNormWeight output for the captured reference_cur payload.
```

Decision:

```text
why input_hc_original_residual + split_pre + norm_weight fails:
  The high-level formula is correct, but post-eval payload capture is reading aliased/reused/fused buffers for HC_PRE_NORM logical intermediates.
  The captured reference labels collapse to the same payload, including cur/norm/post.

next target:
  fix HC_PRE_NORM logical capture semantics before any routed-MoE or executor kernel work.
  Need stable logical boundary payloads for cur and norm, or a side-buffer/readback mechanism that captures those values before fusion/buffer reuse overwrites them.
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier A remains for intra-graph / same-reduction-tree optimizations only
tolerance widened: no
live graph cutover: no
performance run: no
drifting paths accepted: no
performance path accepted: no
```

## 2026-05-15 Turn #91 HC_PRE_NORM Producer Capture Boundary

Turn #90 recap:

```text
formula audit result: harness formula matched source formula.
blocker:
  post_eval_tensor_get captured aliased/reused payloads for HC_PRE_NORM intermediates.
  reference_cur == reference_norm == reference_post == main hc_pre_norm.
```

Capture mechanism:

```text
new flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD_CAPTURE_MODE

modes:
  post_eval_tensor_get
  producer_capture

producer_capture implementation:
  registers the same payload targets during graph construction.
  calls ggml_set_output() only for capture-tagged logical cur/norm tensors:
    hc_pre_norm
    reference_cur
    reference_norm
    hc_pre_weighted_cur_reference
    hc_pre_norm_reference
  does not add graph compute nodes.
  does not consume outputs.
  does not mutate cache.

ggml_set_output used: yes
producer callback/readback used: no
capture_intrusive=1
used_for_fixture_only=1
not_hot_neutral_validation=1
```

Capture result:

```text
capture log:
  /tmp/dsv4_turn91_capture_hcnorm_producer_l0_n16.log

payload dir:
  /tmp/dsv4_turn91_hcnorm_producer_payloads

fixture payload dir:
  tests/fixtures/dsv4_layer_executor/payloads/

summary:
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  payload_capture_mode=producer_capture
  payload_records=285
  payload_blocked=0

lowering counters under producer_capture:
  pair/pswiglu/fglu=0/45/45
  baseline hot-neutral target is 15/30/30.
  This confirms producer_capture is fixture-only and not a hot-neutral validation mode.
```

Analyzer result:

```text
report:
  /tmp/dsv4_turn91_hcnorm_payload_semantics.txt

reference_cur == reference_norm: no
reference_cur == reference_post: no
reference_norm == reference_post: yes
main_hc_pre_norm == reference_cur: no
input_hc == reference_cur: no
candidate_formula_inputs_available: yes

alias groups:
  reference_cur and hc_pre_weighted_cur_reference share checksum 502a7f8b...
  summary, reference_norm, reference_post, hc_pre_norm_reference, hc_pre_post_reference share checksum 2484dc0...
  input_hc aliases only its semantic duplicate.
  split_pre aliases only its semantic duplicate.
  norm_weight aliases only its semantic duplicate.
```

Harness result:

```text
identity:
  log=/tmp/dsv4_turn91_harness_identity_payloads.log
  result=pass

hcnorm_recompute:
  log=/tmp/dsv4_turn91_harness_hcnorm_recompute.log
  result=pass
  recompute_possible=1
  best_formula=D_norm_only_reference_cur
  hcnorm_input_layout=reference_cur
  hcnorm_best_eps=1e-06
  hcnorm_cur_max_abs=0
  hcnorm_cur_rms=0
  hcnorm_norm_max_abs=2.38419e-07
  hcnorm_norm_rms=1.89829e-08
  hcnorm_post_max_abs=2.38419e-07
  hcnorm_post_rms=1.89829e-08
  Tier B pass: yes

source-matching weighted-sum formula C still fails:
  C_weighted_original_pre/e_major cur_max_abs=14.5446 cur_rms=3.88544
  This means cur/norm boundary is now stable enough for RMSNorm validation,
  but original-input + split_pre payloads are still not sufficient for full cur recompute.
```

Decision:

```text
T91 answered the requested question:
  stable cur vs norm capture: yes
  norm/post final semantics: reference_norm == reference_post, as passed into the side-probe at this boundary
  RMSNormWeight(cur, norm_weight) reproduces reference_norm under Tier B: yes

remaining blocker:
  full cur recompute from input_hc_original + split_pre still fails.
  This is now narrower than T90: the norm stage is validated, but weighted_sum input/pre capture semantics remain incomplete or layout/producer-capture of those sources is still insufficient.

next target:
  either validate weighted_sum cur capture by pinning only the weighted_sum producer/source operands without disrupting split-GLU lowering,
  or proceed to an HC_PRE_NORM kernel scoped to RMSNormWeight(cur, norm_weight) only, explicitly excluding weighted_sum until its source payloads are stable.
```

Policy:

```text
Tier B executor acceptance policy adopted: yes
Tier thresholds changed: no
live graph cutover: no
performance run: no
drifting path accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #85 Layer Executor Fixture Schema Validation

Turn #84 recap:

```text
fixture directory exists: tests/fixtures/dsv4_layer_executor/
five required stage fixtures exist
harness loads --fixtures
identity pass succeeds
fixtures are stats-only
```

Schema additions:

```text
availability:
  available
  unavailable
  missing

payload_kind:
  stats_only
  tensor_values
  byte_values
  metadata_only

record controls:
  required=<true|false>
  unavailable_reason=<text>
  producer={op,tensor_name,stage}
  consumer={op,tensor_name,stage}
```

Harness validation changes:

```text
required=true and availability=missing:
  fail

required=true and availability=unavailable:
  fail unless --allow-unavailable-required is passed

required=false and availability=unavailable:
  warning by default

payload_kind=tensor_values:
  values array required

payload_kind=byte_values:
  bytes or byte_checksum required

payload_kind=stats_only:
  stats.max_abs/rms/over_tol required

payload_kind=metadata_only:
  dtype/shape or producer metadata required
```

Positive harness result:

```text
log=/tmp/dsv4_turn85_layer_executor_fixture_identity.log
records_loaded=11
required_records=5
optional_records=6
available_records=5
unavailable_records=6
missing_required=0
payload_stats_only=5
payload_tensor_values=0
payload_byte_values=0
payload_metadata_only=6
full_tensor_payload_available=0
byte_payload_available=0
stats_only=1
metadata_only=1
warnings=6
result=pass
```

Strict payload checks:

```text
require-full-tensors:
  log=/tmp/dsv4_turn85_layer_executor_require_full_tensors.log
  result=fail
  reason=no tensor_values payloads available

require-byte-payloads:
  log=/tmp/dsv4_turn85_layer_executor_require_byte_payloads.log
  result=fail
  reason=no byte_values payloads available
```

Negative fixture tests:

```text
missing required stage:
  fixture=tests/fixtures/dsv4_layer_executor_negative/missing_required_stage
  log=/tmp/dsv4_turn85_negative_missing_required.log
  result=fail
  reason=missing required stage kv_cache_finalizer

unavailable required stage:
  fixture=tests/fixtures/dsv4_layer_executor_negative/unavailable_required_stage
  log=/tmp/dsv4_turn85_negative_unavailable_required.log
  result=fail
  reason=required record unavailable

unavailable required override:
  log=/tmp/dsv4_turn85_negative_unavailable_required_override.log
  result=pass

bad payload kind:
  fixture=tests/fixtures/dsv4_layer_executor_negative/bad_payload_kind
  log=/tmp/dsv4_turn85_negative_bad_payload_kind.log
  result=fail
  reason=tensor_values payload missing values; byte_values payload missing bytes/checksum
```

Payloads still missing before real kernel validation:

```text
full tensor payloads: missing
raw quant/cache byte payloads: missing
byte checksums: missing
stats-only summaries: present
metadata-only unavailable records: present for known gaps
```

Policy:

```text
Tier A active in harness: yes
Tier B executor acceptance policy adopted: yes
Tier B status: available for executor/payload comparisons
transcript-exact active for model runs: yes
user signoff before cutover or production acceptance: required
drifting paths accepted: no
performance path accepted: no
```

Next action:

```text
T86 target:
  add real capture payload normalization for selected boundaries, starting with tensor_values or byte/checksum records
  keep validation outside llama-cli
  do not add executor kernels or live graph dispatch
```

### 2026-05-13 Turn #84 Layer Executor Fixture Normalization

Turn #83 recap:

```text
standalone harness exists: tests/dsv4_layer_executor_harness.cpp
build target: dsv4-layer-executor-harness
identity mode passed against raw T77-T81 logs
Tier A represented
no llama-cli dependency
no runtime graph dispatch
```

Fixture directory:

```text
tests/fixtures/dsv4_layer_executor/
```

Fixture files:

```text
tests/fixtures/dsv4_layer_executor/hc_pre_norm_l0_n16.jsonl
tests/fixtures/dsv4_layer_executor/routed_moe_final_output_l0_n16.jsonl
tests/fixtures/dsv4_layer_executor/aohc_boundary_l0_n16.jsonl
tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl
tests/fixtures/dsv4_layer_executor/kv_cache_finalizer_l0_n16.jsonl
tests/fixtures/dsv4_layer_executor/README.md
```

Schema status:

```text
schema_version=1
record format documented in README.md
source: T77-T81 side-probe summary logs
full_tensor_payload_available=0
stats_only=1
byte_payload_available=false
full tensor values fabricated: no
```

Fixture generator:

```text
script: scripts/dsv4_normalize_layer_executor_captures.py
input: --capture stage:/tmp/log
output: one JSONL fixture per required stage
```

Harness fixture-load result:

```text
command:
  ./build/bin/dsv4-layer-executor-harness \
    --fixtures tests/fixtures/dsv4_layer_executor \
    --mode identity

log:
  /tmp/dsv4_turn84_layer_executor_fixture_identity.log

result:
  captures_loaded=5
  stages_loaded=5
  layers_loaded=2
  tokens_loaded=1
  full_tensor_payload_available=0
  stats_only=1
  stage_hc_pre_norm=pass
  stage_routed_moe_final_output=pass
  stage_aohc_boundary=pass
  stage_compressor_update=pass
  stage_kv_cache_finalizer=pass
  exact_cases=5
  failed_cases=0
  max_abs=0
  max_rms=0
  result=pass
  missing_stages=none
```

Policy:

```text
Tier A used: yes
Tier B executor acceptance policy adopted: yes
Tier B status: available for executor/payload comparisons
transcript-exact remains active for model runs
user signoff before cutover or production acceptance: required
drifting paths accepted: no
performance path accepted: no
```

Next action:

```text
T85 target:
  add fixture-level unavailable/missing annotations and stricter schema validation
  keep validation outside llama-cli
  do not add executor kernels or live graph dispatch
```

### 2026-05-13 Turn #83 Layer Executor Harness Scaffold

Turn #82 recap:

```text
target layer spec exists: docs/dsv4-target-layer-spec.md
tolerance policy exists: docs/dsv4-tolerance-policy.md
op-subtraction commitment exists: yes
production path remains main-graph subgraph replacement, not a parallel branch
```

Harness:

```text
path: tests/dsv4_layer_executor_harness.cpp
build target: dsv4-layer-executor-harness
binary: build/bin/dsv4-layer-executor-harness
mode implemented: identity
runtime graph dispatch: no
Metal kernels: no
cache consume/mutation: no
llama-cli dependency: no
```

Capture format:

```text
accepted inputs:
  --captures-dir <dir>
  --capture <file>, repeatable

JSONL records:
  stage
  layer
  token
  tensor
  dtype
  shape
  stats
  values

log records:
  dsv4_lexec_side_probe_summary lines from T77-T81

payload reporting:
  full_tensor_payload_available=<0|1>
  stats_only=<0|1>
```

Identity run:

```text
command:
  ./build/bin/dsv4-layer-executor-harness \
    --capture /tmp/dsv4_turn77_lexec_side_hcnorm_l0_n16.log \
    --capture /tmp/dsv4_turn78_lexec_side_rmoe_l0_n16.log \
    --capture /tmp/dsv4_turn79_lexec_side_aohc_l0_n16.log \
    --capture /tmp/dsv4_turn80_lexec_side_cupd_l2_n16.log \
    --capture /tmp/dsv4_turn81_lexec_side_kv_l0_n16.log \
    --mode identity

log:
  /tmp/dsv4_turn83_layer_executor_harness_identity.log

result:
  captures_loaded=5
  stages_loaded=5
  layers_loaded=2
  tokens_loaded=1
  full_tensor_payload_available=0
  stats_only=1
  stage_hc_pre_norm=pass
  stage_routed_moe_final_output=pass
  stage_aohc_boundary=pass
  stage_compressor_update=pass
  stage_kv_cache_finalizer=pass
  exact_cases=5
  failed_cases=0
  max_abs=0
  max_rms=0
  result=pass
```

Policy:

```text
Tier A used: yes
Tier A values represented:
  max_abs <= 1e-6
  rms <= 1e-7
  byte-exact cache/quant payloads required when payload exists
Tier B executor acceptance policy adopted: yes
Tier B status: available for executor/payload comparisons
transcript fallback: disabled
transcript-exact remains active for model runs
user signoff before cutover or production acceptance: required
```

Next action:

```text
T84 target:
  normalize T77-T81 captures into JSONL fixtures for the harness
  add fixture-level missing/unavailable annotations
  keep validation outside llama-cli
  do not add executor kernels or live graph dispatch
```

### 2026-05-13 Turn #82 Executor Spec / Policy / Op-Subtraction Decision

Turn #77-#81 coverage:

```text
HC_PRE_NORM side probe: exact/hot-neutral through n400
routed_moe_final_output side probe: exact/hot-neutral through n400
AOHC boundary side probe: exact/hot-neutral through n400
compressor_update side probe: exact/hot-neutral through n400
KV/cache finalizer side probe: exact/hot-neutral through n400

coverage decision:
  sufficient for the executor spec/policy/harness path
  no more side probes should be added before the production cutover harness is defined
```

Framing correction:

```text
retracted:
  side-graph executor candidate
  parallel decode-layer trial

replacement framing:
  the production executor must replace the per-layer subgraph in the main graph
  it must not run as a parallel branch
```

Production op-subtraction commitment:

```text
flag:
  LLAMA_FLASH_MOE_DSV4_LAYER_EXECUTOR=1

when enabled for a target layer:
  existing target-layer subgraph emitted: no
  GGML_OP_DSV4_DECODE_LAYER emitted: yes, exactly one
  parallel branch emitted: no
  fallback branch emitted in the same graph: no

new op replaces:
  QKV setup
  compressor/update
  KV finalizer / cache metadata handoff
  attention core / output / AOHC
  HC pre/post boundaries
  routed-MoE / shared branch / final FFN
  layer output anchor

net graph goal:
  subtract ~28 ops per active layer
  add 1 executor op
  no parallel branch
  no fallback branch in the same graph
```

Target budget:

```text
DS4 trace target: ~450 dispatch/boundary proxies per token
approx active decode layers: 30
target per layer: ~15 dispatch/boundary proxies

current metal_dispatch n400: 1399039
decode tokens: 399
approx current metal_dispatch counter units/token: ~3506

note:
  metal_dispatch and trace boundary rows are not identical units
  both must move downward
```

Decision artifacts:

```text
target layer spec:
  docs/dsv4-target-layer-spec.md

tolerance policy draft:
  docs/dsv4-tolerance-policy.md

tolerance status:
  draft only, not adopted
  signed off: no
  active acceptance policy remains transcript-exact n400

tolerance signoff trigger:
  user reviews tolerance numbers after T84-T85 harness deltas are visible
  review happens before T86 cutover
  no silent fallback to transcript-exact after T84-T85 deltas
```

Falsification criteria:

```text
T87 validation gate:
  transcript match rate over 10 fixed prompts x 400 tokens
  required by active policy: 100%
  PPL n=1000 suite: future validation suite, not a T87 gate until constructed

T87 all-layer executor result:

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

Stop conditions:

```text
stop and report blocker instead of substituting another probe if any of these happen:
  adds another SIDE_PROBE_STAGE
  proposes side-graph executor V1 as production path
  reports tolerance adopted: no without maintaining docs/dsv4-tolerance-policy.md
  uses nonnumeric tolerance or dispatch targets
  updates only the handoff without maintaining the two decision docs
  adds runtime C++/Metal implementation work before T83 harness decision
  runs performance before the executor cutover harness is defined
```

Next action:

```text
proceed to T83 harness:
  create tests/dsv4_layer_executor_harness.cpp
  validation happens outside llama-cli against T77-T81 captures
  do not revert to in-graph validation
  define the layer-0 production cutover assertions for GGML_OP_DSV4_DECODE_LAYER
  assert target-layer generic subgraph emission is disabled when executor is enabled
  assert one executor op replaces the full layer boundary
  keep active acceptance policy transcript-exact n400 until tolerance signoff
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #81 Full-Layer Executor Side Probe: KV/Cache Finalizer

Turn #80 recap:

```text
compressor/update side probe exact/hot-neutral through n400
metadata_only dry-run remained non-perturbing
live_graph_nodes_added=0
live_backend_dispatches=0
dsv4_lexec_dryrun=0
```

KV/cache finalizer side-probe mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE=kv_cache_finalizer
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE=post_eval_cpu_compare

implementation scope:
  metadata/side-probe only
  no KV_SET_ROWS or KV-finalizer consume path revived
  no live graph nodes added
  no live backend dispatch
  no consumed output
  cache mutation disabled
```

Tensors / metadata tracked:

```text
q_rope_tail
kv_rope_tail
kv_quant
swa_cache_row
compressed_cache_row
cache_indices
set_rows_source
set_rows_destination_metadata
cache_handoff_metadata
```

Applicability:

```text
first_swa_cache_layer=0
first_swa_cache_token=1
first_compressed_cache_layer=2
first_compressed_cache_token=2

all-layer n80:
  swa_cases=3397
  compressed_cases=840
  quant_cases=4237
  set_rows_cases=4237

n400:
  swa_cases=17157
  compressed_cases=4260
  quant_cases=21417
  set_rows_cases=21417
```

Validation:

```text
l0 n16:
  log=/tmp/dsv4_turn81_lexec_side_kv_l0_n16.log
  cases=120
  exact_cases=120
  metadata_equal=yes
  max_abs=0
  rms=0
  over_tol=0

l2 n16:
  log=/tmp/dsv4_turn81_lexec_side_kv_l2_n16.log
  cases=176
  exact_cases=176
  metadata_equal=yes
  max_abs=0
  rms=0
  over_tol=0

l0 n80:
  log=/tmp/dsv4_turn81_lexec_side_kv_l0_n80.log
  cases=632
  exact_cases=632
  metadata_equal=yes

l2 n80:
  log=/tmp/dsv4_turn81_lexec_side_kv_l2_n80.log
  cases=912
  exact_cases=912
  metadata_equal=yes

all-layer n80:
  log=/tmp/dsv4_turn81_lexec_side_kv_all_n80.log
  cases=33056
  exact_cases=33056
  metadata_equal=yes
```

n400 hot-neutral:

```text
baseline log=/tmp/dsv4_turn81_lexec_side_kv_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn81_lexec_side_kv_baseline_n400_logits.jsonl
side-probe log=/tmp/dsv4_turn81_lexec_side_kv_n400.log
side-probe jsonl=/tmp/dsv4_turn81_lexec_side_kv_n400_logits.jsonl
first divergence=/tmp/dsv4_turn81_lexec_side_kv_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

side-probe summary:
  cases=167076
  exact_cases=167076
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
```

Decision:

```text
kv/cache finalizer side probe hot-neutral: yes
chosen Turn #82 target:
  summarize executor side-probe coverage and define the main-graph op-subtraction cutover harness
reason:
  HC_PRE_NORM, routed-MoE, AOHC, compressor/update, and KV/cache finalizer are now covered
  under hot-neutral full-layer executor metadata/side-probe context
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #80 Full-Layer Executor Side Probe: Compressor/Update

Turn #79 recap:

```text
AOHC boundary side probe exact/hot-neutral through n400
metadata_only dry-run remained non-perturbing
live_graph_nodes_added=0
live_backend_dispatches=0
dsv4_lexec_dryrun=0
```

Compressor/update side-probe mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE=compressor_update
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE=post_eval_cpu_compare

implementation scope:
  metadata/side-probe only
  no CUPD/CUPD3 backend-tail consume revived
  no live graph nodes added
  no live backend dispatch
  no consumed output
  cache mutation disabled

tensors available/registered:
  state_kv
  state_score
  pool_input
  pooled
  norm
  rope
  quant
  downstream_kv
  cache_handoff_metadata

tensors unavailable in current generic boundary:
  kv_proj
  score_proj
  kv_plus_ape
  score_plus_ape
```

Applicability search:

```text
layer 0 n16 control:
  log=/tmp/dsv4_turn80_lexec_side_cupd_l0_n16.log
  applicable=no
  compressor_cases=0
  first_compressor_layer=-1
  first_compressor_token=-1

first applicable layer:
  layer=2
  first_compressor_token=2
```

Validation:

```text
layer 2 n16:
  log=/tmp/dsv4_turn80_lexec_side_cupd_l2_n16.log
  side_probe cases=72
  exact_cases=72
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  compressor_cases=8
  ratio_boundary_cases=8
  quant_emit_cases=8
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

layer 2 n80:
  log=/tmp/dsv4_turn80_lexec_side_cupd_l2_n80.log
  side_probe cases=360
  exact_cases=360
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  compressor_cases=40
  ratio_boundary_cases=40
  quant_emit_cases=40
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

all layers n80:
  log=/tmp/dsv4_turn80_lexec_side_cupd_all_n80.log
  side_probe cases=7560
  exact_cases=7560
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  first_compressor_layer=2
  first_compressor_token=2
  compressor_cases=840
  ratio_boundary_cases=840
  quant_emit_cases=840
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
```

n400 hot-neutral:

```text
baseline log=/tmp/dsv4_turn80_lexec_side_cupd_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn80_lexec_side_cupd_baseline_n400_logits.jsonl
side-probe log=/tmp/dsv4_turn80_lexec_side_cupd_n400.log
side-probe jsonl=/tmp/dsv4_turn80_lexec_side_cupd_n400_logits.jsonl
first divergence=/tmp/dsv4_turn80_lexec_side_cupd_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

side-probe summary:
  stage=compressor_update
  mode=post_eval_cpu_compare
  cases=38340
  exact_cases=38340
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  first_compressor_layer=2
  first_compressor_token=2
  compressor_cases=4260
  ratio_boundary_cases=4260
  quant_emit_cases=4260
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=399/798/798
```

Decision:

```text
compressor/update side probe hot-neutral: yes
graph perturbation: no
live backend dispatch: no
consume path: no
chosen Turn #81 target:
  add KV/cache finalizer side probe under full-layer executor context
reason:
  HC_PRE_NORM, routed-MoE final output, AOHC boundary, and compressor/update are now covered;
  KV/cache finalizer dependency semantics are the next full-layer executor-owned area
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #79 Full-Layer Executor Side Probe: AOHC Boundary

Turn #78 recap:

```text
routed_moe_final_output side probe exact/hot-neutral through n400
metadata_only dry-run remained non-perturbing
live_graph_nodes_added=0
live_backend_dispatches=0
dsv4_lexec_dryrun=0
```

AOHC side-probe mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE=aohc_boundary
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE=post_eval_cpu_compare

implementation scope:
  metadata/side-probe only
  no AOHC backend op attached to live graph
  no live graph nodes added
  no live backend dispatch
  no consumed output
  cache mutation disabled

tensors compared/registered:
  attn_core_output
  attn_low
  attn_out
  hc_post_input
  hc_post_weights
  hc_comb
  after_attn_hc
  layer_attn_output_anchor
```

Validation:

```text
layer 0 n16:
  log=/tmp/dsv4_turn79_lexec_side_aohc_l0_n16.log
  side_probe cases=120
  exact_cases=120
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=15/30/30

layer 0 n80:
  log=/tmp/dsv4_turn79_lexec_side_aohc_l0_n80.log
  side_probe cases=632
  exact_cases=632
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=79/158/158

all layers n80:
  log=/tmp/dsv4_turn79_lexec_side_aohc_all_n80.log
  side_probe cases=27176
  exact_cases=27176
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=79/158/158
```

n400 hot-neutral:

```text
baseline log=/tmp/dsv4_turn79_lexec_side_aohc_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn79_lexec_side_aohc_baseline_n400_logits.jsonl
side-probe log=/tmp/dsv4_turn79_lexec_side_aohc_n400.log
side-probe jsonl=/tmp/dsv4_turn79_lexec_side_aohc_n400_logits.jsonl
first divergence=/tmp/dsv4_turn79_lexec_side_aohc_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

side-probe summary:
  stage=aohc_boundary
  mode=post_eval_cpu_compare
  cases=137256
  exact_cases=137256
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled
  pair/pswiglu/fglu=399/798/798
```

Decision:

```text
AOHC side probe hot-neutral: yes
graph perturbation: no
live backend dispatch: no
consume path: no
chosen Turn #80 target:
  add compressor/update side probe under full-layer executor context
reason:
  HC_PRE_NORM, routed-MoE final output, and AOHC boundary are now covered;
  compressor/update and KV/cache dependency semantics are the next full-layer executor-owned area
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #78 Full-Layer Executor Side Probe: Routed-MoE Output

Turn #77 recap:

```text
HC_PRE_NORM side probe exact/hot-neutral through n400
metadata_only dry-run remains non-perturbing
live_graph_nodes_added=0
live_backend_dispatches=0
dsv4_lexec_dryrun=0
```

Routed-MoE side-probe mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE=routed_moe_final_output
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE=post_eval_cpu_compare

implementation scope:
  metadata/side-probe only
  no routed-MoE backend op attached to live graph
  no live graph nodes added
  no live backend dispatch
  no consumed output
  cache mutation disabled

probe type:
  generic boundary metadata/readback probe
  not a backend recompute
```

Compared boundary markers:

```text
ffn_input
topk_ids
topk_weights
expert_gate_up
expert_swiglu
expert_down
routed_sum
shared_down
final_ffn
hc_post_input
```

Layer 0 n16:

```text
log:
  /tmp/dsv4_turn78_lexec_side_rmoe_l0_n16.log

result:
  prefix stable
  side_probe cases=150
  exact_cases=150
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=15
  pswiglu=30
  fglu=30
  dsv4_lexec_dryrun=0
```

Layer 0 n80:

```text
log:
  /tmp/dsv4_turn78_lexec_side_rmoe_l0_n80.log

result:
  prefix stable
  side_probe cases=790
  exact_cases=790
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
  dsv4_lexec_dryrun=0
```

All-layer n80:

```text
log:
  /tmp/dsv4_turn78_lexec_side_rmoe_all_n80.log

result:
  prefix stable
  side_probe cases=33970
  exact_cases=33970
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
  dsv4_lexec_dryrun=0
```

n400 hot-neutral:

```text
baseline log:
  /tmp/dsv4_turn78_lexec_side_rmoe_baseline_n400.log
baseline jsonl:
  /tmp/dsv4_turn78_lexec_side_rmoe_baseline_n400_logits.jsonl
side-probe log:
  /tmp/dsv4_turn78_lexec_side_rmoe_n400.log
side-probe jsonl:
  /tmp/dsv4_turn78_lexec_side_rmoe_n400_logits.jsonl
first divergence:
  /tmp/dsv4_turn78_lexec_side_rmoe_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

side-probe summary:
  cases=171570
  exact_cases=171570
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=399
  pswiglu=798
  fglu=798
  metal_dispatch=1399039
  dsv4_lexec_dryrun=0
```

Decision:

```text
side probe hot-neutral: yes
chosen Turn #79 target:
  AOHC boundary side probe under full-layer executor context

reason:
  HC_PRE_NORM and routed-MoE final-output boundaries are covered in hot-neutral side-probe form.
  AOHC was exact/dispatch-reducing locally but not wall-clock useful;
  under the full-layer executor it should be treated as another coherent stage boundary,
  not as a standalone consume path.
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #77 Full-Layer Executor Side Probe: HC_PRE_NORM

Turn #76 recap:

```text
metadata_only and side_graph_plan dry-run paths are hot-neutral
metadata_only n400 exact
live_graph_nodes_added=0
live_backend_dispatches=0
dsv4_lexec_dryrun=0
```

Side-probe mode added:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE=hc_pre_norm
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE=post_eval_cpu_compare

scope:
  metadata/side context only
  no live decode graph nodes
  no live backend dispatch
  no consumed output
  cache mutation disabled

compared boundary markers:
  hc_pre_cur
  hc_pre_norm
  hc_pre_comb
  hc_post_input
```

Layer 0 n16:

```text
log:
  /tmp/dsv4_turn77_lexec_side_hcnorm_l0_n16.log

result:
  prefix stable
  side_probe cases=120
  exact_cases=120
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=15
  pswiglu=30
  fglu=30
  dsv4_lexec_dryrun=0
```

Layer 0 n80:

```text
log:
  /tmp/dsv4_turn77_lexec_side_hcnorm_l0_n80.log

result:
  prefix stable
  side_probe cases=632
  exact_cases=632
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
  dsv4_lexec_dryrun=0
```

All-layer n80:

```text
log:
  /tmp/dsv4_turn77_lexec_side_hcnorm_all_n80.log

result:
  prefix stable
  side_probe cases=27176
  exact_cases=27176
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
  dsv4_lexec_dryrun=0
```

n400 hot-neutral:

```text
baseline log:
  /tmp/dsv4_turn77_lexec_side_baseline_n400.log
baseline jsonl:
  /tmp/dsv4_turn77_lexec_side_baseline_n400_logits.jsonl
side-probe log:
  /tmp/dsv4_turn77_lexec_side_hcnorm_n400.log
side-probe jsonl:
  /tmp/dsv4_turn77_lexec_side_hcnorm_n400_logits.jsonl
first divergence:
  /tmp/dsv4_turn77_lexec_side_hcnorm_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

side-probe summary:
  cases=137256
  exact_cases=137256
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=399
  pswiglu=798
  fglu=798
  metal_dispatch=1399039
  dsv4_lexec_dryrun=0
```

Decision:

```text
side probe hot-neutral: yes
chosen Turn #78 target:
  routed_moe_final_output side-probe under full-layer executor metadata/side context

reason:
  HC_PRE_NORM side-probe plumbing is hot-neutral and exact.
  The next side-stage should use the routed-MoE final output because it was the largest trace gap,
  but it must remain side-graph/post-eval and not live graph dispatch.
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #76 Full Decode-Layer Executor Metadata / Side-Graph Dry Run

Turn #75 recap:

```text
full-layer dry-run op:
  input eligibility complete
  all-layer n80 eligible_cases=3397/3397

blocker:
  live no-consume dry-run dispatch perturbed split-GLU lowering
  n80 pair/pswiglu/fglu became 0/237/237

required pivot:
  validate executor input packing and metadata without adding live decode graph nodes
```

Dry-run modes added:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_MODE=<mode>

modes:
  live_graph_dispatch
  metadata_only
  side_graph_plan
  side_graph_dispatch

Turn #76 validated:
  metadata_only
  side_graph_plan
```

metadata_only n16 layer 0:

```text
log:
  /tmp/dsv4_turn76_lexec_metadata_l0_n16.log

result:
  prefix stable
  cases=15
  eligible_cases=15
  rejected_cases=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  side_graph_created=0
  side_graph_dispatched=0
  dsv4_lexec_dryrun=0

graph/lowering counters:
  pair=15
  pswiglu=30
  fglu=30
  metal_dispatch=66675
```

metadata_only n80 layer 0:

```text
log:
  /tmp/dsv4_turn76_lexec_metadata_l0_n80.log

result:
  prefix stable
  cases=79
  eligible_cases=79
  rejected_cases=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  dsv4_lexec_dryrun=0

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
```

metadata_only n80 all layers:

```text
log:
  /tmp/dsv4_turn76_lexec_metadata_all_n80.log

result:
  prefix stable
  cases=3397
  eligible_cases=3397
  rejected_cases=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  dsv4_lexec_dryrun=0

graph/lowering counters:
  pair=79
  pswiglu=158
  fglu=158
  metal_dispatch=285811
```

side_graph_plan n16 layer 0:

```text
log:
  /tmp/dsv4_turn76_lexec_side_plan_l0_n16.log

result:
  prefix stable
  cases=15
  eligible_cases=15
  rejected_cases=0
  side_graph_created=1
  side_graph_dispatched=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  dsv4_lexec_dryrun=0

graph/lowering counters:
  pair=15
  pswiglu=30
  fglu=30
  metal_dispatch=66675
```

n400 hot-neutral metadata_only:

```text
baseline log:
  /tmp/dsv4_turn76_lexec_baseline_n400.log
baseline jsonl:
  /tmp/dsv4_turn76_lexec_baseline_n400_logits.jsonl
metadata log:
  /tmp/dsv4_turn76_lexec_metadata_n400.log
metadata jsonl:
  /tmp/dsv4_turn76_lexec_metadata_n400_logits.jsonl
first divergence:
  /tmp/dsv4_turn76_lexec_metadata_first_divergence_n400.txt

result:
  first divergent token: none through 400 records
  max_abs=0
  rms=0
  top20 overlap=20/20
  transcript exact=yes
  path_accepted=false

metadata run:
  cases=17157
  eligible_cases=17157
  rejected_cases=0
  live_graph_nodes_added=0
  live_backend_dispatches=0
  dsv4_lexec_dryrun=0
  pair=399
  pswiglu=798
  fglu=798
  metal_dispatch=1399039
```

Decision:

```text
hot-neutral mode:
  metadata_only

side-graph plan status:
  graph-neutral planning path works
  no side dispatch attempted

chosen Turn #77 target:
  implement first numerical full-layer executor substage only in metadata/side-graph context
  recommended: HC_PRE_NORM under full-layer executor metadata context

guard:
  do not use live graph dispatch until a full replacement path is ready
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #75 Full Decode-Layer Executor Dry-Run Op

Turn #74 recap:

```text
full-layer shadow envelope:
  stage_probe all-layer n80 exact
  contract_dry_run had no hard input visibility blocker

chosen step:
  GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN
```

Dry-run op plumbing:

```text
new ggml op:
  GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN

new API:
  ggml_dsv4_decode_layer_executor_dryrun(...)

new Metal kernel:
  kernel_dsv4_decode_layer_executor_dryrun

new flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TRACE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_LAYER=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MIN=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MAX=<n>
```

Dry-run contract:

```text
decode-only metadata/input validation:
  output_consumed=0
  cache_mutation=disabled
  side_effects=disabled
  no replacement layer output computed

validated handles:
  layer input
  attention q/kv handles
  compressor/cache metadata
  attention output
  HC tensors
  routed-MoE ids/weights
  expert/shared weights
  layer output anchor
```

n16 layer-0 dry-run op:

```text
log:
  /tmp/dsv4_turn75_lexec_dryrun_op_l0_n16.log

result:
  prefix stable
  cases=15
  eligible_cases=15
  rejected_cases=0
  first_reject_reason=none
  op_added=1
  backend_op_dispatched=1
  output_consumed=0
  cache_mutation=disabled
  dsv4_lexec_dryrun=15

graph/lowering counters:
  pair=0
  pswiglu=45
  fglu=45
  metal_dispatch=66630

important:
  live dry-run dispatch perturbs split-GLU lowering
  expected n16 baseline shape was pair/pswiglu/fglu=15/30/30
```

n80 layer-0 dry-run op:

```text
log:
  /tmp/dsv4_turn75_lexec_dryrun_op_l0_n80.log

result:
  prefix stable
  cases=79
  eligible_cases=79
  rejected_cases=0
  dsv4_lexec_dryrun=79

graph/lowering counters:
  pair=0
  pswiglu=237
  fglu=237
  metal_dispatch=285574
```

All-layer n80 dry-run op:

```text
log:
  /tmp/dsv4_turn75_lexec_dryrun_op_all_n80.log

result:
  prefix stable
  cases=3397
  eligible_cases=3397
  rejected_cases=0
  dsv4_lexec_dryrun=3397
  output_consumed=0
  cache_mutation=disabled

graph/lowering counters:
  pair=0
  pswiglu=237
  fglu=237
  metal_dispatch=288892
```

n400 hot-neutral baseline, dry-run flags off:

```text
log:
  /tmp/dsv4_turn75_baseline_n400_hotneutral.log
jsonl:
  /tmp/dsv4_turn75_baseline_n400_hotneutral.jsonl

result:
  transcript stable
  dsv4_lexec_dryrun=0
  pair=399
  pswiglu=798
  fglu=798
  metal_dispatch=1399039
```

Decision:

```text
input eligibility:
  pass
  no missing full-layer dry-run handles observed

live dry-run op stability:
  not graph-safe
  adding the live no-consume dispatch changes split-GLU lowering

Turn #76 target:
  move full-layer executor dry-run to separate side graph / metadata-only packing
  do not start numerical full-layer executor work until the dry-run envelope is graph-neutral

reason:
  this repeats the routed-MoE branch-order result:
  extra live graph dispatch can perturb generic split-GLU lowering even with no consumed output.
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #74 Full Decode-Layer Executor Shadow Envelope

Turn #73 recap:

```text
Selected next target:
  Candidate C: DSV4_LAYER_EXECUTOR_FULL_DECODE_LAYER

Reason:
  local MoE/AOHC/CUPD replacements were too sensitive to ggml graph/lowering shape.
  The next viable boundary is a coherent DS4-style decode-layer executor, not
  another standalone local replacement.
```

Full-layer executor contract:

```text
contract path:
  /tmp/dsv4_turn74_full_layer_executor_contract.txt

shadow target:
  DSV4_LAYER_EXECUTOR_FULL_DECODE_LAYER_SHADOW

one-layer boundary inputs:
  layer input / residual
  attention norm inputs
  Q/K/V projection weights
  compressor state inputs
  SWA/compressed cache views
  attention masks / top-k indices
  HC state / HC weights
  FFN norm input
  routed MoE topk ids/weights
  expert/shared weights
  cache position / token index metadata

outputs:
  layer output
  updated HC state candidate
  KV/cache rows candidate, shadow only
  optional stage outputs for compare

side effects in Turn #74:
  none
  cache mutation disabled
  consume disabled
```

Modes added:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_MODE=<mode>

modes:
  generic_envelope
  stage_probe
  contract_dry_run

report fields:
  dsv4_lexec_full_shadow
  dsv4_lexec_full_exact
  dsv4_lexec_full_summary
  dsv4_lexec_full_stage
  dsv4_lexec_full_contract
```

n16 generic envelope:

```text
log:
  /tmp/dsv4_turn74_lexec_full_generic_l0_n16.log

result:
  prefix stable
  cases=165
  exact_cases=165
  non_exact_cases=0
  max_abs=0
  max_rms=0
  graph_nodes_added=0
  backend_dispatches=0
  consume_path=disabled

visible stages:
  attn_qkv
  kv_cache_handoff
  attn_core
  attn_output
  attn_hc_post
  ffn_hc_pre
  routed_moe
  ffn_hc_post
  layer_output

missing stage in this layer-0 n16 envelope:
  compressor_update=0
```

n16 stage probe:

```text
log:
  /tmp/dsv4_turn74_lexec_full_stage_probe_l0_n16.log

result:
  prefix stable
  cases=165
  exact_cases=165
  non_exact_cases=0
  max_abs=0
  max_rms=0
  graph_nodes_added=0
  backend_dispatches=0
  consume_path=disabled

stage map:
  attn_qkv=1
  compressor_update=0
  kv_cache_handoff=1
  attn_core=1
  attn_output=1
  attn_hc_post=1
  ffn_hc_pre=1
  routed_moe=1
  ffn_hc_post=1
  layer_output=1
```

n16 contract dry-run:

```text
log:
  /tmp/dsv4_turn74_lexec_full_contract_l0_n16.log

result:
  prefix stable
  cases=165
  exact_cases=165
  non_exact_cases=0
  graph_nodes_added=0
  backend_dispatches=0
  consume_path=disabled
  cache_mutation=disabled

inputs visible:
  layer_input=1
  attention_weights=1
  compressor_state=1
  cache_row_metadata=1
  hc_tensors=1
  routed_moe_ids_weights=1
  expert_shared_weights=1
  layer_output=1

missing inputs:
  none

blockers:
  no hard input-visibility blocker in contract_dry_run
  compressor_update stage output is not observed in the layer-0 n16 stage probe
```

All-layer n80 stage probe:

```text
log:
  /tmp/dsv4_turn74_lexec_full_stage_probe_all_n80.log

result:
  prefix stable
  cases=48003
  exact_cases=48003
  non_exact_cases=0
  max_abs=0
  max_rms=0
  graph_nodes_added=0
  backend_dispatches=0
  consume_path=disabled

stage map:
  attn_qkv=1
  compressor_update=1
  kv_cache_handoff=1
  attn_core=1
  attn_output=1
  attn_hc_post=1
  ffn_hc_pre=1
  routed_moe=1
  ffn_hc_post=1
  layer_output=1
```

Chosen Turn #75 step:

```text
Option 1:
  full-layer executor dry-run op
  add GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN

Reason:
  contract_dry_run shows the major future executor inputs are visible and stable.
  The next useful step is backend input packing and layer metadata validation,
  still with no numerical output, no consume, and no cache mutation.

Rejected next steps:
  no local routed-MoE replacement
  no local AOHC/CUPD replacement
  no performance run
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #73 DS4-Style Decode-Layer Executor Planning

Turn #72 recap:

```text
Routed-MoE local replacement work is closed as a performance path.
One-tensor routed-MoE shadow proved substage math exact, but local consume/replacement
is blocked or performance-negative under the current ggml graph/lowering shape.

Accepted:
  HC_PRE_NORM

Correct/support-only:
  AOHC exact variants
  CUPD2 safe primitive
  CUPD3 graph-level tail canary
  routed-MoE one-tensor shadow
  routed-MoE shared/final exact canary

Rejected as performance paths:
  CUPD2_FUSED_COMP
  CUPD3 backend-tail consume
  DOWN_SUM6
  FFN/MoE V2 full consume
  sparse indexed attention consume
  AOHC layer-set backend fusion
  routed-MoE full replacement
  routed-MoE shared/final replacement
```

DS4 boundary audit:

```text
audit path:
  /tmp/dsv4_turn73_decode_layer_executor_audit.txt

external DS4 inspected:
  /Users/anemll/SourceRelease/GITHUB/ML_playground/ds4

note:
  ds4_metal.h was requested by the original search pattern but is absent in this checkout.
  ds4.c, ds4_metal.m, and metal/*.metal were inspected.

mapped stages:
  qkv setup
  attention compression/update
  KV/cache finalization
  indexed/mixed attention
  attention output
  HC pre/post
  routed MoE
  result_hc / result_norm / head path

key finding:
  DS4 owns decode work as coherent staged routines with scratch/cache ownership.
  The current repo has many exact DS4 primitives, but local consume paths repeatedly
  change graph lowering, expose cache side-effect ordering, or fail to reduce dispatch/speed.
```

Candidate executor contracts:

```text
contract path:
  /tmp/dsv4_turn73_decode_layer_executor_contract.txt

Candidate A: DSV4_LAYER_EXECUTOR_FFN_ONLY
  owns:
    FFN norm input
    routed MoE
    shared branch
    final FFN output
    HC post after FFN
  status:
    useful only as all-layer/full-consistency FFN executor work
    not suitable as another single-layer local replacement
  blocker:
    routed-MoE local replacement changes split-GLU lowering

Candidate B: DSV4_LAYER_EXECUTOR_ATTENTION_BLOCK
  owns:
    attention q/k/v setup
    compressor/update
    cache finalization
    indexed/mixed attention
    attention output
    HC post after attention
  status:
    feasible intermediate if full-layer scope is too large
  blocker:
    cache/update dependency semantics must be owned as a coherent boundary

Candidate C: DSV4_LAYER_EXECUTOR_FULL_DECODE_LAYER
  owns:
    attention block
    HC pre/post
    routed MoE
    cache/update dependencies
    layer output
  status:
    recommended next target
  reason:
    closest to DS4-style ownership and most likely to avoid graph/lowering mismatch
```

Planning-only runtime flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MAX=<n>

properties:
  report-only
  graph_nodes_added=0
  backend_dispatches=0
  consume=0

summary:
  dsv4_layer_executor_plan_summary
```

Planning smoke:

```text
n16 layer-0 log:
  /tmp/dsv4_turn73_layer_executor_plan_l0_n16.log

all-layer n80 log:
  /tmp/dsv4_turn73_layer_executor_plan_all_n80.log

expected:
  prefix stable
  no graph perturbation
  no consume
  no new backend dispatch
```

Decision:

```text
chosen Turn #74 target:
  Candidate C: DSV4_LAYER_EXECUTOR_FULL_DECODE_LAYER

implementation shape:
  start with a full-layer executor shadow/planning skeleton
  define stage packet inputs/outputs and compare boundaries
  do not consume yet

explicit stops:
  no local routed-MoE partial replacement should continue
  no local AOHC/CUPD small replacement should continue without executor context
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #72 Routed-MoE Line Closure / Decision

Scope:

```text
No new kernel work.
No performance run.
No layer-set or all-layer consume.
This is a decision and handoff cleanup pass.
```

Timeline:

```text
Turn #32 / #33:
  paired trace showed routed-MoE / FFN as the largest structural gap
  ours coarse FFN rows: 8724
  DS4 routed-MoE boundary rows: 1419
  ratio: 6.15x

Turn #34-46:
  built routed-MoE one-tensor shadow infrastructure
  exposed topk ids/weights
  added GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
  implemented exact substages:
    gate/up
    SwiGLU
    q2_K expert down
    routed weighted sum
    q8_0 shared branch
    final FFN output
  all-layer n400 shadow exact:
    600495/600495
    max_abs=0
    rms=0

Turn #47-66:
  true single-layer consume/replacement repeatedly drifted
  root cause narrowed to graph/lowering sensitivity:
    removing or changing generic paired split-GLU lowering breaks transcript exactness
  generic-only lowering parity reproduced drift
  force_replace_like_node_order stayed exact
  conclusion:
    backend math was not the main blocker
    graph/lowering shape was

Turn #67:
  pair-preserving graph-level canary:
    n400 exact
    pair/pswiglu/fglu preserved

Turn #68-69:
  real backend pair-preserve attachment broke matcher:
    target n16 pair/pswiglu/fglu: 15/30/30
    backend variant: 0/45/45
  no attach mode preserved target lowering

Turn #70:
  shared/final-only replacement:
    n400 exact
    pair/pswiglu/fglu preserved
    generic_down_built=1
    generic_shared_built=0
    backend_shared_built=1
    backend_output_consumed=1

Turn #71:
  no-logit performance:
    baseline avg: 21.4 tok/s
    shared/final avg: 15.7 tok/s
    metal_dispatch unchanged: 1399039
    not performance-useful
```

Decision:

```text
Routed-MoE backend work is valuable as validation infrastructure and confirms
the structural gap, but no routed-MoE local replacement is accepted as a
performance path.

Do not continue local routed-MoE substitutions unless the design moves to a
full decode-layer executor or DS4-style executor that owns lowering consistently.
```

Status table:

```text
Accepted:
  HC_PRE_NORM

Correct but not performance-useful / support-only:
  AOHC exact / dispatch-reducing variants
  CUPD2 safe support primitive
  CUPD3 graph-level tail canary
  Routed-MoE one-tensor shadow infrastructure
  Routed-MoE shared/final-only exact canary

Rejected as performance paths:
  CUPD2_FUSED_COMP
  CUPD3 backend-tail consume
  DOWN_SUM6
  FFN/MoE V2 full consume
  sparse indexed attention consume
  AOHC layer-set backend fusion
  routed-MoE full replacement
  routed-MoE shared/final-only replacement
```

Next recommended direction:

```text
Do not continue local MoE replacements.

Either:

A. Full DS4-style decode-layer executor:
   owns FFN/MoE, HC post, cache/update ordering, and graph lowering
   as one coherent boundary.

B. Dedicated whole-stage compressor/KV finalizer:
   only if it owns cache/dependency semantics cleanly.

C. Return to paired Metal trace and Instruments:
   use actual GPU command buffer / kernel timing to pick the next
   whole-stage executor boundary.

Reason:
  A single-layer local replacement is the wrong validation shape for routed-MoE,
  because changing only one layer's lowering creates deterministic transcript
  sensitivity.
```

Policy:

```text
Transcript-exact n400 remains active.
Tolerance remains diagnostic-only.
No drifting path is accepted.
No path is accepted as performance unless exact and no-logit speed improves.
```

### 2026-05-13 Turn #71 Routed-MoE Shared/Final-Only Performance Check

Turn #70 recap:

```text
shared/final-only layer-0 replacement:
  n16 stable
  n80 stable
  n400 hot-neutral exact
  pair/pswiglu/fglu preserved at 399/798/798
  generic_down_built=1
  generic_shared_built=0
  backend_shared_built=1
  backend_output_consumed=1
```

Patch:

```text
summary/report changes:
  dsv4_rmoe_shared_final_summary now reports:
    metal_dispatch=-1
    performance_eligible=<0|1>

note:
  authoritative pair/pswiglu/fglu and metal_dispatch still come from Metal stats.
```

Paired no-logit performance:

```text
directory:
  /tmp/dsv4_turn71_rmoe_shared_perf

baseline A:
  log=/tmp/dsv4_turn71_rmoe_shared_perf/baseline_A_n400.log
  generation=21.5 tok/s
  transcript stable=yes

baseline B:
  log=/tmp/dsv4_turn71_rmoe_shared_perf/baseline_B_n400.log
  generation=21.3 tok/s
  transcript stable=yes

shared/final A:
  log=/tmp/dsv4_turn71_rmoe_shared_perf/shared_A_n400.log
  generation=15.4 tok/s
  transcript stable=yes

shared/final B:
  log=/tmp/dsv4_turn71_rmoe_shared_perf/shared_B_n400.log
  generation=16.0 tok/s
  transcript stable=yes

baseline_avg:
  21.4 tok/s
shared_avg:
  15.7 tok/s
delta:
  -5.7 tok/s
  -26.6%
```

Dispatch shape:

```text
baseline:
  metal_dispatch=1399039
  pair/pswiglu/fglu=399/798/798
  dec_mv=1197
  gen_mv=48267
  mul_mat mv_ext/mv/mm=1956/221848/0

shared/final-only:
  metal_dispatch=1399039
  pair/pswiglu/fglu=399/798/798
  dec_mv=1197
  gen_mv=48267
  mul_mat mv_ext/mv/mm=1956/221848/0
  dsv4_rmoe_shared=399
  generic_shared_built=0
  backend_shared_built=1
  backend_output_consumed=1
  performance_eligible=1
```

Decision:

```text
performance useful: no
scale to layer set: no in current form
performance path accepted: no

reason:
  the path remains correctness-valid, but it does not reduce Metal dispatch count
  and is materially slower in paired no-logit runs.
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #70 Routed-MoE Shared/Final-Only Replacement

Turn #69 recap:

```text
backend down attach cannot preserve split-GLU lowering:
  target n16 pair/pswiglu/fglu=15/30/30
  real pair-preserve backend op produced 0/45/45 or 30/15/15 depending on anchor
  no attachment mode was prefix-stable
```

Patch:

```text
new shared/final-only flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_ONLY=1

new mode:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_MODE=shared_down_plus_final_add

new under-test:
  rmoe_backend_shared_final_single_layer

analyzer/metadata:
  scripts/dsv4_first_divergence.py allows --allow-under-test rmoe_backend_shared_final_single_layer
  CLI JSONL metadata records shared_final_only/shared_final_mode
```

Target graph:

```text
generic routed expert path kept:
  generic router/top-k: yes
  generic gate/up: yes
  generic packed SwiGLU: yes
  generic expert down: yes
  generic weighted routed sum: yes

replacement:
  generic shared branch: skipped
  backend shared branch: built
  backend final add routed_sum + shared_down: consumed
```

Guard:

```text
negative missing-layer log:
  /tmp/dsv4_turn70_rmoe_shared_final_guard_missing_layer.log
  allowed=0
  reason=missing_layer
  dsv4_rmoe_consume=0
```

n16:

```text
log:
  /tmp/dsv4_turn70_rmoe_shared_final_l0_n16.log
prefix stable:
  yes
pair/pswiglu/fglu:
  15/30/30
summary:
  generic_gate_up_built=1
  generic_swiglu_built=1
  generic_down_built=1
  generic_weighted_sum_built=1
  generic_shared_built=0
  backend_shared_built=1
  backend_final_built=1
  backend_output_consumed=1
  dsv4_rmoe_shared=15
  dsv4_rmoe_consume=15
```

n80:

```text
log:
  /tmp/dsv4_turn70_rmoe_shared_final_l0_n80.log
prefix stable:
  yes
pair/pswiglu/fglu:
  79/158/158
summary:
  generic_down_built=1
  generic_shared_built=0
  backend_shared_built=1
  backend_output_consumed=1
  dsv4_rmoe_shared=79
  dsv4_rmoe_consume=79
```

n400 hot-neutral:

```text
baseline:
  log=/tmp/dsv4_turn70_rmoe_shared_final_baseline_n400.log
  jsonl=/tmp/dsv4_turn70_rmoe_shared_final_baseline_n400_logits.jsonl
  pair/pswiglu/fglu=399/798/798
  metal_dispatch=1399039

shared/final-only:
  log=/tmp/dsv4_turn70_rmoe_shared_final_l0_n400.log
  jsonl=/tmp/dsv4_turn70_rmoe_shared_final_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn70_rmoe_shared_final_first_divergence_l0_n400.txt
  pair/pswiglu/fglu=399/798/798
  dsv4_rmoe_shared=399
  dsv4_rmoe_consume=399

result:
  divergence=none through n400
  max_abs=0
  rms=0
  top20_overlap=20/20
  path_accepted=false
```

Decision:

```text
shared/final-only replacement exact: yes
next action:
  paired no-logit performance and dispatch-shape analysis can be considered next;
  do not extrapolate this result to routed expert down replacement.
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #69 Routed-MoE Pair-Preserve Backend Attachment

Turn #68 recap:

```text
real pair-preserve backend op dispatched:
  backend_op_dispatched=1
  backend_output_consumed=1
  generic_swiglu_built=1
  generic_down_built=0

failure:
  n16 prefix drifted
  pair/pswiglu/fglu changed from target 15/30/30 to 0/45/45
```

Patch:

```text
new matcher trace:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_MATCH_TRACE=1

new attach modes:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE=inline_backend_consumer
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE=after_pair_marker
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE=after_swiglu_marker
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE=generic_down_anchor
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE=generic_down_dry_anchor
```

Matcher trace:

```text
representative layer-0 reason:
  pair_decision=accepted_unweighted_swiglu
  pswiglu_decision=accepted_unweighted_swiglu
  fglu_decision=accepted_unweighted_swiglu
  reject_reason=missing_generic_down_consumer

interpretation:
  when the real backend op replaces the generic down-shaped consumer,
  the split-GLU matcher no longer sees the baseline generic ffn_moe_down consumer shape.
  It accepts the selected layer as an unweighted SwiGLU path instead of the baseline pair-preserving lowering.
```

Attachment matrix:

```text
inline_backend_consumer:
  log=/tmp/dsv4_turn69_rmoe_pair_attach_inline_backend_consumer_l0_n16.log
  prefix stable=no
  pair/pswiglu/fglu=0/45/45
  dsv4_rmoe_pair=15
  generic_down_built=0
  backend_op_dispatched=1
  backend_output_consumed=1
  matcher reason=missing_generic_down_consumer

after_pair_marker:
  log=/tmp/dsv4_turn69_rmoe_pair_attach_after_pair_marker_l0_n16.log
  prefix stable=no
  pair/pswiglu/fglu=0/45/45
  dsv4_rmoe_pair=15
  generic_down_built=0
  backend_op_dispatched=1
  backend_output_consumed=1
  matcher reason=missing_generic_down_consumer

after_swiglu_marker:
  log=/tmp/dsv4_turn69_rmoe_pair_attach_after_swiglu_marker_l0_n16.log
  prefix stable=no
  pair/pswiglu/fglu=0/45/45
  dsv4_rmoe_pair=15
  generic_down_built=0
  backend_op_dispatched=1
  backend_output_consumed=1
  matcher reason=missing_generic_down_consumer

generic_down_anchor:
  log=/tmp/dsv4_turn69_rmoe_pair_attach_generic_down_anchor_l0_n16.log
  prefix stable=no
  pair/pswiglu/fglu=30/15/15
  dsv4_rmoe_pair=15
  generic_down_built=1
  generic_down_consumed=0
  backend_op_dispatched=1
  backend_output_consumed=1
  matcher result=still not target lowering

generic_down_dry_anchor:
  log=/tmp/dsv4_turn69_rmoe_pair_attach_generic_down_dry_anchor_l0_n16.log
  prefix stable=no
  pair/pswiglu/fglu=0/45/45
  dsv4_rmoe_pair=15
  generic_down_built=1 in summary, but not live-expanded
  backend_op_dispatched=1
  backend_output_consumed=1
  matcher reason=missing_generic_down_consumer
```

n80:

```text
not run
reason:
  no n16 attach mode satisfied prefix stability and pair/pswiglu/fglu=15/30/30
```

Decision:

```text
lowering-preserving backend attach found: no
routed-MoE backend path eligible: no

root cause:
  a real pair-preserve backend dispatch cannot currently replace the generic down consumer while preserving the selected-layer split-GLU lowering.

practical options:
  1. keep generic gate/up/SwiGLU/down and target shared/final only
  2. move to a full-layer executor where lowering changes happen consistently
  3. mark routed-MoE backend replacement blocked under transcript-exact single-layer policy
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #68 Routed-MoE Pair-Preserve Backend Variant

Turn #67 recap:

```text
pair-preserving graph shape was n400 exact:
  generic gate/up built
  generic packed SwiGLU built
  generic down/shared skipped
  replacement output consumed
  pair/pswiglu/fglu preserved at 399/798/798

limitation:
  dsv4_rmoe=0
  not yet a clean backend-dispatch replacement path
```

Backend pair-preserve op:

```text
API:
  ggml_dsv4_routed_moe_pair_preserve_decode

implementation:
  extends GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE with pair_preserve mode
  consumes generic packed SwiGLU through src9
  consumes ffn_input for shared branch through src1
  consumes topk ids/weights
  owns expert down q2_K
  owns route-weighted routed sum
  owns q8_0 shared branch
  owns final add

does not own:
  router/top-k
  expert gate/up
  expert SwiGLU
```

Dispatch/counter result:

```text
log=/tmp/dsv4_turn68_rmoe_pair_backend_l0_n16.log
debug trace=/tmp/dsv4_turn68_rmoe_pair_backend_fusion_debug_after_rename.log

backend_op_dispatched=1
generic_swiglu_built=1
generic_down_built=0
backend_down_built=1
backend_weighted_sum_built=1
backend_shared_built=1
backend_output_consumed=1
dsv4_rmoe_pair=15

observed counters:
  pair/pswiglu/fglu=0/45/45
  dsv4_rmoe=15

required counters:
  pair/pswiglu/fglu=15/30/30 for n16
  pair/pswiglu/fglu=399/798/798 for n400
```

n16:

```text
result:
  prefix stable=no
  output begins: "This is a great question..."

baseline control:
  output begins: "This is an excellent question..."

classification:
  real backend dispatch variant is not transcript-exact
  adding the backend op as the weighted-SwiGLU consumer still prevents the generic split-GLU matcher from preserving pair lowering
```

n80/n400:

```text
not run
reason:
  n16 already failed transcript exactness and pair counter preservation
```

Precise blocker:

```text
backend op dispatch can own down/shared/final and consume generic packed SwiGLU, but the Metal split-GLU pairing path still keys on the generic ffn_moe_down MUL_MAT_ID consumer shape.

Replacing that consumer with DSV4_ROUTED_MOE_ONE_TENSOR_DECODE changes lowering:
  generic gate/up/SwiGLU remain in graph
  weighted SwiGLU anchor exists
  backend op dispatches
  pair counter remains 0 for the selected layer

To proceed, either:
  teach the split-GLU matcher to treat the pair-preserve backend op as a valid down consumer without changing pair/pswiglu/fglu, or
  conclude that a real backend dispatch cannot preserve the baseline split-GLU lowering without a generic down-shaped anchor.
```

Performance:

```text
not run
reason:
  correctness/counter precondition failed
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #66 Routed-MoE Generic Lowering Parity Test

Turn #65 recap:

```text
replace-generic layer-0 path changes graph/lowering:
  baseline n80 pair/pswiglu/fglu=158/79/79
  replace  n80 pair/pswiglu/fglu=79/79/79
  first graph difference inserts DSV4_ROUTED_MOE_ONE_TENSOR_DECODE and removes two generic MUL_MAT_ID nodes
```

Patch:

```text
new generic-only lowering parity flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY=1

filters:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_LAYER=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_TOKEN_MIN=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_TOKEN_MAX=<n>

modes:
  disable_pair_for_layer
  disable_pair_and_use_unpaired_swiglu
  force_replace_like_node_order

analyzer:
  scripts/dsv4_first_divergence.py allows --allow-under-test rmoe_generic_lowering_parity
```

n400 baseline:

```text
log=/tmp/dsv4_turn66_rmoe_baseline_normal_n400.log
jsonl=/tmp/dsv4_turn66_rmoe_baseline_normal_n400_logits.jsonl
transcript stable=yes
dsv4_rmoe=0
pair/pswiglu/fglu=399/798/798
metal_dispatch=1399039
```

Generic-only lowering parity matrix:

```text
disable_pair_for_layer:
  log=/tmp/dsv4_turn66_rmoe_lowering_disable_pair_for_layer_n400.log
  jsonl=/tmp/dsv4_turn66_rmoe_lowering_disable_pair_for_layer_n400_logits.jsonl
  first-divergence=/tmp/dsv4_turn66_rmoe_first_divergence_lowering_disable_pair_for_layer.txt
  first divergent token index=104
  max_abs_logit_err=0.414131165
  rms_logit_err=0.149363412
  pair/pswiglu/fglu=399/399/399
  dsv4_rmoe=0
  generic_ffn_built=1
  backend_ffn_built=0

disable_pair_and_use_unpaired_swiglu:
  log=/tmp/dsv4_turn66_rmoe_lowering_disable_pair_and_use_unpaired_swiglu_n400.log
  jsonl=/tmp/dsv4_turn66_rmoe_lowering_disable_pair_and_use_unpaired_swiglu_n400_logits.jsonl
  first-divergence=/tmp/dsv4_turn66_rmoe_first_divergence_lowering_disable_pair_and_use_unpaired_swiglu.txt
  first divergent token index=104
  max_abs_logit_err=0.414131165
  rms_logit_err=0.149363412
  pair/pswiglu/fglu=399/399/399
  dsv4_rmoe=0
  generic_ffn_built=1
  backend_ffn_built=0

force_replace_like_node_order:
  log=/tmp/dsv4_turn66_rmoe_lowering_force_replace_like_node_order_n400.log
  jsonl=/tmp/dsv4_turn66_rmoe_lowering_force_replace_like_node_order_n400_logits.jsonl
  first-divergence=/tmp/dsv4_turn66_rmoe_first_divergence_lowering_force_replace_like_node_order.txt
  first divergent token index=none through n400
  max_abs_logit_err=0
  rms_logit_err=0
  pair/pswiglu/fglu=399/798/798
  dsv4_rmoe=0
```

Comparison to replace:

```text
replace n400 from Turn #61:
  first divergent token index=35
  pair/pswiglu/fglu=399/399/399
  dsv4_rmoe=399

matching generic-only mode:
  disable_pair_for_layer and disable_pair_and_use_unpaired_swiglu match pair/pswiglu/fglu=399/399/399
  both drift without backend op or backend consume

interpretation:
  changing/removing the selected-layer paired split-GLU lowering is sufficient to break transcript exactness
  backend replacement still adds an earlier perturbation because replace diverges at token 35 while generic-only parity diverges at token 104
  force_replace_like_node_order alone is exact, so top-k exposure/order without lowering change is not sufficient
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #67 Routed-MoE Pair-Preserving Partial Replacement

Turn #66 recap:

```text
removing/changing the selected-layer paired split-GLU lowering breaks transcript exactness:
  generic-only disable_pair_for_layer drifted at token 104
  force_replace_like_node_order stayed exact

direction:
  preserve generic gate/up + packed SwiGLU lowering
  replace only downstream routed down/sum/shared/final
```

Patch:

```text
new flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE_MODE=down_shared_from_generic_swiglu

new under-test:
  rmoe_backend_pair_preserve_single_layer

implementation:
  selected layer builds generic route/topk, gate/up, and packed CLAMP/SILU/CLAMP/MUL SwiGLU
  build_moe_ffn returns at generic SwiGLU for the selected site
  generic routed down / weighted sum / generic shared branch are skipped for that site
  replacement path consumes generic packed SwiGLU and builds routed down, routed sum, shared branch, final FFN

important limitation:
  this canary deliberately does not attach GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
  dsv4_rmoe=0 in this mode
  reason: the current one-tensor backend op cannot accept generic packed SwiGLU as an input without reintroducing the live backend dispatch/lowering perturbation
```

Guard:

```text
log=/tmp/dsv4_turn67_rmoe_pair_preserve_guard_missing_layer.log
result:
  allowed=0
  reason=missing_layer
  dsv4_rmoe_consume=0
```

n16:

```text
log=/tmp/dsv4_turn67_rmoe_pair_preserve_l0_n16.log
prefix stable=yes
generic_gate_up_built=1
generic_swiglu_built=1
generic_down_built=0
generic_weighted_sum_built=0
generic_shared_built=0
backend_down_built=1
backend_weighted_sum_built=1
backend_shared_built=1
backend_final_built=1
backend_output_consumed=1
dsv4_rmoe_consume=15
```

n80:

```text
log=/tmp/dsv4_turn67_rmoe_pair_preserve_l0_n80.log
prefix stable=yes
pair/pswiglu/fglu=79/158/158
generic_swiglu_built=1
generic_down_built=0
backend_output_consumed=1
dsv4_rmoe_consume=79
dsv4_rmoe=0
```

n400 hot-neutral:

```text
baseline log=/tmp/dsv4_turn67_rmoe_pair_preserve_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn67_rmoe_pair_preserve_baseline_n400_logits.jsonl
pair-preserve log=/tmp/dsv4_turn67_rmoe_pair_preserve_l0_n400.log
pair-preserve jsonl=/tmp/dsv4_turn67_rmoe_pair_preserve_l0_n400_logits.jsonl
first-divergence=/tmp/dsv4_turn67_rmoe_pair_preserve_first_divergence_l0_n400.txt

result:
  first divergent token index=none through n400
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact=yes
  path_accepted=false

counters:
  baseline pair/pswiglu/fglu=399/798/798
  pair-preserve pair/pswiglu/fglu=399/798/798
  baseline metal_dispatch=1399039
  pair-preserve metal_dispatch=1399039
  pair-preserve dsv4_rmoe=0
  pair-preserve dsv4_rmoe_consume=399
```

Conclusion:

```text
pair-preserving partial replacement exact: yes
root cause confirmed:
  transcript drift came from disturbing/removing generic paired split-GLU lowering
  preserving that lowering stabilizes the single-layer downstream replacement

backend-op eligibility:
  not yet a true one-tensor backend-op consume
  next action is a backend op variant/signature that consumes generic packed SwiGLU directly and owns down/shared/final without rebuilding gate/up/SwiGLU or attaching an extra live dispatch branch
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #63 Routed-MoE Result Chain Localization

Turn #62 recap:

```text
layer-0 routed-MoE replacement boundary:
  final_ffn exact
  hc_post_input exact
  hc_post_output exact
  layer_output_after_ffn exact
  next_layer_input exact

previous observed mismatch:
  first stage=result_hc
  first token=3
```

result_hc mapping:

```text
mapping note=/tmp/dsv4_turn63_result_hc_mapping.txt
producer=dsv4_hc_head(final inpL after all layers)
consumer=result_norm / output RMS norm
semantics=final output HC head transform, layer=-1
emission=once per single-token decode graph/eval
depends_on=all layers via final inpL
```

result-chain dump:

```text
new flag=LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_DUMP
new analyzer=scripts/dsv4_compare_result_chain_dump.py

n80 baseline dump=/tmp/dsv4_turn63_baseline_result_chain_l0_n80.jsonl
n80 replace dump=/tmp/dsv4_turn63_replace_result_chain_l0_n80.jsonl
n80 report=/tmp/dsv4_turn63_result_chain_diff.txt
n80 token alignment confirmed=yes
n80 first differing token=none
n80 first differing stage=none

n400 baseline dump=/tmp/dsv4_turn63_baseline_result_chain_l0_n400.jsonl
n400 replace dump=/tmp/dsv4_turn63_replace_result_chain_l0_n400.jsonl
n400 report=/tmp/dsv4_turn63_result_chain_diff_n400.txt
n400 token alignment confirmed=yes
n400 first differing token=none
n400 first differing stage=none
```

Stages exact in paired dumps:

```text
layer0_next_input=exact
layer1_input=exact
layer1_after_attn=exact
layer1_after_ffn=exact
last_layer_input=exact
last_layer_output=exact
result_hc=exact
result_norm=exact
logits_input=exact
```

Classification:

```text
root cause=result_hc mismatch is not a stable value mismatch in the result chain under explicit dump/readback
interpretation=diagnostic materialization/readback makes the replacement run exact through logits_input
remaining blocker=unmaterialized hot-neutral path remains scheduling/materialization sensitive
replace path accepted=no
performance run=no
all-layer consume run=no
```

Policy:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #61 Routed-MoE Replace-Generic n400 Hot-Neutral Validation

Turn #60 recap:

```text
generic-boundary SwiGLU downstream fixed layer-0 replace-generic n16/n80:
  SWIGLU_MODE=generic_graph_boundary
  DOWN_INPUT=generic_graph_boundary
  final_ffn exact
  hc_post_input exact
  generic_ffn_built=0
  backend_output_consumed=1
```

n400 baseline:

```text
log=/tmp/dsv4_turn61_rmoe_replace_baseline_n400.log
jsonl=/tmp/dsv4_turn61_rmoe_replace_baseline_n400_logits.jsonl
transcript stable: yes
metal_dispatch=1399039
dsv4_rmoe=0
pair/pswiglu/fglu=399/798/798
```

n400 replace-generic:

```text
log=/tmp/dsv4_turn61_rmoe_replace_l0_n400.log
jsonl=/tmp/dsv4_turn61_rmoe_replace_l0_n400_logits.jsonl
first-divergence=/tmp/dsv4_turn61_rmoe_replace_first_divergence_l0_n400.txt

result:
  divergence: token index 35 / position 45
  baseline token: " built"
  replace token: " ("
  max_abs=0.323467255
  rms=0.14525186
  top20_overlap=20/20
  transcript exact: no

replace summary:
  generic_ffn_built=0
  backend_ffn_built=1
  backend_output_consumed=1
  dsv4_rmoe=399
  dsv4_rmoe_consume=399
  swiglu_mode=generic_graph_boundary
  down_input=generic_graph_boundary
  path_accepted=false
```

Bounded localization:

```text
baseline dump=/tmp/dsv4_turn61_rmoe_baseline_l0_n80_substage.jsonl
replace dump=/tmp/dsv4_turn61_rmoe_replace_l0_n80_substage.jsonl
diff=/tmp/dsv4_turn61_rmoe_substage_diff_l0_n80.txt

local layer-0 FFN boundary:
  ffn_input exact
  topk_ids exact
  topk_weights exact
  down exact
  routed_sum exact
  shared_gate/up/swiglu/down exact
  final_ffn exact
  hc_post_input exact

note:
  gate/up/swiglu aggregate rows report baseline=None in this analyzer,
  but downstream exactness confirms the layer-0 FFN output boundary remains exact.
```

Decision:

```text
single-layer replace canary exact through n400: no
replace-generic canary accepted: no
performance run: skipped
next action: localize downstream/logit drift after exact layer-0 FFN replacement
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #62 Routed-MoE Downstream Drift Localization

Turn #61 recap:

```text
layer-0 replace-generic n400:
  SWIGLU_MODE=generic_graph_boundary
  DOWN_INPUT=generic_graph_boundary
  generic_ffn_built=0
  backend_output_consumed=1
  first logit/token divergence: token index 35 / position 45

local Turn #61 substage dump:
  layer-0 final_ffn exact
  hc_post_input exact
```

Downstream dump:

```text
new flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWNSTREAM_DUMP=1

new analyzer:
  scripts/dsv4_compare_rmoe_downstream_dump.py

baseline dump=/tmp/dsv4_turn62_rmoe_baseline_l0_n80_downstream.jsonl
replace dump=/tmp/dsv4_turn62_rmoe_replace_l0_n80_downstream.jsonl
diff=/tmp/dsv4_turn62_rmoe_downstream_diff.txt
```

Result:

```text
first differing token: 3
first differing layer: 0
first differing stage: result_hc
first differing tensor: result_hc

logits input:
  first differs at token 3

local layer-0 / next-layer markers:
  final_ffn: exact
  hc_post_output: exact
  layer_output_after_ffn: exact
  next_layer_input: exact

later propagation after sampled-token divergence:
  ffn_input mismatch token 38
  topk_ids mismatch token 38
  topk_weights mismatch token 38
```

Root-cause classification:

```text
not a layer-0 routed-MoE final_ffn mismatch
not an hc_ffn_post input/output mismatch
not a layer handoff mismatch through next_layer_input

first observed downstream difference is at the final model head:
  result_hc / result_norm(logits input)

interpretation:
  layer-0 replacement remains locally exact, but the altered live graph
  produces a final-head/logits-input difference before the top-1 token flips.
```

n400 bounded dump:

```text
not needed in this turn:
  n80 downstream dump already found a pre-token-35 mismatch at result_hc token 3
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #59 Routed-MoE SwiGLU Stage Split

Turn #58 recap:

```text
generic routed-MoE SwiGLU path mapped as:
  CLAMP -> SILU -> CLAMP -> MUL

packed_generic layout matched generic stride/layout but did not make the backend
SwiGLU replacement path graph-safe.
```

Substage dump:

```text
new flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_STAGE_DUMP=1

new diagnostic mode:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_MODE=generic_graph_boundary

stages dumped:
  gate_raw
  gate_clamp_pre_silu
  silu_out
  silu_clamp_post (aliased to silu_out when no separate post clamp tensor exists)
  up_raw
  up_clamp_pre_mul
  mul_out
```

Results:

```text
baseline dump:
  /tmp/dsv4_turn59_rmoe_baseline_l0_n16.jsonl

shadow dump:
  /tmp/dsv4_turn59_rmoe_shadow_generic_graph_boundary_l0_n16.jsonl

replace dump:
  /tmp/dsv4_turn59_rmoe_replace_generic_graph_boundary_l0_n16.jsonl

analyzer:
  /tmp/dsv4_turn59_rmoe_swiglu_stage_diff.txt
```

Stage split conclusion:

```text
baseline_vs_shadow:
  first layout difference: gate_raw token 1 slot 0
  gate_raw value_exact=1 byte_exact=1 layout_exact=0
  gate_clamp_pre_silu value_exact=1 byte_exact=1 layout_exact=0
  silu_out value_exact=1 byte_exact=1 layout_exact=1
  mul_out value_exact=1 byte_exact=1 layout_exact=1
  first value-differing stage: none

baseline_vs_replace:
  first layout difference: gate_raw token 1 slot 0
  gate/up values byte-exact for the initial matching-token window
  first value-differing token: 8
  first value-differing stage: gate_raw
  reason: replacement run has already drifted by then; topk/expert ids differ

generic_graph_boundary:
  fixes the original token-1 SwiGLU value mismatch
  proves backend gate/up values are sufficient for generic CLAMP/SILU/CLAMP/MUL
  does not make replace-generic path transcript-stable through n16
```

Replace retest:

```text
n16:
  log=/tmp/dsv4_turn59_rmoe_replace_generic_graph_boundary_l0_n16.log
  result=not prefix-stable
  visible drift: "gets to" baseline vs "gets at" replacement

n80:
  not run because n16 replacement remained drifting
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #60 Routed-MoE Down Input from Generic SwiGLU Boundary

Turn #59 recap:

```text
generic_graph_boundary made the routed expert CLAMP/SILU/CLAMP/MUL path
value-exact and byte-exact, but replacement still consumed the backend op's
internal scratch downstream.
```

New down input mode:

```text
flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWN_INPUT

modes:
  internal_scratch
  generic_graph_boundary

generic_graph_boundary behavior:
  backend owns gate/up matvec scratch
  generic CLAMP/SILU/CLAMP/MUL produces packed SwiGLU
  expert down consumes that packed generic-boundary MUL output
  routed sum, shared branch, and final add are rebuilt from that path
```

Down input report:

```text
dsv4_rmoe_down_input:
  mode=generic_graph_boundary
  swiglu_source=generic_graph_boundary
  swiglu_layout=[2048,6,1,1]/[4,8192,49152,49152]/MUL
  down_consumes_swiglu_source=1
```

Substage diff:

```text
baseline dump:
  /tmp/dsv4_turn60_rmoe_baseline_l0_n16.jsonl

replace dump:
  /tmp/dsv4_turn60_rmoe_replace_l0_n16.jsonl

substage report:
  /tmp/dsv4_turn60_rmoe_substage_diff.txt

stage report:
  /tmp/dsv4_turn60_rmoe_swiglu_stage_diff.txt
```

Results:

```text
n16:
  swiglu exact: yes
  expert_down exact: yes
  routed_sum exact: yes
  shared_down exact: yes
  final_ffn exact: yes
  hc_post_input exact: yes
  transcript prefix stable: yes

n80:
  log=/tmp/dsv4_turn60_rmoe_replace_l0_n80.log
  dsv4_rmoe=79
  dsv4_rmoe_consume=79
  generic_ffn_built=0
  backend_ffn_built=1
  backend_output_consumed=1
  transcript prefix stable: yes
```

Policy unchanged:

```text
transcript-exact active: yes
tolerance adopted: no
drifting paths accepted: no
performance path accepted: no
n400 run: no
performance run: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #58 Routed-MoE SwiGLU Formula Parity

Turn #57 recap:

```text
gate/up slot values: byte-exact at token 1
backend SwiGLU layout: changed from padded VIEW to packed CONT
generic SwiGLU layout: packed contiguous MUL, stride [4,8192,49152,49152]
SwiGLU exactness: still non-exact at token 1 slot 0
```

Generic split-GLU formula mapping:

```text
mapping note=/tmp/dsv4_turn58_swiglu_formula_mapping.txt
generic graph path:
  DeepSeek4 routed expert limited path:
    clamp gate
    ggml_silu gate
    clamp up
    ggml_mul(silu_gate, clamped_up)
generic parent:
  op=MUL
  dtype=f32
  packed contiguous
  stride=[4,8192,49152,49152]
generic scalar formula:
  silu = x / (1 + exp(-x))
  out = silu * up
exp function:
  unqualified Metal exp, not fast::exp
materialization:
  separate f32 SiLU materialization followed by f32 MUL
```

Formula modes added:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_FORMULA=
  backend_current
  generic_exact
  generic_fast_exp
  generic_scalar_order
  generic_kernel_reuse

generic_kernel_reuse:
  implemented as an RMoE-internal SwiGLU-slot kernel over byte-exact backend gate/up slots
  does not attach a generic pSwiGLU/fGLU live graph op
```

Formula matrix:

```text
baseline=/tmp/dsv4_turn57_rmoe_baseline_l0_n16.jsonl

generic_exact:
  shadow=/tmp/dsv4_turn58_rmoe_shadow_generic_exact_l0_n16.jsonl
  replace=/tmp/dsv4_turn58_rmoe_replace_generic_exact_l0_n16.jsonl
  diff=/tmp/dsv4_turn58_rmoe_swiglu_diff_generic_exact.txt
  gate/up token 1: exact
  first mismatch: swiglu token 1 slot 0

generic_fast_exp:
  diff=/tmp/dsv4_turn58_rmoe_swiglu_diff_generic_fast_exp.txt
  gate/up token 1: exact
  first mismatch: swiglu token 1 slot 0

generic_scalar_order:
  diff=/tmp/dsv4_turn58_rmoe_swiglu_diff_generic_scalar_order.txt
  gate/up token 1: exact
  first mismatch: swiglu token 1 slot 0

generic_kernel_reuse:
  diff=/tmp/dsv4_turn58_rmoe_swiglu_diff_generic_kernel_reuse.txt
  gate/up token 1: exact
  first mismatch: swiglu token 1 slot 0
```

Conclusion:

```text
winning formula mode: none
SwiGLU formula parity achieved: no
replace n16/n80 retest: not run
root cause classification:
  not top-k
  not gate/up source
  not packed parent stride
  not simple scalar expression order
  not fast::exp vs exp
remaining blocker:
  backend scratch SwiGLU materialization still differs from generic f32 SiLU+MUL graph materialization
next action:
  reproduce or reuse the generic SiLU producer + MUL materialization boundary exactly, likely by staging a backend-compatible f32 SiLU slot before MUL rather than computing SwiGLU directly in the gate/up kernel
policy unchanged: yes
```

### 2026-05-13 Turn #48 Routed-MoE Consume Semantic Drift Root Cause

Turn #48 investigated why the routed-MoE backend consume canary from Turn #47 drifted at token index 2 even though short intrusive compare runs were exact. This remains a rejected consume path; no performance run was made.

Turn #47 drift recap:

```text
short intrusive compare:
  n16 exact=525/525
  n80 exact=2765/2765

n400 hot-neutral consume:
  first divergence token index=2
  position=12
  baseline token=" an"
  consume token=" a"
  max_abs=10.101223
  rms=4.13287799
```

New semantic trace controls:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_SEMANTIC=<mode>

modes:
  final_ffn
  materialized_final
  residual_added
  materialized_residual
  same_tensor_control
```

Trace finding for tokens 1..8:

```text
generic_ffn_output:
  name=ffn_out-0
  op=ADD
  dtype=f32
  shape=[4096,1,1,1]
  stride=[4,16384,16384,16384]

backend_ffn_output:
  name=dsv4_rmoe_consume_final_ffn-0
  op=VIEW
  dtype=f32
  shape=[4096,1,1,1]
  stride=[4,16384,16384,16384]

downstream consumer:
  generic=hc_ffn_post / DSV4_HC_POST
  backend=hc_ffn_post / DSV4_HC_POST

direct residual add:
  direct_residual_compatible=0
  residual_semantic_unavailable=1
```

Semantic matrix:

```text
baseline:
  jsonl=/tmp/dsv4_turn47_rmoe_baseline_n400_logits.jsonl

same_tensor_control:
  log=/tmp/dsv4_turn48_rmoe_same_tensor_control_l0_n400.log
  jsonl=/tmp/dsv4_turn48_rmoe_same_tensor_control_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn48_rmoe_first_divergence_same_tensor_control.txt
  result=diverged
  first divergent token index=132
  baseline token=" A"
  consume/control token=" The"
  max_abs=0.305324554
  rms=0.164779033
  conclusion=consume metadata plus extra backend-op branch is not fully hot-neutral even when generic ffn_out is fed downstream

materialized_final:
  log=/tmp/dsv4_turn48_rmoe_materialized_final_l0_n400.log
  jsonl=/tmp/dsv4_turn48_rmoe_materialized_final_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn48_rmoe_first_divergence_materialized_final.txt
  result=diverged
  first divergent token index=2
  baseline token=" an"
  consume token=" a"
  max_abs=10.4645729
  rms=4.27289462
  conclusion=plain CONT materialization of backend final does not fix the early drift

residual_added:
  log=/tmp/dsv4_turn48_rmoe_residual_added_l0_n400.log
  jsonl=/tmp/dsv4_turn48_rmoe_residual_added_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn48_rmoe_first_divergence_residual_added.txt
  result=diverged
  first divergent token index=2
  max_abs=10.101223
  rms=4.13287799
  direct_residual_compatible=0
  conclusion=the residual hypothesis is not directly testable at this insertion point; generic post-FFN residual handling is HC-post, not a shape-compatible final_ffn add

materialized_residual:
  log=/tmp/dsv4_turn48_rmoe_materialized_residual_l0_n400.log
  jsonl=/tmp/dsv4_turn48_rmoe_materialized_residual_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn48_rmoe_first_divergence_materialized_residual.txt
  result=diverged
  first divergent token index=2
  max_abs=10.101223
  rms=4.13287799
  direct_residual_compatible=0
  conclusion=same residual-boundary incompatibility as residual_added
```

Root-cause classification:

```text
consume plumbing:
  implicated for long-context neutrality, because same_tensor_control drifts later at token 132

residual semantic:
  not the immediate fix; direct residual add is shape-incompatible at this graph point

materialization/layout:
  plain CONT materialization does not fix token-2 drift

backend tensor under consume:
  implicated for immediate token-2 drift; replacing generic ADD ffn_out with backend op output VIEW/CONT changes downstream logits even though shadow compare was exact
```

Decision:

```text
routed-MoE backend consume eligible: no
consume canary accepted: no
next action:
  add a dependency-safe/output-finalizer diagnostic for the backend op output, or run a true hot-path local tensor readback around token 1..3 to compare generic ffn_out, backend final, and hc_ffn_post input under consume semantics
performance run: no
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #49 Routed-MoE Backend Consume Boundary Probe

Turn #49 tested whether the routed-MoE backend output can be made semantically equivalent to the generic `ffn_out` tensor consumed by `hc_ffn_post / DSV4_HC_POST`. This remains boundary-correctness work only; no performance run was made.

Turn #48 recap:

```text
same_tensor_control:
  drifted later at token index 132
  conclusion=extra backend branch / consume plumbing is not hot-neutral

backend output consume:
  drifted immediately at token index 2
  conclusion=backend tensor path is not equivalent at hc_ffn_post

generic tensor:
  name=ffn_out-0
  op=ADD
  dtype=f32
  shape=[4096,1,1,1]

backend tensor:
  name=dsv4_rmoe_consume_final_ffn-0
  op=VIEW
  dtype=f32
  shape=[4096,1,1,1]
```

New boundary probe controls:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BOUNDARY_PROBE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_SEMANTIC=<mode>

new modes:
  same_tensor_no_backend_branch
  same_tensor_with_backend_branch
  backend_view
  backend_cont
  backend_add_zero
  backend_alias_like_generic
  backend_rebuild_generic_add
```

Boundary probe finding:

```text
generic ffn_out:
  name=ffn_out-0
  op=ADD
  dtype=f32
  shape=[4096,1,1,1]
  stride=[4,16384,16384,16384]
  view_src=null
  storage_offset=0
  materialized=0
  contiguous=1
  producer_op=ADD

backend final:
  name=dsv4_rmoe_consume_final_ffn-0
  op=VIEW
  dtype=f32
  shape=[4096,1,1,1]
  stride=[4,16384,16384,16384]
  view_src=dsv4_moe_backend_op_shadow-0
  storage_offset=557056
  materialized=0
  contiguous=1
  producer_op=DSV4_ROUTED_MOE_ONE_TENSOR_DECODE

downstream consumer:
  name=hc_ffn_post
  op=DSV4_HC_POST

value compare before hc_post:
  hot-neutral modes did not perform readback
  same_tensor_no_backend_branch is identity-exact by construction
```

Semantic matrix:

```text
baseline:
  log=/tmp/dsv4_turn49_rmoe_baseline_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_baseline_n400_logits.jsonl

same_tensor_no_backend_branch:
  log=/tmp/dsv4_turn49_rmoe_same_tensor_no_backend_branch_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_same_tensor_no_backend_branch_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_same_tensor_no_backend_branch.txt
  result=exact through n400
  max_abs=0
  rms=0
  conclusion=consume metadata/under-test path is neutral when the backend branch is not built

same_tensor_with_backend_branch:
  log=/tmp/dsv4_turn49_rmoe_same_tensor_with_backend_branch_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_same_tensor_with_backend_branch_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_same_tensor_with_backend_branch.txt
  result=diverged
  first divergent token index=132
  baseline token=" A"
  consume/control token=" The"
  max_abs=0.305324554
  rms=0.164779033
  conclusion=the extra backend branch perturbs the hot path even when generic ffn_out is consumed

backend_alias_like_generic:
  log=/tmp/dsv4_turn49_rmoe_backend_alias_like_generic_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_backend_alias_like_generic_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_backend_alias_like_generic.txt
  result=diverged
  first divergent token index=2
  baseline token=" an"
  consume token=" a"
  max_abs=10.101223
  rms=4.13287799
  conclusion=name aliasing is insufficient; the consumed tensor remains a backend VIEW

backend_rebuild_generic_add:
  log=/tmp/dsv4_turn49_rmoe_backend_rebuild_generic_add_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_backend_rebuild_generic_add_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_backend_rebuild_generic_add.txt
  result=diverged
  first divergent token index=11
  baseline token=" what"
  consume token=" how"
  max_abs=4.44015884
  rms=1.89585595
  conclusion=rebuilding an ADD-shaped backend final delays but does not remove drift

backend_cont:
  log=/tmp/dsv4_turn49_rmoe_backend_cont_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_backend_cont_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_backend_cont.txt
  result=diverged
  first divergent token index=2
  baseline token=" an"
  consume token=" a"
  max_abs=10.4645729
  rms=4.27289462
  conclusion=plain CONT/materialization does not fix the boundary

backend_add_zero:
  log=/tmp/dsv4_turn49_rmoe_backend_add_zero_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_backend_add_zero_l0_n400_logits.jsonl
  result=invalid diagnostic
  first_divergence=not parsed; JSONL became invalid after drift
  failure=nlohmann JSON invalid UTF-8 while dumping token output
  boundary metadata: consumed tensor op=ADD and shape/stride matched generic
  conclusion=ADD-shaped metadata alone is still not a graph-safe consume boundary

backend_view:
  log=/tmp/dsv4_turn49_rmoe_backend_view_l0_n400.log
  jsonl=/tmp/dsv4_turn49_rmoe_backend_view_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn49_rmoe_first_divergence_backend_view.txt
  result=diverged
  first divergent token index=2
  baseline token=" an"
  consume token=" a"
  max_abs=10.101223
  rms=4.13287799
  conclusion=current VIEW consume remains rejected
```

Root-cause classification:

```text
consume plumbing alone:
  not the cause; same_tensor_no_backend_branch is exact through n400

extra backend branch:
  confirmed issue; same_tensor_with_backend_branch drifts at token index 132 even while consuming generic ffn_out

backend tensor semantic/materialization:
  confirmed issue; backend VIEW/CONT/name-alias/add-zero modes drift at token index 2 or fail before a valid n400 comparison

hc_post input metadata:
  relevant but not sufficient; backend_rebuild_generic_add improves first divergence from token 2 to token 11 but still drifts

value mismatch under consume:
  still possible at the hc_ffn_post boundary; hot-neutral probes intentionally avoided readback, so the next diagnostic needs a bounded intrusive readback of generic ffn_out, backend final, and hc_ffn_post input/output for tokens 1..8
```

Decision:

```text
graph-safe consume semantic found: no
consume canary accepted: no
routed-MoE backend consume eligible: no
next action:
  first make the extra backend branch hot-neutral when unused, then add a bounded intrusive boundary readback at hc_ffn_post to distinguish value mismatch from producer/dependency metadata mismatch
performance run: no
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #50 Routed-MoE Backend Branch Hot-Neutrality

Turn #50 isolated why the unused routed-MoE backend branch perturbs the n400 hot path even when generic `ffn_out` remains the tensor consumed by `hc_ffn_post`.

Turn #49 recap:

```text
same_tensor_no_backend_branch:
  exact through n400

same_tensor_with_backend_branch:
  diverged at token index 132

backend output consume:
  diverged at token index 2

interpretation:
  consume metadata alone can be neutral
  building the extra backend branch is not neutral
  backend output consumption remains rejected
```

New branch mode control:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_MODE=<mode>

modes:
  none
  metadata_only
  alloc_only
  dispatch_noop
  dispatch_compute_no_read
  dispatch_compute_compare
```

Baseline:

```text
log=/tmp/dsv4_turn50_rmoe_baseline_n400.log
jsonl=/tmp/dsv4_turn50_rmoe_baseline_n400_logits.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
```

Branch matrix:

```text
none:
  log=/tmp/dsv4_turn50_rmoe_branch_none_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_none_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_none.txt
  result=exact through n400
  max_abs=0
  rms=0
  backend_tensors_allocated=0
  backend_dispatches=0
  metal_dispatch=1399039
  dsv4_rmoe=0

metadata_only:
  log=/tmp/dsv4_turn50_rmoe_branch_metadata_only_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_metadata_only_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_metadata_only.txt
  result=exact through n400
  max_abs=0
  rms=0
  backend_tensors_allocated=0
  backend_dispatches=0
  metal_dispatch=1399039
  dsv4_rmoe=0

alloc_only:
  log=/tmp/dsv4_turn50_rmoe_branch_alloc_only_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_alloc_only_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_alloc_only.txt
  result=exact through n400
  max_abs=0
  rms=0
  backend_tensors_allocated=399
  backend_dispatches=0
  scratch_bytes=573440
  metal_dispatch=1399039
  dsv4_rmoe=0

dispatch_noop:
  log=/tmp/dsv4_turn50_rmoe_branch_dispatch_noop_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_dispatch_noop_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_dispatch_noop.txt
  result=diverged
  first divergent token index=35
  baseline token=" built"
  branch token=" ("
  max_abs=0.38425827
  rms=0.183860982
  backend_tensors_allocated=399
  backend_dispatches=399
  scratch_bytes=16384
  metal_dispatch=1397842
  dsv4_rmoe=399
  generic kernel mix changed:
    baseline pair=399 pswiglu=798 fglu=798
    dispatch_noop pair=0 pswiglu=1197 fglu=1197

dispatch_compute_no_read:
  log=/tmp/dsv4_turn50_rmoe_branch_dispatch_compute_no_read_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_dispatch_compute_no_read_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_dispatch_compute_no_read.txt
  result=diverged
  first divergent token index=132
  baseline token=" A"
  branch token=" The"
  max_abs=0.305324554
  rms=0.164779033
  backend_tensors_allocated=399
  backend_dispatches=1596
  scratch_bytes=573440
  metal_dispatch=1400635
  dsv4_rmoe=399

dispatch_compute_compare:
  log=/tmp/dsv4_turn50_rmoe_branch_dispatch_compute_compare_l0_n400.log
  jsonl=/tmp/dsv4_turn50_rmoe_branch_dispatch_compute_compare_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn50_rmoe_first_divergence_branch_dispatch_compute_compare.txt
  result=diverged
  first divergent token index=104
  baseline token=" and"
  branch token=" ("
  max_abs=0.422439575
  rms=0.150950382
  backend_tensors_allocated=399
  backend_dispatches=1596
  compare_enabled=1
  compared_cases=13965
  exact_cases=13965
  scratch_bytes=573440
  metal_dispatch=1461283
  dsv4_rmoe=399
```

First perturbing mode:

```text
mode=dispatch_noop
first divergence token index=35
max_abs=0.38425827
rms=0.183860982
```

Interpretation:

```text
metadata_only exact:
  branch guard/metadata plumbing is neutral

alloc_only exact:
  tensor allocation / graph context growth alone is neutral

dispatch_noop drifts:
  the first perturbation is adding an unused backend op dispatch to the graph
  this changes graph lowering/scheduling even with generic ffn_out consumed

compute_no_read and compute_compare drift:
  expected once dispatch insertion is already non-neutral
  compare/readback is not the first root cause
```

Ordering isolation:

```text
not run in Turn #50
reason=first perturbing mode is dispatch_noop, so the immediate root cause is dispatch/graph scheduling rather than compute body or readback; an ordering-isolation turn should move the branch out of the active decode graph or to a post-layer/end-of-graph diagnostic queue before re-testing consume
```

Decision:

```text
unused backend branch hot-neutral: no
root cause: extra backend op dispatch changes graph scheduling/lowering
backend output consume eligible: no
next action:
  isolate shadow/backend branch execution from the live decode graph before re-testing backend output consumption
performance run: no
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #51 Routed-MoE Backend Branch Ordering / Hot-Neutrality

Turn #51 tested whether the routed-MoE backend branch can be built or dispatched without changing the existing split-GLU lowering for the generic FFN path. Backend output was not consumed, no all-layer consume was attempted, and no performance run was made.

Turn #50 recap:

```text
unused backend branch:
  metadata_only exact through n400
  alloc_only exact through n400
  dispatch_noop diverged at token index 35

generic FFN lowering changed:
  baseline pair=399 pswiglu=798 fglu=798
  dispatch_noop pair=0 pswiglu=1197 fglu=1197

root direction:
  live-graph backend dispatch perturbs split-GLU pairing/lowering
```

New branch order control:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_ORDER=<mode>

modes:
  inline_current
  after_generic_ffn
  after_layer
  end_of_graph
  separate_side_graph
```

Baseline:

```text
log=/tmp/dsv4_turn51_rmoe_baseline_n400.log
jsonl=/tmp/dsv4_turn51_rmoe_baseline_n400_logits.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
pair=399
pswiglu=798
fglu=798
```

Order matrix with `branch_mode=dispatch_noop`:

```text
inline_current:
  log=/tmp/dsv4_turn51_rmoe_order_inline_current_l0_n400.log
  jsonl=/tmp/dsv4_turn51_rmoe_order_inline_current_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_inline_current.txt
  result=diverged
  first divergent token index=35
  baseline token=" built"
  branch token=" ("
  max_abs=0.38425827
  rms=0.183860982
  pair=0
  pswiglu=1197
  fglu=1197
  dsv4_rmoe=399
  backend_dispatches=399

after_generic_ffn:
  log=/tmp/dsv4_turn51_rmoe_order_after_generic_ffn_l0_n400.log
  jsonl=/tmp/dsv4_turn51_rmoe_order_after_generic_ffn_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_after_generic_ffn.txt
  result=diverged
  first divergent token index=35
  max_abs=0.38425827
  rms=0.183860982
  pair=0
  pswiglu=1197
  fglu=1197
  dsv4_rmoe=399
  backend_dispatches=399

after_layer:
  log=/tmp/dsv4_turn51_rmoe_order_after_layer_l0_n400.log
  jsonl=/tmp/dsv4_turn51_rmoe_order_after_layer_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_after_layer.txt
  result=diverged
  first divergent token index=35
  max_abs=0.38425827
  rms=0.183860982
  pair=0
  pswiglu=1197
  fglu=1197
  dsv4_rmoe=399
  backend_dispatches=399

end_of_graph:
  log=/tmp/dsv4_turn51_rmoe_order_end_of_graph_l0_n400.log
  jsonl=/tmp/dsv4_turn51_rmoe_order_end_of_graph_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_end_of_graph.txt
  result=diverged
  first divergent token index=35
  max_abs=0.38425827
  rms=0.183860982
  pair=0
  pswiglu=1197
  fglu=1197
  dsv4_rmoe=399
  backend_dispatches=399

separate_side_graph:
  log=/tmp/dsv4_turn51_rmoe_order_separate_side_graph_l0_n400.log
  jsonl=/tmp/dsv4_turn51_rmoe_order_separate_side_graph_l0_n400_logits.jsonl
  first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_separate_side_graph.txt
  result=exact through n400
  max_abs=0
  rms=0
  pair=399
  pswiglu=798
  fglu=798
  dsv4_rmoe=0
  backend_dispatches=0
  backend_tensors_allocated=399
  scratch_bytes=16384
```

Compute-no-read retest:

```text
order=separate_side_graph
branch_mode=dispatch_compute_no_read
log=/tmp/dsv4_turn51_rmoe_order_separate_side_graph_dispatch_compute_no_read_l0_n400.log
jsonl=/tmp/dsv4_turn51_rmoe_order_separate_side_graph_dispatch_compute_no_read_l0_n400_logits.jsonl
first_divergence=/tmp/dsv4_turn51_rmoe_first_divergence_order_separate_side_graph_dispatch_compute_no_read.txt
result=exact through n400
max_abs=0
rms=0
pair=399
pswiglu=798
fglu=798
dsv4_rmoe=0
backend_dispatches=0
backend_tensors_allocated=399
scratch_bytes=573440
```

Conclusion:

```text
hot-neutral branch order found: separate_side_graph only
root cause:
  any routed-MoE backend dispatch attached to the live decode graph changes split-GLU lowering
  moving the branch later inside the same graph does not preserve pair/pswiglu/fglu counters

important caveat:
  separate_side_graph is hot-neutral because it avoids attaching/dispatching the backend op in the live graph
  it is a safe diagnostic/metadata side path, not a live backend compute path

backend consume canary can resume: no
next action:
  implement a genuinely separate diagnostic execution context, or replace the generic FFN branch outright so the backend op is no longer an extra live-graph dispatch
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #52 Routed-MoE Replace-Generic Single-Layer Canary

Turn #52 implemented a selected-layer replacement canary for the routed-MoE backend op. The selected layer does not build the generic routed-MoE expert/shared FFN branch; it builds the backend V1 routed-MoE output and feeds that tensor into `hc_ffn_post`.

Turn #51 recap:

```text
extra live backend branch:
  not hot-neutral
  changes split-GLU lowering

only neutral branch mode:
  separate_side_graph
  exact because no backend dispatch is attached to the live graph

turn #52 rule:
  consume canary must replace the generic branch
  generic and backend FFN branches must not be built together for the selected layer
```

New flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC=1
```

Implementation notes:

```text
under-test:
  rmoe_backend_single_layer_replace_generic

selected layer:
  layer 0 only

route/top-k:
  replacement now uses build_moe_ffn(..., route_only=true)
  this reuses the generic router/top-k/route-weight construction exactly
  expert/shared FFN branches are not built in route_only mode

backend output boundary:
  default replacement semantic is backend_rebuild_generic_add
  consumes ADD(routed_sum, shared_down), named/callbacked as ffn_out
```

Guard result:

```text
missing layer:
  log=/tmp/dsv4_turn52_rmoe_replace_guard_missing_layer.log
  allowed=0
  reason=missing_layer
  dsv4_rmoe_consume=0
```

n16 replace-generic:

```text
log=/tmp/dsv4_turn52_rmoe_replace_l0_n16.log
prefix stable=yes
generic_ffn_built=0
backend_ffn_built=1
backend_output_consumed=1
output_computed=1
dsv4_rmoe=15
dsv4_rmoe_consume=15
consume_semantic=backend_view before final ADD-boundary revision, then backend_rebuild_generic_add after revision
```

n80 replace-generic:

```text
log=/tmp/dsv4_turn52_rmoe_replace_l0_n80.log
prefix stable=no
generic_ffn_built=0
backend_ffn_built=1
backend_output_consumed=1
output_computed=1
dsv4_rmoe=79
dsv4_rmoe_consume=79
consume_semantic=backend_rebuild_generic_add

observed drift:
  baseline-style prefix expected:
    "This is an excellent question that gets to the heart..."
  replacement generated:
    "This is a great question. The **Apple Neural Engine..."
```

n400 hot-neutral:

```text
not run
reason:
  instructions required n400 only after n80 stability
  n80 replacement was not prefix-stable
```

Decision:

```text
replace-generic canary accepted: no
consume path accepted: no
performance run: no

current blocker:
  backend replacement is graph-shaped correctly and skips generic FFN
  but the consumed backend value/boundary is still not transcript-equivalent

next action:
  add a replacement-only local tensor probe or paired short logit compare to locate whether the mismatch is backend value, final ADD materialization, or hc_ffn_post input semantics
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #53 Routed-MoE Replace-Generic Drift Localization

Turn #53 added a paired JSONL dump for the replacement canary. Baseline and replace-generic are still separate runs, so the diagnostic does not reintroduce a live generic+backend branch in the same graph.

Turn #52 recap:

```text
replace-generic true path:
  generic_ffn_built=0
  backend_ffn_built=1
  backend_output_consumed=1

n80:
  prefix stable=no
  consume path remains rejected
```

New dump flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_DUMP=/tmp/...
```

Paired dump files:

```text
baseline:
  log=/tmp/dsv4_turn53_rmoe_baseline_l0_n80.log
  dump=/tmp/dsv4_turn53_rmoe_baseline_l0_n80.jsonl
  rows=79

replace:
  log=/tmp/dsv4_turn53_rmoe_replace_l0_n80.log
  dump=/tmp/dsv4_turn53_rmoe_replace_l0_n80.jsonl
  rows=79

analyzer:
  script=scripts/dsv4_compare_rmoe_replace_dump.py
  report=/tmp/dsv4_turn53_rmoe_replace_dump_diff.txt
```

Paired dump result:

```text
first topk_ids mismatch:
  token 3

first topk_weights mismatch:
  token 3

first ffn_input mismatch:
  token 3

first final_ffn mismatch:
  token 1

first hc_ffn_post input mismatch:
  token 1

first hc_ffn_post output mismatch:
  unavailable in callback dump
```

Token 1 detail:

```text
topk_ids:
  equal=yes

topk_weights:
  equal=yes

ffn_input:
  equal=yes

final_ffn:
  equal=no
  baseline_hash=6801085f6ac6e305
  replace_hash=e409605ae7f2b25e

final first values:
  baseline=[0.129036039, 0.466552228, -0.223680437, 0.614814579, ...]
  replace=[0.0472793654, 0.329763114, -0.101723596, 0.224023476, ...]
```

Root-cause classification:

```text
backend numerical mismatch:
  yes, in replacement mode before hc_ffn_post

hc_ffn_post producer-op sensitivity:
  not first cause; hc_ffn_post input already differs at token 1

topk mismatch:
  not first cause; topk ids/weights first differ at token 3 after earlier propagation

earlier graph perturbation:
  not first cause; layer-0 ffn_input first differs at token 3

later-layer propagation:
  follows token-1 final_ffn mismatch
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #54 Routed-MoE Replace Substage Diff

Turn #54 extended the replacement dump with substage summaries and compared baseline generic against replace-generic for layer 0 n16. This remained diagnostic only: no performance run, no all-layer consume, and no drift accepted.

Turn #53 recap:

```text
final_ffn differs:
  token 1

hc_post_input differs:
  token 1

ffn_input/topk_ids/topk_weights:
  match until token 3
```

New flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_SUBSTAGE_DUMP=1
```

Substage dump files:

```text
baseline:
  log=/tmp/dsv4_turn54_rmoe_baseline_l0_n16_substage.log
  dump=/tmp/dsv4_turn54_rmoe_baseline_l0_n16_substage.jsonl
  rows=15

replace:
  log=/tmp/dsv4_turn54_rmoe_replace_l0_n16_substage.log
  dump=/tmp/dsv4_turn54_rmoe_replace_l0_n16_substage.jsonl
  rows=15

analyzer:
  script=scripts/dsv4_compare_rmoe_substage_dump.py
  report=/tmp/dsv4_turn54_rmoe_substage_diff.txt
```

Substage diff:

```text
first differing token:
  token 1

first differing stage by checksum:
  gate

gate/up detail:
  shape/stat summaries match
  raw checksum differs
  interpretation: checksum/layout or hidden byte-level difference, not visible in printed scalar stats

first differing stage by numeric stats:
  swiglu

down/routed_sum/shared/final:
  differ at token 1 after the swiglu/stat mismatch

ffn_input/topk_ids/topk_weights:
  first differ at token 8 in this n16 substage run
  not the first cause
```

Root-cause classification:

```text
backend value mismatch under replace:
  yes

first suspect:
  gate/up materialization or byte layout feeding SwiGLU

first visible numeric/stat mismatch:
  SwiGLU

not first cause:
  topk mismatch
  hc_ffn_post producer-op sensitivity
  later-layer propagation
```

Fix:

```text
applied: no
reason: mismatch is not yet a trivial wrong-source/wrong-layout fix; next pass should split gate/up byte exactness from SwiGLU formula/materialization in replace mode
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #55 Routed-MoE Replace Gate/Up Scratch Diff

Turn #55 added a byte-level per-slot gate/up dump and compared three layer 0 n16 runs:

```text
baseline:
  /tmp/dsv4_turn55_rmoe_baseline_l0_n16.jsonl

shadow:
  /tmp/dsv4_turn55_rmoe_shadow_l0_n16.jsonl

replace:
  /tmp/dsv4_turn55_rmoe_replace_l0_n16.jsonl

analyzer:
  scripts/dsv4_compare_rmoe_gate_up_dump.py
  /tmp/dsv4_turn55_rmoe_gate_up_diff.txt
```

Turn #54 recap:

```text
gate/up:
  combined tensor checksum differed at token 1
  printed stats matched

SwiGLU:
  first visible numeric/stat mismatch at token 1
```

Gate/up byte-level result:

```text
baseline_vs_shadow:
  exact: yes

baseline_vs_replace:
  first differing token: 8
  first differing stage: gate
  reason: ffn_input/topk have already diverged by token 8

shadow_vs_replace:
  first differing token: 8
  first differing stage: gate
  reason: ffn_input/topk have already diverged by token 8
```

Token 1 slot detail:

```text
gate:
  all 6 selected expert slots byte-exact across baseline/shadow/replace

up:
  all 6 selected expert slots byte-exact across baseline/shadow/replace

expert order:
  baseline/shadow/replace token 1 ids:
    [234, 216, 130, 13, 17, 124]

layout:
  generic baseline gate/up parent stride:
    [4,8192,49152,49152]
  backend shadow/replace gate/up parent stride:
    [4,16384,98304,98304]
```

Classification:

```text
gate/up value-source mismatch:
  no

gate/up slot order bug:
  no

Turn #54 combined checksum mismatch cause:
  layout/materialization difference
  generic packed [2048,6] slots use 8192-byte slot stride
  backend scratch views use 16384-byte slot stride

first visible numeric mismatch from clean baseline:
  SwiGLU token 1

replace path status:
  still rejected
```

Next target:

```text
split backend SwiGLU/materialization from generic split-GLU semantics:
  compare packed vs padded SwiGLU input layout
  verify whether backend SwiGLU must emit packed [2048,6] to match generic FFN replacement
  separately inspect shared-branch source/layout, which also differs under shadow_vs_replace after routed_sum
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #56 Routed-MoE Replace SwiGLU Diff

Turn #56 extended the routed-MoE replacement dump with per-slot SwiGLU byte/value records.
This was diagnostic only: no performance run, no all-layer consume, and no drift accepted.

New flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_BYTE_DUMP=1
```

Dumps:

```text
baseline:
  /tmp/dsv4_turn56_rmoe_baseline_l0_n16.jsonl

shadow:
  /tmp/dsv4_turn56_rmoe_shadow_l0_n16.jsonl

replace:
  /tmp/dsv4_turn56_rmoe_replace_l0_n16.jsonl

analyzer:
  scripts/dsv4_compare_rmoe_swiglu_dump.py
  /tmp/dsv4_turn56_rmoe_swiglu_diff.txt
```

Turn #55 recap:

```text
gate/up token 1:
  byte-exact per slot across baseline/shadow/replace

first visible numeric mismatch:
  SwiGLU token 1
```

SwiGLU diff:

```text
baseline_vs_shadow:
  first differing token: 1
  first differing stage: swiglu
  first differing slot: 0
  gate exact: yes
  up exact: yes

baseline_vs_replace:
  first differing token: 1
  first differing stage: swiglu
  first differing slot: 0
  gate first mismatch: token 8
  up first mismatch: token 8

shadow_vs_replace:
  first mismatch: token 8
  reason: upstream ffn_input/topk already diverged by then
```

Parent/materialization:

```text
baseline swiglu parent:
  op=MUL
  shape=[2048,6,1,1]
  stride=[4,8192,49152,49152]
  contiguous=1

backend shadow/replace swiglu parent:
  op=VIEW
  view_src=dsv4_moe_backend_op_shadow-0
  shape=[2048,6,1,1]
  stride=[4,16384,98304,98304]
  contiguous=0
  storage_offset=196608
```

Formula/materialization finding:

```text
gate/up inputs:
  byte-exact at token 1

SwiGLU:
  differs at token 1 slot 0 by small f32-level deltas

clean baseline path:
  pair=15
  pswiglu=15
  fglu=15
  producer is generic packed split-GLU/MUL path

backend shadow/replace path:
  dsv4_rmoe=15
  pair=0
  pswiglu=0
  fglu=0
  producer is padded backend scratch VIEW
```

Classification:

```text
root cause:
  backend replacement does not match the clean generic packed split-GLU materialization.
  Gate/up source and slot order are correct; the first mismatch is SwiGLU producer/materialization semantics.

trivial fix:
  no

replace path status:
  rejected
```

Next target:

```text
implement or test a backend SwiGLU mode that matches generic packed split-GLU output:
  same packed parent stride [4,8192,...]
  same fused pSwiGLU/fGLU scalar evaluation/materialization
  no extra live branch
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #57 Routed-MoE Packed SwiGLU Materialization

Turn #57 added a `packed_generic` SwiGLU materialization mode and reran the layer 0 n16
baseline/shadow/replace dump matrix. This was diagnostic only: no performance run, no all-layer
consume, and no drift accepted.

Turn #56 recap:

```text
gate/up:
  byte-exact at token 1

SwiGLU:
  first mismatch at token 1 slot 0

generic parent:
  op=MUL
  packed contiguous stride=[4,8192,49152,49152]

backend parent:
  op=VIEW
  padded scratch stride=[4,16384,98304,98304]
```

Mode:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_MODE=packed_generic
```

Implementation:

```text
compared backend SwiGLU:
  backend padded VIEW -> CONT

reported layout:
  dsv4_rmoe_swiglu_layout:
    mode=packed_generic
    generic_parent_op=MUL
    backend_parent_op=CONT
    generic_stride=[4,8192,49152,49152]
    backend_stride=[4,8192,49152,49152]
    generic_packed=1
    backend_packed=1
    view_src=none
```

Dump files:

```text
baseline:
  /tmp/dsv4_turn57_rmoe_baseline_l0_n16.jsonl

shadow:
  /tmp/dsv4_turn57_rmoe_shadow_l0_n16.jsonl

replace:
  /tmp/dsv4_turn57_rmoe_replace_l0_n16.jsonl

analyzer:
  /tmp/dsv4_turn57_rmoe_swiglu_diff.txt
```

Diff result:

```text
baseline_vs_shadow:
  first differing token: 1
  first differing stage: swiglu
  first differing slot: 0
  gate mismatch: none
  up mismatch: none

baseline_vs_replace:
  first differing token: 1
  first differing stage: swiglu
  first differing slot: 0
  gate first mismatch: token 8
  up first mismatch: token 8

shadow_vs_replace:
  first mismatch: token 8
  reason: upstream ffn_input/topk have diverged by then
```

Classification:

```text
packed layout fixed:
  yes, for compared SwiGLU parent

SwiGLU byte/value exact:
  no

root cause:
  layout was only one part of the mismatch.
  Clean generic pSwiGLU/fGLU computes SwiGLU inside the generic fused split-GLU lowering.
  Backend RMoE computes SwiGLU in its own backend scratch kernel from equivalent gate/up values.
  The resulting f32 materialization differs at token 1 slot 0 even after packing.

exact blocker:
  backend RMoE needs to reuse or exactly mirror the generic pair_limited_swiglu_iq2_xxs_f32
  materialization, not just pack the existing backend scratch result.
```

Replace retest:

```text
n16:
  not run as acceptance retest; SwiGLU still non-exact

n80:
  not run
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
```

### 2026-05-13 Turn #29 CUPD3 Attention-Stream Quant/Cache Row Probe

Turn #28 recap:

```text
CUPD3 backend-tail consume:
  n16/n80 exact in short intrusive checks
  n400 drift at token index 79 / position 89
  barrier did not fix drift
  emit-only did not fix drift
  attn-only diverged with the same logit error
  index-only remained exact through 400 records

policy:
  backend-tail consume remains rejected
  tolerance not adopted
  performance not run
```

New diagnostics:

```text
flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_ROW_PROBE=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_LAYOUT_MODE=<backend_layout|force_contiguous>

scope:
  layer=2
  trace tokens=70..90
  projection_source=generic
  cache_mutation_mode=generic_existing_write
  candidate_cache_side_effect=0
```

Bounded row probe:

```text
log=/tmp/dsv4_turn29_cupd3_attn_row_probe_t70_90.log

result:
  consume_path=disabled
  generic downstream consumed
  emit_cases=50
  exact_cases=50
  drift_trace_cases=10
  attn_row_probe_cases=10

attention stream:
  first_mismatch_stage=none at structural/layout level
  generic_pre_op=DSV4_ROPE_TAIL
  backend_pre_op=DSV4_DECODE_COMPRESS
  generic_pre_type=f32
  backend_pre_type=f32
  generic_pre_shape=[512,1,1,1]
  backend_pre_shape=[512,1,1,1]
  generic_pre_stride=[4,2048,2048,2048]
  backend_pre_stride=[4,2048,2048,2048]
  pre_layout_exact=1
  generic_quant_op=DSV4_FP8_KV_QUANTIZE
  backend_quant_op=DSV4_FP8_KV_QUANTIZE
  generic_quant_shape=[512,1,1,1]
  backend_quant_shape=[512,1,1,1]
  generic_quant_stride=[4,2048,2048,2048]
  backend_quant_stride=[4,2048,2048,2048]
  quant_layout_exact=1
  byte_exact=not_readback

index stream control:
  first_mismatch_stage=none at structural/layout level
  pre_layout_exact=1
  quant_layout_exact=1
```

Layout mode test:

```text
mode=backend_layout:
  structural layout parity already exact in bounded probe

mode=force_contiguous:
  bounded probe log=/tmp/dsv4_turn29_cupd3_attn_row_probe_force_contiguous_t70_90.log
  structural layout parity remains exact
  backend_pre_op=CONT
  backend_quant_src0_op=CONT

n400 hot-neutral attn-only retest:
  baseline log=/tmp/dsv4_turn29_cupd3_baseline_n400.log
  baseline jsonl=/tmp/dsv4_turn29_cupd3_baseline_n400_logits.jsonl
  force-contiguous log=/tmp/dsv4_turn29_cupd3_attn_force_contiguous_l2_n400.log
  force-contiguous jsonl=/tmp/dsv4_turn29_cupd3_attn_force_contiguous_l2_n400_logits.jsonl
  first divergence=/tmp/dsv4_turn29_cupd3_attn_force_contiguous_first_divergence.txt

result:
  first divergent token index=79
  position=89
  baseline token="'s"
  consume token=" has"
  max_abs_logit_err=1.28058624
  rms_logit_err=0.525356713
  fixed drift=no
```

Decision:

```text
first proven non-parity point:
  not tensor shape/stride/storage-offset/header at graph metadata level
  not simple pre-quant contiguity
  not index stream

remaining suspect:
  attention-stream value/content semantics inside GGML_OP_DSV4_DECODE_COMPRESS before or at FP8 quantization

backend-tail consume eligible: no
continue CUPD3 consume: no, until attention-stream value/byte parity can be proven without relying on layout parity alone
next action:
  add a non-consuming readback-limited value/byte probe for the attention stream only, or split DSV4_DECODE_COMPRESS into smaller exact probes to isolate pooled/norm/rope value parity before quant
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
performance run: not run
cache side effects added: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #30 CUPD3 Attention-Stream Value Probe

Turn #29 showed that the attention-stream backend-tail drift is not explained by visible tensor metadata:

```text
pre-quant:
  generic op=DSV4_ROPE_TAIL
  backend op=DSV4_DECODE_COMPRESS
  dtype=f32
  shape=[512,1,1,1]
  stride=[4,2048,2048,2048]
  offset=0
  bytes=2048

quant:
  both op=DSV4_FP8_KV_QUANTIZE
  shape=[512,1,1,1]
  stride=[4,2048,2048,2048]
  offset=0
  bytes=2048

force_contiguous:
  did not fix the n400 attn-only drift
```

Turn #30 added a bounded non-consuming value/byte probe:

```text
flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE=1

scope:
  layer=2
  tokens=70..90
  stream=attn
  consume_path=disabled
  generic downstream consumed
  candidate_cache_side_effect=0
```

Attention-stream value probe:

```text
log=/tmp/dsv4_turn30_cupd3_attn_value_probe_t70_90.log

summary:
  emit_cases=50
  exact_cases=50
  non_exact_cases=0
  value_probe_cases=5
  dsv4_cupd3_backend_tail_consume=0
  consume_path=disabled

current probe limitation:
  pooled/norm/norm_weighted/rope labels all reference the exposed pre-quant row,
  because the current monolithic DSV4_DECODE_COMPRESS path does not expose those
  intermediate rows as independent tensors.
```

First exposed value-level mismatch:

```text
stage=exposed pre-quant row
generic op=DSV4_ROPE_TAIL
backend op=DSV4_DECODE_COMPRESS

token=71:
  max_abs=7.15255737e-07
  rms=9.31196329e-08
  first_bad_index=3
  byte_exact=0

token=79:
  max_abs=4.76837158e-07
  rms=1.00077727e-07
  first_bad_index=0
  generic_value=0.0953176394
  backend_value=0.095317632
  byte_exact=0
```

Downstream FP8 quant/cache handoff also differs for attention stream:

```text
token=79 quant/cache_handoff:
  value_exact=0
  byte_exact=0
  max_abs=4.76837158e-07
  rms=4.21748164e-08
  first_bad_index=448
```

Index-stream control:

```text
log=/tmp/dsv4_turn30_cupd3_index_value_probe_t70_78.log

token=71 index:
  exposed pre-quant row max_abs=4.76837158e-07
  exposed pre-quant row rms=1.05725344e-07
  quant value_exact=1
  quant byte_exact=1
  cache_handoff value_exact=1
  cache_handoff byte_exact=1

token=75 index:
  exposed pre-quant row max_abs=7.15255737e-07
  exposed pre-quant row rms=1.61541317e-07
  quant value_exact=1
  quant byte_exact=1
  cache_handoff value_exact=1
  cache_handoff byte_exact=1
```

Root-cause classification:

```text
first mismatch stage:
  exposed pre-quant DSV4_DECODE_COMPRESS output, before FP8 quant/cache handoff

not first cause:
  visible shape/stride/header/layout
  generic cache write side effect
  cache handoff as an independent layout problem

classification:
  attention-stream DSV4_DECODE_COMPRESS value arithmetic/materialization differs
  slightly from generic DSV4_ROPE_TAIL; FP8 quant/cache row then becomes non-exact.
  The index stream tolerates similar pre-quant float-level differences because its
  HFP4 quant/cache row remains byte-exact in the bounded probe.

trivial fix available:
  no
```

Decision:

```text
backend-tail consume eligible: no
backend-tail consume remains rejected: yes
continue CUPD3 backend-tail consume: no, until DSV4_DECODE_COMPRESS is split or made value-exact against the generic pool/norm/RoPE path before quant
next action:
  split DSV4_DECODE_COMPRESS into independent pool/norm/rope/quant probes, or
  make the backend-tail attention stream reproduce generic pre-quant materialization exactly before any consume retry
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
performance run: not run
cache side effects added: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #31 DSV4_DECODE_COMPRESS Internal Probe

Turn #30 localized the backend-tail consume drift to the attention-stream pre-quant value row from `DSV4_DECODE_COMPRESS`, with visible shape/stride/header/layout and cache handoff layout ruled out.

Turn #31 added diagnostic-only internal probe stages:

```text
flag:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_DECODE_COMPRESS_INTERNAL_PROBE=1

stages:
  pooled
  norm
  norm_weighted
  rope_in
  rope_out
  pre_quant

normal outputs changed:
  no
consume path:
  disabled
candidate_cache_side_effect:
  0
```

Attention-stream probe:

```text
log=/tmp/dsv4_turn31_cupd3_dcomp_internal_probe_attn_t70_90.log

first_non_exact_stage:
  pooled
first_non_exact_token:
  71
first_non_exact_index:
  3

token 71 pooled:
  max_abs=4.47034836e-08
  rms=7.08172351e-09
  exact=no

token 79 pooled:
  max_abs=4.47034836e-08
  rms=7.11245e-09
  exact=no
  first_bad_index=0
  generic_value=0.00753568672
  backend_value=0.00753568625

token 79 downstream:
  norm max_abs=7.15255737e-07 rms=1.06274822e-07 exact=no
  norm_weighted max_abs=4.76837158e-07 rms=9.09144604e-08 exact=no
  rope_in max_abs=4.76837158e-07 rms=9.09144604e-08 exact=no
  rope_out max_abs=4.76837158e-07 rms=9.18295277e-08 exact=no
  pre_quant max_abs=4.76837158e-07 rms=9.18295277e-08 exact=no
```

Index-stream control:

```text
log=/tmp/dsv4_turn31_cupd3_dcomp_internal_probe_index_t70_90.log

first_non_exact_stage:
  pooled

token 79 pooled:
  max_abs=2.98023224e-08
  rms=1.05882559e-08
  exact=no

control interpretation:
  index also shows tiny pre-quant/internal float differences, matching Turn #30.
  The important distinction remains that index quant/cache rows were byte-exact in
  the value probe, while attention FP8 quant/cache rows were not.
```

Root-cause classification:

```text
first mismatching internal stage:
  pooled

classification:
  pool softmax / weighted-pool order, dtype, or materialization differs inside
  the backend-tail DSV4_DECODE_COMPRESS path. RMSNorm, norm-weight multiply,
  RoPE, pre-quant, and FP8 quant/cache differences are downstream consequences,
  not the first observed mismatch.

trivial fix available:
  no

backend-tail consume eligible:
  no
backend-tail consume remains rejected:
  yes
```

Decision:

```text
continue CUPD3 backend-tail consume:
  no, until the attention-stream pool/weighted-pool stage is made value-exact
  against the generic path before quantization.

next action:
  compare pool softmax weights and weighted-sum accumulation order/dtype for the
  attention stream, or route the backend-tail attention stream through a
  generic-exact pooled-row materialization before retrying any consume canary.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
performance run: not run
cache side effects added: no
all-layer consume attempted: no
```

### 2026-05-14 Turn #32 Paired Metal Trace Measurement

Turn #32 added measurement-only Metal trace machinery for this repo and a first paired comparison against the local external DS4 checkout:

```text
ours:
  trace flag=LLAMA_FLASH_MOE_DSV4_TRACE=1
  token window flags:
    LLAMA_FLASH_MOE_DSV4_TRACE_TOKEN_MIN=32
    LLAMA_FLASH_MOE_DSV4_TRACE_TOKEN_MAX=64
  jsonl flag:
    LLAMA_FLASH_MOE_DSV4_TRACE_JSONL=/tmp/dsv4_ours_trace.jsonl

external DS4:
  checkout=/Users/anemll/SourceRelease/GITHUB/ML_playground/ds4
  trace flag=DS4_METAL_TRACE=1
  token window flags:
    DS4_METAL_TRACE_TOKEN_MIN=32
    DS4_METAL_TRACE_TOKEN_MAX=64
  jsonl flag:
    DS4_METAL_TRACE_JSONL=/tmp/dsv4_antirez_trace.jsonl
```

Trace implementation scope:

```text
ours:
  one JSONL row per ggml Metal node/encoder boundary with dispatch_count delta
  fields include token graph index, layer if name-derived, stage_bucket, kernel/op name,
  tensor name, dispatch index range, command-buffer slot, encoder slot, and dims

external DS4:
  one JSONL row per selected DS4 Metal function boundary
  includes routed MoE, Q8 matmul/QKV setup, attention output, compressor update,
  indexed attention, FP8 KV store, HC pre, and HC post calls

comparison caveat:
  ours is lower-level dispatch-boundary accounting.
  DS4 trace is high-level function-boundary accounting and is a lower-bound proxy
  for DS4 dispatch structure, not complete per-dispatch kernel telemetry.
```

Analyzer:

```text
script=scripts/dsv4_compare_metal_traces.py
comparison=/tmp/dsv4_trace_comparison.md

command:
  python3 scripts/dsv4_compare_metal_traces.py \
    --ours /tmp/dsv4_ours_trace.jsonl \
    --ds4 /tmp/dsv4_antirez_trace.jsonl \
    --out /tmp/dsv4_trace_comparison.md
```

Baseline speed:

```text
ours:
  log=/tmp/dsv4_turn32_ours_baseline_n400.log
  generation=21.9 tok/s
  metal_dispatch=1399039
  major regression=no; consistent with recent no-logit baseline band

external DS4:
  log=/tmp/dsv4_turn32_antirez_baseline_n400.log
  generation=36.49 tok/s
  note=DS4 CLI uses its own no-thinking prompt/runtime path, so this is a
       practical speed comparator rather than transcript-equivalent validation

ratio:
  DS4 / ours ~= 1.67x on this fixed short-context n400 run
```

Hot-neutral sanity:

```text
ours log=/tmp/dsv4_turn32_ours_hotneutral_n400.log
ours jsonl=/tmp/dsv4_turn32_ours_hotneutral_n400_logits.jsonl
metal_dispatch=1399039
trace flags off=yes
intrusive stage profiler off=yes
```

Trace window:

```text
token_min=32
token_max=64
note=ours token field is the Metal graph-compute index proxy; DS4 token field is
     generated-token index from its decode loop

ours JSONL=/tmp/dsv4_ours_trace.jsonl
DS4 JSONL=/tmp/dsv4_antirez_trace.jsonl
ours Instruments trace=/tmp/dsv4_ours_metal.trace
DS4 Instruments trace=/tmp/dsv4_antirez_metal.trace
```

Dispatch comparison from JSONL:

```text
ours:
  rows=26684
  tokens=33
  dispatches=27447
  dispatches/token=831.73
  command_buffers/token=1.76

DS4:
  rows=14850
  tokens=33
  function-boundary dispatch proxy=14850
  function-boundaries/token=450.00
  command_buffers/token=1.00 proxy/unknown

ratio:
  ours / DS4 proxy ~= 1.85x
```

Stage comparison:

```text
ffn:
  ours=9069
  DS4 proxy=1419
  ratio=6.39

attn_qkv:
  ours=538
  DS4 proxy=5709
  note=DS4 high-level Q8 matmul/QKV rows are not directly comparable to ours'
       lower-level node dispatch rows

attn_kv:
  ours=1455
  DS4 proxy=0

attn_compress:
  ours=496
  DS4 proxy=2046

attn_core:
  ours=858
  DS4 proxy=0

attn_out:
  ours=345
  DS4 proxy=1419

attn_hc_pre:
  ours=1043
  DS4 proxy=2838

attn_hc_post:
  ours=690
  DS4 proxy=0

kv_cache:
  ours=1379
  DS4 proxy=1419

other:
  ours=11482
  DS4 proxy=0
```

Top loss signals:

```text
#1 graph fragmentation / generic node overhead:
  ours has 831.73 traced dispatches/token and 1.76 command-buffer slots/token
  versus DS4's 450 high-level function boundaries/token. Because DS4 is traced
  at a coarser boundary, the true gap is at least graph/executor-shape visible
  and likely larger inside FFN/cache/tail stages.

#2 FFN executor shape:
  ours FFN bucket=9069 dispatches in the window, with many MUL_MAT,
  MUL_MAT_ID, CLAMP, ADD/MUL/DIV/GLU/GET_ROWS/SUM_ROWS fragments.
  DS4 exposes a routed-MoE whole-stage call boundary instead.

#3 generic uncategorized fragments:
  ours other bucket=11482, dominated by RMS_NORM, CONT, MUL_MAT, CPY,
  ARANGE, SET_ROWS, ADD, CONCAT, GET_ROWS, REPEAT, and UNARY.
  This indicates the remaining gap is not one local peephole; it is broad graph
  fragmentation around the decode executor.
```

Decision:

```text
primary loss source:
  graph fragmentation / decode executor shape, with routed MoE/FFN as the
  strongest concrete bucket from this first paired trace.

next recommended optimization:
  build a measurement-first routed MoE whole-stage executor/candidate, or add
  a more complete model-specific decode-layer executor trace that collapses
  FFN plus adjacent generic fragments before attempting another consume path.

performance path accepted:
  no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
new optimization path accepted: no
```

### 2026-05-13 Turn #33 Routed-MoE Trace Decomposition

Turn #33 refined the paired trace from Turn #32 for FFN/routed-MoE decomposition only. No MoE consume path was added and no performance path was accepted.

Trace fields added:

```text
ours:
  stage_bucket remains the coarse bucket
  stage adds FFN detail:
    ffn_norm
    router
    topk
    expert_gate_up
    expert_swiglu
    expert_down
    expert_weighted_sum
    shared_gate_up
    shared_swiglu
    shared_down
    ffn_residual
    ffn_other
  aliases/metadata:
    impl
    tensor
    shape
    dtype
    expert_count
    topk
    expert_id
    slot
    is_shared
    route_weight_applied

external DS4:
  ds4_gpu_routed_moe_one_tensor now reports:
    stage_bucket=ffn
    stage=ffn_moe_boundary
    topk=6
    expert_count=<n>
    uses_one_tensor=1
```

Trace files:

```text
ours JSONL=/tmp/dsv4_turn33_ours_trace.jsonl
ours log=/tmp/dsv4_turn33_ours_trace.log
DS4 JSONL=/tmp/dsv4_turn33_antirez_trace.jsonl
DS4 log=/tmp/dsv4_turn33_antirez_trace.log
comparison report=/tmp/dsv4_turn33_trace_comparison.md
token_min=32
token_max=64
```

Trace speed context:

```text
ours trace run generation=21.5 tok/s
DS4 trace run generation=36.43 tok/s
note=trace runs are measurement context only, not performance acceptance runs
```

Summary:

```text
ours:
  rows=26684
  tokens=33
  dispatches=27447
  dispatches/token=831.73
  command_buffers/token=1.76

DS4:
  rows=14850
  tokens=33
  boundary proxy/token=450.00
  command_buffers/token=1.00 proxy/unknown
```

FFN/MoE decomposition:

```text
substage                  ours rows/token  DS4 rows/token  ours rows  DS4 rows
ffn_norm                  0.00             0.00            0          0
router                    30.61            0.00            1010       0
topk                      9.70             0.00            320        0
expert_gate_up            64.73            0.00            2136       0
expert_swiglu             40.79            0.00            1346       0
expert_down               14.24            0.00            470        0
expert_weighted_sum       52.27            0.00            1725       0
shared_gate_up            0.00             0.00            0          0
shared_swiglu             0.00             0.00            0          0
shared_down               10.45            0.00            345        0
ffn_residual              31.12            0.00            1027       0
ffn_moe_boundary          0.00             43.00           0          1419
ffn_other                 10.45            0.00            345        0

coarse FFN:
  ours=8724
  DS4 routed-MoE boundary proxy=1419
  ratio=6.15x
```

Top ours FFN fragments:

```text
#1  expert_gate_up       MUL_MAT_ID  ffn_moe_gate              470  14.24/token
#2  expert_down          MUL_MAT_ID  ffn_moe_down              470  14.24/token
#3  expert_weighted_sum  GET_ROWS    ffn_moe_weights           345  10.45/token
#4  expert_weighted_sum  SUM_ROWS    ffn_moe_weights_sum       345  10.45/token
#5  expert_gate_up       MUL_MAT     ffn_gate                  345  10.45/token
#6  expert_gate_up       MUL_MAT     ffn_up                    345  10.45/token
#7  expert_weighted_sum  CLAMP       ffn_moe_weights_sum_clamped 345 10.45/token
#8  expert_swiglu        GLU         ffn_swiglu                345  10.45/token
#9  expert_weighted_sum  DIV         ffn_moe_weights_norm      345  10.45/token
#10 shared_down          MUL_MAT     ffn_shexp                 345  10.45/token
```

Generic-fragmentation summary:

```text
class           ours rows/token  ours rows
MUL_MAT         135.09           4458
MUL_MAT_ID      48.64            1605
RMS_NORM        55.30            1825
CPY             87.12            2875
CONT            69.55            2295
RESHAPE/VIEW    0.00             0
elementwise     187.33           6182
other           248.70           8207
```

Decision:

```text
recommended next exact shadow target:
  Option C: DSV4_ROUTED_MOE_ONE_TENSOR_SHADOW

reason:
  fragmentation is spread across router/topk, expert gate/up, SwiGLU,
  weighted-sum, shared expert, residual/HC-post-adjacent pieces, and generic
  elementwise/copy/cont nodes. A down-only or gate-up-only target would cover
  one visible piece but would not match the DS4 one-tensor boundary that is
  43 rows/token versus ours' 264.36 FFN rows/token.

implementation started:
  no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
new MoE consume path added: no
```

### 2026-05-13 Turn #34 Routed-MoE One-Tensor Shadow

Turn #34 started the routed-MoE whole-stage line as a shadow/probe path only. It did not add a consume path and did not enable the older FFN full-consume or DOWN_SUM6 paths.

Turn #33 recap:

```text
paired trace token window=32..64
ours FFN rows=8724
DS4 routed-MoE boundary rows=1419
ratio=6.15x

excess spread across:
  router
  topk
  expert_gate_up
  expert_swiglu
  expert_down
  expert_weighted_sum
  shared_down
  ffn_residual
  copy/cont/elementwise fragments
```

Boundary mapping:

```text
mapping note=/tmp/dsv4_turn34_routed_moe_boundary_mapping.txt

DS4 routed-MoE one-tensor boundary:
  inputs:
    ffn input / norm row
    selected expert ids
    route weights
    expert gate/up/down weights and model-map offsets
  output:
    routed MoE output tensor
  not owned:
    cache side effects
    HC post side effects

current repo graph:
  ffn_input
  ffn_moe_hash_topk / ffn_moe_argsort / ffn_moe_topk
  ffn_moe_weights / weights_sum / clamp / div / scale
  ffn_moe_gate / ffn_moe_up
  ffn_swiglu / ffn_moe_weighted_swiglu
  ffn_moe_down / ffn_moe_out
  ffn_shexp shared branch
  ffn_out

candidate boundary:
  FFN norm input
  -> router/top-k
  -> selected expert gate/up
  -> SwiGLU
  -> expert down
  -> route-weighted sum
  -> shared branch
  -> final FFN output
```

New flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_MODE=<generic_shadow|stage_probe|one_tensor_shadow>
```

Counters/report fields:

```text
dsv4_moe_shadow
dsv4_moe_shadow_exact
dsv4_moe_shadow_summary
```

Results:

```text
generic_shadow layer 0 n16:
  log=/tmp/dsv4_turn34_moe_generic_shadow_l0_n16.log
  cases=75
  exact_cases=75
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  consume_path=disabled

stage_probe layer 0 n16:
  log=/tmp/dsv4_turn34_moe_stage_probe_l0_n16.log
  cases=75
  exact_cases=75
  non_exact_cases=0
  tensors_found=topk,down,weighted_sum,shared,final
  limitation=router/gate_up/swiglu internals remain inside build_moe_ffn and are mapped by trace names, not separately probed tensors
  consume_path=disabled

one_tensor_shadow layer 0 n16:
  log=/tmp/dsv4_turn34_moe_one_tensor_shadow_l0_n16.log
  cases=75
  exact_cases=75
  non_exact_cases=0
  one_tensor_cases=15
  final_ffn max_abs=0
  final_ffn max_rms=0
  over_tol=0
  consume_path=disabled

one_tensor_shadow layer 0 n80:
  log=/tmp/dsv4_turn34_moe_one_tensor_shadow_l0_n80.log
  cases=395
  exact_cases=395
  non_exact_cases=0
  one_tensor_cases=79
  consume_path=disabled

one_tensor_shadow all-layer n80:
  log=/tmp/dsv4_turn34_moe_one_tensor_shadow_all_n80.log
  cases=13825
  exact_cases=13825
  non_exact_cases=0
  one_tensor_cases=3397
  consume_path=disabled
```

Dependency audit:

```text
generic_shadow/stage_probe:
  router_source=generic
  topk_source=generic
  gate_up_source=generic
  swiglu_source=generic
  down_source=generic
  weighted_sum_source=generic
  candidate_uses_generic_final_ffn=0
  candidate_kind=probe_only

one_tensor_shadow:
  router_source=candidate
  topk_source=candidate
  gate_up_source=candidate
  swiglu_source=candidate
  down_source=candidate
  weighted_sum_source=candidate
  candidate_uses_generic_final_ffn=0
  forbidden_inputs_seen=none
  candidate_kind=graph_level_one_tensor_shadow

status:
  true backend one-tensor kernel=no
  graph-level shadow=yes
```

n400 hot-neutral baseline with all MoE shadow flags off:

```text
log=/tmp/dsv4_turn34_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn34_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
no MoE shadow summary=yes
transcript stable=yes
```

Decision:

```text
one_tensor_shadow exact: yes, as a graph-level duplicated-arithmetic shadow
true one-tensor backend candidate: no
eligible next step:
  backend-oriented DSV4_ROUTED_MOE_ONE_TENSOR_SHADOW that exposes or owns
  router/topk/gate_up/swiglu/down/weighted/shared as one explicit op boundary,
  still shadow-only before any consume canary.
performance path accepted: no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #35 Routed-MoE Internal Probe Map

Turn #35 kept the routed-MoE line measurement/probe-only. It added internal probe reporting around the existing `build_moe_ffn` callback points and did not add a consume path.

Turn #34 recap:

```text
graph-level one_tensor_shadow:
  layer 0 n16 exact=75/75
  layer 0 n80 exact=395/395
  all-layer n80 exact=13825/13825
  candidate_uses_generic_final_ffn=0
  consume_path=disabled

limitation:
  true backend one-tensor executor=no
  router/gate_up/SwiGLU/down/shared internals still generic graph nodes
```

New flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_INTERNAL_PROBE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_INTERNAL_TRACE=1
```

Internal probe results:

```text
layer 0 n16:
  log=/tmp/dsv4_turn35_moe_internal_probe_l0_n16.log
  dsv4_moe_shadow exact=75/75
  dsv4_moe_internal_probe cases=375
  ffn_input=15
  router=30
  topk=30
  topk_weights=75
  expert_gate_proj=30
  expert_up_proj=30
  expert_swiglu=45
  expert_down=45
  expert_weighted_down=0
  shared_gate_proj=15
  shared_up_proj=15
  shared_swiglu=15
  shared_down=15
  final_ffn=15
  consume_path=disabled

layer 0 n80:
  log=/tmp/dsv4_turn35_moe_internal_probe_l0_n80.log
  dsv4_moe_shadow exact=395/395
  dsv4_moe_internal_probe cases=1975
  ffn_input=79
  router=158
  topk=158
  topk_weights=395
  expert_gate_proj=158
  expert_up_proj=158
  expert_swiglu=237
  expert_down=237
  expert_weighted_down=0
  shared_gate_proj=79
  shared_up_proj=79
  shared_swiglu=79
  shared_down=79
  final_ffn=79
  consume_path=disabled
```

Visible current-repo routed-MoE internals:

```text
ffn_input: ffn_norm
router_logits: ffn_moe_logits
topk_ids: ffn_moe_hash_topk / ffn_moe_topk
topk_weights: ffn_moe_weights / weights_sum / clamp / norm / scale
expert_gate_proj: ffn_moe_gate / ffn_moe_gate_clamped
expert_up_proj: ffn_moe_up / ffn_moe_up_clamped
expert_swiglu: ffn_moe_silu / ffn_moe_swiglu_limited / ffn_moe_weighted_swiglu
expert_down: ffn_moe_down / ffn_moe_out
shared_gate_proj: ffn_gate
shared_up_proj: ffn_up
shared_swiglu: ffn_swiglu
shared_down: ffn_shexp
final_ffn_output: ffn_out
```

Unavailable or not separately materialized:

```text
expert_partial_sum: not exposed as a separate stable callback tensor
shared_output: represented by shared_down/final add rather than a separate shared_output callback
residual_after_ffn: not separately exposed in this probe
true backend one-tensor source: unavailable
```

Dependency-source map:

```text
stage_probe:
  router_source=generic
  topk_source=generic
  gate_up_source=generic
  swiglu_source=generic
  down_source=generic
  weighted_sum_source=generic
  shared_source=generic
  final_ffn_source=generic
  candidate_kind=probe_only

turn #34 one_tensor_shadow:
  router_source=candidate
  topk_source=candidate
  gate_up_source=candidate
  swiglu_source=candidate
  down_source=candidate
  weighted_sum_source=candidate
  shared_source=candidate
  final_ffn_source=candidate
  candidate_kind=graph_level_one_tensor_shadow
```

n400 hot-neutral baseline with all MoE probe flags off:

```text
log=/tmp/dsv4_turn35_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn35_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
no MoE internal probe summary=yes
transcript stable=yes
```

Decision:

```text
recommended turn #36 exact shadow target:
  Option C: DSV4_ROUTED_MOE_ONE_TENSOR_BACKEND_SHADOW

reason:
  the probe confirms the fragmentation is spread across router/topk,
  selected expert gate/up/SwiGLU/down, route weighting, shared branch,
  and final add. Down-only or gate/up-only would leave most of the DS4-vs-ggml
  executor-shape gap intact.

implementation status:
  backend implementation not started
  consume path added=no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #36 Routed-MoE Backend Shadow Candidate

Turn #36 added a backend-shadow reporting path for the routed-MoE one-tensor boundary. It did not add a consume path and did not enable old FFN full-consume, DOWN_SUM6, weighted/shared SwiGLU consume, or tolerance acceptance.

Turn #35 recap:

```text
visible generic internals:
  ffn_input
  router_logits
  topk_ids
  topk_weights
  expert_gate_proj
  expert_up_proj
  expert_swiglu
  expert_down
  shared_gate_proj
  shared_up_proj
  shared_swiglu
  shared_down
  final_ffn_output

not separately stable:
  expert_partial_sum
  shared_output
  residual_after_ffn
  true backend one-tensor source

decision from turn #35:
  next target=DSV4_ROUTED_MOE_ONE_TENSOR_BACKEND_SHADOW
```

New backend-shadow flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_MODE=<generic_one_tensor_control|backend_candidate_shadow|ds4_shape_shadow>
```

New report fields:

```text
dsv4_moe_backend_shadow
dsv4_moe_backend_exact
dsv4_moe_backend_summary
dsv4_moe_backend_dep
```

`generic_one_tensor_control` result:

```text
log=/tmp/dsv4_turn36_moe_backend_generic_control_l0_n16.log
mode=generic_one_tensor_control
cases=15
exact_cases=15
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
backend_candidate_built=1
backend_op_dispatched=0
backend_owned=0
graph_arithmetic=1
consume_path=disabled
```

Dependency/ownership audit for `generic_one_tensor_control`:

```text
router_source=candidate
topk_source=candidate
gate_up_source=candidate
swiglu_source=candidate
down_source=candidate
weighted_sum_source=candidate
shared_source=candidate
final_ffn_source=candidate
backend_owned=0
forbidden_inputs_seen=none
missing_inputs=backend_op,expert_weight_layout,topk_route_pack,shared_branch_pack
```

`backend_candidate_shadow` result:

```text
log=/tmp/dsv4_turn36_moe_backend_candidate_l0_n16.log
mode=backend_candidate_shadow
cases=0
exact_cases=0
non_exact_cases=0
backend_candidate_built=0
backend_op_dispatched=0
backend_owned=0
infeasible_cases=15
infeasible_reason=missing_backend_one_tensor_op_requires_new_op_kernel_with_router_topk_gate_up_swiglu_down_weighted_shared_inputs
consume_path=disabled
```

Dependency/ownership audit for `backend_candidate_shadow`:

```text
router_source=unavailable
topk_source=unavailable
gate_up_source=unavailable
swiglu_source=unavailable
down_source=unavailable
weighted_sum_source=unavailable
shared_source=unavailable
final_ffn_source=unavailable
backend_owned=0
forbidden_inputs_seen=missing_backend_one_tensor_op_requires_new_op_kernel_with_router_topk_gate_up_swiglu_down_weighted_shared_inputs
missing_inputs=backend_op,expert_weight_layout,topk_route_pack,shared_branch_pack
```

n80/all-layer:

```text
backend_candidate_shadow n80 not run:
  reason=no exact backend-owned n16 candidate exists; mode reported precise infeasibility instead.
```

n400 hot-neutral baseline with all MoE backend-shadow flags off:

```text
log=/tmp/dsv4_turn36_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn36_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
transcript stable=yes
```

Decision:

```text
backend MoE shadow exact:
  exact only for graph-arithmetic generic_one_tensor_control

true backend-owned candidate:
  no

conclusion:
  a real routed-MoE one-tensor backend candidate requires a new op/kernel that owns
  router/topk, route packing, expert gate/up/SwiGLU/down, weighted sum, shared branch,
  and final output. Current graph callback/probe machinery can prove the boundary and
  exact graph arithmetic, but it cannot dispatch a DS4-style one-tensor backend op.

eligible for future consume:
  no, not until a true backend-owned exact shadow exists.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #37 Routed-MoE Backend Op Contract / Dry Run

Turn #37 converted the routed-MoE backend-shadow finding into a concrete op contract and model-graph dry-run. It did not add a consume path, did not run performance, and did not enable old FFN consume paths.

Turn #36 recap:

```text
generic_one_tensor_control:
  exact=yes
  backend_owned=0
  backend_op_dispatched=0

backend_candidate_shadow:
  infeasible
  reason=missing_backend_one_tensor_op_requires_new_op_kernel_with_router_topk_gate_up_swiglu_down_weighted_shared_inputs
```

Contract note:

```text
path=/tmp/dsv4_turn37_moe_backend_op_contract.txt
op_name=GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
ggml enum added=no
reason=no enum yet:
  V1 requires stable topk_weights/route weights as graph input, but those are
  still internal to build_moe_ffn. Adding an enum before exposing route weights
  would create an incomplete placeholder op.
```

V1 contract:

```text
scope:
  decode-only
  n_tokens == 1
  DeepSeek V4 Flash only
  moe_topk == 6
  Metal backend only

consumes:
  ffn input / normalized activation
  precomputed topk_ids
  precomputed topk_weights / route weights
  expert gate/up/down weights
  shared gate/up/down weights

owns:
  selected expert gate/up
  SwiGLU
  expert down
  route-weighted sum
  shared branch
  final FFN output

does not own:
  router logits
  top-k selection

output:
  final FFN output [4096,1,1,1]
```

V2 contract:

```text
owns router logits and top-k selection too.
```

New dry-run flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DRY_RUN=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX=<n>
```

Dry-run summary fields:

```text
dsv4_moe_backend_op
dsv4_moe_backend_op_dryrun
dsv4_moe_backend_op_eligible
dsv4_moe_backend_op_rejected
dsv4_moe_backend_op_summary
```

Layer 0 n16 dry-run:

```text
log=/tmp/dsv4_turn37_moe_backend_op_dryrun_l0_n16.log
cases=15
eligible_cases=0
rejected_cases=15
first_reject_reason=missing_topk_weights
consume_path=disabled

observed shapes:
  ffn_input=[4096,1,1,1]
  selected_expert_ids=[6,1,1,1]
  selected_expert_weights=missing
  gate_weight=iq2_xxs [4096,2048,256,1]
  up_weight=iq2_xxs [4096,2048,256,1]
  down_weight=q2_K [2048,4096,256,1]
  shared_gate=q8_0 [4096,2048,1,1]
  shared_up=q8_0 [4096,2048,1,1]
  shared_down=q8_0 [2048,4096,1,1]
  output=[4096,1,1,1]
```

Layer 0 n80 dry-run:

```text
log=/tmp/dsv4_turn37_moe_backend_op_dryrun_l0_n80.log
cases=79
eligible_cases=0
rejected_cases=79
first_reject_reason=missing_topk_weights
```

All-layer n80 dry-run:

```text
log=/tmp/dsv4_turn37_moe_backend_op_dryrun_all_n80.log
cases=3397
eligible_cases=0
rejected_cases=3397
first_reject_reason=missing_topk_weights
```

n400 hot-neutral baseline with backend-op flags off:

```text
log=/tmp/dsv4_turn37_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn37_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
transcript stable=yes
```

Decision:

```text
backend op contract ready:
  yes, V1/V2 split is defined.

implementation blocker:
  topk_ids are visible, but topk_weights/route weights are not available
  outside build_moe_ffn. V1 cannot be constructed as a real op input yet.

next implementation step:
  expose route weights/topk_weights as a stable graph tensor or split
  build_moe_ffn so the V1 backend op can consume selected ids + weights.
  After that, add the actual ggml enum/op/kernel and run shadow-only exactness.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #39 Routed-MoE Top-K ID Exposure

Turn #39 resolved the Turn #38 all-layer blocker by surfacing the selected expert id tensor from `build_moe_ffn`. It did not add a backend kernel, did not add a consume path, and did not enable old FFN consume paths.

Turn #38 recap:

```text
topk_weights:
  exposed=yes
  shape=[1,6,1,1]
  dtype=f32
  normalized=1

all-layer n80 dry-run:
  eligible_cases=237
  rejected_cases=3160
  first_reject_reason=missing_topk_ids
```

Top-k id mapping:

```text
note=/tmp/dsv4_turn39_moe_topk_ids_mapping.txt

hash layers:
  source=ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens)
  callback=ffn_moe_hash_topk

non-hash layers:
  source=ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)
  callback=ffn_moe_topk

reason layer 0 was already visible:
  layer 0 is in the hash-top-k path, so selected ids were built in
  src/models/deepseek4.cpp before build_moe_ffn.

reason most layers were missing:
  non-hash routed layers compute selected_experts inside build_moe_ffn.
```

New flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_EXPOSE_TOPK_IDS=1
```

New report fields:

```text
dsv4_moe_topk_ids_exposed
dsv4_moe_topk_ids_summary:
  enabled
  cases
  shape
  dtype
  source
  consumed_by_generic

dsv4_moe_backend_op_summary additions:
  topk_ids_visible
  topk_ids_shape
  topk_ids_dtype
```

Layer 0 n16 expose ids+weights + dry-run:

```text
log=/tmp/dsv4_turn39_moe_topk_ids_l0_n16.log
dsv4_moe_topk_ids_exposed=15
topk_ids_shape=[6,1,1,1]
topk_ids_dtype=i32
topk_ids_source=ffn_moe_hash_topk
dsv4_moe_topk_weights_exposed=15
topk_weights_shape=[1,6,1,1]
eligible_cases=15
rejected_cases=0
first_reject_reason=none
consume_path=disabled
```

Layer 0 n80 expose ids+weights + dry-run:

```text
log=/tmp/dsv4_turn39_moe_topk_ids_l0_n80.log
dsv4_moe_topk_ids_exposed=79
topk_ids_shape=[6,1,1,1]
topk_ids_dtype=i32
dsv4_moe_topk_weights_exposed=79
eligible_cases=79
rejected_cases=0
first_reject_reason=none
consume_path=disabled
```

All-layer n80 expose ids+weights + dry-run:

```text
log=/tmp/dsv4_turn39_moe_topk_ids_all_n80.log
dsv4_moe_topk_ids_exposed=3397
topk_ids_shape=[6,1,1,1]
topk_ids_dtype=i32
topk_ids_source=ffn_moe_topk
dsv4_moe_topk_weights_exposed=3397
topk_weights_shape=[1,6,1,1]
eligible_cases=3397
rejected_cases=0
first_reject_reason=none
```

Interpretation:

```text
missing_topk_ids is resolved.
V1 backend-op dry-run is eligible for all observed routed-MoE decode cases.
The next step can define/add the real backend op/kernel as shadow-only.
```

n400 hot-neutral baseline with expose/backend flags off:

```text
log=/tmp/dsv4_turn39_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn39_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
no topk/backend summary: yes
transcript stable: yes
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #40 Routed-MoE Backend Op Shadow

Turn #40 added the real ggml/Metal op plumbing for `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE`, but only as a shadow dry-run backend dispatch. It does not compute the final FFN tensor yet, does not consume the output, and does not enable any old FFN consume paths.

Turn #39 recap:

```text
topk_ids:
  exposed=yes
  shape=[6,1,1,1]
  dtype=i32

topk_weights:
  exposed=yes
  shape=[1,6,1,1]
  dtype=f32
  normalized=1

V1 dry-run eligibility:
  all-layer n80 eligible=3397
  rejected=0
```

Op enum/plumbing:

```text
added=yes
op=GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
cpu fallback=abort-only
Metal support=added
Metal kernel=kernel_dsv4_routed_moe_one_tensor_decode_dryrun
backend output computed=no
output_not_computed=1
consume_path=disabled
```

Backend op V1 scope:

```text
consumes:
  ffn_input
  topk_ids
  topk_weights
  expert gate/up/down weights
  shared gate/up/down weights

owns, contractually:
  selected expert gate/up
  SwiGLU
  expert down
  route-weighted sum
  shared branch
  final FFN output

does not own:
  router logits
  top-k selection
```

Layer 0 n16 backend-op shadow:

```text
log=/tmp/dsv4_turn40_moe_backend_op_shadow_l0_n16.log
dsv4_rmoe=15
backend_op_dispatched=1
internal_dispatch_count=1
eligible_cases=15
rejected_cases=0
exact_cases=0
output_not_computed=1
first_non_exact_tensor=final_ffn_output_not_computed
consume_path=disabled
prefix stable=yes
```

Layer 0 n80 backend-op shadow:

```text
log=/tmp/dsv4_turn40_moe_backend_op_shadow_l0_n80.log
dsv4_rmoe=79
backend_op_dispatched=1
internal_dispatch_count=1
eligible_cases=79
rejected_cases=0
exact_cases=0
output_not_computed=1
consume_path=disabled
prefix stable=yes
```

All-layer n80:

```text
run=no
reason=layer 0 backend shadow output is not computed, so exactness is not established
```

n400 hot-neutral baseline with backend-op flags off:

```text
log=/tmp/dsv4_turn40_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn40_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
no backend-op summary: yes
transcript stable: yes
```

Decision:

```text
backend op shadow exists: yes
backend op dispatched: yes
final_ffn exact: no, not computed yet
eligible for consume: no
next implementation step:
  implement numerical Metal body or internal sequence for V1:
    gate/up matvec for selected 6 experts
    SwiGLU
    route-weighted down projection/sum
    shared branch
    final FFN output
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #41 Routed-MoE Backend Numerical Body

Turn #41 attempted the first numerical body for `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE`. The backend op still dispatches in shadow mode and all V1 inputs are visible, but the first real numerical implementation is blocked by the required intermediate state contract.

Turn #40 recap:

```text
op=GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE
kernel=kernel_dsv4_routed_moe_one_tensor_decode_dryrun
backend dispatch active=yes
output_computed=no
consume_path=disabled
```

Numerical body scope attempted:

```text
target output:
  final_ffn_output [4096,1,1,1]

inputs available:
  ffn_input
  topk_ids [6,1,1,1] i32
  topk_weights [1,6,1,1] f32 normalized
  expert gate/up/down weights
  shared gate/up/down weights

supported expert gate/up:
  yes, existing routed stage supports iq2_xxs gate/up partials

supported expert down:
  yes, existing routed stage supports q2_K down partials

supported shared branch:
  no, q8_0 shared branch is not integrated into the new one-tensor output contract

blocker:
  requires_intermediate_scratch_for_iq2_xxs_gate_up_q2_K_down_plus_q8_0_shared_branch
```

The important implementation detail is that the existing partial routed-MoE Metal path uses an intermediate 24-slot scratch layout for gate/up/SwiGLU/down partials. The new op contract currently exposes only the final FFN output tensor, so it has nowhere exact and dependency-safe to store those internal partials while also integrating the q8_0 shared branch. Approximating or aliasing the generic final output was not accepted.

Layer 0 n16 numerical shadow:

```text
log=/tmp/dsv4_turn41_moe_backend_op_numeric_l0_n16.log
dsv4_rmoe=15
backend_op_dispatched=1
eligible_cases=15
rejected_cases=0
output_computed=0
output_not_computed=1
partial_output_only=0
internal_dispatch_count=1
supported_expert_gate_up=1
supported_expert_down=1
supported_shared_branch=0
unsupported_blocker=requires_intermediate_scratch_for_iq2_xxs_gate_up_q2_K_down_plus_q8_0_shared_branch
final_ffn exact=not tested, output not computed
consume_path=disabled
prefix stable=yes
```

Layer 0 n80:

```text
run=no
reason=n16 did not compute final_ffn_output, so exactness cannot be tested
```

n400 hot-neutral baseline with backend-op flags off:

```text
log=/tmp/dsv4_turn41_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn41_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
transcript stable=yes
```

Decision:

```text
numerical backend body exists: no
precise blocker identified: yes
backend op shadow dispatched: yes
eligible for consume: no
next action:
  extend the backend op contract with an explicit intermediate scratch/output strategy,
  or implement a true monolithic V1 kernel that computes gate/up, SwiGLU, down,
  weighted sum, q8_0 shared branch, and final output without exposing partials.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #42 Routed-MoE Backend Scratch / Gate-Up Substage

Turn #42 added explicit scratch-mode plumbing for `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE` and used it to compute the selected expert gate/up substage in shadow mode. It still does not compute or consume the final FFN output.

Turn #41 blocker:

```text
requires_intermediate_scratch_for_iq2_xxs_gate_up_q2_K_down_plus_q8_0_shared_branch
```

Scratch layout:

```text
topk=6
hidden_dim=4096
ffn_dim=2048
expert_count=256
gate_scratch_shape=[2048,6]
up_scratch_shape=[2048,6]
swiglu_scratch_shape=[2048,6]
down_scratch_shape=[4096,6]
streaming_down=0
routed_sum_shape=[4096]
shared_gate_shape=[2048]
shared_up_shape=[2048]
shared_swiglu_shape=[2048]
shared_down_shape=[4096]
final_output_shape=[4096]
scratch_bytes_estimate=393216
scratch_allocation_mode=ggml_tensor
```

Implementation:

```text
new flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SCRATCH=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SUBSTAGE=gate_up

op behavior:
  scratch mode output shape=[4096,24]
  slots 0..5: gate projection
  slots 6..11: up projection
  slots 12..17: SwiGLU scratch
  slots 18..23: reserved for down/partial output strategy

kernel:
  reuses kernel_dsv4_ffn_moe_stage_gate_up_iq2_xxs_f32 from the routed-MoE backend op
  computes IQ2_XXS selected expert gate/up projections
  also writes SwiGLU scratch
  final FFN output remains not computed
```

Layer 0 n16 gate/up substage:

```text
log=/tmp/dsv4_turn42_moe_backend_gate_up_l0_n16.log
dsv4_rmoe=15
cases=15
eligible_cases=15
rejected_cases=0
gate_up_compare_cases=180
exact_cases=180
non_exact_cases=0
gate max_abs=0
gate rms=0
gate over_tol=0
up max_abs=0
up rms=0
up over_tol=0
gate_up_substage_computed=1
swiglu_substage_computed=1
output_computed=0
output_not_computed=1
unsupported_blocker=final_ffn_output_not_computed_gate_up_scratch_only
consume_path=disabled
prefix stable=yes
```

Layer 0 n80 gate/up substage:

```text
log=/tmp/dsv4_turn42_moe_backend_gate_up_l0_n80.log
dsv4_rmoe=79
cases=79
eligible_cases=79
rejected_cases=0
gate_up_compare_cases=948
exact_cases=948
non_exact_cases=0
gate_up_substage_computed=1
swiglu_substage_computed=1
output_computed=0
output_not_computed=1
consume_path=disabled
prefix stable=yes
```

n400 hot-neutral baseline with backend flags off:

```text
log=/tmp/dsv4_turn42_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn42_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
transcript stable=yes
```

Decision:

```text
gate/up backend substage implemented: yes
gate/up exact through layer 0 n80: yes
final FFN output computed: no
eligible for consume: no
next action:
  extend scratch mode to down/weighted-sum using slots 12..23,
  then integrate q8_0 shared branch before any final-output compare.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #43 Routed-MoE Backend SwiGLU Substage

Turn #43 extended `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE` scratch mode from exact selected-expert gate/up projections to exact selected-expert SwiGLU scratch. It remains shadow-only: final FFN output is not computed, the candidate is not consumed, and no performance path was tested.

Turn #42 recap:

```text
gate/up backend substage implemented: yes
gate/up exact:
  n16: 180/180
  n80: 948/948
scratch output shape=[4096,24]
slots 0..5: gate projection
slots 6..11: up projection
slots 12..17: SwiGLU
slots 18..23: reserved/down strategy
consume_path=disabled
```

SwiGLU implementation:

```text
formula=silu(gate) * up
dtype=f32
materialization=slots_12_17
layout=[2048,6]
kernel=kernel_dsv4_ffn_moe_stage_gate_up_iq2_xxs_f32
final_output_computed=0
```

Layer 0 n16 SwiGLU substage:

```text
log=/tmp/dsv4_turn43_moe_backend_swiglu_l0_n16.log
dsv4_rmoe=15
cases=15
eligible_cases=15
rejected_cases=0
exact_cases=270
non_exact_cases=0
gate_up_compare_cases=180
swiglu_compare_cases=90
gate max_abs=0
gate rms=0
gate over_tol=0
up max_abs=0
up rms=0
up over_tol=0
swiglu max_abs=0
swiglu rms=0
swiglu over_tol=0
gate_up_substage_computed=1
swiglu_substage_computed=1
gate_up_substage_exact=1
swiglu_exact=1
output_computed=0
unsupported_blocker=final_ffn_output_not_computed_gate_up_scratch_only
consume_path=disabled
prefix stable=yes
```

Layer 0 n80 SwiGLU substage:

```text
log=/tmp/dsv4_turn43_moe_backend_swiglu_l0_n80.log
dsv4_rmoe=79
cases=79
eligible_cases=79
rejected_cases=0
exact_cases=1422
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=948
swiglu_compare_cases=474
gate_up_substage_computed=1
swiglu_substage_computed=1
gate_up_substage_exact=1
swiglu_exact=1
output_computed=0
consume_path=disabled
prefix stable=yes
```

n400 hot-neutral baseline with backend flags off:

```text
log=/tmp/dsv4_turn43_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn43_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
transcript stable=yes
```

Remaining blocker:

```text
down / weighted sum / q8_0 shared branch / final FFN output are not computed yet.
```

Decision:

```text
swiglu backend substage implemented: yes
swiglu exact through layer 0 n80: yes
final FFN output computed: no
eligible for consume: no
next action:
  implement q2_K expert down scratch/streaming strategy and route-weighted sum compare,
  then add q8_0 shared branch before any final-output or consume canary.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #44 Routed-MoE Backend Down / Weighted Sum Substage

Turn #44 extended `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE` scratch mode from exact gate/up plus SwiGLU to exact q2_K selected-expert down and routed weighted partial sums. It remains shadow-only: the shared branch and final FFN output are not computed, the candidate is not consumed, and no performance path was tested.

Turn #43 recap:

```text
gate/up exact through layer 0 n80: yes
SwiGLU exact through layer 0 n80: yes
final FFN output computed: no
consume_path=disabled
```

Down implementation:

```text
q2_K support: yes
kernel=kernel_dsv4_routed_moe_down_slots_q2_K_f32
internal_dispatch_count=2
scratch output shape=[4096,30]
slots 0..5: gate projection
slots 6..11: up projection
slots 12..17: SwiGLU
slots 18..23: per-slot weighted expert down
slots 24..29: cumulative routed partial sums
slot 29: routed sum
route weight order=route_weight_before_down
sum order=left_associative_slots_0_to_5
shared_computed=0
final_output_computed=0
```

Layer 0 n16 down / routed sum substage:

```text
log=/tmp/dsv4_turn44_moe_backend_down_l0_n16.log
dsv4_rmoe=15
cases=15
eligible_cases=15
rejected_cases=0
exact_cases=450
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=180
swiglu_compare_cases=90
down_compare_cases=90
routed_sum_compare_cases=90
gate_up_substage_computed=1
swiglu_substage_computed=1
down_computed=1
routed_sum_computed=1
gate_up_substage_exact=1
swiglu_exact=1
down_exact=1
routed_sum_exact=1
unsupported_blocker=final_ffn_output_not_computed_shared_branch_missing
consume_path=disabled
prefix stable=yes
```

Layer 0 n80 down / routed sum substage:

```text
log=/tmp/dsv4_turn44_moe_backend_down_l0_n80.log
dsv4_rmoe=79
cases=79
eligible_cases=79
rejected_cases=0
exact_cases=2370
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=948
swiglu_compare_cases=474
down_compare_cases=474
routed_sum_compare_cases=474
down_scratch_strategy=slots_18_23_partials_24_29
scratch_bytes_estimate=491520
output_computed=0
consume_path=disabled
prefix stable=yes
```

n400 hot-neutral baseline with backend flags off:

```text
log=/tmp/dsv4_turn44_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn44_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
transcript stable=yes
```

Remaining blocker:

```text
q8_0 shared branch / final FFN output are not computed yet.
```

Decision:

```text
down backend substage implemented: yes
weighted routed sum exact through layer 0 n80: yes
final FFN output computed: no
eligible for consume: no
next action:
  implement q8_0 shared gate/up, shared SwiGLU, shared down, then compare final routed+shared FFN output.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #45 Routed-MoE Backend Shared Branch / Final Output

Turn #45 extended `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE` scratch mode from the exact routed expert path to the q8_0 shared branch and final FFN output. The path remains shadow-only: the candidate output is not consumed, no no-logit performance was run, and no old FFN consume paths were enabled.

Turn #44 recap:

```text
selected expert gate/up exact through layer 0 n80: yes
selected expert SwiGLU exact through layer 0 n80: yes
expert down q2_K exact through layer 0 n80: yes
route-weighted routed sum exact through layer 0 n80: yes
final FFN output computed: no
consume_path=disabled
```

Shared branch implementation:

```text
q8_0 support: yes
new kernels:
  kernel_dsv4_routed_moe_shared_gate_up_swiglu_q8_0_f32
  kernel_dsv4_routed_moe_shared_down_final_q8_0_f32
internal_dispatch_count=4
scratch output shape=[4096,35]
slots 0..5: selected expert gate projection
slots 6..11: selected expert up projection
slots 12..17: selected expert SwiGLU
slots 18..23: per-slot weighted expert down
slots 24..29: cumulative routed partial sums
slot 29: routed sum
slot 30: shared gate projection
slot 31: shared up projection
slot 32: shared SwiGLU
slot 33: shared down
slot 34: final_ffn_output = routed_sum + shared_down
scratch_bytes_estimate=573440
consume_path=disabled
```

Layer 0 n16 shared / final output:

```text
log=/tmp/dsv4_turn45_moe_backend_shared_l0_n16.log
dsv4_rmoe=15
cases=15
eligible_cases=15
rejected_cases=0
exact_cases=525
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=180
swiglu_compare_cases=90
down_compare_cases=90
routed_sum_compare_cases=90
shared_compare_cases=60
final_output_compare_cases=15
shared_exact=1
final_output_exact=1
output_computed=1
unsupported_blocker=none
consume_path=disabled
prefix stable=yes
```

Layer 0 n80 shared / final output:

```text
log=/tmp/dsv4_turn45_moe_backend_shared_l0_n80.log
dsv4_rmoe=79
cases=79
eligible_cases=79
rejected_cases=0
exact_cases=2765
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=948
swiglu_compare_cases=474
down_compare_cases=474
routed_sum_compare_cases=474
shared_compare_cases=316
final_output_compare_cases=79
shared_exact=1
final_output_exact=1
output_computed=1
unsupported_blocker=none
consume_path=disabled
prefix stable=yes
```

All-layer n80 shared / final output:

```text
log=/tmp/dsv4_turn45_moe_backend_shared_all_n80.log
dsv4_rmoe=3397
cases=3397
eligible_cases=3397
rejected_cases=0
exact_cases=118895
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
gate_up_compare_cases=40764
swiglu_compare_cases=20382
down_compare_cases=20382
routed_sum_compare_cases=20382
shared_compare_cases=13588
final_output_compare_cases=3397
shared_exact=1
final_output_exact=1
output_computed=1
unsupported_blocker=none
consume_path=disabled
first failing layer/token/tensor=none
```

n400 hot-neutral baseline with backend flags off:

```text
log=/tmp/dsv4_turn45_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn45_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_rmoe=0
transcript stable=yes
```

Decision:

```text
full V1 backend shadow output computed: yes
exact through all-layer n80: yes
eligible for future consume: not yet; next turn should run n400 hot-neutral shadow/under-test or a single-layer consume canary only under transcript-exact policy
performance path accepted: no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #46 Routed-MoE Backend n400 Shadow Validation

Turn #46 validated the complete V1 routed-MoE backend shadow through n400 shadow conditions. The backend output is still not consumed; generic downstream remains the active path.

Turn #45 recap:

```text
full V1 backend shadow final_ffn_output computed: yes
layer 0 n16 exact: 525/525
layer 0 n80 exact: 2765/2765
all-layer n80 exact: 118895/118895
dsv4_rmoe all-layer n80: 3397
consume_path=disabled
```

Summary metadata added:

```text
dsv4_moe_backend_op_summary now reports:
  shadow_only
  compared_cases
  dsv4_rmoe
  backend_op_dispatched_count
  consume_guard_requested
  consume_guard_allowed=0
  consume_guard_reason
```

Future consume guardrails were added as reporting-only scaffolding:

```text
planned flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME=1
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX=<n>
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_ABORT_ON_MISMATCH=1

guard result in Turn #46:
  consume_guard_requested=0
  consume_guard_allowed=0
  consume_guard_reason=not_requested
  consume_path=disabled
```

n400 layer 0 shadow:

```text
log=/tmp/dsv4_turn46_moe_backend_shadow_l0_n400.log
layer_filter=0
token_min=1
token_max=399
dsv4_rmoe=399
cases=399
eligible_cases=399
rejected_cases=0
exact_cases=13965
compared_cases=13965
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
backend_op_dispatched_count=399
output_computed=1
final_output_exact=1
generic downstream consumed=yes
consume_path=disabled
transcript completed=yes
```

n400 all-layer shadow:

```text
log=/tmp/dsv4_turn46_moe_backend_shadow_all_n400.log
layer_filter=all
dsv4_rmoe=17157
cases=17157
eligible_cases=17157
rejected_cases=0
exact_cases=600495
compared_cases=600495
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
backend_op_dispatched_count=17157
output_computed=1
final_output_exact=1
generic downstream consumed=yes
consume_path=disabled
transcript completed=yes
```

Hot-neutral baseline with backend flags off:

```text
log=/tmp/dsv4_turn46_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn46_baseline_n400_hotneutral.jsonl
jsonl_records=401
metal_dispatch=1399039
dsv4_rmoe=0
backend summaries present=no
transcript stable=yes
```

Decision:

```text
n400 shadow ready: yes
eligible for future single-layer consume canary: yes, only as a guarded transcript-exact under-test
consume run in Turn #46: no
performance run in Turn #46: no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #47 Routed-MoE Backend Single-Layer Consume Canary

Turn #47 implemented a strictly guarded single-layer consume canary for `GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE`. The canary consumes only layer 0 backend final FFN output and leaves all other layers on the generic FFN path. No all-layer consume and no no-logit performance were run.

Turn #46 recap:

```text
layer 0 n400 shadow exact: yes, 13965/13965
all-layer n400 shadow exact: yes, 600495/600495
full V1 backend final_ffn_output computed: yes
consume_path=disabled
```

New consume flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_ABORT_ON_MISMATCH=1
```

Guard result:

```text
missing-layer log=/tmp/dsv4_turn47_rmoe_guard_missing_layer.log
allowed=0
reason=missing_layer
dsv4_rmoe_consume=0
backend_op_dispatched_count=0
all_layer_blocked=1
rejected_paths_active=0
```

Layer 0 n16 intrusive compare + consume:

```text
log=/tmp/dsv4_turn47_rmoe_consume_l0_n16.log
dsv4_rmoe=15
dsv4_rmoe_consume=15
compared_cases=525
exact_cases=525
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
backend_output_computed=1
backend_output_consumed=1
generic_ffn_built=1
backend_ffn_built=1
consume_path=single_layer
```

Layer 0 n80 intrusive compare + consume:

```text
log=/tmp/dsv4_turn47_rmoe_consume_l0_n80.log
dsv4_rmoe=79
dsv4_rmoe_consume=79
compared_cases=2765
exact_cases=2765
non_exact_cases=0
max_abs=0
max_rms=0
over_tol=0
backend_output_computed=1
backend_output_consumed=1
generic_ffn_built=1
backend_ffn_built=1
consume_path=single_layer
```

n400 hot-neutral consume validation:

```text
baseline log=/tmp/dsv4_turn47_rmoe_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn47_rmoe_baseline_n400_logits.jsonl
consume log=/tmp/dsv4_turn47_rmoe_consume_l0_n400.log
consume jsonl=/tmp/dsv4_turn47_rmoe_consume_l0_n400_logits.jsonl
first divergence=/tmp/dsv4_turn47_rmoe_first_divergence_l0_n400.txt

baseline records=400
consume records=325
baseline metal_dispatch=1399039
consume metal_dispatch=1137982
dsv4_rmoe_consume=324
path_accepted=false
```

First divergence:

```text
first divergent token index=2
position=12
baseline token=" an"
consume token=" a"
baseline top1 logit=43.124855
consume top1 logit=33.7358818
max_abs_logit_err=10.101223
rms_logit_err=4.13287799
top20 overlap=12/20
classification=Case B real numerical bug signal
```

Decision:

```text
single-layer consume canary accepted: no
reason: n400 hot-neutral transcript/logit drift
performance run: no
all-layer consume run: no
next action: root-cause consumed backend final_ffn_output semantics versus generic downstream; likely the shadow compare path is not sufficient for replacement semantics
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path accepted: no
```

### 2026-05-13 Turn #38 Routed-MoE Top-K Weight Exposure

Turn #38 resolved the Turn #37 `missing_topk_weights` blocker for the hash-top-k routed-MoE decode path. It did not add a backend kernel, did not add a consume path, and did not enable old FFN consume paths.

Route-weight mapping:

```text
note=/tmp/dsv4_turn38_moe_topk_weights_mapping.txt

source:
  llm_graph_context::build_moe_ffn

route-weight flow:
  weights = ggml_get_rows(ctx0, probs, selected_experts)
  callback: ffn_moe_weights

  if norm_w:
    weights_sum = ggml_sum_rows(weights)
    weights_sum = ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)
    weights = ggml_div(weights, weights_sum)
    callback: ffn_moe_weights_norm

  if expert_weights_scale applies:
    weights = ggml_scale(weights, w_scale)
    callback: ffn_moe_weights_scaled

exposed tensor:
  callback/name=dsv4_moe_topk_weights
  shape=[1,6,1,1]
  dtype=f32
  source=ffn_moe_weights_scaled
  normalized=1
  consumed_by_generic=1
```

New flag:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_EXPOSE_TOPK_WEIGHTS=1
```

New report fields:

```text
dsv4_moe_topk_weights_exposed
dsv4_moe_topk_weights_summary:
  enabled
  cases
  shape
  dtype
  source
  normalized
  consumed_by_generic

dsv4_moe_backend_op_summary additions:
  topk_weights_visible
  topk_weights_shape
```

Layer 0 n16 expose + dry-run:

```text
log=/tmp/dsv4_turn38_moe_topk_weights_l0_n16.log
dsv4_moe_topk_weights_exposed=15
shape=[1,6,1,1]
dtype=f32
eligible_cases=15
rejected_cases=0
first_reject_reason=none
topk_weights_visible=1
consume_path=disabled
```

Layer 0 n80 expose + dry-run:

```text
log=/tmp/dsv4_turn38_moe_topk_weights_l0_n80.log
dsv4_moe_topk_weights_exposed=79
shape=[1,6,1,1]
dtype=f32
eligible_cases=79
rejected_cases=0
first_reject_reason=none
topk_weights_visible=1
consume_path=disabled
```

All-layer n80 expose + dry-run:

```text
log=/tmp/dsv4_turn38_moe_topk_weights_all_n80.log
dsv4_moe_topk_weights_exposed=3397
shape=[1,6,1,1]
dtype=f32
eligible_cases=237
rejected_cases=3160
first_reject_reason=missing_topk_ids
topk_weights_visible=1
```

Interpretation:

```text
missing_topk_weights is resolved.

V1 backend-op dry-run is now eligible for layers/tokens that already expose
stable selected expert ids. The all-layer dry-run moves to the next concrete
blocker: missing_topk_ids on non-hash layers.
```

n400 hot-neutral baseline with expose/backend flags off:

```text
log=/tmp/dsv4_turn38_baseline_n400_hotneutral.log
jsonl=/tmp/dsv4_turn38_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
no topk/backend summary: yes
transcript stable: yes
```

Decision:

```text
backend op V1 unblocked for hash-top-k routed-MoE layers: yes
backend op V1 unblocked for all routed-MoE layers: no
new blocker: missing_topk_ids on non-hash layers
next action:
  either surface stable topk_ids for the remaining routed-MoE layers, or move V2
  forward so the backend op owns router/top-k selection.
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
```

### 2026-05-13 Turn #27 CUPD3 Backend Tail Consume Canary

Turn #26 established a shadow-only backend-tail candidate for:

```text
pool softmax -> weighted pool -> RMSNorm -> norm weight -> RoPE -> FP8/HFP4 quant row
```

Turn #27 tested whether that candidate can replace the generic tail for one layer while preserving generic projections and the existing generic cache write path.

Patch / mode:

```text
new consume flags:
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_MODE=pool_norm_rope_quant_generic_cache
  LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ABORT_ON_MISMATCH

under-test:
  cupd3_backend_tail_single_layer_consume

scope:
  layer=2 only
  projection_source=generic
  cache_mutation_mode=generic_existing_write
  candidate_cache_side_effect=0
  no all-layer consume
```

Guard result:

```text
missing-layer log=/tmp/dsv4_turn27_cupd3_backend_tail_guard_missing_layer.log
allowed=0
reason=missing_layer
dsv4_cupd3_backend_tail_consume=0

rejected-path log=/tmp/dsv4_turn27_cupd3_backend_tail_guard_rejected_path.log
allowed=0
reason=rejected_paths_enabled:LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6
rejected_paths_active=1
dsv4_cupd3_backend_tail_consume=0
```

Short exactness:

```text
n16 log=/tmp/dsv4_turn27_cupd3_backend_tail_consume_l2_n16.log
  emit_cases=4
  exact_cases=4
  non_exact_cases=0
  dsv4_dcomp=8
  dsv4_cupd3_backend_tail_consume=4
  generic_tail_built=0
  backend_tail_built=1
  backend_tail_consumed=1
  generic_cache_write_built=1
  candidate_cache_side_effect=0
  cache_mutation_mode=generic_existing_write
  max_abs=0
  max_rms=0
  over_tol=0

n80 log=/tmp/dsv4_turn27_cupd3_backend_tail_consume_l2_n80.log
  emit_cases=36
  exact_cases=36
  non_exact_cases=0
  dsv4_dcomp=40
  dsv4_cupd3_backend_tail_consume=36
  generic_tail_built=0
  backend_tail_built=1
  backend_tail_consumed=1
  generic_cache_write_built=1
  candidate_cache_side_effect=0
  cache_mutation_mode=generic_existing_write
  max_abs=0
  max_rms=0
  over_tol=0
```

n400 hot-path-neutral:

```text
baseline log=/tmp/dsv4_turn27_cupd3_backend_tail_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn27_cupd3_backend_tail_baseline_n400_logits.jsonl
consume log=/tmp/dsv4_turn27_cupd3_backend_tail_consume_l2_n400.log
consume jsonl=/tmp/dsv4_turn27_cupd3_backend_tail_consume_l2_n400_logits.jsonl
first divergence=/tmp/dsv4_turn27_cupd3_backend_tail_first_divergence_l2_n400.txt

baseline metal_dispatch=1399039
consume metal_dispatch=1398235
observed dispatch delta=-804
consume dsv4_dcomp=200
summary emit_cases=196
summary dsv4_cupd3_backend_tail_consume=196
generic_tail_built=0
backend_tail_consumed=1
generic_cache_write_built=1
candidate_cache_side_effect=0
cache_mutation_mode=generic_existing_write
path_accepted=false

divergence:
  first divergent token index=79
  position=89
  baseline token="'s"
  consume token=" has"
  max_abs_logit_err=1.28058624 over overlapping top20
  rms_logit_err=0.525356713 over overlapping top20
  top20_overlap=18/20
```

Decision:

```text
backend-tail consume exact/cache-safe in short intrusive n16/n80: yes
n400 transcript/logit exact: no
dispatch-collapsing: yes, but rejected because n400 drifts
performance run: skipped
performance path accepted: no
next action: root-cause graph/cache dependency before any further CUPD3 backend-tail consume or scaling
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
new cache side effects added: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #28 CUPD3 Backend Tail Drift Root Cause

Turn #27 recap:

```text
CUPD3 backend-tail consume:
  n16/n80 short compare: exact
  n400 hot-neutral: drift
  first divergent token index=79
  position=89
  baseline token="'s"
  consume token=" has"
  max_abs_logit_err=1.28058624
  rms_logit_err=0.525356713
  dispatch delta=-804
  path_accepted=false
```

New diagnostic controls:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DRIFT_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DEP_BARRIER=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_EMIT_ONLY=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_STREAM=attn|index|all
```

Shadow/no-consume window:

```text
log=/tmp/dsv4_turn28_cupd3_backend_tail_shadow_t70_90.log
mode=backend-tail shadow compare, no consume
token window=70..90
emit_cases=10
exact_cases=10
attn_emits=5
index_emits=5
generic_tail_built=1
backend_tail_consumed=0
drift_trace_cases=10
shadow no-consume exact=yes
generic downstream consumed=yes
```

Same-build baseline:

```text
log=/tmp/dsv4_turn28_cupd3_backend_tail_baseline_n400.log
jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_baseline_n400_logits.jsonl
metal_dispatch=1399039
dsv4_dcomp=0
```

Consume trace:

```text
log=/tmp/dsv4_turn28_cupd3_backend_tail_consume_l2_n400_trace_t70_90.log
jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_consume_l2_n400_logits.jsonl
first divergence=/tmp/dsv4_turn28_cupd3_backend_tail_consume_first_divergence.txt
trace window=70..90
traced emits=10
first divergent token index=79
position=89
max_abs_logit_err=1.28058624
rms_logit_err=0.525356713
metal_dispatch=1398235
dsv4_dcomp=200
cache_mutation_mode=generic_existing_write
candidate_cache_side_effect=0
```

Dependency barrier:

```text
log=/tmp/dsv4_turn28_cupd3_backend_tail_barrier_l2_n400.log
jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_barrier_l2_n400_logits.jsonl
first divergence=/tmp/dsv4_turn28_cupd3_backend_tail_barrier_first_divergence.txt
dep_barrier=1
first divergent token index=79
position=89
max_abs_logit_err=1.28058624
rms_logit_err=0.525356713
result=fails the same way

conclusion:
  not fixed by explicit quant-row -> cache-write dependency edge
  simple graph/cache scheduling barrier is not sufficient
```

Emit-only:

```text
log=/tmp/dsv4_turn28_cupd3_backend_tail_emit_only_l2_n400.log
jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_emit_only_l2_n400_logits.jsonl
first divergence=/tmp/dsv4_turn28_cupd3_backend_tail_emit_only_first_divergence.txt
emit_only=1
first divergent token index=79
position=89
max_abs_logit_err=1.28058624
rms_logit_err=0.525356713
result=fails the same way

conclusion:
  non-emit state/update activation is not the cause
```

Stream isolation:

```text
attn-only:
  log=/tmp/dsv4_turn28_cupd3_backend_tail_attn_only_l2_n400.log
  jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_attn_only_l2_n400_logits.jsonl
  first divergence=/tmp/dsv4_turn28_cupd3_backend_tail_attn_only_first_divergence.txt
  dsv4_cupd3_backend_tail_consume=98
  first divergent token index=79
  max_abs_logit_err=1.28058624
  rms_logit_err=0.525356713
  result=reproduces full drift

index-only:
  log=/tmp/dsv4_turn28_cupd3_backend_tail_index_only_l2_n400.log
  jsonl=/tmp/dsv4_turn28_cupd3_backend_tail_index_only_l2_n400_logits.jsonl
  first divergence=/tmp/dsv4_turn28_cupd3_backend_tail_index_only_first_divergence.txt
  dsv4_cupd3_backend_tail_consume=98
  first divergent token index=none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  result=exact
```

Root-cause classification:

```text
backend-tail arithmetic mismatch: likely, specifically attention compressed-KV emitted row
cache handoff ordering: unlikely; dependency barrier did not fix drift
non-emit state update consumption: no; emit-only still drifted
index stream: no; index-only exact
attn stream: yes; attn-only reproduces full drift
cache side effects: none added
```

Decision:

```text
backend-tail consume eligible: no
performance run: no
performance path accepted: no
continue by root-causing attn stream backend-tail numeric/layout mismatch before any consume/scaling
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
new cache side effects added: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #25 Compressor/Update V3 Tail Attribution

Turn #25 preserved the Turn #24 result:

```text
V3 tail consume:
  exact/cache-safe: yes
  projection_source: generic
  cache writes: existing generic write path
  metal_dispatch delta: 0
  no-logit speed: baseline avg 22.2 tok/s, consume avg 21.9 tok/s
  performance path accepted: no

turn #25 action:
  no consume path added
  no cache side-effect op added
  no V3 projection ownership added
  no all-layer scaling attempted
```

New attribution flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX=<n>
```

Attribution scope:

```text
layer/token filtered decode emits only
projection_source=generic
cache_mutation=disabled
candidate_cache_side_effect=0
consume_path=disabled

counted tail pieces per compressor emit:
  pool_softmax
  weighted_pool
  rms_norm
  norm_weight
  rope
  quant
  cache_handoff

estimated dispatch-class nodes:
  10 per emit
  softmax + mul + sum_rows + reshape + rms_norm + norm_weight_mul + reshape + rope + quant + cache_handoff

observed per-op dispatch isolation:
  unavailable from current counters
```

Guard result:

```text
missing layer log: /tmp/dsv4_turn25_cupd3_tail_attrib_guard_missing_layer.log
result:
  dsv4_cupd3_tail_attrib_guard:
    allowed=0
    reason=missing_layer
    selected_layer=-1
    token_min=-1
    token_max=-1
    cache_side_effect=0
    consume_path=disabled
```

Layer 2 n16 attribution:

```text
log: /tmp/dsv4_turn25_cupd3_tail_attrib_l2_n16.log
guard:
  allowed=1
  reason=allowed
summary:
  emit_count=4
  attn_emits=2
  index_emits=2
  first_layer=2
  first_token=11
  pool_softmax=4
  weighted_pool=4
  rms_norm=4
  norm_weight=4
  rope=4
  quant=4
  cache_handoff=4
  estimated_tail_dispatch_nodes=40
  projection_source=generic
  cache_mutation=disabled
  candidate_cache_side_effect=0
  consume_path=disabled
prefix stable: yes
```

Layer 2 n80 attribution:

```text
log: /tmp/dsv4_turn25_cupd3_tail_attrib_l2_n80.log
summary:
  emit_count=40
  attn_emits=20
  index_emits=20
  first_layer=2
  first_token=11
  pool_softmax=40
  weighted_pool=40
  rms_norm=40
  norm_weight=40
  rope=40
  quant=40
  cache_handoff=40
  estimated_tail_dispatch_nodes=400
  observed_dispatch_contribution=unknown
  per_op_dispatch_isolation=unavailable
  next_plausible_boundary=pool_norm_rope_quant_tail
prefix stable: yes
```

n400 hot-neutral attribution:

```text
baseline log: /tmp/dsv4_turn25_cupd3_baseline_n400.log
baseline jsonl: /tmp/dsv4_turn25_cupd3_baseline_n400_logits.jsonl
attrib log: /tmp/dsv4_turn25_cupd3_tail_attrib_l2_n400.log
attrib jsonl: /tmp/dsv4_turn25_cupd3_tail_attrib_l2_n400_logits.jsonl
first divergence: /tmp/dsv4_turn25_cupd3_tail_attrib_first_divergence_l2_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  mean_abs_logit_err=0
  rms_logit_err=0
  top10 overlap=10/10
  top20 overlap=20/20
  transcript exact/stable: yes
  hot-path-neutral guard: ok
  rejected paths active: no
  intrusive flags active: no

attribution summary:
  emit_count=200
  attn_emits=100
  index_emits=100
  first_layer=2
  first_token=11
  pool_softmax=200
  weighted_pool=200
  rms_norm=200
  norm_weight=200
  rope=200
  quant=200
  cache_handoff=200
  estimated_tail_dispatch_nodes=2000
  observed_dispatch_contribution=unknown
  per_op_dispatch_isolation=unavailable
  projection_source=generic
  cache_mutation=disabled
  candidate_cache_side_effect=0
  consume_path=disabled
```

Decision:

```text
next plausible dispatch-reducing boundary:
  pool_norm_rope_quant_tail

why:
  layer 2 n400 emits 200 compressor tail events
  each emit traverses the same contiguous pool/weighted-pool/norm/RoPE/quant/cache-handoff tail
  cache handoff alone is too small and must keep explicit generic cache-write semantics for now
  projection ownership is not justified by this attribution alone

recommended next exact step:
  build a backend-fused V3 tail candidate covering pool softmax/weighted pool/RMSNorm/norm-weight/RoPE/quant
  keep projection_source=generic
  keep cache writes explicit/generic
  prove n16/n80 exactness before any consume
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
cache side-effect path added: no
all-layer V3 consume attempted: no
```

### 2026-05-13 Turn #26 CUPD3 Backend Tail Candidate

Turn #26 implemented a shadow-only backend-tail candidate for the contiguous compressor tail identified in Turn #25:

```text
candidate scope:
  pool softmax / weighted pool / RMSNorm / norm weight / RoPE via GGML_OP_DSV4_DECODE_COMPRESS
  FP8/HFP4 quant row via existing quant ops

projection_source:
  generic

cache writes:
  existing generic cache write path only
  candidate_cache_side_effect=0

consume:
  disabled
```

New flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX=<n>
```

Backend activation note:

```text
GGML_OP_DSV4_DECODE_COMPRESS backend lowering is enabled by the new BACKEND_TAIL flag.
The old LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_COMPRESS flag remains separate.
No old compressor/update consume path was enabled.
```

Layer 2 n16 shadow:

```text
log: /tmp/dsv4_turn26_cupd3_backend_tail_l2_n16.log
summary:
  scope=pool_norm_rope_quant
  emit_cases=4
  exact_cases=4
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  first_mismatch=none
  attn_emits=2
  index_emits=2
  first_layer=2
  first_token=11
  projection_source=generic
  generic_tail_built=1
  backend_tail_built=1
  backend_tail_consumed=0
  generic_cache_write_built=1
  candidate_cache_side_effect=0
  cache_mutation_mode=generic_existing_write
  consume_path=disabled
backend counter:
  dsv4_dcomp=4
```

Layer 2 n80 shadow:

```text
log: /tmp/dsv4_turn26_cupd3_backend_tail_l2_n80.log
summary:
  scope=pool_norm_rope_quant
  emit_cases=40
  exact_cases=40
  non_exact_cases=0
  max_abs=0
  max_rms=0
  over_tol=0
  first_mismatch=none
  attn_emits=20
  index_emits=20
  projection_source=generic
  generic_tail_built=1
  backend_tail_built=1
  backend_tail_consumed=0
  generic_cache_write_built=1
  candidate_cache_side_effect=0
  cache_mutation_mode=generic_existing_write
  consume_path=disabled
backend counter:
  dsv4_dcomp=40
```

Short dispatch audit:

```text
baseline n80 log: /tmp/dsv4_turn26_cupd3_backend_tail_baseline_n80.log
backend-tail n80 log: /tmp/dsv4_turn26_cupd3_backend_tail_l2_n80.log

baseline:
  metal_dispatch=285811
  dsv4_dcomp=0
  dsv4_rope_hfp4=462
  dsv4_rope_fp8=624

backend-tail shadow:
  metal_dispatch=286051
  dsv4_dcomp=40
  dsv4_rope_hfp4=442
  dsv4_rope_fp8=604

observed whole-run dispatch delta:
  +240

interpretation:
  backend candidate activates and collapses its own pool/norm/RoPE unit to dsv4_dcomp.
  Because this is shadow-only, the generic tail still runs and the extra candidate branch increases total dispatch.
  This is not a performance result and not a consume candidate yet.
```

n400 hot-neutral baseline:

```text
log: /tmp/dsv4_turn26_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn26_baseline_n400_hotneutral.jsonl
metal_dispatch=1399039
dsv4_dcomp=0
no CUPD3 backend-tail summary
transcript stable: yes
```

Decision:

```text
backend tail exact as a shadow candidate: yes, for the graph-level compare path
backend op activated: yes, dsv4_dcomp > 0
candidate cache side effects: no
consume eligible now: no
dispatch-collapsing as consumed replacement: not proven
continue CUPD3: yes, but next step must be a guarded consume canary only after exactness is paired with cache-safe generic writes
scale layers: no
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
consume path added: no
cache side-effect path added: no
all-layer V3 consume attempted: no
```

### 2026-05-13 Turn #23 Compressor/Update V3 Stage Shadow

Turn #23 pivoted away from AOHC after the accepted Turn #22 result:

```text
AOHC backend fusion:
  exact: yes
  dispatch-reducing: yes
  four-layer dispatch delta: -1596
  wall-clock useful: no, effectively flat (+0.15 tok/s)
  performance path accepted: no
```

Boundary audit:

```text
audit path: /tmp/dsv4_turn23_compressor_v3_boundary_audit.txt

DS4 compressor/update boundary:
  projection inputs
  APE/add
  state_kv update
  state_score update
  ratio/window pooling
  RMSNorm
  norm weight
  RoPE
  FP8/HFP4 quant row
  cache-store handoff

current repo boundary:
  generic projection nodes
  CUPD2-owned decode row update/current packing/pool input support
  generic tail still owns pool softmax, weighted pool, RMSNorm, norm weight, RoPE, quant op, and cache writes

prior compressor results:
  CUPD2 safe exact but flat
  CUPD2 + compressor pair exact but flat
  CUPD2_FUSED_COMP drifted and remains rejected
```

New V3 flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SHADOW=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SEARCH=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_MODE=<generic_shadow|cupd2_tail_shadow|ds4_shape_shadow>
```

New report fields:

```text
dsv4_cupd3_summary
dsv4_cupd3_search_summary
dsv4_cupd3_dep
cache_mutation=disabled
consume_path=disabled
```

Search:

```text
log: /tmp/dsv4_turn23_cupd3_search_n80.log
first_compressor_layer: 2
first_compressor_token: 10
compressor_cases: 4898
ratio_boundary_cases: 840
quant_emit_cases: 840
search compare summary:
  cases: 16516
  exact_cases: 16516
  max_abs: 0
  max_rms: 0
  over_tol: 0
```

Layer 0 control note:

```text
/tmp/dsv4_turn23_cupd3_generic_shadow_l0_n16.log had no V3 compare cases because layer 0 does not exercise the compressor/update path for the fixed prompt.
The first compressor-active layer is layer 2, so the exact generic-shadow control was rerun on layer 2.
```

Generic shadow control:

```text
log: /tmp/dsv4_turn23_cupd3_generic_shadow_l2_n16.log
layer: 2
token window: 1..16
cases: 60
exact_cases: 60
max_abs: 0
max_rms: 0
over_tol: 0
projection_source: generic
cache_mutation: disabled
consume_path: disabled
```

CUPD2 tail shadow:

```text
log: /tmp/dsv4_turn23_cupd3_tail_l2_n16.log
layer: 2
token window: 1..16
compared tensors:
  attn/index state_kv
  attn/index state_score
  attn/index pool_input
  attn/index pooled
  attn/index pre_norm
  attn/index norm
  attn/index norm_w
  attn/index rope
  attn/index quant
  attn/index downstream_kv
cases: 60
exact_cases: 60
max_abs: 0
max_rms: 0
over_tol: 0
dependency note:
  CUPD2 handoff covers state_kv/state_score/pool_input support
  generic tail still represents pool softmax, weighted pool, RMSNorm, norm weight, RoPE, and quant
```

DS4-shaped V3 shadow:

```text
n16 log: /tmp/dsv4_turn23_cupd3_ds4_shape_l2_n16.log
layer: 2
token window: 1..16
cases: 60
exact_cases: 60
max_abs: 0
max_rms: 0
over_tol: 0
first_non_exact_tensor: none

n80 layer 2 log: /tmp/dsv4_turn23_cupd3_ds4_shape_l2_n80.log
cases: 636
exact_cases: 636
max_abs: 0
max_rms: 0
over_tol: 0

all applicable n80 log: /tmp/dsv4_turn23_cupd3_ds4_shape_all_n80.log
cases: 16516
exact_cases: 16516
non_exact_cases: 0
max_abs: 0
max_rms: 0
over_tol: 0
first failing case: none
```

Dependency audit:

```text
projection_source: generic
candidate_uses_generic_state_kv: 0
candidate_uses_generic_state_score: 0
candidate_uses_generic_pool_output: 0
candidate_uses_generic_norm_output: 0
candidate_uses_generic_rope_output: 0
candidate_uses_generic_quant_output: 0
allowed_inputs: projection, prev_state, ape, norm, pos, rope_cfg
forbidden_inputs_seen: none
cache_mutation: disabled
consume_path: disabled
```

Important limitation:

```text
This V3 pass is an exact shadow envelope with projection_source=generic.
It compares the generic-boundary tensors and quant outputs exactly, but it is not a consume path and does not mutate cache directly.
It is not yet a full independent compressor replacement because projection ownership remains generic.
```

n400 hot-neutral baseline:

```text
log: /tmp/dsv4_turn23_baseline_n400_hotneutral.log
jsonl: /tmp/dsv4_turn23_baseline_n400_hotneutral.jsonl
metal_dispatch: 1399039
generation: 19.4 tok/s
CUPD3 summary: absent
intrusive V3 flags: absent
transcript stable: yes
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance paths accepted: no
consume path added: no
cache mutation added: no
```

Conclusion:

```text
V3 shadow exact: yes, for the observed state/pool/norm/RoPE/quant boundary tensors
V3 candidate independent enough for consume: no, because projection_source remains generic and this is still a shadow envelope
eligible future step: single-layer V3 consume canary only after cache side effects are explicitly dependency-safe
next exact step: either make the V3 candidate own the projection/tail more concretely, or design a cache-safe consume canary with generic cache writes preserved
```

### 2026-05-13 Turn #24 Compressor/Update V3 Tail Consume Canary

Turn #24 implemented a single-layer cache-safe V3 tail consume canary:

```text
projection_source: generic
candidate scope:
  APE/state row
  pool input
  pooled row
  RMSNorm
  norm weight
  RoPE
  FP8/HFP4 quant row
cache writes:
  existing generic cache write path
new cache side-effect op: no
consume path accepted as performance path: no
```

New flags and under-test:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_MODE=tail_candidate_generic_cache
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_ABORT_ON_MISMATCH=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL=1

hot-neutral under-test:
  cupd3_tail_single_layer_consume
```

Guard result:

```text
missing layer log: /tmp/dsv4_turn24_cupd3_guard_missing_layer.log
result:
  consume_allowed=0
  reason=missing_layer
  dsv4_cupd3_consume=0

rejected path log: /tmp/dsv4_turn24_cupd3_guard_rejected_path.log
result:
  consume_allowed=0
  reason=rejected_paths_enabled:LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6
  rejected_paths_active=1
```

Layer 2 n16 compare + consume:

```text
log: /tmp/dsv4_turn24_cupd3_consume_l2_n16.log
compare summary:
  cases: 3204
  exact_cases: 3204
  non_exact_cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
consume summary:
  dsv4_cupd3_consume: 8
  generic_projection_built: 1
  generic_tail_built: 0
  candidate_tail_built: 1
  candidate_tail_consumed: 1
  generic_cache_write_built: 1
  candidate_cache_side_effect: 0
  cache_mutation_mode: generic_existing_write
prefix stable: yes
```

Layer 2 n80 compare + consume:

```text
log: /tmp/dsv4_turn24_cupd3_consume_l2_n80.log
compare summary:
  cases: 16516
  exact_cases: 16516
  non_exact_cases: 0
  max_abs: 0
  max_rms: 0
  over_tol: 0
consume summary:
  dsv4_cupd3_consume: 40
  generic_tail_built: 0
  candidate_tail_built: 1
  candidate_tail_consumed: 1
  generic_cache_write_built: 1
  candidate_cache_side_effect: 0
  cache_mutation_mode: generic_existing_write
prefix stable: yes
```

n400 hot-neutral exactness:

```text
baseline log: /tmp/dsv4_turn24_cupd3_baseline_n400.log
baseline jsonl: /tmp/dsv4_turn24_cupd3_baseline_n400_logits.jsonl
consume log: /tmp/dsv4_turn24_cupd3_consume_l2_n400.log
consume jsonl: /tmp/dsv4_turn24_cupd3_consume_l2_n400_logits.jsonl
first divergence: /tmp/dsv4_turn24_cupd3_first_divergence_l2_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err: 0
  rms_logit_err: 0
  top20 overlap: 20/20
  transcript exact/stable: yes
  path_accepted: false

consume summary:
  dsv4_cupd3_consume: 200
  generic_projection_built: 1
  generic_tail_built: 0
  candidate_tail_built: 1
  candidate_tail_consumed: 1
  generic_cache_write_built: 1
  candidate_cache_side_effect: 0
  cache_mutation_mode: generic_existing_write
```

No-logit performance:

```text
baseline A log: /tmp/dsv4_turn24_cupd3_perf/baseline_A_n400.log
baseline B log: /tmp/dsv4_turn24_cupd3_perf/baseline_B_n400.log
consume A log: /tmp/dsv4_turn24_cupd3_perf/consume_l2_A_n400.log
consume B log: /tmp/dsv4_turn24_cupd3_perf/consume_l2_B_n400.log

baseline tok/s:
  A: 22.2
  B: 22.2
  average: 22.2
consume tok/s:
  A: 22.0
  B: 21.8
  average: 21.9
metal_dispatch:
  baseline: 1399039
  consume: 1399039
delta:
  tok/s: -0.3
  dispatch: 0
performance useful: no
```

Decision:

```text
graph-safe: yes
cache-safe: yes, using generic_existing_write
generic tail skipped: yes
performance useful as single layer: no
expand beyond one layer now: no
next step: either make V3 own projections or build a real backend tail fusion before scaling layer count
```

Policy:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
new cache side-effect op added: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #22 AOHC Backend Fusion Layer-Set Scaling

Turn #21 recap:

```text
AOHC backend fused layer 0:
  transcript/logit exact=yes
  dsv4_q8hc=399
  metal_dispatch delta=-399
  no-logit performance useful=no
  baseline avg=21.8 tok/s
  fused layer 0 avg=21.3 tok/s
```

Turn #22 added bounded explicit layer-list support:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYERS=0,14,28,42
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_MAX_LAYERS=<n>

default max layer count=4
all-layer wildcard=blocked
duplicate/invalid/too-large layer sets=blocked
AOHC_FUSED_LAYER and AOHC_FUSED_LAYERS together=rejected
```

Layer-set guard:

```text
valid layer-set guard:
  layers=0,14,28,42
  layer_count=4
  max_layers=4
  all_layer_blocked=0
  rejected_paths_active=0

negative guard log=/tmp/dsv4_turn22_aohc_layer_set_guard_too_many.log
  layers=0,1,2,3,4
  layer_count=5
  max_layers=4
  allowed=0
  reason=too_many_layers
  dsv4_q8hc=0
```

Short exactness:

```text
n16 compare log=/tmp/dsv4_turn22_aohc_fused_layers_0_14_28_42_n16.log
  compared_cases=180
  exact_cases=180
  max_abs=0
  max_rms=0
  over_tol=0
  dsv4_aohc_fused=60
  dsv4_q8hc=0 in intrusive compare mode
  reason=next adjacency, expected for compare mode

n80 compare log=/tmp/dsv4_turn22_aohc_fused_layers_0_14_28_42_n80.log
  compared_cases=948
  exact_cases=948
  max_abs=0
  max_rms=0
  over_tol=0
  dsv4_aohc_fused=316
  dsv4_q8hc=0 in intrusive compare mode
  reason=next adjacency, expected for compare mode
```

n400 hot-path-neutral layer-set consume:

```text
baseline log=/tmp/dsv4_turn22_aohc_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn22_aohc_baseline_n400_logits.jsonl
fused log=/tmp/dsv4_turn22_aohc_fused_layers_0_14_28_42_n400.log
fused jsonl=/tmp/dsv4_turn22_aohc_fused_layers_0_14_28_42_n400_logits.jsonl
first divergence=/tmp/dsv4_turn22_aohc_first_divergence_layers_0_14_28_42_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  mean_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

backend fused summary:
  layers=0,14,28,42
  layer_count=4
  consumed=1596
  expected_consumed=1596
  dsv4_q8hc=1596
  expected_q8hc=1596
  q8hc_eligible=1596
  q8hc_rejected=0
  q8hc_dispatched=1596
  generic_high_projection_built=0
  generic_hc_expand_built=0
  candidate_branch_built=1
  fused_downstream_consumed=1
  dispatch_collapsed=1
```

Dispatch:

```text
baseline metal_dispatch=1399039
fused metal_dispatch=1397443
expected dispatch delta=-1596
observed dispatch delta=-1596
dispatch reduction scales: yes
```

No-logit performance:

```text
baseline A log=/tmp/dsv4_turn22_aohc_perf/baseline_A_n400.log
baseline A generation=19.3 tok/s
baseline A metal_dispatch=1399039

baseline B log=/tmp/dsv4_turn22_aohc_perf/baseline_B_n400.log
baseline B generation=20.2 tok/s
baseline B metal_dispatch=1399039

fused A log=/tmp/dsv4_turn22_aohc_perf/fused_layers_A_n400.log
fused A generation=19.4 tok/s
fused A metal_dispatch=1397443
fused A dsv4_q8hc=1596

fused B log=/tmp/dsv4_turn22_aohc_perf/fused_layers_B_n400.log
fused B generation=20.4 tok/s
fused B metal_dispatch=1397443
fused B dsv4_q8hc=1596

baseline avg=19.75 tok/s
fused avg=19.9 tok/s
delta=+0.15 tok/s
```

Decision:

```text
AOHC exact: yes
AOHC dispatch reduction scales: yes
AOHC wall-clock useful: no material gain; result is effectively flat
expand to 8 layers: no
performance path accepted: no
recommended next action: stop AOHC as a primary line and pivot to another exact whole-stage boundary
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #21 AOHC Backend Fusion Eligibility / Activation

Turn #20 proved the partial AOHC fused candidate was exact but dispatch-neutral:

```text
dsv4_aohc_fused=399
generic_branch_built=0
fused_branch_built=1
fused_downstream_consumed=1
dsv4_q8hc=0
metal_dispatch baseline=1399039
metal_dispatch fused=1399039
```

Turn #21 added backend eligibility tracing for AOHC and Q8HC:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_ELIG_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8HC_ELIG_TRACE=1
```

Eligibility trace result:

```text
compare trace log=/tmp/dsv4_turn21_aohc_elig_l0_n16.log
consume trace before fix=/tmp/dsv4_turn21_aohc_elig_consume_l0_n16.log
consume trace after fix=/tmp/dsv4_turn21_aohc_elig_consume_l0_n16_after_namefix.log

compare mode:
  candidate_cases=15
  q8hc_eligible=0
  q8hc_rejected=15
  first_reject_reason=next

consume mode before fix:
  dsv4_aohc_fused=15
  generic_branch_built=0
  fused_branch_built=1
  fused_downstream_consumed=1
  candidate_cases=0
  dsv4_q8hc=0
```

Root cause:

```text
the AOHC high-projection op was built with the marker name:
  dsv4_aohc_fused_high-l0-p<token>

but the graph callback later renamed it to:
  attn_out

the existing Metal Q8HC matcher is name-based for this experimental path,
so consume mode never exposed a dsv4_aohc_fused_high op to the backend matcher.
```

The fix restores the AOHC marker name after the callback/probe naming step for the fused branch.
No new GGML op and no new Metal kernel were added.

Activation after fix:

```text
n16 consume trace log=/tmp/dsv4_turn21_aohc_elig_consume_l0_n16_after_namefix.log
  dsv4_q8hc=15
  candidate_cases=15
  q8hc_eligible=15
  q8hc_rejected=0
  q8hc_dispatched=15
  metal_dispatch=66660

n80 consume log=/tmp/dsv4_turn21_aohc_backend_fused_l0_n80.log
  dsv4_q8hc=79
  candidate_cases=79
  q8hc_eligible=79
  q8hc_rejected=0
  q8hc_dispatched=79
  metal_dispatch=285732
```

Short exactness:

```text
n16 compare log=/tmp/dsv4_turn21_aohc_backend_fused_l0_n16.log
  compared_cases=45
  exact_cases=45
  max_abs=0
  max_rms=0
  over_tol=0
  dsv4_q8hc=0 in compare mode
  q8hc reject reason=next

note:
  intrusive compare mode still breaks the immediate high-proj -> HC-expand adjacency
  that the existing Q8HC matcher requires, so backend activation was validated in
  consume mode and final correctness was validated with hot-neutral logits.
```

n400 hot-path-neutral backend-fused consume:

```text
baseline log=/tmp/dsv4_turn21_aohc_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn21_aohc_baseline_n400_logits.jsonl
fused log=/tmp/dsv4_turn21_aohc_backend_fused_l0_n400.log
fused jsonl=/tmp/dsv4_turn21_aohc_backend_fused_l0_n400_logits.jsonl
first divergence=/tmp/dsv4_turn21_aohc_backend_fused_first_divergence_l0_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  mean_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

backend fused summary:
  dsv4_q8hc=399
  dsv4_aohc_fused=399
  candidate_cases=399
  q8hc_eligible=399
  q8hc_rejected=0
  q8hc_dispatched=399
  generic_branch_built=0
  fused_branch_built=1
  fused_downstream_consumed=1
```

Dispatch result:

```text
baseline metal_dispatch=1399039
backend-fused metal_dispatch=1398640
observed dispatch delta=-399
dispatch-collapsing=yes, one dispatch per consumed decode token
```

No-logit performance:

```text
baseline A log=/tmp/dsv4_turn21_aohc_perf/baseline_A_n400.log
baseline A generation=21.9 tok/s
baseline A metal_dispatch=1399039

baseline B log=/tmp/dsv4_turn21_aohc_perf/baseline_B_n400.log
baseline B generation=21.7 tok/s
baseline B metal_dispatch=1399039

backend-fused A log=/tmp/dsv4_turn21_aohc_perf/backend_fused_l0_A_n400.log
backend-fused A generation=21.2 tok/s
backend-fused A metal_dispatch=1398640
backend-fused A dsv4_q8hc=399

backend-fused B log=/tmp/dsv4_turn21_aohc_perf/backend_fused_l0_B_n400.log
backend-fused B generation=21.4 tok/s
backend-fused B metal_dispatch=1398640
backend-fused B dsv4_q8hc=399

baseline avg=21.8 tok/s
backend-fused avg=21.3 tok/s
delta=-0.5 tok/s
```

Decision:

```text
AOHC backend fusion exact: yes
AOHC dispatch collapsed: yes, for layer 0 single-layer consume
performance useful: no, layer 0 A/B was slower despite -399 dispatches
performance path accepted: no
expand beyond one layer: not accepted in this turn
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

### 2026-05-13 Turn #20 AOHC Dispatch-Collapsing Fused Candidate

Turn #19 proved that a selected-layer AOHC graph replacement can be exact and can skip
the generic AOHC branch, but it remained dispatch-neutral:

```text
layer 0 skip-generic replacement:
  transcript/logit exact: yes
  generic_branch_built=0
  candidate_branch_built=1
  candidate_downstream_consumed=1
  metal_dispatch: 1399039 -> 1399039
  avg tok/s: 19.55 -> 19.55
```

Dispatch-shape audit:

```text
mapping note=/tmp/dsv4_turn20_aohc_dispatch_shape_audit.txt

baseline AOHC:
  attention core output
  dsv4_grouped_out low projection
  Q8 high/output projection
  dsv4_hc_expand
  after_attn_hc

Turn #19 graph replacement:
  generic branch skipped for selected layer
  candidate branch built from original inputs
  backend dispatch shape remained equivalent to baseline
```

Turn #20 added a partial fused AOHC candidate:

```text
scope=partial_high_projection_plus_hc_post
new op enum: no
new Metal kernel: no
backend target: existing q8 high-projection + HC expand fusion
counter: dsv4_aohc_fused
under-test name: aohc_fused_single_layer_consume
```

New flags:

```text
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_COMPARE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TRACE=1
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX=<n>
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME=1
```

Short exactness:

```text
layer 0 n16:
  log=/tmp/dsv4_turn20_aohc_fused_l0_n16.log
  dsv4_aohc_fused=15
  compared_cases=45
  exact_cases=45
  max_abs=0
  max_rms=0
  over_tol=0
  dependency_audit_pass=1
  consume_path=disabled

layer 0 n80:
  log=/tmp/dsv4_turn20_aohc_fused_l0_n80.log
  dsv4_aohc_fused=79
  compared_cases=237
  exact_cases=237
  max_abs=0
  max_rms=0
  over_tol=0
  dependency_audit_pass=1
  consume_path=disabled

all layers n80:
  log=/tmp/dsv4_turn20_aohc_fused_all_n80.log
  dsv4_aohc_fused=3397
  compared_cases=10191
  exact_cases=10191
  max_abs=0
  max_rms=0
  over_tol=0
  dependency_audit_pass=1
  consume_path=disabled
```

n400 hot-path-neutral fused consume:

```text
baseline log=/tmp/dsv4_turn20_aohc_baseline_n400.log
baseline jsonl=/tmp/dsv4_turn20_aohc_baseline_n400_logits.jsonl
fused log=/tmp/dsv4_turn20_aohc_fused_l0_n400.log
fused jsonl=/tmp/dsv4_turn20_aohc_fused_l0_n400_logits.jsonl
first divergence=/tmp/dsv4_turn20_aohc_fused_first_divergence_l0_n400.txt

result:
  first divergent token: none through 400 records
  max_abs_logit_err=0
  rms_logit_err=0
  top20 overlap=20/20
  transcript exact/stable=yes
  path_accepted=false

fused summary:
  dsv4_aohc_fused=399
  generic_branch_built=0
  fused_branch_built=1
  fused_downstream_consumed=1
  compare_enabled=0
  readback_enabled=0
  hotpath_neutral_validate=1
  consume_path=single_layer
```

Dispatch result:

```text
baseline metal_dispatch=1399039
fused metal_dispatch=1399039
baseline dsv4_q8hc=0
fused dsv4_q8hc=0
observed dispatch delta=0
```

Decision:

```text
fused exact: yes, for the partial AOHC candidate
dispatch-collapsing: no
performance run: skipped, because dispatch did not drop
performance useful: no evidence
expand beyond one layer: no
```

Policy result:

```text
transcript-exact active: yes
tolerance diagnostic only: yes
drifting paths accepted: no
performance path accepted: no
all-layer consume attempted: no
```

## 2026-05-15 Turn #92 HC_PRE_NORM Weighted-Sum Capture/Layout Closure

Turn #91 recap:
- Producer capture fixed the `cur` vs `norm` alias collapse.
- `RMSNormWeight(reference_cur, norm_weight)` passed Tier B.
- The full source formula `cur = dsv4_hc_weighted_sum(inpL, split_pre)` still failed because the unpinned source input/pre payloads did not match the op-consumed buffers.

Weighted-sum mapping:
- Source: `src/models/deepseek4.cpp:dsv4_hc_pre_from_mixes`.
- `split = ggml_dsv4_hc_split_sinkhorn(mixes, scale, base, n_hc, ...)`.
- `pre = ggml_view_2d(split, n_hc, n_tokens, split->nb[1], 0)`.
- Decode `n_tokens == 1` keeps `pre` as a view.
- `cur = ggml_dsv4_hc_weighted_sum(ctx, x, pre)`.
- Metal op computes `acc = sum_h x[d*x_nb0 + h*x_nb1 + t*x_nb2] * weights[h*w_nb0 + t*w_nb1]`.
- For contiguous `[n_embd,n_hc,n_tokens]`, the source-matching layout is `hidden_major`: `input[h*n_embd + e]`, not `e*n_hc + h`.
- Mapping note: `/tmp/dsv4_turn92_hc_weighted_sum_mapping.txt`.

Capture changes:
- Added explicit weighted-sum labels:
  - `hc_ws_input_inpL_raw`
  - `hc_ws_input_inpL_view`
  - `hc_ws_input_inpL_contiguous` unavailable, not created
  - `hc_ws_split_full_raw`
  - `hc_ws_split_pre_raw`
  - `hc_ws_split_pre_view`
  - `hc_ws_split_pre_contiguous` unavailable, not created
  - `hc_ws_reference_cur`
  - `hc_ws_reference_cur_pre_reshape`
  - `hc_ws_reference_cur_post_reshape`
- Producer capture pins weighted-sum source/reference labels with `ggml_set_output()` for fixture capture only.
- Capture run remains intrusive and is not a hot-neutral validation mode:
  - `capture_intrusive=1`
  - `used_for_fixture_only=1`
  - `not_hot_neutral_validation=1`
  - capture-run `pair/pswiglu/fglu=0/45/45`
- No runtime cutover, no consume, no cache mutation, and no live backend dispatch was added by the side-probe machinery.

Analyzer summary:
- Weighted-sum analyzer: `/tmp/dsv4_turn92_hc_weighted_sum_analysis.txt`.
- `hc_ws_input_inpL_raw`: shape `[4096,4,1,1]`, stride `[4,16384,65536,65536]`, payload `65536` bytes.
- `hc_ws_split_pre_raw`: shape `[4,1,1,1]`, stride `[4,96,96,96]`, view source `split`, offset `0`, payload `16` bytes.
- `hc_ws_reference_cur`: shape `[4096,1,1,1]`, stride `[4,16384,16384,16384]`, payload `16384` bytes.
- `split_pre == split_full[0:4]`: yes.
- `split_pre` values look like mixture weights: yes.
- `input_hc` matches expected `n_embd*n_hc = 16384` payload: yes.

Formula sweep:
- Harness log: `/tmp/dsv4_turn92_harness_hcnorm_recompute.log`.
- Best source formula: `C0_weighted_x_d_h_pre_h / metal_stride_hidden_major`.
- Weighted-sum result:
  - `cur_max_abs=2.98023e-08`
  - `cur_rms=3.79816e-09`
  - `cur_max_rel=2.43394e-06`
  - Tier B: pass
- Full HC_PRE_NORM result:
  - `norm_max_abs=3.57628e-07`
  - `norm_rms=2.34908e-08`
  - `post_max_abs=3.57628e-07`
  - `post_rms=2.34908e-08`
  - Tier B: pass
- E-major/transposed candidate remains wrong:
  - `C3_weighted_input_h_plus_e_hc/e_major cur_max_abs=0.396929`, Tier B fail
- Post/comb/flat alternatives fail as expected.

Decision:
- Weighted-sum mismatch is closed.
- Root cause was capture semantics: the source operands for weighted sum (`inpL` and `pre`) were not producer-pinned, so payloads did not represent the op-consumed logical tensors. Once `inpL`, `pre`, split parent, and cur references are fixture-pinned, the source-contract formula passes Tier B.
- Norm-only regression still passes Tier B.
- HC_PRE_NORM is now fully validated in the standalone harness for layer 0 token 1 fixture semantics.

Policy:
- Tier B adopted: yes.
- Tier thresholds changed: no.
- Transcript-exact remains active for model runs until signed-off cutover.
- Live graph cutover: no.
- Performance run: no.
- Drifting path accepted: no.
- Performance path accepted: no.

### 2026-05-13 Turn #93 Harness Metal HC_PRE_NORM Recompute Kernel

Turn #92 recap:
- CPU harness recompute closed the HC_PRE_NORM weighted-sum mismatch.
- Validated formula/layout:
  - `cur[e] = sum_h input_hc[h*n_embd + e] * split_pre[h]`
  - `norm = RMSNormWeight(cur, norm_weight)`
  - `post = norm`
  - layout: `C0_weighted_x_d_h_pre_h / metal_stride_hidden_major`
  - epsilon: `1e-6`
- CPU result passed Tier B with `cur_max_abs=2.98023e-08`, `norm_max_abs=3.57628e-07`, `post_max_abs=3.57628e-07`.

Metal harness kernel:
- Added a harness-only Metal recompute path for `hc_pre_norm`.
- Inputs:
  - `input_hc_original_residual`
  - `split_pre`
  - `norm_weight`
- Outputs:
  - `candidate_cur`
  - `candidate_norm`
  - `candidate_post`
- Formula: `C0_weighted_x_d_h_pre_h`.
- Input layout: `metal_stride_hidden_major`.
- Epsilon: `1e-6`.
- The default `--mode kernel --stage hc_pre_norm` path performs real Metal recompute.
- The old copy smoke is retained only as explicit `--kernel-mode copy_smoke`.
- Copy guard result:
  - `--kernel-mode copy_smoke --forbid-copy-smoke` fails with `copy_smoke_forbidden`.
- `copied_reference_output=0` for the real Metal recompute path.

Results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn93_harness_identity_payloads.log`
  - result: pass
  - records loaded: `33`
  - required stages: all present
- CPU recompute regression:
  - log: `/tmp/dsv4_turn93_harness_hcnorm_cpu_recompute.log`
  - result: pass
  - `cur_max_abs=2.98023e-08`
  - `norm_max_abs=3.57628e-07`
  - `post_max_abs=3.57628e-07`
- Metal recompute:
  - log: `/tmp/dsv4_turn93_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `cur_max_abs=0`
  - `cur_rms=0`
  - `norm_max_abs=4.76837e-07`
  - `norm_rms=4.65561e-08`
  - `post_max_abs=4.76837e-07`
  - `post_rms=4.65561e-08`
  - Tier B: pass
  - Tier A exact: pass for the aggregate harness thresholds
- The exact `cur` match is limited to the four-term weighted sum stage; norm/post still show small non-zero FP-scale deltas and remain well within Tier B.

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Performance run: no.
- Drifting path accepted: no.
- Performance path accepted: no.

### 2026-05-13 Turn #94 Harness Routed-MoE Source-Contract Recompute

Turn #93 recap:
- `HC_PRE_NORM` has a harness-only Metal recompute kernel.
- The Metal path computes from captured inputs, reports `metal_recompute=1`, `copied_reference_output=0`, and passes Tier B.
- No live graph cutover or performance path was added.

Routed-MoE source contract:
- Mapping note: `/tmp/dsv4_turn94_rmoe_source_contract.txt`.
- Source path maps to:
  - `ffn_input`
  - router top-k ids/weights
  - per-slot expert gate/up/SwiGLU/down outputs
  - routed cumulative sum
  - shared gate/up/SwiGLU/down outputs
  - `final_ffn_output = routed_sum + shared_down`
- T94 does not decode quantized expert/shared weights.
- T94 validates the captured source-contract boundary:
  - `routed_sum = sum(expert_down slots)`
  - `final_ffn = routed_sum + shared_down`
- `expert_weight_recompute=0`
- `weights_not_decoded=1`
- `copied_reference_output=0`

Producer-capture payloads:
- Capture log: `/tmp/dsv4_turn94_capture_rmoe_l0n16.log`.
- Fixture: `tests/fixtures/dsv4_layer_executor/routed_moe_final_output_l0_n16.jsonl`.
- Payload side-files: `tests/fixtures/dsv4_layer_executor/dsv4_turn94_rmoe_payloads/`.
- Producer capture is fixture-only intrusive:
  - `capture_intrusive=1`
  - `used_for_fixture_only=1`
  - `not_hot_neutral_validation=1`
- Side-probe summary reported:
  - `live_graph_nodes_added=0`
  - `live_backend_dispatches=0`
  - `output_consumed=0`
  - `cache_mutation=disabled`
  - `payload_records=300`
  - `payload_blocked=0`

Captured stages:
- Available tensor payloads:
  - `rmoe_ffn_input`
  - `rmoe_topk_ids`
  - `rmoe_topk_weights`
  - `rmoe_expert_gate`
  - `rmoe_expert_up`
  - `rmoe_expert_swiglu`
  - `rmoe_expert_down`
  - `rmoe_routed_sum`
  - `rmoe_shared_gate`
  - `rmoe_shared_up`
  - `rmoe_shared_swiglu`
  - `rmoe_shared_down`
  - `rmoe_final_ffn_reference`
- Analyzer log: `/tmp/dsv4_turn94_rmoe_payload_semantics.txt`.
- Top-k ids: `[234, 216, 130, 13, 17, 124]`.
- Top-k weights: `[0.1377616823, 0.0584524572, 0.3220014572, 0.3859136105, 0.2330851257, 0.3627856672]`.
- Top-k weight sum: `1.5`.
- Alias groups:
  - `rmoe_weighted_down_slot5,rmoe_routed_sum`
- `routed_sum == final_ffn`: `0`.
- `shared_down == final_ffn`: `0`.

Harness results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn94_harness_identity_payloads.log`
  - result: pass
  - records loaded: `53`
  - stages loaded: `5`
  - full tensor payload available: `1`
  - byte payload available: `1`
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn94_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE source-contract CPU recompute:
  - log: `/tmp/dsv4_turn94_harness_rmoe_recompute.log`
  - result: pass
  - `recompute_possible=1`
  - `partial_rmoe_recompute=1`
  - `full_rmoe_recompute=1` for captured substage outputs
  - `routed_sum_only_recompute=0`
  - `expert_weight_recompute=0`
  - `weights_not_decoded=1`
  - `rmoe_routed_sum_max_abs=2.98023e-08`
  - `rmoe_routed_sum_rms=5.61747e-09`
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - `rmoe_final_ffn_rms=9.08791e-09`
  - Tier B: pass

Policy:
- No routed-MoE Metal kernel in T94.
- No live graph dispatch or cutover.
- No performance run.
- No tolerance changes.
- No copied-reference acceptance.
- Tier B remains the official executor acceptance policy.
- Tier A remains limited to same-reduction-tree / intra-graph optimizations.
- Transcript-exact remains active for model runs until signed-off cutover.
- User signoff is still required before production acceptance.

Next target:
- Turn #95 should implement the harness-only routed-MoE Metal kernel for the validated captured source contract.

## 2026-05-13 Turn #95 Harness Metal Routed-MoE Final-Output Kernel

Turn #94 recap:
- Routed-MoE source-contract CPU recompute passes Tier B.
- Captured substage payloads available:
  - `rmoe_expert_down`
  - `rmoe_routed_sum`
  - `rmoe_shared_down`
  - `rmoe_final_ffn_reference`
- T94 result:
  - `partial_rmoe_recompute=1`
  - `full_rmoe_recompute=1` for captured substage outputs
  - `weights_not_decoded=1`
  - `copied_reference_output=0`
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - Tier B: pass

Metal harness kernel:
- Added harness-only Metal entry point:
  - `dsv4_layer_executor_metal_rmoe_final_from_substages`
- Kernel name:
  - `dsv4_harness_rmoe_final_from_substages`
- Inputs:
  - weighted-down slots from `rmoe_expert_down`
  - `rmoe_shared_down`
- Outputs:
  - `candidate_routed_sum`
  - `candidate_final_ffn`
- Formula:
  - `routed_sum[e] = sum_{slot=0..5} weighted_down[slot*n_embd + e]`
  - `final_ffn[e] = routed_sum[e] + shared_down[e]`
- Layout:
  - `slot_major`
  - `n_embd=4096`
  - `topk=6`
- This is still a source-contract substage kernel:
  - `expert_weight_recompute=0`
  - `weights_not_decoded=1`
  - no quantized weight decoding
- Copy protection:
  - real kernel path reports `copied_reference_output=0`
  - explicit copy-smoke guard fails with `copy_smoke_forbidden`

Results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn95_harness_identity_payloads.log`
  - result: pass
  - records loaded: `53`
  - stages loaded: `5`
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn95_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE CPU recompute regression:
  - log: `/tmp/dsv4_turn95_harness_rmoe_cpu_recompute.log`
  - result: pass
  - `partial_rmoe_recompute=1`
  - `full_rmoe_recompute=1`
  - `weights_not_decoded=1`
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - Tier B: pass
- Routed-MoE Metal kernel:
  - log: `/tmp/dsv4_turn95_harness_rmoe_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `partial_rmoe_recompute=1`
  - `full_rmoe_recompute=1`
  - `weights_not_decoded=1`
  - `weighted_down_available=1`
  - `shared_down_available=1`
  - `reference_routed_sum_available=1`
  - `reference_final_ffn_available=1`
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_routed_sum_rms=0`
  - `rmoe_final_ffn_max_abs=0`
  - `rmoe_final_ffn_rms=0`
  - Tier B: pass
  - Tier A exact: pass
- Zero-delta note:
  - The routed-MoE Metal kernel is not a copy path.
  - It receives only weighted-down slots and shared-down as inputs.
  - Exact zero deltas indicate the kernel matched the captured float add order for this source-contract substage.
- Copy-smoke guard:
  - log: `/tmp/dsv4_turn95_harness_rmoe_forbid_copy_smoke.log`
  - result: expected fail
  - reason: `copy_smoke_forbidden`

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Performance run: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #96 should move to the next bounded executor stage under the same pattern: source contract, fixture capture/recompute if needed, then harness-only kernel.

### 2026-05-13 Turn #96 Harness AOHC Boundary Source Contract / Kernel

Turn #95 recap:
- HC_PRE_NORM Metal harness kernel passes Tier B.
- Routed-MoE source-contract CPU recompute passes Tier B.
- Routed-MoE harness-only Metal source-contract kernel passes Tier B.
- No live graph cutover, no live graph executor dispatch, and no performance path accepted.

AOHC source contract audited:
- Source path:
  - `src/models/deepseek4.cpp`
  - attention core output is reshaped/RoPE-tail adjusted, then projected through `dsv4_grouped_out`.
  - `dsv4_grouped_out` emits `attn_low` and `attn_out`.
  - `dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens)` emits `after_attn_hc`.
- GGML op:
  - `GGML_OP_DSV4_HC_EXPAND`
  - CPU implementation: `ggml_compute_forward_dsv4_hc_expand`
  - Metal implementation: `kernel_dsv4_hc_expand`
- Formula audited:
  - `after_attn_hc[d,dst_hc,t] = attn_out[d,t] * post[dst_hc,t] + sum_src comb[dst_hc,src_hc,t] * residual[d,src_hc,t]`
- Fixture source contract for T96:
  - projection weights are not decoded.
  - AOHC recompute starts from captured logical `attn_out`.
  - `weights_not_decoded=1`.
  - `partial_aohc_recompute=1`, `full_aohc_recompute=0`.

Capture additions:
- Added producer-capture payload targets for:
  - `aohc_attn_core_output`
  - `aohc_attn_low`
  - `aohc_attn_out`
  - `aohc_hc_post_input`
  - `aohc_hc_post_residual`
  - `aohc_hc_split_full`
  - `aohc_hc_post_weights`
  - `aohc_hc_comb`
  - `aohc_after_attn_hc_reference`
  - `aohc_layer_attn_output_anchor`
- `aohc_` payload targets are pinned for fixture capture with `ggml_set_output()`.
- Producer capture is fixture-only intrusive:
  - `capture_intrusive=1`
  - `used_for_fixture_only=1`
  - `not_hot_neutral_validation=1`
- Normal model execution remains hot-neutral:
  - `live_graph_nodes_added=0`
  - `live_backend_dispatches=0`
  - `output_consumed=0`
  - `cache_mutation=disabled`

Captured AOHC tensor facts:
- `aohc_attn_out`:
  - op: `MUL_MAT`
  - shape: `[4096,1,1,1]`
  - required: yes
- `aohc_hc_post_residual`:
  - op: `RESHAPE`
  - shape: `[4096,4,1,1]`
  - required: yes
- `aohc_hc_split_full`:
  - op: `DSV4_HC_SPLIT_SINKHORN`
  - shape: `[24,1,1,1]`
  - used to derive post and comb because direct view capture can be ambiguous.
- `aohc_hc_post_weights`:
  - op: `VIEW`
  - shape: `[4,1,1,1]`
  - required: yes
- `aohc_hc_comb`:
  - op: `RESHAPE`
  - shape: `[4,4,1,1]`
  - required: yes
- `aohc_after_attn_hc_reference`:
  - op: `DSV4_HC_EXPAND`
  - shape: `[4096,4,1,1]`
  - required: yes
- Chain consistency:
  - `aohc_after_attn_hc_reference` matches the downstream HC_PRE_NORM input.
  - harness reports `aohc_matches_hc_pre_norm_input=1`.

AOHC CPU recompute:
- log: `/tmp/dsv4_turn96_harness_aohc_cpu_recompute.log`
- result: fail
- `recompute_possible=1`
- `copied_reference_output=0`
- `partial_aohc_recompute=1`
- `full_aohc_recompute=0`
- `aohc_attn_out_available=1`
- `aohc_residual_available=1`
- `aohc_post_weights_available=1`
- `aohc_comb_available=1`
- `aohc_reference_available=1`
- `aohc_matches_hc_pre_norm_input=1`
- `aohc_output_max_abs=38.2618`
- `aohc_output_rms=1.96209`
- `aohc_output_max_rel=50957.5`
- Tier B: fail

AOHC Metal harness kernel:
- Added harness-only entry point:
  - `dsv4_layer_executor_metal_aohc_hc_post_from_substages`
- Kernel formula:
  - `candidate_after_attn_hc = HC_EXPAND(attn_out, residual, post, comb)`
- Inputs:
  - captured `aohc_attn_out`
  - captured `aohc_hc_post_residual`
  - captured or full-split-derived post weights
  - captured or full-split-derived comb weights
- Output:
  - `candidate_after_attn_hc`
- log: `/tmp/dsv4_turn96_harness_aohc_metal_kernel.log`
- result: fail
- `metal_recompute=1`
- `copied_reference_output=0`
- `aohc_output_max_abs=38.2618`
- `aohc_output_rms=1.96209`
- Tier B: fail
- Interpretation:
  - Metal agrees with the CPU recompute failure.
  - The blocker is not Metal arithmetic; it is AOHC source-contract capture semantics.

Blocker:
- `aohc_after_attn_hc_reference` is stable and chain-consistent with HC_PRE_NORM input.
- Captured `attn_out`, `residual`, `post`, and `comb` are all present as full tensor payloads.
- Recomputing the audited `DSV4_HC_EXPAND` formula from those payloads does not reproduce the reference.
- Full split capture did not resolve the mismatch.
- Minimal next fix:
  - capture the exact `GGML_OP_DSV4_HC_EXPAND` source tensors from `after_attn_hc->src[0..3]` at the op boundary, including source pointer identity, view offsets, and payloads, instead of relying only on surrounding logical variables.
  - If those exact op-source payloads still fail, inspect payload readback layout for the `RESHAPE` residual source and `VIEW/RESHAPE` split sources.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn96_harness_identity_payloads.log`
  - result: pass
  - records loaded: `64`
  - stages loaded: `5`
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn96_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE CPU recompute regression:
  - log: `/tmp/dsv4_turn96_harness_rmoe_cpu_recompute.log`
  - result: pass
  - `partial_rmoe_recompute=1`
  - `full_rmoe_recompute=1`
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - Tier B: pass
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn96_harness_rmoe_metal_kernel.log`
  - result: pass
  - `metal_recompute=1`
  - `copied_reference_output=0`
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_final_ffn_max_abs=0`
  - Tier B: pass
- AOHC copy-smoke guard:
  - log: `/tmp/dsv4_turn96_harness_aohc_forbid_copy_smoke.log`
  - expected result: fail
  - reason: `copy_smoke_forbidden`

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #97 should fix AOHC source-contract capture at the exact `DSV4_HC_EXPAND` op-source boundary before adding any new stage.

### 2026-05-13 Turn #97 AOHC Exact HC_EXPAND Source-Boundary Capture

Turn #96 recap:
- AOHC final boundary payload is stable:
  - `aohc_after_attn_hc_reference` matches downstream HC_PRE_NORM input.
  - `aohc_matches_hc_pre_norm_input=1`.
- AOHC CPU and Metal recompute both failed from surrounding logical source labels:
  - `aohc_output_max_abs=38.2618`
  - `aohc_output_rms=1.96209`
  - `aohc_output_max_rel=50957.5`
- HC_PRE_NORM Metal and routed-MoE CPU/Metal regressions remained passing.

Payload readback audit:
- Existing payload flush uses:
  - `ggml_backend_tensor_get(entry.tensor, data.data(), 0, ggml_nbytes(entry.tensor))`
- For Metal buffers, `ggml_metal_buffer_get_id()` resolves the buffer offset from `tensor->data`.
- For views, `tensor->data` already includes `view_offs`, so offset `0` is view-relative for Metal shared/private buffer readback.
- The original hypothesis that offset `0` necessarily reads from the backing buffer base was not supported by the local implementation.
- The remaining issue is not the call offset alone.

Exact `DSV4_HC_EXPAND` source capture:
- Added payload targets directly from `after_attn_hc->src[0..3]`:
  - `aohc_hcexpand_src0_block`
  - `aohc_hcexpand_src1_residual`
  - `aohc_hcexpand_src2_post`
  - `aohc_hcexpand_src3_comb`
- Normalized fixture now preserves shape/stride/view metadata for these exact source records:
  - `aohc_hcexpand_src0_block`: op `MUL_MAT`, shape `[4096,1,1,1]`, stride `[4,16384,16384,16384]`, offset `0`
  - `aohc_hcexpand_src1_residual`: op `RESHAPE`, shape `[4096,4,1,1]`, stride `[4,16384,65536,65536]`, view source `node_2`, offset `0`
  - `aohc_hcexpand_src2_post`: op `VIEW`, shape `[4,1,1,1]`, stride `[4,96,96,96]`, view source `node_8`, offset `16`
  - `aohc_hcexpand_src3_comb`: op `RESHAPE`, shape `[4,4,1,1]`, stride `[4,16,64,64]`, view source `node_8`, offset `32`
- Exact source payload equality:
  - `aohc_hcexpand_src0_block == aohc_attn_out`
  - `aohc_hcexpand_src1_residual == aohc_hc_post_residual`
  - `aohc_hcexpand_src2_post == aohc_hc_post_weights`
  - `aohc_hcexpand_src3_comb == aohc_hc_comb`

AOHC CPU recompute:
- log: `/tmp/dsv4_turn97_harness_aohc_cpu_recompute.log`
- result: fail
- `recompute_possible=1`
- `copied_reference_output=0`
- `partial_aohc_recompute=1`
- `full_aohc_recompute=0`
- `aohc_attn_out_available=1`
- `aohc_residual_available=1`
- `aohc_post_weights_available=1`
- `aohc_comb_available=1`
- `aohc_reference_available=1`
- `aohc_matches_hc_pre_norm_input=1`
- `aohc_output_max_abs=38.2618`
- `aohc_output_rms=1.96209`
- `aohc_output_max_rel=50957.5`
- Tier B: fail

AOHC Metal recompute:
- log: `/tmp/dsv4_turn97_harness_aohc_metal_kernel.log`
- result: fail
- `metal_recompute=1`
- `copied_reference_output=0`
- metrics match CPU failure:
  - `aohc_output_max_abs=38.2618`
  - `aohc_output_rms=1.96209`
- Interpretation:
  - Metal kernel parity with CPU recompute is confirmed.
  - AOHC remains blocked before kernel acceptance.

Remaining blocker:
- Capturing exact `DSV4_HC_EXPAND` op-source tensor handles after eval still does not reproduce the `DSV4_HC_EXPAND` output.
- The exact source records are byte-identical to the prior surrounding logical labels, so T96 was already pointing at the same source objects.
- Since `after_attn_hc` itself matches downstream HC_PRE_NORM input, the output payload is trustworthy.
- The source payloads read after eval are therefore not sufficient evidence of the values consumed by the Metal `DSV4_HC_EXPAND` kernel.
- Narrow remaining possibilities:
  - source buffers are reused or overwritten after `DSV4_HC_EXPAND` consumes them, despite output pinning;
  - `ggml_set_output()` preserves the final AOHC output but does not preserve all source-buffer values at the consumer boundary;
  - source payload readback needs op-time preservation or an explicit fixture-only materialization of `after_attn_hc->src[0..3]`.
- Minimal next fix:
  - add fixture-only source preservation at the `DSV4_HC_EXPAND` consumer boundary.
  - The preservation must not be used to claim runtime performance or normal hot-neutral validation.
  - It must not add live executor dispatch or consume executor output.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn97_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn97_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE CPU recompute regression:
  - log: `/tmp/dsv4_turn97_harness_rmoe_cpu_recompute.log`
  - result: pass
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - Tier B: pass
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn97_harness_rmoe_metal_kernel.log`
  - result: pass
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_final_ffn_max_abs=0`
  - Tier B: pass
- AOHC copy-smoke guard:
  - log: `/tmp/dsv4_turn97_harness_aohc_forbid_copy_smoke.log`
  - expected result: fail
  - reason: `copy_smoke_forbidden`

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #98 should remain on AOHC and preserve/read back the `DSV4_HC_EXPAND` source values at the consumer boundary before moving to any later executor stage.

## 2026-05-15 Turn #98 AOHC HC_EXPAND Dispatch-Time Source Capture

Turn #97 recap:
- Exact post-eval `DSV4_HC_EXPAND` source handles were captured, but AOHC recompute still failed Tier B.
- Failure metrics were `38.2618 / 1.96209 / 50957.5` for max_abs / rms / max_rel.
- `after_attn_hc` still matched the downstream HC_PRE_NORM input, so the output reference was trusted.

Metal/source contract:
- Source audit written to `/tmp/dsv4_turn98_hc_expand_metal_contract.txt`.
- Generic Metal kernel: `kernel_dsv4_hc_expand`.
- Optimized n_hc=4 kernel: `kernel_dsv4_hc_expand4`.
- Thread mapping: `d = tid % n_embd`, `dst_hc = (tid / n_embd) % n_hc`, `t = tid / (n_embd * n_hc)`.
- Formula:
  - `acc = block[d,t] * post[dst_hc,t]`
  - `acc += comb[dst_hc,src_hc,t] * residual[d,src_hc,t]` for each source HC.
  - `dst[d,dst_hc,t] = acc`
- Layout:
  - src0/block is `[d,t]`.
  - src1/residual is hidden-major `[d,src_hc,t]`.
  - src2/post is `[dst_hc,t]`.
  - src3/comb is consumed as `[dst_hc,src_hc,t]`.
  - output is `[d,dst_hc,t]`.

Dispatch/consumer-boundary capture:
- Added `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD_CAPTURE_MODE=consumer_dispatch`.
- Added optional flag alias `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_HC_EXPAND_DISPATCH_CAPTURE=1`.
- Capture method:
  - fixture-only intrusive `ggml_scale(x, 1.0f)` preservation for exact tensors consumed by `DSV4_HC_EXPAND`;
  - preserved tensors are used as the `DSV4_HC_EXPAND` source operands during the fixture run;
  - no live executor cutover and no executor output consumption.
- Reported policy for this mode:
  - `capture_intrusive=1`
  - `used_for_fixture_only=1`
  - `not_hot_neutral_validation=1`

Captured AOHC labels:
- `aohc_hcexpand_dispatch_src0`
- `aohc_hcexpand_dispatch_src1`
- `aohc_hcexpand_dispatch_src2`
- `aohc_hcexpand_dispatch_src3`
- `aohc_hcexpand_dispatch_output`

Post-eval vs consumer-dispatch comparison:
- Analyzer: `/tmp/dsv4_turn98_aohc_hc_expand_dispatch_semantics.txt`
- `src0 post_eval == dispatch_time`: 1
- `src1 post_eval == dispatch_time`: 1
- `src2 post_eval == dispatch_time`: 1
- `src3 post_eval == dispatch_time`: 1
- `output post_eval == dispatch_time`: 1
- First differing source: none.
- Interpretation:
  - T97 values were not stale after all once the source records were preserved/materialized as tensors with the exact consumer contract.
  - The remaining mismatch was harness/fixture loading from non-repo `/tmp` payload paths plus use of the old post-eval source labels.

AOHC CPU recompute:
- log: `/tmp/dsv4_turn98_harness_aohc_recompute.log`
- result: pass
- `copied_reference_output=0`
- `partial_aohc_recompute=1`
- `full_aohc_recompute=0`
- `aohc_attn_out_available=1`
- `aohc_residual_available=1`
- `aohc_post_weights_available=1`
- `aohc_comb_available=1`
- `aohc_reference_available=1`
- `aohc_matches_hc_pre_norm_input=1`
- `aohc_output_max_abs=1.49012e-08`
- `aohc_output_rms=2.42407e-09`
- `aohc_output_max_rel=9.0421e-05`
- Tier B: pass.
- Tier A exact: pass under the current harness Tier A max_abs/rms gates.

AOHC Metal recompute:
- log: `/tmp/dsv4_turn98_harness_aohc_metal_kernel.log`
- result: pass
- `metal_recompute=1`
- `copied_reference_output=0`
- `aohc_output_max_abs=0`
- `aohc_output_rms=0`
- `aohc_output_max_rel=0`
- Tier B: pass.
- Tier A exact: pass.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn98_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn98_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE CPU recompute regression:
  - log: `/tmp/dsv4_turn98_harness_rmoe_cpu_recompute.log`
  - result: pass
  - `rmoe_final_ffn_max_abs=1.19209e-07`
  - Tier B: pass
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn98_harness_rmoe_metal_kernel.log`
  - result: pass
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_final_ffn_max_abs=0`
  - Tier B: pass
- AOHC copy-smoke guard:
  - log: `/tmp/dsv4_turn98_harness_aohc_forbid_copy_smoke.log`
  - expected result: fail
  - reason: `copy_smoke_forbidden`

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #99 can move to the next bounded executor stage only after preserving the current regressions: HC_PRE_NORM Metal, routed-MoE CPU/Metal, AOHC CPU/Metal, identity payload, and copy-smoke guards.

## 2026-05-15 Turn #99 Harness Compressor/Update Source-Contract Recompute

Turn #98 recap:
- AOHC boundary source contract, CPU recompute, and harness-only Metal kernel pass Tier B.
- Accepted regressions before T99: HC_PRE_NORM Metal, routed-MoE Metal, AOHC Metal, and identity payload gate.

Source-contract note:
- Path: `/tmp/dsv4_turn99_compressor_update_source_contract.txt`
- Audited source sites:
  - `src/models/deepseek4.cpp`
  - `ggml/include/ggml.h`
  - `ggml/src/ggml-cpu/ops.cpp`
  - `ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ggml/src/ggml-metal/ggml-metal.metal`
- Bounded T99 formula:
  - `candidate_norm_weighted[e] = compressed_norm[e] * compressed_norm_weight[e]`
  - Compare against captured `compressed_norm_weighted`.
- Scope:
  - `partial_cupd_recompute=1`
  - `full_cupd_recompute=0`
  - `weights_not_decoded=1`
  - cache mutation disabled
  - no projection decode
  - no quant/cache row production by the harness

Capture:
- Capture directory: `/tmp/dsv4_turn99_cupd_payloads`
- Capture log: `/tmp/dsv4_turn99_capture_cupd_l2_n16.log`
- Layer/token window: layer 2, tokens 1..16.
- Fixture-only producer capture:
  - `capture_intrusive=1`
  - `used_for_fixture_only=1`
  - `not_hot_neutral_validation=1`
- Side-probe summary:
  - `stage=compressor_update`
  - `mode=source_contract`
  - `live_graph_nodes_added=0`
  - `live_backend_dispatches=0`
  - `output_consumed=0`
  - `cache_mutation=disabled`
  - `cases=72`
  - `exact_cases=72`
  - `payload_records=92`
  - `payload_blocked=4`
  - `first_compressor_layer=2`
  - `first_compressor_token=2`
  - `compressor_cases=8`
  - `ratio_boundary_cases=8`
  - `quant_emit_cases=8`

Fixture and analyzer:
- Fixture path: `tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`
- Analyzer path: `scripts/dsv4_analyze_cupd_payload_semantics.py`
- Analyzer log: `/tmp/dsv4_turn99_cupd_payload_semantics.txt`
- Analyzer summary:
  - norm inputs available: `compressed_norm`, `compressed_norm_weight`, `compressed_norm_weighted`
  - state payloads available: `state_kv`, `state_score`
  - pool payloads available: `pool_input`, `pooled_compressed_row`
  - rope payload available: `compressed_rope`
  - byte rows available: `compressed_quant_fp8_or_nfp4`, `downstream_compressed_kv`, `downstream_kv`
  - `enough_for_norm_weighted_recompute=1`
  - missing norm-weighted inputs: none

CPU recompute result:
- Log: `/tmp/dsv4_turn99_harness_cupd_recompute.log`
- Result: pass
- `recompute_possible=1`
- `copied_reference_output=0`
- `partial_cupd_recompute=1`
- `full_cupd_recompute=0`
- `weights_not_decoded=1`
- `cupd_compressed_norm_available=1`
- `cupd_norm_weight_available=1`
- `cupd_norm_weighted_available=1`
- `cupd_output_max_abs=0`
- `cupd_output_rms=0`
- `cupd_output_max_rel=0`
- Tier B: pass

Metal:
- Compressor/update Metal kernel: not implemented in Turn #99.
- Reason: T99 target was source-contract and CPU-reference validation. Kernel work can start from the bounded norm-weighted relation next.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn99_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn99_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn99_harness_rmoe_metal_kernel.log`
  - result: pass
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_final_ffn_max_abs=0`
  - Tier B: pass
- AOHC Metal regression:
  - log: `/tmp/dsv4_turn99_harness_aohc_metal_kernel.log`
  - result: pass
  - `aohc_output_max_abs=0`
  - Tier B: pass

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

## 2026-05-15 Turn #100 Harness Metal Compressor/Update Norm-Weighted Kernel

Turn #99 recap:
- Compressor/update CPU source-contract recompute passes Tier B for the bounded norm-weighted relation.
- Validated relation:
  - `candidate_norm_weighted[e] = compressed_norm[e] * compressed_norm_weight[e]`
- This is a partial compressor/update boundary only.

Metal harness kernel:
- Files:
  - `tests/dsv4_layer_executor_metal.h`
  - `tests/dsv4_layer_executor_metal.m`
  - `tests/dsv4_layer_executor_harness.cpp`
- Kernel name: `kernel_dsv4_harness_cupd_norm_weighted`
- Harness entry:
  - `--mode kernel --stage compressor_update`
- Inputs:
  - `compressed_norm`
  - `compressed_norm_weight`
- Output:
  - `candidate_norm_weighted`
- Reference:
  - `compressed_norm_weighted`
- Formula:
  - `candidate_norm_weighted[e] = compressed_norm[e] * compressed_norm_weight[e]`
- `copied_reference_output=0`
- `metal_recompute=1`
- `weights_not_decoded=1`
- `cache_mutation=disabled`

Compressor/update Metal result:
- Log: `/tmp/dsv4_turn100_harness_cupd_metal_kernel.log`
- Result: pass
- `kernel_mode=cupd_norm_weighted`
- `partial_cupd_recompute=1`
- `full_cupd_recompute=0`
- `compressed_norm_available=1`
- `compressed_norm_weight_available=1`
- `compressed_norm_weighted_reference_available=1`
- `cupd_output_max_abs=0`
- `cupd_output_rms=0`
- `cupd_output_max_rel=0`
- Tier B: pass
- Tier A exact: pass

Copy-smoke guard:
- Log: `/tmp/dsv4_turn100_harness_cupd_forbid_copy_smoke.log`
- Expected result: fail
- Reason: `copy_smoke_forbidden`

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn100_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn100_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
  - Tier B: pass
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn100_harness_rmoe_metal_kernel.log`
  - result: pass
  - `rmoe_routed_sum_max_abs=0`
  - `rmoe_final_ffn_max_abs=0`
  - Tier B: pass
- AOHC Metal regression:
  - log: `/tmp/dsv4_turn100_harness_aohc_metal_kernel.log`
  - result: pass
  - `aohc_output_max_abs=0`
  - Tier B: pass
- Compressor/update CPU recompute regression:
  - log: `/tmp/dsv4_turn100_harness_cupd_cpu_recompute.log`
  - result: pass
  - `cupd_output_max_abs=0`
  - Tier B: pass

Bounded-vs-full compressor/update gap:
- T100 validates only `compressed_norm * compressed_norm_weight -> compressed_norm_weighted`.
- Full compressor/update still requires source-contract validation and harness kernels for:
  - pooling / pooled row
  - RoPE tail
  - quantization / packed row
  - cache row metadata and write handoff
- No live graph cutover can treat T100 as full CUPD coverage.

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

## 2026-05-15 Turn #101 Harness Compressor/Update RoPE-Tail Source Contract

Turn #100 recap:
- Compressor/update norm-weighted Metal harness kernel passes Tier B.
- That remains a bounded CUPD substage only.

Source contract:
- Mapping note: `/tmp/dsv4_turn101_compressor_rope_source_contract.txt`
- Input semantic: `compressed_norm_weighted` / `cupd_rope_input`.
- Output semantic: `compressed_rope` / `cupd_rope_reference`.
- Implementation audit:
  - Generic path calls `dsv4_apply_rope_tail()` after pooling, RMS norm, and norm weight.
  - Backend path uses `GGML_OP_DSV4_DECODE_COMPRESS` with output stage 4 for rope input and stage 5 for rope output.
  - RoPE tail rotates only `i >= head_dim - n_rot`.
  - Standard layout pairs adjacent tail elements; NEOX layout pairs the two halves of the tail.
  - Position is generated by `comp_pos = pos + 1 - compress_ratio`.
  - Compressed RoPE uses the compressed RoPE frequency base; local audit records the known default `160000`.

Capture and fixture:
- Capture log: `/tmp/dsv4_turn101_capture_cupd_rope_l2_n16.log`
- Payload directory: `/tmp/dsv4_turn101_cupd_rope_payloads`
- Fixture path: `tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`
- Capture mode: `producer_capture`
- `capture_intrusive=1`
- `used_for_fixture_only=1`
- `not_hot_neutral_validation=1`
- Added explicit payload labels:
  - `cupd_norm_weighted`
  - `cupd_rope_input`
  - `cupd_rope_reference`
  - `cupd_rope_position`
  - `cupd_rope_n_rot`
  - `cupd_rope_metadata`
  - `cupd_rope_cos`
  - `cupd_rope_sin`

Analyzer:
- Analyzer path: `scripts/dsv4_analyze_cupd_payload_semantics.py`
- Log: `/tmp/dsv4_turn101_cupd_rope_payload_semantics.txt`
- Result:
  - `norm_weighted_available=1`
  - `rope_input_available=1`
  - `rope_reference_available=1`
  - `position_metadata_available=0`
  - `n_rot_metadata_available=0`
  - `cos_available=0`
  - `sin_available=0`
  - `byte_rows_available=compressed_quant_fp8_or_nfp4,downstream_compressed_kv,downstream_kv`

CPU RoPE recompute:
- Harness mode: `--mode cupd_recompute --cupd-stage rope`
- Log: `/tmp/dsv4_turn101_harness_cupd_rope_cpu_recompute.log`
- Result: pass for available position-zero source-contract records.
- `recompute_possible=1`
- `copied_reference_output=0`
- `partial_cupd_recompute=1`
- `full_cupd_recompute=0`
- Best formula: `R0_standard_tail`
- Best layout: `standard_tail`
- Best reported position: `0`
- Best `n_rot`: `64`
- `cupd_output_max_abs=0`
- `cupd_output_rms=0`
- `cupd_output_max_rel=0`
- Tier B: pass
- Metadata gap:
  - `missing_formula_params=[cupd_rope_position_metadata,cupd_rope_n_rot_metadata]`
- Nonzero-position candidate records still show sub-Tier-B mismatches without explicit metadata; this turn validates the bounded no-op/position-zero RoPE relation but does not close full nonzero RoPE-tail semantics.

Metal RoPE kernel:
- Not implemented in T101.
- Reason: the CPU pass is currently limited to position-zero/no-op records and explicit `position` / `n_rot` metadata is still unavailable. The next Metal kernel should wait for a nontrivial RoPE source-contract pass or explicit metadata capture.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn101_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn101_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn101_harness_rmoe_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=0`
- AOHC Metal regression:
  - log: `/tmp/dsv4_turn101_harness_aohc_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=0`
- Compressor/update norm-weighted Metal regression:
  - log: `/tmp/dsv4_turn101_harness_cupd_norm_weighted_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=0`

Bounded-vs-full compressor/update gap:
- T101 validates only the position-zero/no-op compressor/update RoPE source-contract records.
- Full compressor/update still requires:
  - nonzero-position RoPE metadata capture and recompute
  - pooling / pooled row validation
  - quantization / packed row validation
  - cache row metadata and write handoff validation
- No live graph cutover can treat T101 as full CUPD coverage.

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #102 should capture explicit compressor/update RoPE metadata (`position`, `n_rot`, and preferably cos/sin or op params) and rerun the nonzero-position RoPE-tail recompute before any Metal RoPE kernel.

## 2026-05-15 Turn #102 Compressor/Update RoPE Metadata Capture

Turn #101 recap:
- Compressor/update norm-weighted CPU and Metal checks passed Tier B.
- RoPE recompute only passed position-zero / no-op records.
- Explicit position and `n_rot` metadata were unavailable, so nonzero RoPE-tail semantics were not accepted.

Source contract and metadata audit:
- Mapping note: `/tmp/dsv4_turn102_cupd_rope_metadata_contract.txt`
- Exact op: `GGML_OP_DSV4_ROPE_TAIL` via `dsv4_apply_rope_tail()`.
- Source tensors: `src0=input`, `src1=I32 position`, `src2=freq_factors` when present. The compressor/update path currently uses formula-derived cos/sin from op params rather than materialized cos/sin tensors.
- Op params decoded:
  - `n_rot = 64`
  - `rope_type = 0`
  - `n_ctx_orig = 65536`
  - `freq_base = 160000`
  - `freq_scale = 0.0625`
  - `ext_factor = 1`
  - `attn_factor = 0.782926619`
  - `beta_fast = 32`
  - `beta_slow = 1`
- Tail contract:
  - `width = 128`
  - `tail_offset = 64`
  - dims before tail pass through unchanged
  - tail dims rotate using the captured position and op params

Capture and fixture:
- Capture log: `/tmp/dsv4_turn102_capture_cupd_rope_l2_n16.log`
- Payload directory: `/tmp/dsv4_turn102_cupd_rope_metadata_payloads`
- Fixture path: `tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`
- Capture mode: `producer_capture`
- `capture_intrusive=1`
- `used_for_fixture_only=1`
- `not_hot_neutral_validation=1`
- New metadata records:
  - `cupd_rope_op_params`
  - `cupd_rope_position`
  - `cupd_rope_cache_position`
  - `cupd_rope_n_rot`
  - `cupd_rope_width`
  - `cupd_rope_freq_base`
  - `cupd_rope_freq_scale`
  - `cupd_rope_mode`
  - `cupd_rope_tail_offset`
  - `cupd_rope_metadata`

Analyzer:
- Analyzer path: `scripts/dsv4_analyze_cupd_payload_semantics.py`
- Log: `/tmp/dsv4_turn102_cupd_rope_payload_semantics.txt`
- Result:
  - `rope_input_available=1`
  - `rope_reference_available=1`
  - `position_metadata_available=1`
  - `cache_position_metadata_available=1`
  - `n_rot_metadata_available=1`
  - `freq_base_available=1`
  - `freq_scale_available=1`
  - `rope_mode_available=1`
  - `tail_offset_available=1`
  - `cos_available=0`
  - `sin_available=0`
  - `cos_sin_materialized_or_formula_derived=1`
  - `nonzero_position_records_count=4`
  - `position_list=8,12,16,20`
  - `n_rot_list=64`
  - `rope_mode_list=normal`
  - `tail_offset_list=64`
  - `rope_input_equals_rope_reference=0`
  - `missing_rope_inputs=none`

CPU RoPE recompute:
- Harness mode: `--mode cupd_recompute --cupd-stage rope`
- Log: `/tmp/dsv4_turn102_harness_cupd_rope_cpu_recompute.log`
- Result: fail for nonzero-position source-contract records.
- `recompute_possible=1`
- `copied_reference_output=0`
- `partial_cupd_recompute=1`
- `full_cupd_recompute=0`
- Nonzero records tested: 4
- Best formula: `R2_neox_tail`
- Best position: `16`
- Best `n_rot`: `64`
- Best `tail_offset`: `64`
- `cupd_output_max_abs=3.06067`
- `cupd_output_rms=1.27891`
- `cupd_output_max_rel=280542`
- Tier B: fail
- Missing inputs: none.
- Remaining blocker: metadata is now available and nonzero, but captured `cupd_rope_input` / `cupd_rope_reference` still do not satisfy the audited RoPE-tail formulas. The next fix should make the compressor/update RoPE fixture stream-disambiguated and atomic around a single `GGML_OP_DSV4_ROPE_TAIL` instance so input, position/op params, and output cannot be cross-paired between attn/index streams.

Metal RoPE kernel:
- Not implemented in T102.
- Reason: CPU nonzero-position RoPE recompute did not pass Tier B.

Regression results:
- Identity payload gate:
  - log: `/tmp/dsv4_turn102_harness_identity_payloads.log`
  - result: pass
- HC_PRE_NORM Metal regression:
  - log: `/tmp/dsv4_turn102_harness_hcnorm_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=4.76837e-07`
- Routed-MoE Metal regression:
  - log: `/tmp/dsv4_turn102_harness_rmoe_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=0`
- AOHC Metal regression:
  - log: `/tmp/dsv4_turn102_harness_aohc_metal_kernel.log`
  - result: pass
  - `kernel_max_abs=0`
- Compressor/update norm-weighted Metal regression:
  - log: `/tmp/dsv4_turn102_harness_cupd_norm_weighted_metal_kernel.log`
  - result: fail
  - blocker: this deeper RoPE metadata fixture does not expose `compressed_norm`; `compressed_norm_weight` and `compressed_norm_weighted` remain available.

Bounded-vs-full compressor/update gap:
- T102 validates explicit RoPE metadata capture and proves nonzero records are present.
- T102 does not validate nonzero RoPE-tail numerics and does not add a RoPE Metal kernel.
- Full compressor/update still requires:
  - stream-stable nonzero RoPE input/reference pairing
  - pooling / pooled row validation
  - quantization / packed row validation
  - cache row metadata and write handoff validation
- Existing ggml RoPE, quant, and cache-write paths may still be used by a future hybrid orchestrator. This turn is a deeper validation path and should not be framed as proof that every existing ggml substep must be independently revalidated before an orchestrator can exist.

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #103 should stay on compressor/update RoPE and capture an atomic, stream-disambiguated `GGML_OP_DSV4_ROPE_TAIL` source-contract record for one stream: input, position tensor, op params, and output from the same op instance. It should also restore or preserve the `compressed_norm` fixture needed by the accepted norm-weighted regression.

## 2026-05-15 Turn #103 CUPD Fixture Restoration + RoPE-Tail Deferral

Turn #102 recap:
- Deeper RoPE metadata capture wrote a fixture (`tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`) that exposed `cupd_rope_*` metadata fields but no longer surfaced `compressed_norm` payload data.
- T100 CUPD norm-weighted Metal kernel regression that depended on `compressed_norm` regressed to fail (`kernel_reason=missing_required_compressor_update_metal_kernel_payloads`).
- Three consecutive turns (T101, T102, and the originally planned T103 path) had failed to close nonzero RoPE-tail Tier B from the captured input/reference pair (`max_abs=3.06` at T102), indicating the RoPE-tail recompute path is not converging under the current capture mode.

T103 scope (Option B / executor-issued corrective prompt path):
- Restore the regressed `compressed_norm` fixture without losing the T102 RoPE metadata.
- Mark CUPD RoPE-tail as orchestrator-invokes-existing — no separate harness recompute is required for an orchestrator that wraps existing `GGML_OP_DSV4_ROPE_TAIL` dispatches.
- Defer kv_cache_finalizer source-contract work and orchestrator construction to T104+ (re-scoped from full Option B per scope review).

CUPD RoPE-tail status:
- Orchestrator will invoke the existing `GGML_OP_DSV4_ROPE_TAIL` op directly via its current ggml/Metal dispatch path.
- No separate harness recompute / parallel CPU re-implementation is required to construct the planned `GGML_OP_DSV4_DECODE_LAYER` hybrid orchestrator.
- T101 and T102 RoPE mapping notes (`/tmp/dsv4_turn101_cupd_rope_metadata_contract.txt`, `/tmp/dsv4_turn102_cupd_rope_metadata_contract.txt`) are retained as reference. The bounded position-zero / no-op CUPD RoPE relation that passed at T101 remains in the fixture; the nonzero recompute path is closed out as not-required-for-orchestrator-construction.

Fixture restoration:
- Diagnosed T102 capture path bug: at tokens 2 and 6, `compressor_update_compressed_norm_weighted_l2_t{2,6}.bin` and `compressor_update_compressed_pre_norm_l2_t{2,6}.bin` in `/tmp/dsv4_turn102_cupd_rope_metadata_payloads/` are byte-identical (same MD5). The T102 intrusive RoPE capture path overwrote two distinct tensor slots with the same data at those tokens. At tokens 10 and 14 the same binaries match T99/T101 (`d9e25b…`, `27fbcc…`, `edcbae…`), so the capture path was deterministic for late tokens.
- T101's payload directory (`/tmp/dsv4_turn101_cupd_rope_payloads/`) contains a strict superset of T102's binaries (the only delta is the two missing `compressed_norm_l2_t{2,6}.bin` files) and matches T99 (`md5(t99) == md5(t101)` for `compressed_norm`, `compressed_norm_weight`, `compressed_norm_weighted`, `compressed_pre_norm`). T101 is the deterministic reference baseline.
- Built `/tmp/dsv4_turn103_cupd_norm_weighted_payloads/` as a copy of T101's payload directory (58 files) and re-ran `scripts/dsv4_normalize_layer_executor_captures.py` with `--capture compressor_update:/tmp/dsv4_turn102_capture_cupd_rope_l2_n16.log --payload-dir /tmp/dsv4_turn103_cupd_norm_weighted_payloads --out /tmp/dsv4_turn103_fixture_out/` to merge T102's RoPE metadata (from the T102 capture log) with T101's clean tensor payloads.
- Installed the merged jsonl to `tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`. Backup of the T102 jsonl saved to `/tmp/dsv4_turn103_fixture_backup_t102.jsonl`.
- New fixture: `compressed_norm` is `availability=available, payload_kind=tensor_values` at tokens 2 and 6 (the harness `find_record` returns the first available record, which is sufficient for the regression); tokens 10 and 14 remain `unavailable` because T99/T101 did not capture them. RoPE metadata (`cupd_rope_position`, `cupd_rope_n_rot`, `cupd_rope_freq_base`, etc.) is preserved at all four tokens.

Regression results (T103, post-restore):
- HC_PRE_NORM Metal kernel: pass (`kernel_max_abs=4.76837e-07`, Tier B pass) — `/tmp/dsv4_turn103_harness_hcnorm_metal_kernel.log`
- routed-MoE Metal kernel: pass (`kernel_max_abs=0`, Tier A exact) — `/tmp/dsv4_turn103_harness_rmoe_metal_kernel.log`
- AOHC Metal kernel: pass (`kernel_max_abs=0`, Tier A exact) — `/tmp/dsv4_turn103_harness_aohc_metal_kernel.log`
- CUPD norm-weighted Metal kernel: pass (`kernel_max_abs=0`, Tier A exact, `cupd_compressed_norm_available=1`, `cupd_norm_weight_available=1`, `cupd_norm_weighted_available=1`, `missing_inputs=[]`) — `/tmp/dsv4_turn103_harness_cupd_metal_kernel.log`
- Identity payload gate: pass — `/tmp/dsv4_turn103_harness_identity_payloads.log`
- Copy-smoke negative guards (all 4 stages: hc_pre_norm, routed_moe_final_output, aohc_boundary, compressor_update): correctly fail with `kernel_reason=copy_smoke_forbidden` — `/tmp/dsv4_turn103_harness_cupd_forbid_copy_smoke.log`

Bounded-vs-full compressor/update gap:
- After T103 fixture restoration, the validated bounded relation for CUPD is `compressed_norm * compressed_norm_weight -> compressed_norm_weighted` (T100 norm-weighted formula, restored).
- The position-zero / no-op CUPD RoPE bounded relation from T101 remains validated in the fixture metadata records (cos/sin not materialized; formula-derived from op params).
- Nonzero CUPD RoPE-tail recompute is **deferred indefinitely** for the orchestrator-construction path: the existing `GGML_OP_DSV4_ROPE_TAIL` op already produces correct production output, so a hybrid orchestrator can call it directly without a separate harness reimplementation.
- Remaining bounded-vs-full CUPD gaps (pooling validation, quantization validation, cache-row write validation) are similarly deferred to a "trust existing ggml/DSV4 ops via hybrid wrapper" path. They can be re-opened post-hoc if the eventual perf measurement shows the executor approach delivers signal but per-stage accuracy still drifts.

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required before cutover or production acceptance.
- Transcript-exact remains active for model runs.
- Live graph cutover: no.
- Live graph executor dispatch: no.
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no (deferred to a future turn once the orchestrator op exists).
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #104 should begin the hybrid orchestrator: add a new `GGML_OP_DSV4_DECODE_LAYER` ggml op (alongside the existing `GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN` marker) plus its helper and CPU/Metal stubs, behind `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER=1` and disabled by default in the deepseek4 graph. Wiring of the per-stage kernel dispatch sequence and layer-0 transcript-identical comparison are T105 work.

## 2026-05-15 Turn #104 Decode-Layer Orchestrator Op Skeleton

T103 recap:
- CUPD norm-weighted Metal fixture restored via a T101-base payload directory merged with T102's RoPE-metadata capture log.
- CUPD RoPE-tail formally deferred to orchestrator-invokes-existing.
- All four validated Metal kernel regressions (HC_PRE_NORM, routed_moe final-sum, AOHC HC_EXPAND, CUPD norm_weighted) plus identity gate and four copy-smoke negative guards pass.

T104 scope: ggml op skeleton only. Pure scaffolding for the hybrid wrapper orchestrator. Per the corrective prompt, no per-stage kernel dispatch is wired inside the op at T104 — that is T105's job.

GGML_OP_DSV4_DECODE_LAYER enum:
- Added at the end of the op enum, immediately after `GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN`. `GGML_OP_COUNT` bumped from 115 to 116; both static asserts in `ggml/src/ggml.c` updated.
- `GGML_OP_NAME` entry: `"DSV4_DECODE_LAYER"`. `GGML_OP_SYMBOL` entry: `"dsv4_decode_layer(x)"`.

Stage mask bits (declared in `ggml/include/ggml.h`):

```c
enum {
    DSV4_LAYER_STAGE_HC_PRE_NORM   = 1 << 0,
    DSV4_LAYER_STAGE_ROUTED_MOE    = 1 << 1,
    DSV4_LAYER_STAGE_AOHC          = 1 << 2,
    DSV4_LAYER_STAGE_CUPD          = 1 << 3,
    DSV4_LAYER_STAGE_ROPE_TAIL     = 1 << 4, // orchestrator-invokes-existing
    DSV4_LAYER_STAGE_KV_FINALIZER  = 1 << 5, // orchestrator-invokes-existing
};
```

At T104 the helper is always invoked with `stage_mask = 0` (passthrough stub). T105 will set bits to enable per-stage dispatch from inside the op.

Helper:

```c
GGML_API struct ggml_tensor * ggml_dsv4_decode_layer(
        struct ggml_context * ctx,
        struct ggml_tensor  * layer_input,
        int                   layer_index,
        uint32_t              stage_mask);
```

Op storage:
- `src[0]` = layer input tensor.
- `op_params[0]` = `layer_index` (int32).
- `op_params[1]` = `stage_mask` (uint32 reinterpreted as int32).
- `op_params[2..3]` = reserved (0 at T104, future-proof for T105+ wiring).
- Output: `ggml_dup_tensor(ctx, layer_input)` — same shape and dtype as the input.

CPU handler:
- `ggml_compute_forward_dsv4_decode_layer` in `ggml/src/ggml-cpu/ops.cpp` and declared in `ggml/src/ggml-cpu/ops.h`.
- Dispatch case added in `ggml/src/ggml-cpu/ggml-cpu.c` (`compute_forward` switch and `n_tasks = 1` switch).
- Behavior: passthrough `memcpy(dst->data, src[0]->data, ggml_nbytes(dst))` when `ith == 0`. Asserts same shape, same dtype, contiguous src and dst.

Metal handler:
- Kernel `kernel_dsv4_decode_layer_stub` added in `ggml/src/ggml-metal/ggml-metal.metal`. Body: one thread per f32 element, `dst[gid] = src0[gid]` while `gid < total_elems_f32`.
- Pipeline getter `ggml_metal_library_get_pipeline_dsv4_decode_layer` added in `ggml/src/ggml-metal/ggml-metal-device.cpp` and `.h`.
- `supports_op` case for `GGML_OP_DSV4_DECODE_LAYER` added in `ggml/src/ggml-metal/ggml-metal-device.m`: requires `op->type == GGML_TYPE_F32`, `src[0]` non-null, `src[0]->type == GGML_TYPE_F32`, contiguous src and dst, same shape.
- Kargs struct `ggml_metal_kargs_dsv4_decode_layer` added in `ggml/src/ggml-metal/ggml-metal-impl.h` (`layer_index`, `stage_mask`, `total_elems_f32`).
- Dispatch case and handler `ggml_metal_op_dsv4_decode_layer` added in `ggml/src/ggml-metal/ggml-metal-ops.cpp` and declared in `ggml/src/ggml-metal/ggml-metal-ops.h`. Uses 256-thread threadgroups, `ceil(total_elems / 256)` groups.
- Atomic counters `g_ggml_metal_dsv4_decode_layer_count` and `_trace_count` registered in the dispatch stats log (`dsv4_lexec=<n>` line) and reset path.

Flag-gated emission in `src/models/deepseek4.cpp`:
- Master flag: `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER` (any truthy value enables).
- Layer selector: `LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER_LAYERS`. Empty / `all` / `ALL` enables every decode layer. Otherwise a comma-separated list of layer indices (e.g. `0`, `0,3,7`).
- Site predicate `dsv4_experimental_decode_layer_site_enabled(il, pos, n_tokens)` mirrors the existing `dryrun_op` predicate: site fires only on decode-only steps (`n_tokens == 1`, `pos > 0`) and when the per-layer selector matches.
- Phase 3 implementation choice: **ADDITIVE**. When the site fires, an additional `ggml_dsv4_decode_layer(ctx, layer_input, il, /*stage_mask=*/0)` tensor is emitted and forced into the graph via `ggml_build_forward_expand`, but no downstream op consumes its output. The new op is a dead-end at T104. Rationale: passthrough math at T104 would diverge the layer's residual stream if it took over the layer output; ADDITIVE preserves transcript-exact for flag-ON while still proving the op infrastructure dispatches end-to-end. T105 switches to REPLACEMENT once the wrapper logic produces the correct residual.

Flag-OFF regression matrix (T104, fresh build, identical to T103 expectations):
- llama-cli build: pass.
- dsv4-layer-executor-harness build: pass.
- identity payload gate (`--require-full-tensors --require-byte-payloads`): pass — `/tmp/dsv4_turn104_harness_identity_payloads.log`.
- HC_PRE_NORM Metal: pass, `kernel_max_abs=4.76837e-07`, Tier B pass — `/tmp/dsv4_turn104_harness_hcnorm_metal_kernel.log`.
- routed-MoE Metal: pass, `kernel_max_abs=0`, Tier A exact — `/tmp/dsv4_turn104_harness_rmoe_metal_kernel.log`.
- AOHC Metal: pass, `kernel_max_abs=0`, Tier A exact — `/tmp/dsv4_turn104_harness_aohc_metal_kernel.log`.
- CUPD norm-weighted Metal: pass, `kernel_max_abs=0`, Tier A exact — `/tmp/dsv4_turn104_harness_cupd_metal_kernel.log`.
- Copy-smoke negative guards (all 4 stages): correctly fail with `kernel_reason=copy_smoke_forbidden` — `/tmp/dsv4_turn104_harness_copy_smoke_guards.log`, `/tmp/dsv4_turn104_harness_cupd_copy_smoke.log`.
- `git diff --check`: pass.
- `py_compile` on `scripts/dsv4_normalize_layer_executor_captures.py`: pass.
- markdown fences / placeholder-marker sanity on the four required docs: pass.

llama-cli sanity:
- Flag OFF (`n=8`, prompt "What is Apple Neural engine?", `--temp 0 --seed 42 --moe-mode stock --moe-topk 6 --no-warmup`): decode runs to completion, 7 tokens decoded, 21.3 t/s generation, output `|- This is an excellent question that gets to`. Stats line: `metal_dispatch=41011`, `dsv4_lexec_dryrun=0`, `dsv4_lexec=0`. Log: `/tmp/dsv4_turn104_llama_cli_flag_off_n8.log`.
- Flag ON for layer 0 (`LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER=1 LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER_LAYERS=0`, same args): decode runs to completion, 7 tokens decoded, 19.3 t/s generation, output **byte-identical to flag-OFF** (`|- This is an excellent question that gets to`). Stats line: `metal_dispatch=41018` (exactly +7 vs flag-OFF — one extra dispatch per decoded token at layer 0), `dsv4_lexec=7`. Log: `/tmp/dsv4_turn104_llama_cli_flag_on_l0_n8.log`.
- Diff of the two log files restricted to the chat-output region: identical except for the stats counters and the t/s numbers (transcript-exact preserved under ADDITIVE).

Stub semantics:
- The op is mathematically a no-op for layer flow because the output is not consumed downstream at T104 (ADDITIVE). When T105 switches to REPLACEMENT, the passthrough stub would diverge model output if invoked on a real decode layer; the wrapper logic must therefore be in place before REPLACEMENT is flipped on.

Files changed at T104:
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cpu/ops.h`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-metal/ggml-metal-impl.h`
- `ggml/src/ggml-metal/ggml-metal.metal`
- `ggml/src/ggml-metal/ggml-metal-device.h`
- `ggml/src/ggml-metal/ggml-metal-device.cpp`
- `ggml/src/ggml-metal/ggml-metal-device.m`
- `ggml/src/ggml-metal/ggml-metal-ops.h`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`
- `src/models/deepseek4.cpp`
- `docs/dsv4-m5-metal-performance-handoff.md`

Policy:
- Tier B official executor policy: yes.
- Tier A scope: same-reduction-tree / intra-graph only.
- User signoff still required.
- Transcript-exact active for model runs: yes (preserved with flag OFF and, under ADDITIVE, with flag ON).
- Live graph cutover: no (op is emitted as dead-end under ADDITIVE; REPLACEMENT deferred to T105).
- Live graph executor dispatch: no (no validated kernel calls from inside the new op yet).
- Cache mutation by harness: disabled.
- Output consumed: no.
- Performance run: no.
- Tolerance thresholds changed: no.
- Reference-copy acceptance: no.
- Drifting path accepted: no.
- Performance path accepted: no.

Next target:
- Turn #105 replaces the passthrough stub body with the actual wrapper logic. Per-stage dispatch: HC_PRE_NORM, routed_moe final-sum + shared add, AOHC HC_EXPAND, CUPD norm_weighted call the validated Metal kernels from inside the op. RoPE_TAIL, kv_finalizer, compressor pooling/quant/cache-write, and routed_moe gate/up/swiglu/down invoke existing per-layer ops directly (orchestrator-invokes-existing). T105 must produce transcript-identical output to the flag-OFF baseline (Tier A required for the wrapper). Switch from ADDITIVE to REPLACEMENT only after the wrapper produces the correct residual.
- Turn #106 enables the orchestrator for all layers and runs the same 16-token transcript-identical check.
- Turn #107 runs the n=400 paired no-logit performance measurement = FIRST PERF NUMBER.
