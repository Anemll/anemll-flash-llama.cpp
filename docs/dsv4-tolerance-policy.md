# DSV4 Tolerance Policy Draft

## Current active policy

Current model-run safety policy remains transcript-exact n400 for existing graph paths. Tier B is now the official DSV4 executor acceptance policy for cross-graph executor validation.

```text
transcript-exact n400 active: yes
Tier B executor acceptance policy adopted: yes
drifting path accepted: no
performance path accepted without exactness and speed: no
user signoff before cutover or production acceptance: required
```

No executor cutover or production acceptance is allowed until the user reviews the T84-T85 harness deltas and explicitly signs off before T86/T87 cutover decisions. There is no silent fallback to transcript-exact.

## Why transcript-exact alone is insufficient

Transcript-exact n400 is the correct active gate for current production decisions, but it is not enough to design a full decode-layer executor:

```text
1. it only validates one prompt/window unless expanded into a prompt set
2. it can reject numerically harmless reorderings before we know their statistical impact
3. it does not localize tensor, cache, or distribution-level drift
4. it does not define whether a future non-exact executor is safe enough to benchmark or ship
```

This policy defines the executor thresholds used by the standalone harness and future executor kernels. User signoff is still required before cutover or production acceptance.

## Acceptance tiers

Tier 0: active policy

```text
requirement: transcript-exact n400
status: active
drift allowed: no
```

Tier A: intra-graph, preserves reduction order

```text
requirement: strict tensor/logit/distribution thresholds pass
intended scope:
  optimizations that keep the existing reduction tree
  metadata-only validation
  exact-boundary probes
threshold profile: strict
status: active for intra-graph / same-reduction-tree optimizations
requires signoff before production acceptance: yes
```

Tier B: cross-graph executor, reduction order may differ

```text
requirement: executor tensor/logit/distribution thresholds pass
intended scope:
  GGML_OP_DSV4_DECODE_LAYER cutover
  full-layer executor replacement
  reduction order may differ from generic graph
threshold profile: executor
executor evaluated against this tier: yes
status: official executor acceptance policy
requires user signoff before cutover or production acceptance: yes
```

Tier C: performance acceptance

```text
requirement: Tier B plus user signoff plus paired no-logit speed improvement
status: draft only
requires signoff: yes
```

Tier A remains the strict gate for optimizations that keep the existing reduction tree. The DSV4 decode-layer executor will be evaluated against Tier B.

## Tensor tolerances

Per-layer tensor output thresholds:

```text
Tier A, intra-graph strict:
  f32 boundary max_abs <= 1e-6
  f32 boundary rms <= 1e-7
  f32 boundary max_rel <= 1e-5 where denominator is stable
  bit/exact-required boundary over_tol = 0

Tier B, cross-graph executor:
  fp16 layer output max_abs <= 5e-4
  fp32 layer output max_abs <= 1e-5
  rms reported for every compared tensor
  max_rel reported where denominator is stable
```

Quant/cache byte row thresholds:

```text
quant/cache byte rows: byte_exact required
cache metadata: metadata_equal required
SET_ROWS destination metadata: metadata_equal required
cache row index/count/width/position: metadata_equal required
```

## Logit tolerances

Deterministic smoke thresholds:

```text
Tier A, intra-graph strict:
  top20_overlap >= 20/20
  max_abs_logit_err <= 1e-4
  rms_logit_err <= 1e-5
  baseline top1 rank in candidate <= 1 unless top1-top2 margin < 1e-4
  candidate top1 rank in baseline <= 1 unless top1-top2 margin < 1e-4

Tier B, cross-graph executor:
  top5_overlap >= 5/5
  max_abs_logit_err <= 1e-3
  rms_logit_err reported
  baseline top1 rank in candidate reported
  candidate top1 rank in baseline reported
```

Any top1 change outside the margin exception is a hard diagnostic failure under this draft.

## Distribution tolerances

Distribution thresholds over normalized top-k probabilities:

```text
Tier A, intra-graph strict:
  KL divergence <= 1e-6
  top1 probability delta <= 1e-5
  top-k probability vector contains no NaN/Inf

Tier B, cross-graph executor:
  KL divergence <= 1e-4
  top1 probability delta reported
  top-k probability vector contains no NaN/Inf
```

The default top-k for smoke comparison is 20.

## PPL / text drift gates

Validation prompt set:

```text
T87 gate:
  transcript match rate over 10 fixed prompts x 400 tokens
  required transcript match rate: evaluated against Tier B executor acceptance

PPL future suite:
  fixed prompt count: n=1000
  relative PPL drift <= 0.1%
  status: future suite, not a T87 gate until constructed

no NaN/Inf in logits or probabilities
no tokenizer mismatch
no prompt truncation mismatch
```

Text drift gate:

```text
deterministic decode seed fixed
temperature fixed at 0 for smoke runs
prompt set transcript changes require tolerance-policy signoff before performance acceptance
PPL is not a T87 gate until the n=1000 suite is constructed
```

## Hard rejects

Hard reject conditions:

```text
NaN/Inf
cache metadata mismatch
byte mismatch in quant/cache rows
missing hot-neutral metadata
extra live graph branch in validation mode
unbounded all-layer consume before layer-0 gates pass
transcript drift outside Tier B executor policy
fallback branch emitted in the same production graph
parallel production executor branch
```

## Required reports

Every tolerance-based candidate report must include:

```text
tensor max_abs
tensor rms
tensor max_rel where denominator is stable
over_tol count
first_bad_index
cache byte_exact result
cache metadata_equal result
logit top20 overlap for Tier A
logit top5 overlap for Tier B
max_abs_logit_err
rms_logit_err
top1 rank cross-check
top1-top2 margin
KL divergence
top1 probability delta
transcript match rate over 10 fixed prompts x 400 tokens
PPL relative drift on n=1000 prompt set only after the future suite is constructed
hot-neutral metadata
graph branch audit
```

## Signoff status

```text
Tier B executor acceptance policy adopted: yes.
Signed off for cutover / production acceptance: no.
Active model-run safety policy remains transcript-exact n400 for existing graph paths.
Review trigger: user reviews tolerance numbers after T84-T85 harness deltas are visible, before T86 cutover.
Silent fallback to transcript-exact after T84-T85 deltas: no.
```
