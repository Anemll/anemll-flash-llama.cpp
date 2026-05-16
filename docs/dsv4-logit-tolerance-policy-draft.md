# DSV4 Logit / Tolerance Policy Draft

Status: draft only, not adopted.

Current active policy:
deterministic n400 transcript-exact validation remains the acceptance gate.

This document does not accept any drifting optimization path.

Accepted today:

```text
HC_PRE_NORM
```

Not accepted:

```text
FFN/MoE V2 full consume
CUPD2_FUSED_COMP
DOWN_SUM6
weighted/shared SwiGLU
mixed-attention variants
```

## Motivation

Hot-path-neutral n400 validation showed that some rejected paths diverge with high top-k overlap and modest logit deltas. This raises a policy question:

```text
Are these paths numerically unsafe,
or are they acceptable whole-stage DS4-style numerical variants?
```

The answer is not automatic. A tolerance policy would be an explicit owner decision, not a side effect of a faster run or a close-looking transcript.

Turn #5 evidence:

```text
Hot-neutral baseline A/B:
  no divergence through 400 records
  max_abs = 0
  rms = 0
  metal_dispatch unchanged at 1399039

FFN/MoE V2:
  first divergence token 104, position 114, " and" vs " ("
  top10/top20 overlap 10/10, 20/20
  max_abs = 0.422439575
  rms = 0.150950382

CUPD2_FUSED_COMP:
  first divergence token 104, position 114, " and" vs " ("
  top10/top20 overlap 10/10, 20/20
  max_abs = 0.402101517
  rms = 0.128725288

DOWN_SUM6:
  first divergence token 104, position 114, " and" vs " ("
  top10/top20 overlap 10/10, 20/20
  max_abs = 0.1621418
  rms = 0.0774944226
```

## Non-Negotiable Validation Requirements

Any tolerance-based candidate must still pass all of these requirements:

```text
build passes
git diff --check passes
hot-path-neutral baseline A/B passes
no intrusive validation flags
no graph readbacks
no layer executor shadow/compare/consume
no stage profiler
no multiple rejected paths stacked
exactly one under-test path declared
under-test metadata says path_accepted=false
n400 completed
no crash / OOM
no malformed JSONL
first-divergence analyzer passes
```

Any candidate that passes logit-policy review must then run a normal no-logit performance smoke:

```text
logit-dump runs are validation runs, not performance baselines
```

## Required Metrics

Every candidate report must include:

```text
first divergent token index
position
baseline token id/text
under-test token id/text

baseline top1 id/logit
baseline top2 id/logit
baseline top1-top2 margin

under-test top1 id/logit
under-test top2 id/logit
under-test top1-top2 margin

baseline top1 rank in under-test distribution
under-test top1 rank in baseline distribution

top10 overlap
top20 overlap

max_abs logit error
mean_abs logit error
rms logit error

whether error grows over time
whether divergence is early or late
active counters
active experimental flags
metal_dispatch
```

Semantic smoke is also required:

```text
n400 text must not collapse, loop, emit garbage, or ignore the prompt
```

Semantic smoke alone is not sufficient.

## Proposed Policy Tiers

These tiers are proposed only. They are not active acceptance criteria.

### Tier 0: Transcript-Exact Accepted

Criteria:

```text
tokens match through n400
logit errors zero or negligible
normal no-logit perf smoke passes
```

Status:

```text
accepted under current policy
```

### Tier 1: Tolerance Candidate

Use this tier for close paths like the turn #5 results.

Proposed criteria:

```text
first divergence at token >= 80
top20 overlap >= 19/20 at first divergence
top10 overlap >= 9/10 at first divergence
baseline top1 rank in under-test <= 3
under-test top1 rank in baseline <= 3
rms top20-overlap logit error <= 0.20
max_abs top20-overlap logit error <= 0.60
no monotonic growth to large errors before divergence
no output collapse in n400
```

Status:

```text
candidate for owner approval only
not accepted by default
```

### Tier 2: Tolerance Reject

Proposed reject criteria:

```text
first divergence before token 40
top20 overlap < 18/20
top10 overlap < 8/10
baseline top1 rank in under-test > 5
under-test top1 rank in baseline > 5
rms top20-overlap logit error > 0.35
max_abs top20-overlap logit error > 1.0
visible output collapse / repetition / garbage
```

Status:

```text
rejected
```

### Tier 3: Hard Reject

Always reject:

```text
OOM
crash
invalid tokens
NaNs/Infs
multiple rejected paths stacked
missing metadata
non-hot-path-neutral validation
path accepted flag set by the under-test run
cache mutation outside an accepted path
stage profiler active
layer executor shadow/readback active
```

## Current Candidate Classification Under This Draft

| Path | First divergence | Top10 | Top20 | Max abs | RMS | Draft tier | Current status |
|---|---:|---:|---:|---:|---:|---|---|
| FFN/MoE V2 full consume | 104 | 10/10 | 20/20 | 0.422439575 | 0.150950382 | Tier 1 candidate | rejected under current transcript-exact policy |
| CUPD2_FUSED_COMP | 104 | 10/10 | 20/20 | 0.402101517 | 0.128725288 | Tier 1 candidate | rejected under current transcript-exact policy |
| DOWN_SUM6 | 104 | 10/10 | 20/20 | 0.1621418 | 0.0774944226 | Tier 1 candidate | rejected under current transcript-exact policy |

These classifications are draft-only. They do not enable or accept these paths.

## 2026-05-13 Turn #7 No-Logit Performance Reality Check

Normal no-logit n400 performance smokes were run for the Tier 1 draft tolerance candidates. These runs did not use logit dump, hot-path-neutral validation, layer-executor flags, stage profiling, graph probes, or stacked rejected paths.

Baseline:

```text
baseline A:
  log = /tmp/dsv4_turn7_tolerance_perf/baseline_A_n400.log
  generation = 21.0 tok/s
  metal_dispatch = 1399039

baseline B:
  log = /tmp/dsv4_turn7_tolerance_perf/baseline_B_n400.log
  generation = 21.4 tok/s
  metal_dispatch = 1399039

baseline average = 21.2 tok/s
```

Candidate logs:

```text
DOWN_SUM6:
  log = /tmp/dsv4_turn7_tolerance_perf/down_sum6_n400.log
  generation = 21.5 tok/s
  metal_dispatch = 1391857
  dsum6 = 1197
  dec_mv = 0

CUPD2_FUSED_COMP:
  log = /tmp/dsv4_turn7_tolerance_perf/cupd2_fused_comp_n400.log
  generation = 21.8 tok/s
  metal_dispatch = 1249275
  dsv4_cupd2 = 24738
  rerun log = /tmp/dsv4_turn7_tolerance_perf/cupd2_fused_comp_rerun_n400.log
  rerun generation = 21.2 tok/s

FFN/MoE V2 full consume:
  log = /tmp/dsv4_turn7_tolerance_perf/ffnmoe_v2_full_consume_n400.log
  generation = 20.7 tok/s
  metal_dispatch = 1374301
  dsv4_ffnmoe = 34314
  dec_mv = 0
  pair = 0
  pswiglu = 0
  fglu = 0
```

Decision table:

| Path | Hot-neutral tier | Normal n400 tok/s | Baseline avg tok/s | Delta | Transcript-exact? | Current status | Tolerance discussion value |
|---|---|---:|---:|---:|---|---|---|
| DOWN_SUM6 | Tier 1 candidate | 21.5 | 21.2 | +0.3 | no | rejected | Weak; small one-run uplift only |
| CUPD2_FUSED_COMP | Tier 1 candidate | 21.8, rerun 21.2 | 21.2 | +0.6, rerun 0.0 | no | rejected | Weak/unstable; first run faster, rerun at baseline |
| FFN/MoE V2 full consume | Tier 1 candidate | 20.7 | 21.2 | -0.5 | no | rejected | None; slower than baseline |

Interpretation:

```text
If a path is not faster in normal no-logit runs, there is no practical reason to relax transcript-exact policy for it.
If a path is materially faster, it may justify owner review under the draft tolerance policy, but it remains rejected until explicitly approved.
```

Turn #7 conclusion:

```text
No Tier 1 tolerance candidate showed a stable, material no-logit speedup.
CUPD2_FUSED_COMP had one faster run, but the required rerun fell to baseline.
DOWN_SUM6 was only slightly faster and not enough to justify policy relaxation by itself.
FFN/MoE V2 was slower.
```

## Proposed Approval Workflow

1. Candidate passes hot-path-neutral n400 logit validation.
2. Candidate is classified as Tier 1.
3. Owner explicitly approves tolerance-based validation for that named path.
4. Candidate gets a dedicated acceptance section in the handoff.
5. Candidate is rerun with normal no-logit performance smoke.
6. Candidate is compared against transcript-exact baseline for quality.
7. Only then can it be marked `accepted-under-tolerance`.

The accepted label must be explicit:

```text
accepted-under-tolerance
```

It must not be collapsed into:

```text
accepted
```

## What Is Still Not Allowed

No path may be accepted because:

```text
n80 passed
dispatch dropped
tok/s improved but n400 drifted
generated text looked okay
top-k overlap was high in one run only
multiple rejected paths together seemed faster
```

No tolerance policy should be applied to cache-mutating paths without separate cache correctness proof.

## Recommended Next Engineering Step After Policy Draft

If transcript-exact remains required:

```text
do not revisit FFN/MoE V2, CUPD2_FUSED_COMP, or DOWN_SUM6 as performance paths
move to DS4-style whole-stage executor design or external DS4 comparison
```

If tolerance policy is adopted:

```text
first candidate should be DOWN_SUM6 or CUPD2_FUSED_COMP only after normal no-logit perf smoke
do not stack multiple tolerance candidates until each is individually accepted
```

Draft ordering for a future owner decision:

```text
1. DOWN_SUM6, because it has the smallest hot-neutral logit error.
2. CUPD2_FUSED_COMP.
3. FFN/MoE V2.
```

This is a draft recommendation, not an acceptance decision.
