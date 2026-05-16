# Flash-MoE Multi-SSD Sidecar Handoff

Last updated: 2026-05-11

## Goal

Understand why Flash-MoE sidecar reads remain the main blocker even with multiple high-end SSDs, and establish what actually worked for DeepSeek V4 Flash sidecar decode on M5 Max.

Dense model must stay on the internal SSD:

```text
/Users/anemll/Models/DeepSeek-V4-Flash-SSD/dense/model-dense.gguf
```

Sidecar paths used in the tests:

```text
Internal family-major: /Users/anemll/Models/DeepSeek-V4-Flash-SSD/sidecar
Internal expert-major: /Users/anemll/Models/DeepSeek-V4-Flash-SSD/sidecar_emj
Optane family-major:   /Volumes/optane/DeepSeek-V4-Flash-SSD/sidecar
Optane expert-major:   /Volumes/optane/DS_EMJ
```

## Executive Summary

Multiple SSDs did not meaningfully increase decode speed because the workload is not a simple aggregate-bandwidth problem. Decode sidecar demand is a dependency-heavy sequence of routed expert choices, small random-ish expert reads, slot-bank install/upload, and GPU execution. The internal SSD wins most individual expert reads in the measured workload, while Optane wins only a small minority.

The major win was changing sidecar layout to expert-major. That reduced read fragmentation by making a routed expert load one contiguous expert record instead of separate family slices. Multi-drive striping, whole-expert distribution, concurrent racing, and oracle replay were neutral or slower.

The practical rule right now:

- Use internal SSD for dense and default demand sidecar reads.
- Use expert-major sidecars everywhere.
- Avoid always-racing internal vs Optane.
- Consider Optane only for selectively predicted hard random misses, not as a blanket second lane.

## First Principles

Raw SSD bandwidth is not the active bottleneck in decode. The hot path is closer to:

```text
top-k route -> resolve slots -> read missing experts -> stage bytes -> upload/install slot -> GPU graph uses slot -> next layer
```

This has several consequences:

- The model cannot stream huge contiguous files during decode. It asks for routed experts selected layer-by-layer.
- A single token has many small dependent reads, not one big sequential read.
- Cache hits remove most demand reads, so the remaining misses are sparse and random.
- Racing two drives doubles physical read work. It only wins if the secondary drive wins often enough to offset duplicate reads.
- Striping a small expert record can force the request to wait for the slowest lane and can destroy local sequentiality.
- Upload/install and Metal execution still sit on the critical path after bytes arrive.

In short: multiple drives only help if the schedule can issue independent, high-confidence reads to each drive without waiting on duplicated or split subreads. Our current demand path rarely has that shape.

## Sidecar Layout Ground Truth

Expert-major sidecar export produced:

```text
model.sidecar_layout = expert-major
layout               = layer_major_expert
entries              = 129
expert_stride        = 13369344
bytes_per_expert     = 4456448
family offsets       = 0 / 4456448 / 8912896 for gate / up / down
layer file size      = 3422552064
total sidecar        = about 137 GiB
```

Why this matters:

- Family-major layout stores all gate experts, all up experts, all down experts as separate regions.
- A routed expert load becomes three separate logical ranges: gate, up, down.
- Expert-major interleaves `gate/up/down` for the same expert, so the runtime can load one contiguous expert record.

Representative result:

| Layout | Drive | Decode | Source Wall | pread ops | Note |
| --- | --- | ---: | ---: | ---: | --- |
| family-major | internal | `6.8 t/s` | `113.297s` | `432192` | old baseline |
| expert-major | internal | `7.3 t/s` | `8.892s` | `144064` | large I/O reduction |

The exact logs for these two early layout checks were not preserved under a single named `/tmp` file, but the result was observed before the multi-drive sweep and drove the expert-major tests below.

## Multi-Drive Strategy Results

The clean no-prefetch sweep used 399 decode tokens, the same prompt, and internal dense model.

Raw logs:

```text
/tmp/flashmoe_optane_noprefetch_internal.log
/tmp/flashmoe_optane_noprefetch_optane.log
/tmp/flashmoe_optane_noprefetch_lastmiss.log
/tmp/flashmoe_optane_noprefetch_dist_7_1.log
/tmp/flashmoe_optane_noprefetch_dist_3_1.log
/tmp/flashmoe_optane_noprefetch_dist_1_1.log
/tmp/flashmoe_optane_noprefetch_stripe_3_1.log
/tmp/flashmoe_optane_noprefetch_concurrent.log
```

| Strategy | Decode | Source Wall | Install | Key Result |
| --- | ---: | ---: | ---: | --- |
| internal only | `8.8 t/s` | `11.569s` | `18.521s` | best simple baseline |
| Optane only | `8.2 t/s` | `17.555s` | `24.520s` | slower |
| secondary last-miss | `8.8 t/s` | `12.310s` | `19.273s` | roughly neutral |
| distribute `7:1` | `8.8 t/s` | `11.628s` | `18.623s` | roughly neutral |
| distribute `3:1` | `8.7 t/s` | `12.369s` | `19.360s` | slower |
| distribute `1:1` | `8.6 t/s` | `13.697s` | `20.668s` | slower |
| stripe `3:1` | `8.5 t/s` | `23.256s` | `20.816s` | bad |
| concurrent race | `7.6 t/s` | `19.830s` | `28.505s` | bad; Optane won only `497/9839 = 5.05%` |

Concurrent race breakdown from `/tmp/flashmoe_optane_noprefetch_concurrent.log`:

```text
Flash-MoE concurrent demand races=9839 primary-win=9342 secondary-win=497
miss1=2.3%(108/4785), miss2=4.4%(118/2690), miss3=11.0%(177/1608), miss4=12.4%(94/756)
cold=2.2%(89/3992), evict=7.0%(408/5847)
top-layers L0=19.2%(126/655), L2=17.4%(109/627), L1=15.1%(96/634)
```

Interpretation:

- Internal wins the overwhelming majority of demand reads.
- Optane looks better on larger miss groups and early layers, but the global win rate is too low.
- Racing doubles read work to get about 5% secondary wins, so it loses.
- Striping is especially poor because it turns one logical expert into a wait-for-all-lanes operation.

## TB5 Optane Race Results

After moving Optane to TB5, concurrent race improved only modestly.

Raw logs:

```text
/tmp/flashmoe_tb5_concurrent_2000.log
/tmp/flashmoe_tb5_concurrent_space_2000.log
/tmp/flashmoe_tb5_concurrent_oracle_800.log
```

| Run | Decode Tokens | Decode | Races | Optane Wins | Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| short TB5 concurrent | 520 | `7.1 t/s` | 11463 | 1076 | `9.39%` |
| Space Invaders prompt | 1999 | `7.5 t/s` | 38183 | 2039 | `5.34%` |
| instrumented oracle-ish | 799 | `7.3 t/s` | 15944 | 1021 | `6.40%` |

Instrumented 799-token breakdown:

```text
miss1=3.1%(257/8393), miss2=5.9%(244/4122), miss3=14.6%(353/2424), miss4=16.6%(167/1005)
cold=3.5%(141/3992), evict=7.4%(880/11952)
single=3.1%(257/8393), adjacent=14.0%(13/93), gap<=4=15.9%(44/277), gap<=16=10.3%(95/923), gap>16=9.8%(612/6258)
top-layers L0=22.5%(251/1116), L1=19.8%(211/1068), L2=19.1%(198/1036)
```

Takeaway: TB5 made Optane more plausible, but not enough. The total race win rate stayed low, and the race overhead still outweighed the benefit.

## Prefetch-On Strategy Results

With `--moe-prefetch-temporal`, demand misses become rarer and more random. This is the condition where Optane should theoretically help. It still did not produce a clear win.

Raw logs:

```text
/tmp/flashmoe_optane_strategy_internal.log
/tmp/flashmoe_optane_strategy_optane.log
/tmp/flashmoe_optane_strategy_lastmiss.log
/tmp/flashmoe_optane_strategy_dist_7_1.log
```

| Strategy | Decode Tokens | Decode | Source Wall | Install |
| --- | ---: | ---: | ---: | ---: |
| internal only | 799 | `8.8 t/s` | `18.142s` | `28.768s` |
| Optane only | 799 | `8.5 t/s` | `26.021s` | `36.273s` |
| secondary last-miss | 799 | `9.0 t/s` | `18.905s` | `29.114s` |
| distribute `7:1` | 657 | `9.1 t/s` | `15.440s` | `24.366s` |

The `dist_7_1` run stopped at 657 decode tokens, so do not overinterpret its `9.1 t/s`. It is directionally consistent with “tiny Optane share is not terrible,” but not proof of improvement.

## Oracle Replay

Oracle replay tested whether a perfect history of race winners could beat internal-only routing by replaying the winning lane without racing.

Raw logs:

```text
/tmp/flashmoe_oracle_record_noprefetch_400.log
/tmp/flashmoe_oracle_replay_noprefetch_400.log
/tmp/flashmoe_oracle_record_prefetch_800.log
/tmp/flashmoe_oracle_replay_prefetch_800.log
```

No-prefetch, 399 decode:

```text
race record: secondary-rate=5.99%, generation=7.2 t/s
oracle replay: secondary-rate=5.99%, generation=8.5 t/s
internal-only baseline: generation=8.8 t/s
```

Prefetch-on, 799 decode:

```text
race record: secondary-rate=6.16%, generation=7.6 t/s
oracle replay: secondary-rate=6.16%, generation=8.8 t/s
internal-only baseline: generation=8.8 t/s
```

Interpretation:

- Perfect replay of observed race winners did not beat internal-only.
- Race winner identity is not a stable standalone routing oracle.
- The secondary win rate is too low to justify heavy Optane use.

## What Worked

### Expert-major sidecar layout

This is the highest-confidence I/O improvement.

Code:

```text
tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py:811
tools/flashmoe-sidecar/flashmoe_sidecar.py:423
src/llama-model.cpp:617
src/llama-context.cpp:8377
src/llama-context.cpp:8444
```

Important runtime logic:

```text
src/llama-context.cpp:8377  can_single_read_expert_major(...)
src/llama-context.cpp:8444  load_expert_major_slot(...)
src/llama-context.cpp:8464  read_sidecar_bytes_concurrent_contiguous(...) when racing expert-major records
```

Expert-major changes the read unit from separate family tensors to one contiguous expert record. This directly attacks the `gate/up/down` fragmentation problem.

### Small Optane share can be neutral

Whole-expert distribution with a small Optane share, such as `7:1`, was close to internal-only. This suggests Optane is not useless, but the useful window is narrow.

### Instrumentation

The concurrent-race instrumentation is useful even though always-race is not.

Code:

```text
src/llama-context.cpp:6972  concurrent demand summary
src/llama-context.cpp:7079  demand oracle record summary
src/llama-context.cpp:7089  demand oracle replay summary
```

It provides exactly the features we need to build a selective policy:

- miss count
- cold vs eviction
- reuse age
- expert-id gap
- layer

## What Did Not Work

### Optane-only demand

Optane-only was slower than internal-only in both no-prefetch and prefetch-on tests.

### Striping

Demand striping performed poorly. It is the wrong shape for this workload because it splits small expert reads and waits for all parts.

Relevant code and flags:

```text
common/arg.cpp: demand stripe / distribute / concurrent flags around the Flash-MoE CLI options
src/llama-model.cpp: demand stripe/distribute/concurrent parameter plumbing
src/llama-context.cpp: read lane selection and install paths
```

### Always-race

Always-race is currently a diagnostic mode, not a speed mode.

Code:

```text
include/llama.h:396       moe_demand_concurrent
common/arg.cpp:2432       --moe-demand-concurrent
src/llama-model.cpp:805   model stores demand-concurrent enabled
src/llama-context.cpp:7735 read_expert_bytes_concurrent(...)
src/llama-context.cpp:7790 read_sidecar_bytes_concurrent_contiguous(...)
```

Earlier implementation problems were fixed:

- no more per-expert `std::async` thread storm
- no more busy-spin `wait_for(0ms) + yield`
- no more unbounded orphan loser reads
- pooled read buffers to reduce heap churn

But the first-principles issue remains: if Optane only wins about 5-9% of races, racing almost every demand miss is too expensive.

## Related Runtime Fixes From This Pass

These are not multi-SSD wins by themselves, but they remove noise and correctness hazards while testing.

### Fit probe reset

The `--fit` probe now clears multi-sidecar and predictor knobs so the probe does not load irrelevant sidecars or predictors.

```text
common/common.cpp:1128
common/common.cpp:1129
```

### Manifest validation

Expert-major manifest validation now checks the computed final byte range against actual file size.

```text
src/llama-model.cpp:653
src/llama-model.cpp:684
```

### Atomic sidecar/export writes

Sidecar tools now use temp files plus `os.replace(...)` for layer files and manifests.

```text
tools/flashmoe-sidecar/flashmoe_sidecar.py:49
tools/flashmoe-sidecar/export_dsv4_hf_flashssd.py:42
```

### Hidden predictor cleanup

The hidden predictor no longer silently forces Metal readback by default. If `attn_norm` is not CPU-visible, it logs once and skips. If readback is explicitly enabled, scoring is now async on a worker thread.

Code:

```text
src/llama-context.cpp:1809 warning path
src/llama-context.cpp:2004 hidden predictor task structures
src/llama-context.cpp:5675 async scoring
src/llama-context.cpp:5805 harvest completed results
src/llama-context.cpp:7176 async predictor summary
```

Smoke logs:

```text
/tmp/flashmoe_predictor_async_default.log
/tmp/flashmoe_predictor_async_readback_solo.log
/tmp/flashmoe_predictor_badpath.log
```

Observed:

```text
default Metal path:
  warning_count=1
  hidden predictor async submitted=0 completed=0 applied=0 late=0 dropped=0 pending=0

explicit readback:
  hidden predictor async submitted=46 completed=46 applied=46 late=0 dropped=0 pending=0 threads=1
  Flash-MoE predict-hidden-attn calls=46 uniq=184 hit=97.8% miss=4
```

## Current Hypothesis

The blocker is not “we do not have enough SSD bandwidth.” The blocker is that the decode-sidecar access pattern does not expose enough independent, predictable, high-throughput work to the drives.

Internal SSD is probably winning because:

- It has excellent small-read performance in this machine.
- It benefits from unified local path, caching, and lower software overhead.
- Near-sequential expert accesses are common enough that internal retains an advantage.
- Optane’s random-read advantage appears only in a subset: higher miss count, evictions, some early layers, and some gap buckets.

Multiple drives can only help when:

- reads are independent,
- each drive gets whole expert records,
- requests do not wait on slower sibling pieces,
- we avoid duplicate reads,
- and the selector predicts the faster lane with high confidence.

Our current multi-drive approaches fail one or more of those conditions.

## Recommended Next Experiments

1. Keep expert-major as mandatory.

2. Build a selective Optane policy instead of always-race:

```text
Use Optane only when:
  miss_count >= 3
  or load_kind == eviction
  or layer in {0, 1, 2}
  or expert-gap bucket is strongly random
```

3. Validate selective policy against internal-only, not against always-race.

4. Keep recording race diagnostics for short calibration runs, but do not use racing as the production strategy.

5. Add a per-layer/per-bucket policy table generated from race logs:

```text
layer, miss_count, load_kind, gap_bucket -> preferred lane
```

6. Test whether larger prefetch windows can issue Optane reads early enough to hide its latency. If the read is not on the critical path, Optane’s lower random latency may matter less than avoiding internal-drive interference.

7. Re-test with long decode lengths only after the policy is selective. Short runs are noisy, but the previous 1999-token race still showed only `5.34%` Optane wins.

## Representative Command Shape

Internal expert-major baseline:

```sh
./build/bin/llama-cli --perf \
  -m /Users/anemll/Models/DeepSeek-V4-Flash-SSD/dense/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/DeepSeek-V4-Flash-SSD/sidecar_emj \
  --moe-slot-bank 96 \
  --moe-topk 4 \
  --moe-cache-io-split 16 \
  -fit on -ngl 999 -c 2048 -b 512 -ub 1 \
  --no-warmup --seed 123 --temp 0 --no-display-prompt --simple-io \
  -p "What is Apple Neural Engine?" \
  -n 400 -st
```

Concurrent diagnostic race:

```sh
./build/bin/llama-cli --perf \
  -m /Users/anemll/Models/DeepSeek-V4-Flash-SSD/dense/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar /Users/anemll/Models/DeepSeek-V4-Flash-SSD/sidecar_emj \
  --moe-secondary-sidecar /Volumes/optane/DS_EMJ \
  --moe-demand-concurrent \
  --moe-slot-bank 96 \
  --moe-topk 4 \
  --moe-cache-io-split 16 \
  -fit on -ngl 999 -c 2048 -b 512 -ub 1 \
  --no-warmup --seed 123 --temp 0 --no-display-prompt --simple-io \
  -p "What is Apple Neural Engine?" \
  -n 2000 -st
```

Oracle record/replay env knobs:

```sh
LLAMA_FLASH_MOE_DEMAND_ORACLE_RECORD=/tmp/flashmoe_demand_oracle.seq
LLAMA_FLASH_MOE_DEMAND_ORACLE_REPLAY=/tmp/flashmoe_demand_oracle.seq
```

## Bottom Line

The thing that was wrong in the original approach was treating multiple SSDs like additive bandwidth lanes. Flash-MoE decode does not naturally consume storage that way. It consumes small, layer-dependent, routed expert records with install/upload work after each miss. Expert-major layout fixed a real structural inefficiency. Multi-drive only becomes useful if we stop splitting or duplicating reads and instead use Optane selectively where the measured probability of winning is high enough.
