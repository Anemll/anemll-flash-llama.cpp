# Session Export: DS-Flash Metal Optimization

Date: 2026-05-15

Workspace:

`/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-flash-moe/anemll-flash-llama.cpp`

## Scope

This export captures the visible/current session state from the DS-Flash Metal optimization loop through Turn #102. It is not a raw platform transcript dump; Codex does not have direct access to the complete hidden chat log as a file. This file preserves the actionable state, constraints, results, and next target.

## Active Policy

- Tier B is the official executor acceptance policy.
- Tier A is only for same-reduction-tree / intra-graph optimizations.
- Transcript-exact remains active for model runs until user signoff cutover.
- User signoff is required before production acceptance.
- Drifting paths accepted: no.
- Performance path accepted: no.
- Do not widen tolerance thresholds.
- Do not accept copy/reference-output paths as real executor computation.
- Do not add live graph dispatch or cutover before the harness path is validated.

## Accepted Harness Kernels

- Turn #93: `HC_PRE_NORM` Metal harness kernel passes Tier B.
- Turn #95: `routed_moe_final_output` Metal harness kernel passes Tier B.
- Turn #98: `aohc_boundary` Metal harness kernel passes Tier B.
- Turn #100: compressor/update `norm_weighted` Metal harness kernel passes Tier B.

## Turn #101 Status

Turn #101 was accepted as a precise blocker, not RoPE-tail coverage.

Findings:

- CUPD `norm_weighted` CPU recompute passes Tier B.
- CUPD `norm_weighted` Metal kernel passes Tier B.
- RoPE payloads were captured.
- Position metadata unavailable.
- `n_rot` metadata unavailable.
- cos/sin unavailable.
- CPU RoPE recompute passed only position-zero / no-op records.
- Nonzero RoPE-tail semantics were not validated.

## Turn #102 Result

Final reply:

`Turn #102 completed. Accepted: no.`

### Files Changed

- `src/models/deepseek4.cpp`
- `scripts/dsv4_normalize_layer_executor_captures.py`
- `scripts/dsv4_analyze_cupd_payload_semantics.py`
- `tests/dsv4_layer_executor_harness.cpp`
- `tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`
- `docs/dsv4-m5-metal-performance-handoff.md`

### Mapping Note

`/tmp/dsv4_turn102_cupd_rope_metadata_contract.txt`

### Capture

Capture log:

`/tmp/dsv4_turn102_capture_cupd_rope_l2_n16.log`

Payload directory:

`/tmp/dsv4_turn102_cupd_rope_metadata_payloads`

Fixture:

`tests/fixtures/dsv4_layer_executor/compressor_update_l2_n16.jsonl`

Capture mode:

- `producer_capture`
- `capture_intrusive=1`
- `used_for_fixture_only=1`
- `not_hot_neutral_validation=1`

### RoPE Metadata Captured

- position metadata available: yes
- cache position metadata available: yes
- `n_rot` metadata available: yes
- freq base / freq scale available: yes
- rope mode: `normal`
- tail offset: `64`
- nonzero position records: `4`
- positions: `8,12,16,20`
- `n_rot`: `64`
- `freq_base`: `160000`
- `freq_scale`: `0.0625`
- cos/sin source: formula-derived from op params, not materialized tensors

### RoPE Recompute

Harness command:

```bash
./build/bin/dsv4-layer-executor-harness \
  --fixtures tests/fixtures/dsv4_layer_executor \
  --mode cupd_recompute \
  --cupd-stage rope \
  --require-full-tensors \
  2>&1 | tee /tmp/dsv4_turn102_harness_cupd_rope_cpu_recompute.log
```

Result:

- `recompute_possible=1`
- `copied_reference_output=0`
- best formula: `R2_neox_tail`
- best record: token `10`, position `16`, `n_rot=64`, `tail_offset=64`
- max_abs/rms/max_rel: `3.06067 / 1.27891 / 280542`
- Tier B pass: no
- Metal RoPE kernel added: no

Blocker:

Metadata is now present and nonzero, but captured `cupd_rope_input` and `cupd_rope_reference` still do not satisfy the audited RoPE-tail formulas. The likely next fix is atomic, stream-disambiguated capture of one `GGML_OP_DSV4_ROPE_TAIL` instance: input, position tensor, op params, and output from the same attn/index stream.

### Regression Results

- `llama-cli` build: pass
- harness build: pass
- identity payload: pass
  - `/tmp/dsv4_turn102_harness_identity_payloads.log`
- `HC_PRE_NORM` kernel regression: pass
  - `/tmp/dsv4_turn102_harness_hcnorm_metal_kernel.log`
  - max_abs `4.76837e-07`
- routed-MoE kernel regression: pass
  - `/tmp/dsv4_turn102_harness_rmoe_metal_kernel.log`
  - max_abs `0`
- AOHC kernel regression: pass
  - `/tmp/dsv4_turn102_harness_aohc_metal_kernel.log`
  - max_abs `0`
- CUPD `norm_weighted` kernel regression: fail
  - `/tmp/dsv4_turn102_harness_cupd_norm_weighted_metal_kernel.log`
  - missing input: `compressed_norm`
  - reason: the deeper RoPE metadata fixture exposes `compressed_norm_weight` and `compressed_norm_weighted`, but not `compressed_norm`.
- CUPD RoPE analyzer: pass
- CUPD RoPE CPU recompute: fail Tier B
- `git diff --check`: pass
- `py_compile`: pass
- markdown / no-`TBD` sanity: pass

## Next Target

Turn #103 should stay on compressor/update RoPE.

Required next work:

- Capture an atomic, stream-disambiguated `GGML_OP_DSV4_ROPE_TAIL` source-contract record for one stream.
- The record must bind together:
  - input
  - position tensor
  - op params
  - output
  - stream identity, likely `attn` vs `index`
- Restore or preserve the `compressed_norm` fixture needed by the accepted norm-weighted regression.
- Do not move to KV/cache finalizer until CUPD RoPE is either accepted or precisely blocked again.

## Important Constraints For Continuation

- No live graph cutover.
- No live graph dispatch.
- No cache mutation.
- No executor output consume.
- No performance run.
- No tolerance threshold changes.
- No copy/reference-output acceptance.
- No RoPE Metal kernel unless nonzero-position CPU recompute passes Tier B first.

