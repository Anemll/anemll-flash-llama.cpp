# DSV4 Layer Executor Fixtures

These fixtures normalize the T77-T81 full-layer executor side-probe summaries into JSONL files for the standalone harness:

```text
build/bin/dsv4-layer-executor-harness --fixtures tests/fixtures/dsv4_layer_executor --mode identity
```

The fixtures started as stats-only summary captures in T84. T87 added real payload side-files, and T89 added HC_PRE_NORM input/reference payloads for recompute diagnostics. Records must declare their actual payload kind; do not infer full tensors from stats-only metadata.

Required stages:

```text
hc_pre_norm
routed_moe_final_output
aohc_boundary
compressor_update
kv_cache_finalizer
```

JSONL record format:

```json
{
  "schema_version": 1,
  "capture_id": "t77_hc_pre_norm_l0_n16",
  "stage": "hc_pre_norm",
  "layer": 0,
  "token": 1,
  "tensor": "norm",
  "dtype": "f32",
  "shape": [4096, 1, 1, 1],
  "availability": "available",
  "payload_kind": "stats_only",
  "payload_file": "",
  "semantic_label": "hc_pre_norm",
  "payload_checksum": "",
  "payload_numel": 0,
  "payload_bytes": 0,
  "first_8_values": [],
  "last_8_values": [],
  "stride": [],
  "view_src": "",
  "storage_offset": 0,
  "same_payload_as": "",
  "same_tensor_ptr_as": "",
  "same_view_src_as": "",
  "same_storage_offset_as": "",
  "required": true,
  "unavailable_reason": "",
  "producer": {
    "op": "",
    "tensor_name": "",
    "stage": ""
  },
  "consumer": {
    "op": "",
    "tensor_name": "",
    "stage": ""
  },
  "stats_only": true,
  "byte_payload_available": false,
  "stats": {
    "sum": 0.0,
    "sumsq": 0.0,
    "min": 0.0,
    "max": 0.0,
    "max_abs": 0.0,
    "rms": 0.0,
    "over_tol": 0
  },
  "metadata": {
    "source_turn": 77,
    "mode": "post_eval_cpu_compare",
    "live_graph_nodes_added": 0,
    "live_backend_dispatches": 0,
    "output_consumed": 0,
    "cache_mutation": "disabled"
  }
}
```

Availability values:

```text
available: record exists and contains the declared payload_kind
unavailable: boundary is known but tensor payload is not available from T77-T81 captures
missing: required stage has no record and should fail harness validation
```

Payload kind values:

```text
stats_only: numeric summary only; no full values
tensor_values: full numeric tensor payload present
byte_values: raw byte payload present, required for quant/cache byte-exact checks
metadata_only: shape/dtype/op/producer/consumer metadata only
```

Semantic and alias fields:

```text
semantic_label: logical boundary label used by the executor harness/analyzer
payload_checksum: SHA-256 of payload_file when present
payload_numel / payload_bytes: decoded element count and raw byte size
first_8_values / last_8_values: payload preview for diagnostics
stride / view_src / storage_offset: tensor layout metadata when available; empty means not captured
same_payload_as: first semantic label with the same payload checksum
same_tensor_ptr_as / same_view_src_as / same_storage_offset_as: alias metadata when available; empty means unknown or not aliased
```

For large real captures, `payload_file` points to a raw binary side-file, usually under `tests/fixtures/dsv4_layer_executor/payloads/`. The harness loads that file and identity mode compares the bytes after round-trip. Inline `values` or `bytes` are accepted only for small payloads.

Required records must be available by default. Optional unavailable records are warnings, not failures, and identify known capture gaps such as missing full tensor payloads or raw quant/cache bytes.

Current fixture status:

```text
availability for required summary records: available
payload kind for required summary records: tensor_values or byte_values
HC_PRE_NORM T89 recompute inputs: tensor_values present for input_hc_original_residual, split_pre, norm_weight, reference_cur, reference_norm, reference_post
HC_PRE_NORM T91 semantic capture: producer_capture fixture payloads present under payloads/
HC_PRE_NORM T92 weighted-sum capture: `hc_ws_*` tensor_values records preserve input/pre/cur layout metadata for source-contract recompute
producer_capture fixture mode: may use ggml_set_output for capture-tagged tensors; not a hot-neutral validation mode
byte_payload_available: true for byte-oriented cache/quant fixture records
full tensor values fabricated: no
```
