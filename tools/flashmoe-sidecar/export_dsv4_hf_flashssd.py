#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402


ARCH = "deepseek4"
MXFP4_BLOCK = 32
MXFP4_TYPE_SIZE = 17
F8_BLOCK = 128
F8_TYPE_SIZE = 129
EXPERT_FAMILIES = (
    ("ffn_gate_exps", "w1"),
    ("ffn_up_exps", "w3"),
    ("ffn_down_exps", "w2"),
)
LAYER_TENSOR_RE = re.compile(r"layers\.(\d+)\.")
EXPERT_WEIGHT_RE = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight")


@contextmanager
def atomic_write_path(path: Path, mode: str, encoding: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    kwargs: dict[str, str] = {}
    if "b" not in mode:
        kwargs["encoding"] = encoding or "utf-8"
    try:
        with tmp.open(mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class SafeTensorRef:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    offset_begin: int
    offset_end: int

    @property
    def nbytes(self) -> int:
        return self.offset_end - self.offset_begin


@dataclass(frozen=True)
class DenseSpec:
    source_name: str
    tensor_name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    scale_name: str | None = None


class SafeTensorStore:
    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir.expanduser().resolve()
        index_path = self.hf_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise SystemExit(f"missing safetensors index: {index_path}")

        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.weight_map: dict[str, str] = index["weight_map"]
        self.refs: dict[str, SafeTensorRef] = {}
        self._parse_shards(sorted(set(self.weight_map.values())))

    def _parse_shards(self, shard_names: Sequence[str]) -> None:
        for shard_name in shard_names:
            path = self.hf_dir / shard_name
            with path.open("rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            data_start = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, end = meta["data_offsets"]
                self.refs[name] = SafeTensorRef(
                    name=name,
                    path=path,
                    dtype=meta["dtype"],
                    shape=tuple(int(v) for v in meta["shape"]),
                    data_start=data_start,
                    offset_begin=int(begin),
                    offset_end=int(end),
                )

    def has(self, name: str) -> bool:
        return name in self.refs

    def get(self, name: str) -> SafeTensorRef:
        try:
            return self.refs[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor '{name}'") from exc

    def read_bytes(self, name: str) -> bytes:
        ref = self.get(name)
        with ref.path.open("rb") as handle:
            handle.seek(ref.data_start + ref.offset_begin)
            data = handle.read(ref.nbytes)
        if len(data) != ref.nbytes:
            raise IOError(f"short read for tensor '{name}': got {len(data)}, expected {ref.nbytes}")
        return data

    def read_u8(self, name: str) -> np.ndarray:
        return np.frombuffer(self.read_bytes(name), dtype=np.uint8)

    def read_f32(self, name: str) -> np.ndarray:
        ref = self.get(name)
        if ref.dtype != "F32":
            raise ValueError(f"tensor '{name}' is {ref.dtype}, expected F32")
        return np.frombuffer(self.read_bytes(name), dtype="<f4").reshape(ref.shape)

    def read_i64_as_i32(self, name: str) -> np.ndarray:
        ref = self.get(name)
        if ref.dtype != "I64":
            raise ValueError(f"tensor '{name}' is {ref.dtype}, expected I64")
        return np.frombuffer(self.read_bytes(name), dtype="<i8").astype(np.int32, copy=False).reshape(ref.shape)


def human_bytes(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def product(shape: Sequence[int]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def f8_packed_nbytes(shape: Sequence[int]) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    if cols % F8_BLOCK != 0:
        raise ValueError(f"F8 tensor row length {cols} is not divisible by {F8_BLOCK}")
    return rows * (cols // F8_BLOCK) * F8_TYPE_SIZE


def mxfp4_packed_nbytes(shape: Sequence[int]) -> int:
    rows, packed_cols = int(shape[0]), int(shape[1])
    logical_cols = packed_cols * 2
    if logical_cols % MXFP4_BLOCK != 0:
        raise ValueError(f"FP4 logical row length {logical_cols} is not divisible by {MXFP4_BLOCK}")
    return rows * (logical_cols // MXFP4_BLOCK) * MXFP4_TYPE_SIZE


def pack_f8_e4m3_b128(weight: np.ndarray, scale: np.ndarray, shape: Sequence[int], scale_shape: Sequence[int]) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    n_blocks = cols // F8_BLOCK
    expected_scale = (math.ceil(rows / F8_BLOCK), n_blocks)
    if tuple(scale_shape) != expected_scale:
        raise ValueError(f"unexpected FP8 scale shape {tuple(scale_shape)}, expected {expected_scale}")

    w = weight.reshape(rows, cols)
    s = scale.reshape(expected_scale)
    out = np.empty((rows, n_blocks * F8_TYPE_SIZE), dtype=np.uint8)
    for block in range(n_blocks):
        dst = block * F8_TYPE_SIZE
        out[:, dst] = np.repeat(s[:, block], F8_BLOCK)[:rows]
        out[:, dst + 1 : dst + 1 + F8_BLOCK] = w[:, block * F8_BLOCK : (block + 1) * F8_BLOCK]
    return out


def f8_e4m3fn_table() -> np.ndarray:
    q = np.arange(256, dtype=np.uint16)
    aq = q & 0x7F
    exp = aq >> 3
    mant = aq & 0x07
    scales = np.array(
        [2**-9, 2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
         2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5],
        dtype=np.float32,
    )
    mag = np.where(exp == 0, mant, 8 + mant).astype(np.float32) * scales[exp]
    mag = np.where(aq == 0x7F, np.nan, mag)
    return np.where((q & 0x80) != 0, -mag, mag).astype(np.float32)


F8_E4M3FN_TO_F32 = f8_e4m3fn_table()


def e8m0_to_f32(scale: np.ndarray) -> np.ndarray:
    s = np.asarray(scale, dtype=np.uint8)
    bits = np.where(s == 0, np.uint32(0x00400000), s.astype(np.uint32) << np.uint32(23))
    return bits.view(np.float32)


def f32_to_bf16_bytes(values: np.ndarray) -> np.ndarray:
    f32 = np.asarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounded = u32 + np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2", copy=False).view(np.uint8)


def dequant_f8_e4m3_to_bf16_bytes(weight: np.ndarray, scale: np.ndarray, shape: Sequence[int], scale_shape: Sequence[int]) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    n_col_blocks = cols // F8_BLOCK
    expected_scale = (math.ceil(rows / F8_BLOCK), n_col_blocks)
    if tuple(scale_shape) != expected_scale:
        raise ValueError(f"unexpected FP8 scale shape {tuple(scale_shape)}, expected {expected_scale}")

    w = weight.reshape(rows, cols)
    s = e8m0_to_f32(scale.reshape(expected_scale))
    out = np.empty((rows, cols * 2), dtype=np.uint8)

    for row0 in range(0, rows, F8_BLOCK):
        row1 = min(row0 + F8_BLOCK, rows)
        scale_row = row0 // F8_BLOCK
        for col_block in range(n_col_blocks):
            col0 = col_block * F8_BLOCK
            col1 = col0 + F8_BLOCK
            vals = F8_E4M3FN_TO_F32[w[row0:row1, col0:col1]] * s[scale_row, col_block]
            out[row0:row1, col0 * 2:col1 * 2] = f32_to_bf16_bytes(vals).reshape(row1 - row0, F8_BLOCK * 2)

    return out


def pack_mxfp4_from_hf_fp4(weight: np.ndarray, scale: np.ndarray, shape: Sequence[int], scale_shape: Sequence[int]) -> np.ndarray:
    rows, packed_cols = int(shape[0]), int(shape[1])
    logical_cols = packed_cols * 2
    n_blocks = logical_cols // MXFP4_BLOCK
    expected_scale = (rows, n_blocks)
    if tuple(scale_shape) != expected_scale:
        raise ValueError(f"unexpected FP4 scale shape {tuple(scale_shape)}, expected {expected_scale}")

    # HF stores adjacent FP4 pairs: [v0|v1, v2|v3, ...].
    # ggml MXFP4 stores the first 16 values in low nibbles and the next 16 in high nibbles.
    w = weight.reshape(rows, n_blocks, MXFP4_BLOCK // 2)
    low = w & np.uint8(0x0F)
    high = w >> np.uint8(4)
    values = np.empty((rows, n_blocks, MXFP4_BLOCK), dtype=np.uint8)
    values[:, :, 0::2] = low
    values[:, :, 1::2] = high
    qs = values[:, :, : MXFP4_BLOCK // 2] | (values[:, :, MXFP4_BLOCK // 2 :] << np.uint8(4))

    out = np.empty((rows, n_blocks, MXFP4_TYPE_SIZE), dtype=np.uint8)
    out[:, :, 0] = scale.reshape(expected_scale)
    out[:, :, 1:] = qs
    return out.reshape(rows, n_blocks * MXFP4_TYPE_SIZE)


def parse_layers(spec: str | None, n_layers: int) -> list[int]:
    if not spec:
        return list(range(n_layers))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise SystemExit(f"layer(s) out of range 0..{n_layers - 1}: {bad}")
    return sorted(layers)


def infer_available_layer_count(store: SafeTensorStore) -> int:
    layers: set[int] = set()
    for name in store.weight_map.keys():
        match = LAYER_TENSOR_RE.match(name)
        if match is not None:
            layers.add(int(match.group(1)))

    if not layers:
        raise SystemExit("could not infer transformer layer count from safetensors")

    count = max(layers) + 1
    missing = sorted(set(range(count)) - layers)
    if missing:
        preview = ", ".join(str(layer) for layer in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise SystemExit(f"non-contiguous transformer layers in safetensors; missing layer id(s): {preview}{suffix}")

    return count


def infer_available_expert_count(store: SafeTensorStore, layers: Sequence[int]) -> int:
    selected = set(layers)
    per_layer: dict[int, set[int]] = {layer: set() for layer in layers}

    for name in store.weight_map.keys():
        match = EXPERT_WEIGHT_RE.fullmatch(name)
        if match is None:
            continue
        layer = int(match.group(1))
        if layer in selected:
            per_layer[layer].add(int(match.group(2)))

    missing_layers = [layer for layer, experts in per_layer.items() if not experts]
    if missing_layers:
        preview = ", ".join(str(layer) for layer in missing_layers[:8])
        suffix = "..." if len(missing_layers) > 8 else ""
        raise SystemExit(f"missing routed expert tensors for layer(s): {preview}{suffix}")

    counts: dict[int, int] = {}
    for layer, experts in per_layer.items():
        count = max(experts) + 1
        missing = sorted(set(range(count)) - experts)
        if missing:
            preview = ", ".join(str(expert) for expert in missing[:8])
            suffix = "..." if len(missing) > 8 else ""
            raise SystemExit(f"non-contiguous experts in layer {layer}; missing expert id(s): {preview}{suffix}")
        counts[layer] = count

    unique_counts = sorted(set(counts.values()))
    if len(unique_counts) != 1:
        details = ", ".join(f"layer {layer}: {count}" for layer, count in sorted(counts.items())[:8])
        suffix = "..." if len(counts) > 8 else ""
        raise SystemExit(f"inconsistent expert count across layers: {details}{suffix}")

    return unique_counts[0]


def infer_compress_ratios(store: SafeTensorStore, n_layers: int) -> list[int]:
    ratios: list[int] = []
    for layer in range(n_layers):
        ape_name = f"layers.{layer}.attn.compressor.ape"
        if not store.has(ape_name):
            ratios.append(0)
            continue
        ref = store.get(ape_name)
        if len(ref.shape) != 2:
            raise SystemExit(f"unexpected compressor ape shape for '{ape_name}': {ref.shape}")
        # HF stores this as [compress_ratio, compressor_width]; GGUF loader sees
        # the transposed logical shape [compressor_width, compress_ratio].
        ratios.append(int(ref.shape[0]))
    return ratios


def first_existing(store: SafeTensorStore, names: Iterable[str]) -> SafeTensorRef | None:
    for name in names:
        if store.has(name):
            return store.get(name)
    return None


def apply_tensor_geometry_overrides(cfg: dict, store: SafeTensorStore, layers: Sequence[int], actual_expert_count: int) -> dict:
    cfg = dict(cfg)
    overrides: dict[str, int | list[int]] = {}

    emb = store.get("embed.weight")
    if len(emb.shape) != 2:
        raise SystemExit(f"unexpected embed.weight shape: {emb.shape}")
    overrides["vocab_size"] = int(emb.shape[0])
    overrides["hidden_size"] = int(emb.shape[1])

    n_layer = int(cfg["num_hidden_layers"])
    layer0 = layers[0]
    q_norm = store.get(f"layers.{layer0}.attn.q_norm.weight")
    kv_norm = store.get(f"layers.{layer0}.attn.kv_norm.weight")
    sinks = store.get(f"layers.{layer0}.attn.attn_sink")
    expert_w1 = store.get(f"layers.{layer0}.ffn.experts.0.w1.weight")

    overrides["q_lora_rank"] = int(q_norm.shape[0])
    overrides["head_dim"] = int(kv_norm.shape[0])
    overrides["num_attention_heads"] = int(sinks.shape[0])
    overrides["n_routed_experts"] = int(actual_expert_count)
    overrides["moe_intermediate_size"] = int(expert_w1.shape[0])

    shared_w1 = first_existing(store, (f"layers.{layer0}.ffn.shared_experts.w1.weight",))
    if shared_w1 is not None:
        n_ff = int(overrides["moe_intermediate_size"])
        overrides["n_shared_experts"] = max(1, int(shared_w1.shape[0]) // n_ff)

    wo_a = store.get(f"layers.{layer0}.attn.wo_a.weight")
    head_total = int(overrides["num_attention_heads"]) * int(overrides["head_dim"])
    # HF stores wo_a as [groups * o_lora_rank, n_head * head_dim / groups].
    if len(wo_a.shape) == 2 and int(wo_a.shape[1]) > 0 and head_total % int(wo_a.shape[1]) == 0:
        o_groups = head_total // int(wo_a.shape[1])
        if o_groups > 0 and int(wo_a.shape[0]) % o_groups == 0:
            overrides["o_groups"] = int(o_groups)
            overrides["o_lora_rank"] = int(wo_a.shape[0]) // int(o_groups)

    indexer_proj = first_existing(
        store,
        (f"layers.{layer}.attn.indexer.weights_proj.weight" for layer in layers),
    )
    if indexer_proj is not None and len(indexer_proj.shape) == 2:
        # HF stores [indexer_heads, hidden_size].
        overrides["index_n_heads"] = int(indexer_proj.shape[0])

    indexer_norm = first_existing(
        store,
        (f"layers.{layer}.attn.indexer.compressor.norm.weight" for layer in layers),
    )
    if indexer_norm is not None and len(indexer_norm.shape) == 1:
        overrides["index_head_dim"] = int(indexer_norm.shape[0])

    hash_layers = [
        layer
        for layer in layers
        if store.has(f"layers.{layer}.ffn.gate.tid2eid")
    ]
    if hash_layers:
        overrides["num_hash_layers"] = max(hash_layers) + 1

    hc_head_base = first_existing(store, ("hc_head_base",))
    if hc_head_base is not None and len(hc_head_base.shape) == 1:
        overrides["hc_mult"] = int(hc_head_base.shape[0])

    overrides["compress_ratios"] = infer_compress_ratios(store, n_layer)

    changed: list[str] = []
    for key, value in overrides.items():
        if cfg.get(key) != value:
            changed.append(f"{key}={cfg.get(key)!r}->{value!r}")
            cfg[key] = value

    if changed:
        preview = ", ".join(changed[:12])
        suffix = ", ..." if len(changed) > 12 else ""
        print(f"warning: tensor-derived DS4 geometry overrides config: {preview}{suffix}", flush=True)

    return cfg


def root_tensor_name(name: str) -> str | None:
    return {
        "embed.weight": "token_embd.weight",
        "head.weight": "output.weight",
        "norm.weight": "output_norm.weight",
        "hc_head_base": "output_hc_base.weight",
        "hc_head_fn": "output_hc_fn.weight",
        "hc_head_scale": "output_hc_scale.weight",
    }.get(name)


def layer_tensor_name(layer: int, suffix: str) -> str | None:
    mapping = {
        "attn_norm.weight": "attn_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "attn.attn_sink": "attn_sinks.weight",
        "attn.q_norm.weight": "attn_q_a_norm.weight",
        "attn.kv_norm.weight": "attn_kv_a_norm.weight",
        "attn.wq_a.weight": "attn_q_a.weight",
        "attn.wq_b.weight": "attn_q_b.weight",
        "attn.wkv.weight": "attn_kv.weight",
        "attn.wo_a.weight": "attn_wo_a.weight",
        "attn.wo_b.weight": "attn_wo_b.weight",
        "attn.compressor.ape": "attn_compressor_ape.weight",
        "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
        "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
        "attn.compressor.norm.weight": "attn_compressor_norm.weight",
        "attn.indexer.weights_proj.weight": "indexer.proj.weight",
        "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
        "attn.indexer.compressor.ape": "indexer_compressor_ape.weight",
        "attn.indexer.compressor.wkv.weight": "indexer_compressor_kv.weight",
        "attn.indexer.compressor.wgate.weight": "indexer_compressor_gate.weight",
        "attn.indexer.compressor.norm.weight": "indexer_compressor_norm.weight",
        "ffn.gate.weight": "ffn_gate_inp.weight",
        "ffn.gate.bias": "exp_probs_b.bias",
        "ffn.gate.tid2eid": "ffn_gate_tid2eid.weight",
        "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
        "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
        "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
        "hc_attn_base": "hc_attn_base.weight",
        "hc_attn_fn": "hc_attn_fn.weight",
        "hc_attn_scale": "hc_attn_scale.weight",
        "hc_ffn_base": "hc_ffn_base.weight",
        "hc_ffn_fn": "hc_ffn_fn.weight",
        "hc_ffn_scale": "hc_ffn_scale.weight",
    }
    mapped = mapping.get(suffix)
    if mapped is None:
        return None
    return f"blk.{layer}.{mapped}"


def canonical_dense_name(hf_name: str) -> str | None:
    if hf_name.startswith("mtp."):
        return None
    if ".ffn.experts." in hf_name:
        return None
    if hf_name.endswith(".scale"):
        return None
    root = root_tensor_name(hf_name)
    if root is not None:
        return root
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", hf_name)
    if match is None:
        return None
    return layer_tensor_name(int(match.group(1)), match.group(2))


def build_dense_specs(store: SafeTensorStore) -> list[DenseSpec]:
    specs: list[DenseSpec] = []
    for source_name in store.weight_map.keys():
        tensor_name = canonical_dense_name(source_name)
        if tensor_name is None:
            continue
        ref = store.get(source_name)
        if ref.dtype == "F8_E4M3":
            scale_name = source_name.removesuffix(".weight") + ".scale"
            if not store.has(scale_name):
                raise SystemExit(f"missing FP8 scale tensor for '{source_name}': expected '{scale_name}'")
            if source_name.endswith("attn.wo_a.weight"):
                specs.append(DenseSpec(source_name, tensor_name, "f8_to_bf16", ref.dtype, ref.shape, product(ref.shape) * 2, scale_name))
            else:
                specs.append(DenseSpec(source_name, tensor_name, "f8", ref.dtype, ref.shape, f8_packed_nbytes(ref.shape), scale_name))
        elif ref.dtype == "BF16":
            specs.append(DenseSpec(source_name, tensor_name, "bf16", ref.dtype, ref.shape, ref.nbytes))
        elif ref.dtype == "F32":
            specs.append(DenseSpec(source_name, tensor_name, "f32", ref.dtype, ref.shape, ref.nbytes))
        elif ref.dtype == "I64" and source_name.endswith(".tid2eid"):
            specs.append(DenseSpec(source_name, tensor_name, "i64_to_i32", ref.dtype, ref.shape, product(ref.shape) * 4))
        else:
            raise SystemExit(f"unsupported dense tensor '{source_name}' dtype={ref.dtype} shape={ref.shape}")
    return specs


def add_model_metadata(writer: gguf.GGUFWriter, cfg: dict) -> None:
    n_layer = int(cfg["num_hidden_layers"])
    n_vocab = int(cfg["vocab_size"])
    head_dim = int(cfg["head_dim"])
    rope_scaling = cfg.get("rope_scaling", {})
    compress_ratios = [int(v) for v in cfg.get("compress_ratios", [])[:n_layer]]
    if len(compress_ratios) != n_layer:
        raise SystemExit(f"expected {n_layer} compress ratios, got {len(compress_ratios)}")

    writer.add_name("DeepSeek-V4-SSD")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F8_E4M3_MXFP4.value)
    writer.add_quantization_version(2)
    writer.add_vocab_size(n_vocab)
    writer.add_context_length(int(cfg["max_position_embeddings"]))
    writer.add_embedding_length(int(cfg["hidden_size"]))
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(int(cfg["moe_intermediate_size"]))
    writer.add_expert_feed_forward_length(int(cfg["moe_intermediate_size"]))
    writer.add_expert_count(int(cfg["n_routed_experts"]))
    writer.add_expert_used_count(int(cfg["num_experts_per_tok"]))
    writer.add_expert_shared_count(int(cfg["n_shared_experts"]))
    writer.add_expert_weights_scale(float(cfg.get("routed_scaling_factor", 1.0)))
    writer.add_expert_weights_norm(bool(cfg.get("norm_topk_prob", False)))
    writer.add_uint32(f"{ARCH}.expert_gating_func", 4)  # sqrtsoftplus
    writer.add_head_count(int(cfg["num_attention_heads"]))
    writer.add_head_count_kv(int(cfg.get("num_key_value_heads", 1)))
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_key_length_mla(head_dim)
    writer.add_value_length_mla(head_dim)
    writer.add_q_lora_rank(int(cfg["q_lora_rank"]))
    writer.add_rope_dimension_count(int(cfg["qk_rope_head_dim"]))
    writer.add_rope_freq_base(float(cfg["rope_theta"]))
    writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
    writer.add_rope_scaling_factor(float(rope_scaling.get("factor", 1.0)))
    writer.add_rope_scaling_orig_ctx_len(int(rope_scaling.get("original_max_position_embeddings", 0)))
    writer.add_rope_scaling_yarn_beta_fast(float(rope_scaling.get("beta_fast", 32.0)))
    writer.add_rope_scaling_yarn_beta_slow(float(rope_scaling.get("beta_slow", 1.0)))
    writer.add_layer_norm_rms_eps(float(cfg["rms_norm_eps"]))
    writer.add_sliding_window(int(cfg.get("sliding_window", 128)))
    writer.add_indexer_head_count(int(cfg.get("index_n_heads", 64)))
    writer.add_indexer_key_length(int(cfg.get("index_head_dim", 128)))
    writer.add_indexer_top_k(int(cfg.get("index_topk", 1024)))
    writer.add_swiglu_clamp_exp([float(cfg.get("swiglu_limit", 0.0))] * n_layer)

    writer.add_uint32(f"{ARCH}.attention.output_lora_rank", int(cfg.get("o_lora_rank", 1024)))
    writer.add_uint32(f"{ARCH}.attention.output_group_count", int(cfg.get("o_groups", 16)))
    writer.add_float32(f"{ARCH}.attention.compress_rope_freq_base", float(cfg.get("compress_rope_theta", 160000.0)))
    writer.add_array(f"{ARCH}.attention.compress_ratios", compress_ratios)
    writer.add_uint32(f"{ARCH}.hash_layer_count", int(cfg.get("num_hash_layers", 3)))
    writer.add_uint32(f"{ARCH}.hyper_connection.count", int(cfg.get("hc_mult", 4)))
    writer.add_uint32(f"{ARCH}.hyper_connection.sinkhorn_iterations", int(cfg.get("hc_sinkhorn_iters", 20)))
    writer.add_float32(f"{ARCH}.hyper_connection.epsilon", float(cfg.get("hc_eps", 1.0e-6)))
    # MTP tensors are intentionally not exported into this dense+sidecar runtime package.
    writer.add_uint32(f"{ARCH}.nextn_predict_layers", 0)


def add_tokenizer(writer: gguf.GGUFWriter, hf_dir: Path, cfg: dict) -> None:
    tokenizer_path = hf_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise SystemExit(f"missing tokenizer.json: {tokenizer_path}")
    with tokenizer_path.open("r", encoding="utf-8") as handle:
        tok = json.load(handle)

    vocab_size = int(cfg["vocab_size"])
    vocab = tok["model"]["vocab"]
    tokens: list[str] = [""] * vocab_size
    toktypes: list[int] = [gguf.TokenType.UNUSED.value] * vocab_size

    for token, token_id in vocab.items():
        if int(token_id) < vocab_size:
            tokens[int(token_id)] = token
            toktypes[int(token_id)] = gguf.TokenType.NORMAL.value

    for entry in tok.get("added_tokens", []):
        token_id = int(entry["id"])
        if token_id < vocab_size:
            tokens[token_id] = entry["content"]
            toktypes[token_id] = gguf.TokenType.CONTROL.value if entry.get("special", False) else gguf.TokenType.USER_DEFINED.value

    for token_id, token in enumerate(tokens):
        if token == "":
            tokens[token_id] = f"[PAD{token_id}]"

    merges = tok["model"].get("merges", [])
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("deepseek-v3")
    writer.add_token_list(tokens)
    writer.add_token_merges(merges)
    writer.add_token_types(toktypes)
    writer.add_bos_token_id(int(cfg.get("bos_token_id", 0)))
    writer.add_eos_token_id(int(cfg.get("eos_token_id", 1)))
    writer.add_pad_token_id(int(cfg.get("pad_token_id", 2)))
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def dense_tensor_info(spec: DenseSpec) -> tuple[tuple[int, ...], np.dtype, int, gguf.GGMLQuantizationType | None]:
    if spec.kind == "f8":
        rows, cols = spec.shape
        return (rows, (cols // F8_BLOCK) * F8_TYPE_SIZE), np.dtype(np.uint8), spec.nbytes, gguf.GGMLQuantizationType.F8_E4M3_B128
    if spec.kind == "bf16" or spec.kind == "f8_to_bf16":
        return (*spec.shape[:-1], spec.shape[-1] * 2), np.dtype(np.uint8), spec.nbytes, gguf.GGMLQuantizationType.BF16
    if spec.kind == "f32":
        return spec.shape, np.dtype("<f4"), spec.nbytes, None
    if spec.kind == "i64_to_i32":
        return spec.shape, np.dtype("<i4"), spec.nbytes, None
    raise AssertionError(spec.kind)


def materialize_dense_tensor(store: SafeTensorStore, spec: DenseSpec) -> np.ndarray:
    if spec.kind == "f8":
        assert spec.scale_name is not None
        ref = store.get(spec.source_name)
        scale_ref = store.get(spec.scale_name)
        return pack_f8_e4m3_b128(store.read_u8(spec.source_name), store.read_u8(spec.scale_name), ref.shape, scale_ref.shape)
    if spec.kind == "f8_to_bf16":
        assert spec.scale_name is not None
        ref = store.get(spec.source_name)
        scale_ref = store.get(spec.scale_name)
        return dequant_f8_e4m3_to_bf16_bytes(store.read_u8(spec.source_name), store.read_u8(spec.scale_name), ref.shape, scale_ref.shape)
    if spec.kind == "bf16":
        ref = store.get(spec.source_name)
        return store.read_u8(spec.source_name).reshape((*ref.shape[:-1], ref.shape[-1] * 2))
    if spec.kind == "f32":
        return store.read_f32(spec.source_name)
    if spec.kind == "i64_to_i32":
        return store.read_i64_as_i32(spec.source_name)
    raise AssertionError(spec.kind)


def write_dense_gguf(store: SafeTensorStore, cfg: dict, specs: Sequence[DenseSpec], out_path: Path, hf_dir: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(out_path, arch=ARCH)
    add_model_metadata(writer, cfg)
    add_tokenizer(writer, hf_dir, cfg)

    for spec in specs:
        shape, dtype, nbytes, raw_dtype = dense_tensor_info(spec)
        writer.add_tensor_info(spec.tensor_name, shape, dtype, nbytes, raw_dtype=raw_dtype)

    writer.write_header_to_file(out_path)
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for index, spec in enumerate(specs, start=1):
        print(f"dense [{index:04d}/{len(specs):04d}] {spec.tensor_name} <- {spec.source_name} ({human_bytes(spec.nbytes)})", flush=True)
        writer.write_tensor_data(materialize_dense_tensor(store, spec))

    writer.close()


def expert_tensor_names(layer: int, expert: int, short: str) -> tuple[str, str]:
    prefix = f"layers.{layer}.ffn.experts.{expert}.{short}"
    return f"{prefix}.weight", f"{prefix}.scale"


def sidecar_entry_tensor_name(layer: int, family: str) -> str:
    return f"blk.{layer}.{family}.weight"


def sidecar_logical_shape(cfg: dict, family: str) -> list[int]:
    n_embd = int(cfg["hidden_size"])
    n_ff = int(cfg["moe_intermediate_size"])
    n_expert = int(cfg["n_routed_experts"])
    if family == "ffn_down_exps":
        return [n_ff, n_embd, n_expert]
    return [n_embd, n_ff, n_expert]


def sidecar_manifest_layout(sidecar_layout: str) -> str:
    if sidecar_layout == "expert-major":
        return "layer_major_expert"
    return "layer_major_whole_tensor"


def write_sidecar(store: SafeTensorStore, cfg: dict, layers: Sequence[int], out_dir: Path, expert_limit: int | None, sidecar_layout: str) -> dict:
    if sidecar_layout == "expert-major":
        return write_sidecar_expert_major(store, cfg, layers, out_dir, expert_limit)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_expert_total = int(cfg["n_routed_experts"])
    n_expert_written = min(n_expert_total, expert_limit) if expert_limit is not None else n_expert_total
    entries: list[dict] = []

    for layer_index, layer in enumerate(layers, start=1):
        layer_path = out_dir / f"layer_{layer:03d}.bin"
        print(f"sidecar layer {layer} ({layer_index}/{len(layers)}) -> {layer_path}", flush=True)
        with atomic_write_path(layer_path, "wb") as handle:
            for family, short in EXPERT_FAMILIES:
                offset = handle.tell()
                bytes_per_expert: int | None = None
                for expert in range(n_expert_written):
                    weight_name, scale_name = expert_tensor_names(layer, expert, short)
                    weight_ref = store.get(weight_name)
                    scale_ref = store.get(scale_name)
                    if weight_ref.dtype != "I8" or scale_ref.dtype != "F8_E8M0":
                        raise SystemExit(
                            f"expected native FP4 I8/F8_E8M0 expert tensors, got {weight_name}={weight_ref.dtype}, {scale_name}={scale_ref.dtype}"
                        )
                    packed = pack_mxfp4_from_hf_fp4(store.read_u8(weight_name), store.read_u8(scale_name), weight_ref.shape, scale_ref.shape)
                    if bytes_per_expert is None:
                        bytes_per_expert = int(packed.nbytes)
                    elif bytes_per_expert != packed.nbytes:
                        raise SystemExit(f"inconsistent expert byte size for {weight_name}: {packed.nbytes} vs {bytes_per_expert}")
                    packed.tofile(handle)

                assert bytes_per_expert is not None
                exact = bytes_per_expert * n_expert_written
                entries.append(
                    {
                        "tensor_name": sidecar_entry_tensor_name(layer, family),
                        "original_gguf_tensor_name": sidecar_entry_tensor_name(layer, family),
                        "tensor_family": family,
                        "layer": layer,
                        "quant_type": "mxfp4",
                        "shape": sidecar_logical_shape(cfg, family),
                        "expert_count": n_expert_written,
                        "repacked_file": layer_path.name,
                        "repacked_offset": offset,
                        "exact_byte_length": exact,
                        "bytes_per_expert": bytes_per_expert,
                    }
                )
                print(f"  {family}: {n_expert_written} experts, {human_bytes(exact)}", flush=True)

    manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": sidecar_manifest_layout(sidecar_layout),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model_dir": str(store.hf_dir),
            "model_files": sorted(set(store.weight_map.values())),
            "preserve_quant": True,
        },
        "model": {
            "arch": ARCH,
            "layer_count": int(cfg["num_hidden_layers"]),
            "expert_count": n_expert_total,
            "expert_written_count": n_expert_written,
            "expert_used_count": int(cfg["num_experts_per_tok"]),
            "expert_format": "native_fp4_mxfp4",
            "sidecar_layout": sidecar_layout,
        },
        "entries": entries,
    }

    with atomic_write_path(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def write_sidecar_expert_major(store: SafeTensorStore, cfg: dict, layers: Sequence[int], out_dir: Path, expert_limit: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_expert_total = int(cfg["n_routed_experts"])
    n_expert_written = min(n_expert_total, expert_limit) if expert_limit is not None else n_expert_total
    entries: list[dict] = []

    for layer_index, layer in enumerate(layers, start=1):
        layer_path = out_dir / f"layer_{layer:03d}.bin"
        print(f"sidecar layer {layer} ({layer_index}/{len(layers)}) -> {layer_path} [expert-major]", flush=True)

        family_specs: list[dict] = []
        expert_stride = 0
        for family, short in EXPERT_FAMILIES:
            weight_name, scale_name = expert_tensor_names(layer, 0, short)
            weight_ref = store.get(weight_name)
            scale_ref = store.get(scale_name)
            if weight_ref.dtype != "I8" or scale_ref.dtype != "F8_E8M0":
                raise SystemExit(
                    f"expected native FP4 I8/F8_E8M0 expert tensors, got {weight_name}={weight_ref.dtype}, {scale_name}={scale_ref.dtype}"
                )
            bytes_per_expert = mxfp4_packed_nbytes(weight_ref.shape)
            family_specs.append({
                "family": family,
                "short": short,
                "offset": expert_stride,
                "bytes_per_expert": bytes_per_expert,
            })
            expert_stride += bytes_per_expert

        with atomic_write_path(layer_path, "wb") as handle:
            for expert in range(n_expert_written):
                for spec in family_specs:
                    weight_name, scale_name = expert_tensor_names(layer, expert, spec["short"])
                    weight_ref = store.get(weight_name)
                    scale_ref = store.get(scale_name)
                    if weight_ref.dtype != "I8" or scale_ref.dtype != "F8_E8M0":
                        raise SystemExit(
                            f"expected native FP4 I8/F8_E8M0 expert tensors, got {weight_name}={weight_ref.dtype}, {scale_name}={scale_ref.dtype}"
                        )
                    packed = pack_mxfp4_from_hf_fp4(store.read_u8(weight_name), store.read_u8(scale_name), weight_ref.shape, scale_ref.shape)
                    if packed.nbytes != spec["bytes_per_expert"]:
                        raise SystemExit(
                            f"inconsistent expert byte size for {weight_name}: {packed.nbytes} vs {spec['bytes_per_expert']}"
                        )
                    packed.tofile(handle)

        for spec in family_specs:
            family = str(spec["family"])
            bytes_per_expert = int(spec["bytes_per_expert"])
            exact = bytes_per_expert * n_expert_written
            entries.append(
                {
                    "tensor_name": sidecar_entry_tensor_name(layer, family),
                    "original_gguf_tensor_name": sidecar_entry_tensor_name(layer, family),
                    "tensor_family": family,
                    "layer": layer,
                    "quant_type": "mxfp4",
                    "shape": sidecar_logical_shape(cfg, family),
                    "expert_count": n_expert_written,
                    "repacked_file": layer_path.name,
                    "repacked_offset": int(spec["offset"]),
                    "exact_byte_length": exact,
                    "bytes_per_expert": bytes_per_expert,
                    "expert_stride": expert_stride,
                    "expert_major": True,
                }
            )
            print(f"  {family}: {n_expert_written} experts, {human_bytes(exact)}", flush=True)
        print(f"  expert-major stride: {human_bytes(expert_stride)} per expert", flush=True)

    manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": sidecar_manifest_layout("expert-major"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model_dir": str(store.hf_dir),
            "model_files": sorted(set(store.weight_map.values())),
            "preserve_quant": True,
        },
        "model": {
            "arch": ARCH,
            "layer_count": int(cfg["num_hidden_layers"]),
            "expert_count": n_expert_total,
            "expert_written_count": n_expert_written,
            "expert_used_count": int(cfg["num_experts_per_tok"]),
            "expert_format": "native_fp4_mxfp4",
            "sidecar_layout": "expert-major",
        },
        "entries": entries,
    }

    with atomic_write_path(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def refresh_sidecar_manifest(store: SafeTensorStore, cfg: dict, layers: Sequence[int], out_dir: Path, expert_limit: int | None, sidecar_layout: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_expert_total = int(cfg["n_routed_experts"])
    n_expert_written = min(n_expert_total, expert_limit) if expert_limit is not None else n_expert_total
    entries: list[dict] = []

    for layer in layers:
        layer_path = out_dir / f"layer_{layer:03d}.bin"
        if not layer_path.is_file():
            raise SystemExit(f"cannot refresh sidecar manifest; missing existing sidecar layer file: {layer_path}")

        offset = 0
        expert_stride = 0
        family_bytes: list[tuple[str, int]] = []
        for family, short in EXPERT_FAMILIES:
            weight_name, scale_name = expert_tensor_names(layer, 0, short)
            weight_ref = store.get(weight_name)
            scale_ref = store.get(scale_name)
            if weight_ref.dtype != "I8" or scale_ref.dtype != "F8_E8M0":
                raise SystemExit(
                    f"expected native FP4 I8/F8_E8M0 expert tensors, got {weight_name}={weight_ref.dtype}, {scale_name}={scale_ref.dtype}"
                )

            bytes_per_expert = mxfp4_packed_nbytes(weight_ref.shape)
            family_bytes.append((family, bytes_per_expert))
            expert_stride += bytes_per_expert

        for family, bytes_per_expert in family_bytes:
            exact = bytes_per_expert * n_expert_written
            item = {
                "tensor_name": sidecar_entry_tensor_name(layer, family),
                "original_gguf_tensor_name": sidecar_entry_tensor_name(layer, family),
                "tensor_family": family,
                "layer": layer,
                "quant_type": "mxfp4",
                "shape": sidecar_logical_shape(cfg, family),
                "expert_count": n_expert_written,
                "repacked_file": layer_path.name,
                "repacked_offset": offset,
                "exact_byte_length": exact,
                "bytes_per_expert": bytes_per_expert,
            }
            if sidecar_layout == "expert-major":
                item["expert_stride"] = expert_stride
                item["expert_major"] = True
            entries.append(item)
            offset += bytes_per_expert if sidecar_layout == "expert-major" else exact

        actual_size = layer_path.stat().st_size
        expected_size = expert_stride * n_expert_written if sidecar_layout == "expert-major" else offset
        if actual_size != expected_size:
            raise SystemExit(
                f"existing sidecar layer file size mismatch for {layer_path}: got {actual_size}, expected {expected_size}; "
                "rerun without --skip-sidecar to rebuild sidecar binaries"
            )

    manifest = {
        "schema_version": 1,
        "sidecar_kind": "flashmoe_gguf",
        "layout": sidecar_manifest_layout(sidecar_layout),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model_dir": str(store.hf_dir),
            "model_files": sorted(set(store.weight_map.values())),
            "preserve_quant": True,
            "reused_existing_sidecar_bins": True,
        },
        "model": {
            "arch": ARCH,
            "layer_count": int(cfg["num_hidden_layers"]),
            "expert_count": n_expert_total,
            "expert_written_count": n_expert_written,
            "expert_used_count": int(cfg["num_experts_per_tok"]),
            "expert_format": "native_fp4_mxfp4",
            "sidecar_layout": sidecar_layout,
        },
        "entries": entries,
    }

    with atomic_write_path(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def copy_encoding_assets(hf_dir: Path, out_dir: Path) -> bool:
    src = hf_dir / "encoding"
    if not src.is_dir():
        return False

    dst = out_dir / "encoding"
    if dst.exists():
        shutil.rmtree(dst)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {".DS_Store", "__pycache__"} or name.endswith(".pyc")}

    shutil.copytree(src, dst, ignore=ignore)
    return True


def write_package_files(out_dir: Path, cfg: dict, dense_written: bool, sidecar_written: bool, encoding_written: bool, estimate: dict[str, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoding_note = "- DS4 chat encoder: `encoding/encoding_dsv4.py`\n" if encoding_written else ""
    readme = f"""# DeepSeek-V4 Flash-MoE SSD

This package is a Flash-MoE SSD layout exported from the original Hugging Face DeepSeek-V4 weights.

- Dense tensors: native FP8 E4M3 with UE8M0 scales, packed as GGUF `F8_E4M3_B128`.
- Routed expert tensors: native FP4 experts, packed as GGUF `MXFP4` blocks in the sidecar.
- Dense GGUF: `dense/model-dense.gguf`
- Expert sidecar: `sidecar/manifest.json`
{encoding_note}
Build a DS4 chat prompt from this package:

```bash
DSV4_INPUT=$(PYTHONPATH={out_dir}/encoding python3 - <<'PY'
from encoding_dsv4 import encode_messages

print(encode_messages(
    [{{"role": "user", "content": "What is Apple Neural Engine?"}}],
    thinking_mode="chat",
), end="")
PY
)
```

Example:

```bash
./build/bin/llama-cli \\
  -m {out_dir}/dense/model-dense.gguf \\
  --moe-mode slot-bank \\
  --moe-sidecar {out_dir}/sidecar \\
  --moe-slot-bank 64 \\
  --moe-topk {int(cfg["num_experts_per_tok"])} \\
  -ngl 999 -c 8192 -b 2048 -ub 1 \\
  -p "$DSV4_INPUT" -n 128
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    yaml = f"""package: DeepSeek-V4-Flash-MoE-SSD
format: flashmoe_ssd
source_arch: deepseek_v4
dense:
  path: dense/model-dense.gguf
  quantization: native_fp8_e4m3_ue8m0
  written: {str(dense_written).lower()}
experts:
  path: sidecar
  quantization: native_fp4_mxfp4
  written: {str(sidecar_written).lower()}
encoding:
  path: encoding
  written: {str(encoding_written).lower()}
model:
  layers: {int(cfg["num_hidden_layers"])}
  experts: {int(cfg["n_routed_experts"])}
  top_k: {int(cfg["num_experts_per_tok"])}
  hidden_size: {int(cfg["hidden_size"])}
  moe_intermediate_size: {int(cfg["moe_intermediate_size"])}
estimate:
  dense_bytes: {estimate["dense"]}
  sidecar_bytes: {estimate["sidecar"]}
"""
    (out_dir / "flashmoe-ssd.yaml").write_text(yaml, encoding="utf-8")

    package = {
        "package_kind": "flashmoe_ssd",
        "arch": ARCH,
        "dense": "dense/model-dense.gguf",
        "sidecar": "sidecar",
        "encoding": "encoding" if encoding_written else None,
        "dense_quantization": "native_fp8_e4m3_ue8m0",
        "expert_quantization": "native_fp4_mxfp4",
        "written": {
            "dense": dense_written,
            "sidecar": sidecar_written,
            "encoding": encoding_written,
        },
        "estimate": estimate,
    }
    with atomic_write_path(out_dir / "flashmoe-package.json", "w", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2)
        handle.write("\n")


def estimate_sidecar_bytes(store: SafeTensorStore, cfg: dict, layers: Sequence[int], expert_limit: int | None) -> int:
    n_expert = int(cfg["n_routed_experts"])
    n_written = min(n_expert, expert_limit) if expert_limit is not None else n_expert
    layer = layers[0] if layers else 0
    total_per_layer = 0
    for _, short in EXPERT_FAMILIES:
        weight_name, _ = expert_tensor_names(layer, 0, short)
        total_per_layer += mxfp4_packed_nbytes(store.get(weight_name).shape) * n_written
    return total_per_layer * len(layers)


def check_space(out_dir: Path, required: int, allow_low_space: bool) -> None:
    parent = out_dir.expanduser().resolve()
    while not parent.exists() and parent.parent != parent:
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    margin = 16 << 30
    if usage.free < required + margin:
        msg = (
            f"target volume for {out_dir} has {human_bytes(usage.free)} free, "
            f"but export estimate is {human_bytes(required)} plus a {human_bytes(margin)} margin"
        )
        if allow_low_space:
            print(f"warning: {msg}", flush=True)
        else:
            raise SystemExit(f"{msg}; choose a larger --out or pass --allow-low-space")


def prepare_output(out_dir: Path, force: bool, skip_dense: bool, skip_sidecar: bool) -> None:
    if force:
        if not skip_dense:
            shutil.rmtree(out_dir / "dense", ignore_errors=True)
        if not skip_sidecar:
            shutil.rmtree(out_dir / "sidecar", ignore_errors=True)
    elif out_dir.exists():
        conflicts = []
        if not skip_dense and (out_dir / "dense" / "model-dense.gguf").exists():
            conflicts.append(str(out_dir / "dense" / "model-dense.gguf"))
        if not skip_sidecar and (out_dir / "sidecar" / "manifest.json").exists():
            conflicts.append(str(out_dir / "sidecar" / "manifest.json"))
        if conflicts:
            raise SystemExit(f"output already exists: {', '.join(conflicts)}; use --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export original DeepSeek-V4 HF safetensors to a Flash-MoE SSD dense GGUF + FP4 expert sidecar package."
    )
    parser.add_argument("--hf", type=Path, required=True, help="Original Hugging Face model directory")
    parser.add_argument("--out", type=Path, required=True, help="Output package directory")
    parser.add_argument("--preserve-quant", action="store_true", help="Required spelling of the supported export mode")
    parser.add_argument("--experts", choices=("fp4",), default="fp4", help="Expert quantization to preserve")
    parser.add_argument("--dense", choices=("fp8",), default="fp8", help="Dense quantization to preserve")
    parser.add_argument("--layers", help="Optional sidecar layer filter, e.g. 0,2-4")
    parser.add_argument("--expert-limit", type=int, help="Debug/smoke-test limit; omit for full 384 experts")
    parser.add_argument(
        "--sidecar-layout",
        choices=("layer-major", "expert-major"),
        default="layer-major",
        help="Expert sidecar physical layout: layer-major keeps whole tensor-family runs; expert-major stores gate/up/down contiguously per expert",
    )
    parser.add_argument("--skip-dense", action="store_true", help="Only write sidecar/package files")
    parser.add_argument("--skip-sidecar", action="store_true", help="Only write dense GGUF/package files")
    parser.add_argument("--dry-run", action="store_true", help="Validate mappings and print size estimates without writing")
    parser.add_argument("--force", action="store_true", help="Overwrite dense/sidecar outputs under --out")
    parser.add_argument("--allow-low-space", action="store_true", help="Do not fail the early free-space check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.preserve_quant:
        raise SystemExit("this exporter only supports native preservation mode; pass --preserve-quant")
    if args.expert_limit is not None and args.expert_limit <= 0:
        raise SystemExit("--expert-limit must be positive")

    hf_dir = args.hf.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    with (hf_dir / "config.json").open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if cfg.get("model_type") != "deepseek_v4":
        raise SystemExit(f"expected model_type=deepseek_v4, got {cfg.get('model_type')!r}")

    store = SafeTensorStore(hf_dir)
    actual_layer_count = infer_available_layer_count(store)
    config_layer_count = int(cfg["num_hidden_layers"])
    if actual_layer_count != config_layer_count:
        print(
            f"warning: config num_hidden_layers={config_layer_count}, "
            f"but safetensors contain {actual_layer_count} contiguous transformer layers; "
            "using tensor-derived layer count",
            flush=True,
        )
        cfg = dict(cfg)
        cfg["num_hidden_layers"] = actual_layer_count

    layers = parse_layers(args.layers, int(cfg["num_hidden_layers"]))
    actual_expert_count = infer_available_expert_count(store, layers)
    config_expert_count = int(cfg["n_routed_experts"])
    if actual_expert_count != config_expert_count:
        print(
            f"warning: config n_routed_experts={config_expert_count}, "
            f"but safetensors contain {actual_expert_count} routed experts per exported layer; "
            "using tensor-derived expert count",
            flush=True,
        )
        cfg = dict(cfg)
        cfg["n_routed_experts"] = actual_expert_count

    cfg = apply_tensor_geometry_overrides(cfg, store, layers, actual_expert_count)

    dense_specs = build_dense_specs(store)
    dense_bytes = sum(spec.nbytes for spec in dense_specs)
    sidecar_bytes = 0 if args.skip_sidecar else estimate_sidecar_bytes(store, cfg, layers, args.expert_limit)
    estimate = {"dense": 0 if args.skip_dense else dense_bytes, "sidecar": sidecar_bytes}

    print(f"note: DeepSeek-V4 HF source: {hf_dir}")
    print(f"note: dense tensors: {len(dense_specs)} estimated {human_bytes(dense_bytes)}")
    print(f"note: sidecar layers: {len(layers)} estimated {human_bytes(sidecar_bytes)}")
    print(f"note: output: {out_dir}")

    if args.dry_run:
        print("dry-run: validated tensor mapping; no files written")
        return 0

    required = (0 if args.skip_dense else dense_bytes) + sidecar_bytes
    check_space(out_dir, required, args.allow_low_space)
    prepare_output(out_dir, args.force, args.skip_dense, args.skip_sidecar)

    dense_written = False
    sidecar_written = False
    if not args.skip_dense:
        write_dense_gguf(store, cfg, dense_specs, out_dir / "dense" / "model-dense.gguf", hf_dir)
        dense_written = True
    if not args.skip_sidecar:
        write_sidecar(store, cfg, layers, out_dir / "sidecar", args.expert_limit, args.sidecar_layout)
        sidecar_written = True
    elif (out_dir / "sidecar").is_dir():
        refresh_sidecar_manifest(store, cfg, layers, out_dir / "sidecar", args.expert_limit, args.sidecar_layout)
        print(f"note: refreshed sidecar manifest for existing sidecar bins at {out_dir / 'sidecar'}")

    encoding_written = copy_encoding_assets(hf_dir, out_dir)
    if encoding_written:
        print(f"note: copied DS4 encoding assets to {out_dir / 'encoding'}")
    else:
        print(f"warning: no DS4 encoding assets found under {hf_dir / 'encoding'}")

    write_package_files(out_dir, cfg, dense_written, sidecar_written, encoding_written, estimate)
    print(f"done: wrote Flash-MoE SSD package to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
