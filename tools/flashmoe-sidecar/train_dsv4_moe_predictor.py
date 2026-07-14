#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


LABEL_PRIORITY = ("ffn_moe_topk_reduced", "ffn_moe_topk", "ffn_moe_hash_topk")


@dataclass
class TensorRecord:
    eval_index: int
    phase: str
    name: str
    layer: int
    ne: tuple[int, int, int, int]
    dtype: str
    file: Path


@dataclass
class LayerDataset:
    x: np.ndarray
    y_topk: np.ndarray
    temporal_topk: np.ndarray
    y_multihot: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small per-layer DeepSeek V4 Flash-MoE expert predictor from "
            "llama-cli --oracle-dump tensors. The supported runtime feature is "
            "attn_norm, which is available before attention."
        )
    )
    parser.add_argument("--oracle-dir", required=True, type=Path, help="directory written by llama-cli --oracle-dump")
    parser.add_argument("--out", required=True, type=Path, help="output .npz predictor model")
    parser.add_argument(
        "--feature",
        default="attn_norm",
        help="pre-attention tensor used as the hidden-state feature; runtime currently supports attn_norm only",
    )
    parser.add_argument("--topk", type=int, default=4, help="number of experts predicted per token")
    parser.add_argument("--n-experts", type=int, default=256, help="DS4 routed expert count")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "all"),
        default="all",
        help="train on records whose oracle phase matches this filter",
    )
    parser.add_argument(
        "--history",
        choices=("none", "same-layer-prev-token", "prev-layer-same-token", "both"),
        default="both",
        help="append cheap route-history features that are known before layer attention",
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=0,
        help="optional deterministic prefix limit for hidden dimensions after flattening; 0 keeps all",
    )
    parser.add_argument(
        "--feature-stride",
        type=int,
        default=1,
        help="optional deterministic hidden-dimension stride before --feature-limit",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="comma-separated layer ids to train; default trains every layer with feature+topk records",
    )
    parser.add_argument(
        "--accept-min-gain",
        type=float,
        default=0.05,
        help="required validation hit-rate gain over both temporal and no-prefetch baselines",
    )
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="write the model even if the required hit-rate gain is not met",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="optional raw C++ runtime predictor directory; default is OUT.raw",
    )
    parser.add_argument(
        "--max-dataset-gb",
        type=float,
        default=32.0,
        help="abort if estimated in-memory training arrays exceed this many GiB; use 0 to disable",
    )
    args = parser.parse_args()
    if args.feature != "attn_norm":
        raise SystemExit("--feature currently supports attn_norm only; runtime will reject other predictor features")
    if args.topk <= 0:
        raise SystemExit("--topk must be positive")
    if args.n_experts <= 0:
        raise SystemExit("--n-experts must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not (0.0 <= args.valid_frac < 1.0):
        raise SystemExit("--valid-frac must be in [0, 1)")
    if args.feature_stride <= 0:
        raise SystemExit("--feature-stride must be positive")
    if args.max_dataset_gb < 0:
        raise SystemExit("--max-dataset-gb must be non-negative")
    return args


def load_manifest(oracle_dir: Path) -> dict[str, Any]:
    path = oracle_dir / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing oracle manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "llama-cli-oracle-v1":
        raise SystemExit(f"unsupported oracle format: {manifest.get('format')!r}")
    return manifest


def read_records(oracle_dir: Path, manifest: dict[str, Any], phase: str) -> list[TensorRecord]:
    records: list[TensorRecord] = []
    for item in manifest.get("records", []):
        item_phase = str(item.get("phase", ""))
        if phase != "all" and item_phase != phase:
            continue
        layer = int(item.get("layer", -1))
        if layer < 0:
            continue
        ne_raw = item.get("ne", [1, 1, 1, 1])
        if len(ne_raw) != 4:
            raise SystemExit(f"invalid tensor shape in oracle manifest: {ne_raw!r}")
        records.append(
            TensorRecord(
                eval_index=int(item["eval_index"]),
                phase=item_phase,
                name=str(item["name"]),
                layer=layer,
                ne=tuple(max(1, int(v)) for v in ne_raw),
                dtype=str(item.get("dtype", "f32")),
                file=oracle_dir / str(item["file"]),
            )
        )
    return records


def tensor_as_reversed(record: TensorRecord) -> np.ndarray:
    if record.dtype != "f32":
        raise SystemExit(f"unsupported oracle tensor dtype for {record.file}: {record.dtype}")
    expected = math.prod(record.ne)
    arr = np.fromfile(record.file, dtype=np.float32)
    if arr.size != expected:
        raise SystemExit(f"size mismatch for {record.file}: got {arr.size} floats, expected {expected}")
    n0, n1, n2, n3 = record.ne
    return arr.reshape((n3, n2, n1, n0))


def token_axis_for_feature(record: TensorRecord) -> int:
    n0, n1, n2, n3 = record.ne
    if n2 > 1:
        return 2
    if n1 > 1:
        return 1
    if n3 > 1:
        return 3
    # Hidden-state feature tensors store hidden width in ne[0] and token count
    # in ne[1]. A single decode token is therefore shaped [hidden, 1, 1, 1],
    # not [1 token, hidden features].
    if n0 > 1:
        return 1
    return 0


def tensor_to_token_matrix(record: TensorRecord) -> np.ndarray:
    arr = tensor_as_reversed(record)
    token_axis = token_axis_for_feature(record)
    rev_axis = 3 - token_axis
    arr = np.moveaxis(arr, rev_axis, 0)
    return arr.reshape((record.ne[token_axis], -1)).astype(np.float32, copy=False)


def topk_to_token_matrix(record: TensorRecord, topk: int) -> np.ndarray:
    n0, n1, n2, n3 = record.ne
    if n0 < topk:
        raise SystemExit(f"{record.file} has top-k width {n0}, but --topk={topk}")
    arr = tensor_as_reversed(record)
    if n1 > 1:
        mat = np.moveaxis(arr, 2, 0).reshape((n1, -1))
    elif n2 > 1:
        mat = np.moveaxis(arr, 1, 0).reshape((n2, -1))
    else:
        mat = arr.reshape((1, -1))
    return np.rint(mat[:, :topk]).astype(np.int32, copy=False)


def parse_layers(value: str) -> set[int] | None:
    if not value.strip():
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def token_count_for_topk(record: TensorRecord) -> int:
    _, n1, n2, _ = record.ne
    if n1 > 1:
        return n1
    if n2 > 1:
        return n2
    return 1


def feature_column_count(record: TensorRecord, args: argparse.Namespace) -> int:
    token_axis = token_axis_for_feature(record)
    token_count = max(1, record.ne[token_axis])
    raw_width = math.prod(record.ne) // token_count
    width = (raw_width + args.feature_stride - 1) // args.feature_stride
    if args.feature_limit > 0:
        width = min(width, args.feature_limit)
    return width


def token_count_for_feature(record: TensorRecord) -> int:
    return record.ne[token_axis_for_feature(record)]


def estimate_dataset_bytes(
    args: argparse.Namespace,
    features_by_layer: dict[int, list[TensorRecord]],
    labels_by_layer: dict[int, list[TensorRecord]],
) -> int:
    rows_by_layer: dict[int, int] = {}
    total = 0
    max_occurrences = max(
        [len(v) for v in features_by_layer.values()] + [len(v) for v in labels_by_layer.values()] + [0]
    )

    for occurrence in range(max_occurrences):
        layers = sorted(
            layer
            for layer, feature_records in features_by_layer.items()
            if occurrence < len(feature_records) and occurrence < len(labels_by_layer.get(layer, []))
        )
        for layer in layers:
            feature_record = features_by_layer[layer][occurrence]
            label_record = labels_by_layer[layer][occurrence]
            rows = min(token_count_for_topk(label_record), token_count_for_feature(feature_record))
            feature_cols = feature_column_count(feature_record, args)
            total += rows * feature_cols * np.dtype(np.float32).itemsize
            total += rows * args.topk * np.dtype(np.int32).itemsize
            rows_by_layer[layer] = rows_by_layer.get(layer, 0) + rows

    history_components = 0
    if args.history in ("same-layer-prev-token", "both"):
        history_components += 1
    if args.history in ("prev-layer-same-token", "both"):
        history_components += 1

    for rows in rows_by_layer.values():
        total += rows * args.topk * np.dtype(np.int32).itemsize  # y_by_layer_full
        total += rows * args.topk * np.dtype(np.int32).itemsize  # temporal_topk
        total += rows * args.n_experts * np.dtype(np.float32).itemsize  # y_multihot
        total += rows * history_components * args.n_experts * np.dtype(np.float32).itemsize
    return total


def collect_label_records_by_layer(records: list[TensorRecord]) -> dict[int, list[TensorRecord]]:
    by_layer_name: dict[int, dict[str, list[TensorRecord]]] = {}
    for record in records:
        if record.name in LABEL_PRIORITY:
            by_layer_name.setdefault(record.layer, {}).setdefault(record.name, []).append(record)

    labels: dict[int, list[TensorRecord]] = {}
    for layer, by_name in by_layer_name.items():
        for candidate in LABEL_PRIORITY:
            if by_name.get(candidate):
                labels[layer] = by_name[candidate]
                break
    return labels


def multihot(topk_ids: np.ndarray, n_experts: int) -> np.ndarray:
    if np.any(topk_ids < 0) or np.any(topk_ids >= n_experts):
        lo = int(np.min(topk_ids))
        hi = int(np.max(topk_ids))
        raise SystemExit(f"expert id out of range for --n-experts={n_experts}: min={lo} max={hi}")
    out = np.zeros((topk_ids.shape[0], n_experts), dtype=np.float32)
    rows = np.arange(topk_ids.shape[0])[:, None]
    out[rows, topk_ids] = 1.0
    return out


def temporal_previous_topk(y_topk: np.ndarray) -> np.ndarray:
    out = np.full_like(y_topk, -1)
    if y_topk.shape[0] > 1:
        out[1:] = y_topk[:-1]
    return out


def append_history(
    x: np.ndarray,
    y_by_layer: dict[int, np.ndarray],
    layer: int,
    n_experts: int,
    history: str,
) -> np.ndarray:
    if history == "none":
        return x

    extras: list[np.ndarray] = []
    y = y_by_layer[layer]
    if history in ("same-layer-prev-token", "both"):
        prev = np.zeros((y.shape[0], n_experts), dtype=np.float32)
        if y.shape[0] > 1:
            prev[1:] = multihot(y[:-1], n_experts)
        extras.append(prev)

    if history in ("prev-layer-same-token", "both"):
        prev_layer_y = y_by_layer.get(layer - 1)
        prev_layer = np.zeros((y.shape[0], n_experts), dtype=np.float32)
        if prev_layer_y is not None:
            rows = min(y.shape[0], prev_layer_y.shape[0])
            prev_layer[:rows] = multihot(prev_layer_y[:rows], n_experts)
        extras.append(prev_layer)

    if not extras:
        return x
    return np.concatenate([x] + extras, axis=1)


def build_datasets(args: argparse.Namespace) -> dict[int, LayerDataset]:
    manifest = load_manifest(args.oracle_dir)
    records = read_records(args.oracle_dir, manifest, args.phase)
    wanted_layers = parse_layers(args.layers)
    labels_by_layer = collect_label_records_by_layer(records)
    features_by_layer: dict[int, list[TensorRecord]] = {}
    for record in records:
        if record.name == args.feature and (wanted_layers is None or record.layer in wanted_layers):
            features_by_layer.setdefault(record.layer, []).append(record)

    if args.max_dataset_gb > 0:
        estimated_bytes = estimate_dataset_bytes(args, features_by_layer, labels_by_layer)
        max_bytes = int(args.max_dataset_gb * (1024 ** 3))
        if estimated_bytes > max_bytes:
            raise SystemExit(
                f"estimated in-memory dataset is {estimated_bytes / (1024 ** 3):.1f} GiB, "
                f"above --max-dataset-gb={args.max_dataset_gb:.1f}; "
                "reduce --layers/--feature-limit, increase --feature-stride, or pass --max-dataset-gb 0"
            )

    layer_parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    max_occurrences = max(
        [len(v) for v in features_by_layer.values()] + [len(v) for v in labels_by_layer.values()] + [0]
    )
    for occurrence in range(max_occurrences):
        layers = sorted(
            layer
            for layer, feature_records in features_by_layer.items()
            if occurrence < len(feature_records) and occurrence < len(labels_by_layer.get(layer, []))
        )
        y_by_layer: dict[int, np.ndarray] = {}
        for layer in layers:
            label_record = labels_by_layer[layer][occurrence]
            y_by_layer[layer] = topk_to_token_matrix(label_record, args.topk)

        for layer in layers:
            feature_record = features_by_layer[layer][occurrence]
            y_topk = y_by_layer.get(layer)
            if feature_record is None or y_topk is None:
                continue
            x = tensor_to_token_matrix(feature_record)
            rows = min(x.shape[0], y_topk.shape[0])
            x = x[:rows]
            y_topk = y_topk[:rows]
            if args.feature_stride > 1:
                x = x[:, :: args.feature_stride]
            if args.feature_limit > 0:
                x = x[:, : args.feature_limit]
            layer_parts.setdefault(layer, []).append((x, y_topk))

    y_by_layer_full: dict[int, np.ndarray] = {
        layer: np.concatenate([part[1] for part in parts], axis=0)
        for layer, parts in layer_parts.items()
    }

    datasets: dict[int, LayerDataset] = {}
    for layer, parts in sorted(layer_parts.items()):
        x = np.concatenate([part[0] for part in parts], axis=0).astype(np.float32, copy=False)
        y_topk = y_by_layer_full[layer]
        temporal_topk = temporal_previous_topk(y_topk)
        x = append_history(x, y_by_layer_full, layer, args.n_experts, args.history)
        y = multihot(y_topk, args.n_experts)
        datasets[layer] = LayerDataset(x=x, y_topk=y_topk, temporal_topk=temporal_topk, y_multihot=y)
    return datasets


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[row, idx], axis=1)[:, ::-1]
    return idx[row, order]


def topk_metrics(scores: np.ndarray, truth: np.ndarray, k: int) -> dict[str, float]:
    pred = topk_indices(scores, k)
    hits = 0
    full = 0
    for row_pred, row_truth in zip(pred, truth):
        overlap = len(set(int(v) for v in row_pred) & set(int(v) for v in row_truth))
        hits += overlap
        full += int(overlap == k)
    denom = max(1, truth.shape[0] * k)
    return {
        "recall_at_k": hits / float(denom),
        "precision_at_k": hits / float(denom),
        "full_hit_rate": full / float(max(1, truth.shape[0])),
        "hits": float(hits),
        "refs": float(denom),
    }


def fixed_prediction_metrics(pred: np.ndarray, truth: np.ndarray, k: int) -> dict[str, float]:
    hits = 0
    full = 0
    for row_pred, row_truth in zip(pred, truth):
        pred_set = {int(v) for v in row_pred if int(v) >= 0}
        overlap = len(pred_set & set(int(v) for v in row_truth))
        hits += overlap
        full += int(overlap == k)
    denom = max(1, truth.shape[0] * k)
    return {
        "recall_at_k": hits / float(denom),
        "precision_at_k": hits / float(denom),
        "full_hit_rate": full / float(max(1, truth.shape[0])),
        "hits": float(hits),
        "refs": float(denom),
    }


def train_layer(
    dataset: LayerDataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = dataset.x
    y = dataset.y_multihot
    y_topk = dataset.y_topk
    temporal_topk = dataset.temporal_topk
    n = x.shape[0]
    perm = rng.permutation(n)
    valid_n = int(round(n * args.valid_frac))
    valid_idx = perm[:valid_n]
    train_idx = perm[valid_n:] if valid_n < n else perm

    mean = x[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-4)
    x_norm = (x - mean) / std

    d = x_norm.shape[1]
    e = args.n_experts
    w = (rng.normal(0.0, 0.01, size=(d, e))).astype(np.float32)
    b = np.zeros((e,), dtype=np.float32)
    pos_weight = float(max(1.0, (e - args.topk) / float(args.topk)))

    for _epoch in range(args.epochs):
        epoch_idx = rng.permutation(train_idx)
        for start in range(0, epoch_idx.size, args.batch_size):
            batch_idx = epoch_idx[start : start + args.batch_size]
            xb = x_norm[batch_idx]
            yb = y[batch_idx]
            pred = sigmoid(xb @ w + b)
            weight = 1.0 + yb * (pos_weight - 1.0)
            grad = (pred - yb) * weight / float(max(1, batch_idx.size))
            w -= args.lr * (xb.T @ grad + args.weight_decay * w)
            b -= args.lr * grad.sum(axis=0)

    train_scores = x_norm[train_idx] @ w + b
    train_metrics = topk_metrics(train_scores, y_topk[train_idx], args.topk)
    if valid_n > 0:
        valid_scores = x_norm[valid_idx] @ w + b
        valid_metrics = topk_metrics(valid_scores, y_topk[valid_idx], args.topk)
        temporal_metrics = fixed_prediction_metrics(temporal_topk[valid_idx], y_topk[valid_idx], args.topk)
    else:
        valid_metrics = {"recall_at_k": 0.0, "precision_at_k": 0.0, "full_hit_rate": 0.0, "hits": 0.0, "refs": 0.0}
        temporal_metrics = {"recall_at_k": 0.0, "precision_at_k": 0.0, "full_hit_rate": 0.0, "hits": 0.0, "refs": 0.0}
    no_prefetch_metrics = {"recall_at_k": 0.0, "precision_at_k": 0.0, "full_hit_rate": 0.0, "hits": 0.0, "refs": float(max(1, valid_n * args.topk))}

    arrays = {"w": w, "b": b, "mean": mean, "std": std}
    summary = {
        "tokens": int(n),
        "train_tokens": int(train_idx.size),
        "valid_tokens": int(valid_n),
        "input_dim": int(d),
        "train": train_metrics,
        "valid": valid_metrics,
        "baseline_temporal": temporal_metrics,
        "baseline_no_prefetch": no_prefetch_metrics,
    }
    return arrays, summary


def weighted_recall(rows: dict[str, Any], key: str) -> float:
    hits = 0.0
    refs = 0.0
    for layer_summary in rows.values():
        metrics = layer_summary[key]
        hits += float(metrics.get("hits", 0.0))
        refs += float(metrics.get("refs", 0.0))
    return 0.0 if refs <= 0.0 else hits / refs


def write_raw_runtime_model(
    raw_dir: Path,
    summary: dict[str, Any],
    model_arrays: dict[str, np.ndarray],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_layers: dict[str, Any] = {}
    for layer_text, layer_summary in summary["layers"].items():
        layer = int(layer_text)
        prefix = f"layer_{layer}_"
        files: dict[str, str] = {}
        for name in ("w", "b", "mean", "std"):
            arr = np.asarray(model_arrays[prefix + name], dtype=np.float32)
            file_name = f"layer_{layer}_{name}.f32"
            arr.tofile(raw_dir / file_name)
            files[name] = file_name
        raw_layers[layer_text] = {
            "input_dim": layer_summary["input_dim"],
            "weight_rows": int(model_arrays[prefix + "w"].shape[0]),
            "weight_cols": int(model_arrays[prefix + "w"].shape[1]),
            "w": files["w"],
            "b": files["b"],
            "mean": files["mean"],
            "std": files["std"],
        }

    metadata = {
        "format": "dsv4-flash-moe-hidden-predictor-raw-v1",
        "feature": summary["feature"],
        "history": summary["history"],
        "topk": summary["topk"],
        "n_experts": summary["n_experts"],
        "feature_limit": summary["feature_limit"],
        "feature_stride": summary["feature_stride"],
        "acceptance": summary["acceptance"],
        "layers": raw_layers,
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = build_datasets(args)
    if not datasets:
        raise SystemExit(
            "no trainable layers found; collect an oracle dump that includes "
            f"{args.feature} and ffn_moe_topk tensors"
        )

    model_arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "format": "dsv4-flash-moe-hidden-predictor-v1",
        "feature": args.feature,
        "history": args.history,
        "topk": args.topk,
        "n_experts": args.n_experts,
        "feature_limit": args.feature_limit,
        "feature_stride": args.feature_stride,
        "layers": {},
    }

    for layer, dataset in datasets.items():
        arrays, layer_summary = train_layer(dataset, args, rng)
        prefix = f"layer_{layer}_"
        for name, value in arrays.items():
            model_arrays[prefix + name] = value
        summary["layers"][str(layer)] = layer_summary
        valid = layer_summary["valid"]
        print(
            f"layer {layer:02d}: tokens={layer_summary['tokens']} "
            f"dim={layer_summary['input_dim']} "
            f"valid_recall@{args.topk}={valid['recall_at_k']:.3f} "
            f"temporal={layer_summary['baseline_temporal']['recall_at_k']:.3f} "
            f"valid_full={valid['full_hit_rate']:.3f}",
            file=sys.stderr,
        )

    aggregate = {
        "predictor_hit_rate": weighted_recall(summary["layers"], "valid"),
        "temporal_hit_rate": weighted_recall(summary["layers"], "baseline_temporal"),
        "no_prefetch_hit_rate": weighted_recall(summary["layers"], "baseline_no_prefetch"),
    }
    aggregate["gain_vs_temporal"] = aggregate["predictor_hit_rate"] - aggregate["temporal_hit_rate"]
    aggregate["gain_vs_no_prefetch"] = aggregate["predictor_hit_rate"] - aggregate["no_prefetch_hit_rate"]
    aggregate["required_min_gain"] = args.accept_min_gain
    aggregate["accepted"] = (
        aggregate["gain_vs_temporal"] >= args.accept_min_gain and
        aggregate["gain_vs_no_prefetch"] >= args.accept_min_gain
    )
    summary["acceptance"] = aggregate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model_arrays["metadata_json"] = np.array(json.dumps(summary, sort_keys=True), dtype="S")
    np.savez_compressed(args.out, **model_arrays)
    summary_path = args.out.with_suffix(args.out.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_dir = args.raw_dir if args.raw_dir is not None else args.out.with_suffix(args.out.suffix + ".raw")
    write_raw_runtime_model(raw_dir, summary, model_arrays)
    print(
        "acceptance: "
        f"predictor={aggregate['predictor_hit_rate']:.4f} "
        f"temporal={aggregate['temporal_hit_rate']:.4f} "
        f"no_prefetch={aggregate['no_prefetch_hit_rate']:.4f} "
        f"gain_vs_temporal={aggregate['gain_vs_temporal']:.4f} "
        f"gain_vs_no_prefetch={aggregate['gain_vs_no_prefetch']:.4f} "
        f"required={aggregate['required_min_gain']:.4f} "
        f"accepted={aggregate['accepted']}",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"wrote {summary_path}", file=sys.stderr)
    print(f"wrote {raw_dir}", file=sys.stderr)
    if not aggregate["accepted"] and not args.no_enforce_acceptance:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
