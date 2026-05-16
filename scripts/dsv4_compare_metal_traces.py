#!/usr/bin/env python3
"""Compare lightweight DS4 Metal JSONL traces.

The trace format is intentionally loose. Rows may come from this repository or
from an external DS4 runner, and missing fields are treated as unknown rather
than fatal.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


STAGES = [
    "ffn",
    "attn_qkv",
    "attn_kv",
    "attn_compress",
    "attn_core",
    "attn_out",
    "attn_hc_pre",
    "attn_hc_post",
    "kv_cache",
    "head",
    "other",
]

FFN_STAGES = [
    "ffn_norm",
    "router",
    "topk",
    "expert_gate_up",
    "expert_swiglu",
    "expert_down",
    "expert_weighted_sum",
    "shared_gate_up",
    "shared_swiglu",
    "shared_down",
    "ffn_residual",
    "ffn_moe_boundary",
    "ffn_other",
]

ELEMENTWISE_OPS = {
    "ADD", "MUL", "DIV", "SCALE", "CLAMP", "UNARY", "GLU", "SILU", "SQRT",
    "SQR", "NEG", "EXP", "SUM_ROWS", "SOFT_MAX", "ARGMAX", "ARGSORT",
}


def load_rows(path: str | None) -> tuple[list[dict[str, Any]], str | None]:
    if not path:
        return [], "not provided"
    p = Path(path)
    if not p.exists():
        return [], f"missing: {path}"
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                return rows, f"invalid JSON at {path}:{lineno}: {exc}"
    return rows, None


def dispatch_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("dispatch_count", 1))
    except (TypeError, ValueError):
        return 1


def token(row: dict[str, Any]) -> int:
    try:
        return int(row.get("token", -1))
    except (TypeError, ValueError):
        return -1


def command_buffer(row: dict[str, Any]) -> str:
    value = row.get("command_buffer_index", row.get("command_buffer", "unknown"))
    return str(value)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_token: dict[int, int] = collections.Counter()
    cb_by_token: dict[int, set[str]] = collections.defaultdict(set)
    by_stage: dict[str, int] = collections.Counter()
    kernels_by_stage: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    kernels: collections.Counter[str] = collections.Counter()
    ffn_substages: collections.Counter[str] = collections.Counter()
    ffn_kernels: collections.Counter[tuple[str, str, str, str]] = collections.Counter()
    generic_fragments: collections.Counter[str] = collections.Counter()

    for row in rows:
        d = dispatch_count(row)
        t = token(row)
        stage = str(row.get("stage_bucket") or "other")
        if stage not in STAGES:
            stage = "other"
        kernel = str(row.get("kernel") or row.get("function") or row.get("ggml_op") or "unknown")
        by_token[t] += d
        cb_by_token[t].add(command_buffer(row))
        by_stage[stage] += d
        kernels_by_stage[stage][kernel] += d
        kernels[kernel] += d
        op = str(row.get("ggml_op") or "")
        detail_stage = ffn_detail_stage(row)
        if detail_stage:
            ffn_substages[detail_stage] += d
            tensor = str(row.get("tensor") or row.get("tensor_name") or "")
            ffn_kernels[(detail_stage, op, kernel, tensor_prefix(tensor))] += d
        generic_fragments[fragment_class(op)] += d

    tokens = sorted(t for t in by_token if t >= 0)
    dispatch_total = sum(by_token.values())
    cb_total = sum(len(cb_by_token[t]) for t in tokens)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "dispatch_total": dispatch_total,
        "dispatch_per_token": dispatch_total / len(tokens) if tokens else 0.0,
        "command_buffers_total": cb_total,
        "command_buffers_per_token": cb_total / len(tokens) if tokens else 0.0,
        "by_token": by_token,
        "cb_by_token": cb_by_token,
        "by_stage": by_stage,
        "kernels_by_stage": kernels_by_stage,
        "kernels": kernels,
        "ffn_substages": ffn_substages,
        "ffn_kernels": ffn_kernels,
        "generic_fragments": generic_fragments,
    }


def ffn_detail_stage(row: dict[str, Any]) -> str | None:
    stage_bucket = str(row.get("stage_bucket") or "")
    stage = str(row.get("stage") or "")
    if stage in FFN_STAGES:
        return stage
    if stage_bucket != "ffn":
        return None
    return "ffn_other"


def tensor_prefix(name: str) -> str:
    if not name:
        return "unknown"
    s = name.split(" (", 1)[0]
    if "-" in s:
        head, tail = s.rsplit("-", 1)
        if tail.isdigit():
            return head
    return s


def fragment_class(op: str) -> str:
    if op == "MUL_MAT":
        return "MUL_MAT"
    if op == "MUL_MAT_ID":
        return "MUL_MAT_ID"
    if op in {"RMS_NORM", "NORM"}:
        return "RMS_NORM"
    if op == "CPY":
        return "CPY"
    if op == "CONT":
        return "CONT"
    if op in {"RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"}:
        return "RESHAPE/VIEW"
    if op in ELEMENTWISE_OPS:
        return "elementwise"
    return "other"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt_float(v: float) -> str:
    return f"{v:.2f}"


def render(name_a: str, rows_a: list[dict[str, Any]], err_a: str | None,
           name_b: str, rows_b: list[dict[str, Any]], err_b: str | None) -> str:
    a = summarize(rows_a)
    b = summarize(rows_b)
    lines: list[str] = ["# DS4 Metal Trace Comparison", ""]
    if err_a:
        lines += [f"- {name_a}: {err_a}"]
    if err_b:
        lines += [f"- {name_b}: {err_b}"]
    if err_a or err_b:
        lines.append("")

    lines += [
        "## Summary",
        "",
        md_table(
            ["trace", "rows", "tokens", "dispatches", "dispatches/token", "command buffers/token"],
            [
                [name_a, a["rows"], len(a["tokens"]), a["dispatch_total"], fmt_float(a["dispatch_per_token"]), fmt_float(a["command_buffers_per_token"])],
                [name_b, b["rows"], len(b["tokens"]), b["dispatch_total"], fmt_float(b["dispatch_per_token"]), fmt_float(b["command_buffers_per_token"])],
            ],
        ),
        "",
        "## Per-Stage Dispatches",
        "",
    ]

    stage_rows = []
    for stage in STAGES:
        av = a["by_stage"].get(stage, 0)
        bv = b["by_stage"].get(stage, 0)
        ratio = (av / bv) if bv else ""
        stage_rows.append([stage, av, bv, fmt_float(ratio) if ratio != "" else "n/a"])
    lines += [md_table(["stage", name_a, name_b, "ours/ds4"], stage_rows), ""]

    ffn_rows = []
    for stage in FFN_STAGES:
        av = a["ffn_substages"].get(stage, 0)
        bv = b["ffn_substages"].get(stage, 0)
        apt = av / len(a["tokens"]) if a["tokens"] else 0.0
        bpt = bv / len(b["tokens"]) if b["tokens"] else 0.0
        ratio = apt / bpt if bpt else None
        ffn_rows.append([
            stage,
            fmt_float(apt),
            fmt_float(bpt),
            fmt_float(ratio) if ratio is not None else "n/a",
            av,
            bv,
        ])
    lines += [
        "## FFN/MoE Decomposition",
        "",
        md_table(["substage", f"{name_a} rows/token", f"{name_b} rows/token", "ratio", name_a, name_b], ffn_rows),
        "",
    ]

    top_ffn_rows = []
    for rank, ((stage, op, kernel, prefix), count) in enumerate(a["ffn_kernels"].most_common(20), 1):
        cpt = count / len(a["tokens"]) if a["tokens"] else 0.0
        top_ffn_rows.append([rank, stage, op or "unknown", kernel, prefix, count, fmt_float(cpt)])
    lines += [
        "## Top 20 Ours FFN Kernels/Ops",
        "",
        md_table(["rank", "stage", "ggml_op", "kernel", "tensor_prefix", "count", "count/token"], top_ffn_rows),
        "",
    ]

    frag_order = ["MUL_MAT", "MUL_MAT_ID", "RMS_NORM", "CPY", "CONT", "RESHAPE/VIEW", "elementwise", "other"]
    frag_rows = []
    for key in frag_order:
        av = a["generic_fragments"].get(key, 0)
        bv = b["generic_fragments"].get(key, 0)
        apt = av / len(a["tokens"]) if a["tokens"] else 0.0
        bpt = bv / len(b["tokens"]) if b["tokens"] else 0.0
        frag_rows.append([key, fmt_float(apt), fmt_float(bpt), av, bv])
    lines += [
        "## Generic Fragmentation Summary",
        "",
        md_table(["class", f"{name_a} rows/token", f"{name_b} rows/token", name_a, name_b], frag_rows),
        "",
    ]

    common_tokens = sorted(set(a["tokens"]) | set(b["tokens"]))
    token_rows = []
    for t in common_tokens[:80]:
        token_rows.append([
            t,
            a["by_token"].get(t, 0),
            b["by_token"].get(t, 0),
            len(a["cb_by_token"].get(t, set())),
            len(b["cb_by_token"].get(t, set())),
        ])
    lines += ["## Per-Token Totals", "", md_table(["token", f"{name_a} dispatches", f"{name_b} dispatches", f"{name_a} cmd buffers", f"{name_b} cmd buffers"], token_rows), ""]

    lines += ["## Per-Stage Kernel Lists", ""]
    for stage in STAGES:
        ak = a["kernels_by_stage"].get(stage, collections.Counter())
        bk = b["kernels_by_stage"].get(stage, collections.Counter())
        if not ak and not bk:
            continue
        lines += [f"### {stage}", ""]
        keys = sorted(set(ak) | set(bk), key=lambda k: (-(ak.get(k, 0) + bk.get(k, 0)), k))[:32]
        lines += [md_table(["kernel", name_a, name_b], [[k, ak.get(k, 0), bk.get(k, 0)] for k in keys]), ""]

    extra_rows = []
    for kernel, count in a["kernels"].most_common(40):
        delta = count - b["kernels"].get(kernel, 0)
        if delta > 0:
            extra_rows.append([kernel, count, b["kernels"].get(kernel, 0), delta])
    lines += ["## Top Extra Kernels In Our Path", "", md_table(["kernel", name_a, name_b, "extra"], extra_rows[:20]), ""]

    lines += [
        "## Notes",
        "",
        "- Timing fields are optional. No CPU/GPU gap table is emitted unless traces include timing data.",
        "- Dispatch counts are summed from `dispatch_count`; rows without that field count as one dispatch.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ds4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ours_rows, ours_err = load_rows(args.ours)
    ds4_rows, ds4_err = load_rows(args.ds4)
    text = render("ours", ours_rows, ours_err, "ds4", ds4_rows, ds4_err)
    Path(args.out).write_text(text, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
