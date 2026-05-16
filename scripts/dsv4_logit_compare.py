#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_logits(path: Path, include_prefill: bool):
    records = []
    metadata = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("kind") == "metadata":
                metadata.append(obj)
                continue
            if obj.get("kind") != "logits":
                continue
            if not include_prefill and obj.get("phase") != "decode":
                continue
            records.append(obj)
    return metadata, records


def topk_map(record):
    return {int(item["id"]): float(item["logit"]) for item in record.get("topk", [])}


def token_text(record, token_id):
    for item in record.get("topk", []):
        if int(item["id"]) == int(token_id):
            return str(item.get("text", ""))
    return ""


def rank_of(record, token_id):
    for item in record.get("topk", []):
        if int(item["id"]) == int(token_id):
            return int(item.get("rank", 0))
    return None


def overlap_metrics(generic, fused):
    g = topk_map(generic)
    f = topk_map(fused)
    overlap = sorted(set(g) & set(f))
    diffs = [abs(g[token_id] - f[token_id]) for token_id in overlap]
    if diffs:
        mean_abs = sum(diffs) / len(diffs)
        rms = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        max_abs = max(diffs)
        max_rel = max(
            diffs[i] / max(abs(g[overlap[i]]), 1.0e-12)
            for i in range(len(overlap))
        )
    else:
        mean_abs = rms = max_abs = max_rel = None
    return {
        "topk_overlap_count": len(overlap),
        "max_abs_logit_err_topk_overlap": max_abs,
        "mean_abs_logit_err_topk_overlap": mean_abs,
        "rms_logit_err_topk_overlap": rms,
        "max_rel_logit_err_topk_overlap": max_rel,
    }


def fmt_float(value):
    if value is None:
        return "n/a"
    return f"{value:.9g}"


def describe_pair(generic, fused):
    generic_top1 = int(generic["top1_id"])
    fused_top1 = int(fused["top1_id"])
    metrics = overlap_metrics(generic, fused)
    return {
        "token_index": generic.get("token_index"),
        "position": generic.get("position"),
        "generic_top1_id": generic_top1,
        "generic_top1_text": token_text(generic, generic_top1),
        "generic_top1_logit": float(generic["top1_logit"]),
        "generic_top2_id": int(generic["top2_id"]),
        "generic_top2_text": token_text(generic, int(generic["top2_id"])),
        "generic_top2_logit": float(generic["top2_logit"]),
        "generic_top1_top2_margin": float(generic["top1_top2_margin"]),
        "fused_top1_id": fused_top1,
        "fused_top1_text": token_text(fused, fused_top1),
        "fused_top1_logit": float(fused["top1_logit"]),
        "fused_top2_id": int(fused["top2_id"]),
        "fused_top2_text": token_text(fused, int(fused["top2_id"])),
        "fused_top2_logit": float(fused["top2_logit"]),
        "fused_top1_top2_margin": float(fused["top1_top2_margin"]),
        "generic_top1_rank_in_fused": rank_of(fused, generic_top1),
        "fused_top1_rank_in_generic": rank_of(generic, fused_top1),
        **metrics,
    }


def write_report(out_path: Path, baseline_path: Path, fused_path: Path, generic_records, fused_records, first_diff, last_pair, note):
    lines = []
    lines.append("DSV4 first-divergence report")
    lines.append(f"baseline: {baseline_path}")
    lines.append(f"fused:    {fused_path}")
    lines.append(f"generic records: {len(generic_records)}")
    lines.append(f"fused records:   {len(fused_records)}")
    if note:
        lines.append(f"note: {note}")
    lines.append("")

    if first_diff is None:
        lines.append("first divergence: none within aligned records")
        if last_pair is not None:
            lines.append("")
            lines.append("last aligned token:")
            append_pair(lines, last_pair)
    else:
        lines.append("first divergence:")
        append_pair(lines, first_diff)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_pair(lines, pair):
    fields = [
        ("token_index", pair["token_index"]),
        ("position", pair["position"]),
        ("generic token", f'{pair["generic_top1_id"]} {pair["generic_top1_text"]!r}'),
        ("fused token", f'{pair["fused_top1_id"]} {pair["fused_top1_text"]!r}'),
        ("generic top1 logit", fmt_float(pair["generic_top1_logit"])),
        ("generic top2", f'{pair["generic_top2_id"]} {pair["generic_top2_text"]!r} logit={fmt_float(pair["generic_top2_logit"])}'),
        ("generic top1/top2 margin", fmt_float(pair["generic_top1_top2_margin"])),
        ("fused top1 logit", fmt_float(pair["fused_top1_logit"])),
        ("fused top2", f'{pair["fused_top2_id"]} {pair["fused_top2_text"]!r} logit={fmt_float(pair["fused_top2_logit"])}'),
        ("fused top1/top2 margin", fmt_float(pair["fused_top1_top2_margin"])),
        ("generic top1 rank in fused", pair["generic_top1_rank_in_fused"]),
        ("fused top1 rank in generic", pair["fused_top1_rank_in_generic"]),
        ("top-k overlap count", pair["topk_overlap_count"]),
        ("max_abs_logit_err", fmt_float(pair["max_abs_logit_err_topk_overlap"])),
        ("mean_abs_logit_err", fmt_float(pair["mean_abs_logit_err_topk_overlap"])),
        ("rms_logit_err", fmt_float(pair["rms_logit_err_topk_overlap"])),
        ("max_rel_logit_err", fmt_float(pair["max_rel_logit_err_topk_overlap"])),
    ]
    for key, value in fields:
        lines.append(f"  {key}: {value}")
    lines.append("  metric scope: overlapping tokens in the dumped top-k lists")


def main():
    parser = argparse.ArgumentParser(description="Compare DSV4 llama-cli top-logit JSONL dumps.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("--out", type=Path, default=Path("/tmp/dsv4_first_divergence_report.txt"))
    parser.add_argument("--include-prefill", action="store_true")
    parser.add_argument("--stop-at-first-diff", action="store_true")
    args = parser.parse_args()

    _, generic_records = load_logits(args.baseline, args.include_prefill)
    _, fused_records = load_logits(args.fused, args.include_prefill)
    n = min(len(generic_records), len(fused_records))
    note = ""
    if len(generic_records) != len(fused_records):
        note = "record counts differ; compared aligned prefix only"

    first_diff = None
    last_pair = None
    for i in range(n):
        pair = describe_pair(generic_records[i], fused_records[i])
        last_pair = pair
        if pair["generic_top1_id"] != pair["fused_top1_id"]:
            first_diff = pair
            if args.stop_at_first_diff:
                break
            break

    write_report(args.out, args.baseline, args.fused, generic_records, fused_records, first_diff, last_pair, note)
    print(f"wrote {args.out}")
    if first_diff is None:
        print("first divergence: none")
    else:
        print(f"first divergence token_index={first_diff['token_index']} "
              f"generic={first_diff['generic_top1_id']} fused={first_diff['fused_top1_id']} "
              f"generic_margin={fmt_float(first_diff['generic_top1_top2_margin'])} "
              f"fused_margin={fmt_float(first_diff['fused_top1_top2_margin'])}")


if __name__ == "__main__":
    main()
