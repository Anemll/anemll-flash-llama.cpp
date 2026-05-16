#!/usr/bin/env python3
"""Compare paired routed-MoE replace-generic tensor dumps."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


FIELDS = [
    ("topk_ids", "first topk_ids mismatch"),
    ("topk_weights", "first topk_weights mismatch"),
    ("ffn_input.hash", "first ffn_input mismatch"),
    ("final.hash", "first final_ffn mismatch"),
    ("hc_ffn_post_input.hash", "first hc_ffn_post input mismatch"),
    ("hc_ffn_post_output.hash", "first hc_ffn_post output mismatch"),
]


def load(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_token(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            out[int(row["token"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return out


def get_path(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_mismatch(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]], path: str) -> tuple[int | None, Any, Any]:
    for token in sorted(set(a) & set(b)):
        av = get_path(a[token], path)
        bv = get_path(b[token], path)
        if av != bv:
            return token, av, bv
    return None, None, None


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: dsv4_compare_rmoe_replace_dump.py BASELINE.jsonl REPLACE.jsonl", file=sys.stderr)
        return 2
    baseline = by_token(load(sys.argv[1]))
    replace = by_token(load(sys.argv[2]))
    print(f"baseline rows: {len(baseline)}")
    print(f"replace rows: {len(replace)}")
    common = sorted(set(baseline) & set(replace))
    print(f"common tokens: {len(common)}")
    for path, label in FIELDS:
        if common and all(get_path(baseline[t], path) is None and get_path(replace[t], path) is None for t in common):
            print(f"{label}: unavailable")
            continue
        token, av, bv = first_mismatch(baseline, replace, path)
        if token is None:
            print(f"{label}: none")
        else:
            print(f"{label}: token {token}")
            print(f"  baseline: {av}")
            print(f"  replace:  {bv}")
    first_any: int | None = None
    first_labels: list[str] = []
    for path, label in FIELDS:
        token, _av, _bv = first_mismatch(baseline, replace, path)
        if token is None:
            continue
        if first_any is None or token < first_any:
            first_any = token
            first_labels = [label]
        elif token == first_any:
            first_labels.append(label)
    if first_any is None:
        print("classification: no paired dump mismatch")
    else:
        joined = ", ".join(first_labels)
        print(f"classification: first mismatch at token {first_any}: {joined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
