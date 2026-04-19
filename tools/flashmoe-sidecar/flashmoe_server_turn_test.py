#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_prompt(path: str | None, label: str | None) -> str:
    root = Path(__file__).resolve().parent
    if path:
        return Path(path).expanduser().resolve().read_text(encoding="utf-8")
    if label:
        prompt_path = root / "prompts" / "coding" / f"coding_{label}.txt"
        return prompt_path.read_text(encoding="utf-8")
    raise SystemExit("need one of --prompt-file or --prompt-label")


def post_json(url: str, body: dict, api_key: str | None, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {payload}") from exc


def print_summary(label: str, body: dict) -> str:
    content = body.get("content") or ""
    timings = body.get("timings") or {}
    print(f"{label}: content_len={len(content)} stop_type={body.get('stop_type')}")
    if timings:
        print(
            f"{label}: prompt_n={timings.get('prompt_n')} cache_n={timings.get('cache_n')} "
            f"prompt_tps={timings.get('prompt_per_second')} predicted_n={timings.get('predicted_n')} "
            f"decode_tps={timings.get('predicted_per_second')}"
        )
    if "tokens_cached" in body:
        print(f"{label}: tokens_cached={body.get('tokens_cached')}")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a big-prefill + follow-up turn test against a running llama-server")
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-label", choices=["1k", "4k", "16k", "22k"])
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--id-slot", type=int, default=0)
    parser.add_argument("--api-key")
    parser.add_argument("--followup", default="Continue with three more implementation details.")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args()

    prompt = read_prompt(args.prompt_file, args.prompt_label)

    first_req = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "cache_prompt": True,
        "id_slot": args.id_slot,
        "stream": False,
    }
    first = post_json(args.url, first_req, args.api_key, args.request_timeout)
    first_content = print_summary("turn1", first).rstrip()

    second_prompt = prompt.rstrip() + "\n\n" + first_content + "\n\n" + args.followup + "\n"
    second_req = {
        "prompt": second_prompt,
        "n_predict": args.n_predict,
        "cache_prompt": True,
        "id_slot": args.id_slot,
        "stream": False,
    }
    second = post_json(args.url, second_req, args.api_key, args.request_timeout)
    print_summary("turn2", second)
    return 0


if __name__ == "__main__":
    sys.exit(main())
