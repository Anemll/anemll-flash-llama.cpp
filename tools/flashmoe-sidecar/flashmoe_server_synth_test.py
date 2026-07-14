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


def print_completion_summary(body: dict) -> None:
    content = body.get("content") or ""
    timings = body.get("timings") or {}
    print(f"content_len={len(content)} stop_type={body.get('stop_type')}")
    if timings:
        print(
            f"prompt_n={timings.get('prompt_n')} cache_n={timings.get('cache_n')} "
            f"prompt_tps={timings.get('prompt_per_second')} predicted_n={timings.get('predicted_n')} "
            f"decode_tps={timings.get('predicted_per_second')}"
        )
    if "tokens_cached" in body:
        print(f"tokens_cached={body.get('tokens_cached')}")


def print_chat_summary(body: dict) -> None:
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tool_calls = msg.get("tool_calls") or []
    timings = body.get("timings") or {}

    print(
        f"finish_reason={choice.get('finish_reason')} content_len={len(content)} "
        f"reasoning_len={len(reasoning)} tool_calls={len(tool_calls)}"
    )
    if timings:
        print(
            f"prompt_n={timings.get('prompt_n')} cache_n={timings.get('cache_n')} "
            f"prompt_tps={timings.get('prompt_per_second')} predicted_n={timings.get('predicted_n')} "
            f"decode_tps={timings.get('predicted_per_second')}"
        )


def build_completion_request(prompt: str, n_predict: int, id_slot: int, cache_prompt: bool, temperature: float) -> dict:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": cache_prompt,
        "id_slot": id_slot,
        "stream": False,
        "temperature": temperature,
    }


def build_chat_request(model: str, prompt: str, n_predict: int, temperature: float) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": n_predict,
        "stream": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single synthetic long-prompt test against a running llama-server"
    )
    parser.add_argument("--mode", choices=["completion", "chat"], default="completion")
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--model", default="minimax-m2")
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-label", choices=["1k", "4k", "16k", "22k"], default="16k")
    parser.add_argument("--n-predict", type=int, default=128)
    parser.add_argument("--id-slot", type=int, default=0)
    parser.add_argument("--cache-prompt", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    args = parser.parse_args()

    prompt = read_prompt(args.prompt_file, args.prompt_label)

    if args.mode == "completion":
        body = build_completion_request(
            prompt=prompt,
            n_predict=args.n_predict,
            id_slot=args.id_slot,
            cache_prompt=args.cache_prompt,
            temperature=args.temperature,
        )
    else:
        body = build_chat_request(
            model=args.model,
            prompt=prompt,
            n_predict=args.n_predict,
            temperature=args.temperature,
        )

    response = post_json(args.url, body, args.api_key, args.request_timeout)
    if args.mode == "completion":
        print_completion_summary(response)
    else:
        print_chat_summary(response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
