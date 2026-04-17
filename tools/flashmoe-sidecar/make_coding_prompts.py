#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "prompts" / "coding"
DEFAULT_TARGETS = (1024, 4096, 16384, 22528)
DEFAULT_TOKENIZER_REPO = "MiniMaxAI/MiniMax-M2"


@dataclass(frozen=True)
class PromptSpec:
    label: str
    target_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate coding-style benchmark prompts with exact token counts. "
            "By default this uses the MiniMax tokenizer.json directly; if "
            "--llama-model is provided it will first try llama-tokenize --ids "
            "and fall back to the tokenizer.json path if that fails."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for generated prompts (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGETS),
        help="target token counts to emit (default: 1024 4096 16384 22528)",
    )
    parser.add_argument(
        "--tokenizer-repo",
        default=DEFAULT_TOKENIZER_REPO,
        help=f"Hugging Face tokenizer repo for exact counts (default: {DEFAULT_TOKENIZER_REPO})",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="optional local tokenizer.json path; overrides --tokenizer-repo",
    )
    parser.add_argument(
        "--llama-model",
        type=Path,
        help="optional GGUF model path for llama-tokenize --ids",
    )
    parser.add_argument(
        "--llama-tokenize-cmd",
        type=Path,
        default=Path("./build/bin/llama-tokenize"),
        help="path to llama-tokenize binary (default: ./build/bin/llama-tokenize)",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("auto", "llama", "hf"),
        default="auto",
        help="tokenization source preference (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing prompt files",
    )
    return parser.parse_args()


def label_for_target(target: int) -> str:
    if target % 1024 == 0:
        return f"{target // 1024}k"
    if target % 1000 == 0:
        return f"{target // 1000}k"
    return str(target)


def load_tokenizer(args: argparse.Namespace) -> tuple[Tokenizer, str]:
    if args.tokenizer_json:
        return Tokenizer.from_file(str(args.tokenizer_json)), str(args.tokenizer_json)
    path = hf_hub_download(args.tokenizer_repo, "tokenizer.json")
    return Tokenizer.from_file(path), f"{args.tokenizer_repo}:tokenizer.json"


def repo_inventory() -> str:
    modules = [
        ("server/auth/session_cache.ts", "session cache invalidation and tenant-scoped auth"),
        ("server/jobs/build_queue.ts", "bounded build scheduler with retry backoff"),
        ("server/jobs/log_mux.ts", "interleaved job log fan-in and chunk flushing"),
        ("server/repos/git_indexer.ts", "repository scan and incremental file fingerprinting"),
        ("server/search/symbol_graph.ts", "symbol edges, reverse references, and path ranking"),
        ("server/search/query_planner.ts", "heuristics for code search terms and fallback regex"),
        ("server/rpc/ws_transport.ts", "streamed RPC frames and reconnect handling"),
        ("server/rpc/request_store.ts", "request lifecycle and timeout accounting"),
        ("shared/config/runtime_flags.ts", "feature flags and staged rollout rules"),
        ("shared/config/workspace_limits.ts", "token, memory, and concurrency limits"),
        ("apps/web/src/routes/editor.tsx", "editor route, file tabs, and diagnostics panel"),
        ("apps/web/src/routes/search.tsx", "search results grouping and ranking badges"),
        ("apps/web/src/components/diff_view.tsx", "split diff renderer and fold state"),
        ("apps/web/src/components/trace_table.tsx", "runtime trace table with sticky filters"),
        ("apps/web/src/lib/fetch_json.ts", "typed fetch wrapper with retry budget"),
        ("apps/web/src/lib/use_stream.ts", "stream reader and backpressure aware hooks"),
    ]
    lines = [
        "Repository overview:",
        "This benchmark prompt describes a moderately large coding assistant backend with a web IDE, build queue, code search, and streamed traces.",
        "",
        "Important modules:",
    ]
    for idx, (path, desc) in enumerate(modules, start=1):
        lines.append(f"{idx:02d}. {path} - {desc}")
    return "\n".join(lines)


def make_ticket(idx: int) -> str:
    service = f"tenant-{idx % 11:02d}"
    shard = f"shard-{idx % 7}"
    route = f"/api/workspaces/{1000 + idx}/apply"
    file_base = idx % 17
    return f"""
### Coding Ticket {idx:03d}: inconsistent streamed patch apply under retry pressure

Context:
- Active tenant: {service}
- Build shard: {shard}
- Primary route: {route}
- Editor session id: editor-{4000 + idx}
- Search request id: q-{90000 + idx}

Problem statement:
Users report that a streamed patch can succeed in the UI but still leave the repository index stale.
The failure usually happens after a reconnect or after two fast retries from the browser.
The issue is not deterministic; it appears when log chunks, repo indexing, and websocket acks overlap.

Files most likely involved:
- server/jobs/build_queue.ts
- server/repos/git_indexer.ts
- server/rpc/ws_transport.ts
- shared/config/runtime_flags.ts
- apps/web/src/routes/editor.tsx
- apps/web/src/components/diff_view.tsx

Relevant code sketch:
```ts
export async function applyPatchAndRefresh(ctx: RequestContext, patch: PatchChunk[]) {{
    const requestId = ctx.requestId;
    await queueBuild({{
        tenantId: ctx.tenantId,
        requestId,
        onLog(line) {{
            ctx.stream.write({{ type: "log", line }});
        }},
    }});

    await repoIndexer.refreshWorkspace(ctx.workspaceId, {{
        requestId,
        eagerSymbols: ctx.flags.eagerSymbols,
        maxChangedFiles: ctx.flags.maxChangedFiles,
    }});

    return {{
        requestId,
        refreshedAt: Date.now(),
        changedFiles: patch.length,
    }};
}}
```

Observed failure details:
```text
2026-04-17T10:12:{(idx * 7) % 60:02d}.114Z INFO  request_store: begin request=q-{90000 + idx} tenant={service}
2026-04-17T10:12:{(idx * 7 + 1) % 60:02d}.229Z WARN  ws_transport: duplicate ack sequence={(idx % 5) + 1}
2026-04-17T10:12:{(idx * 7 + 2) % 60:02d}.402Z INFO  build_queue: build accepted on {shard} priority={(idx % 4) + 1}
2026-04-17T10:12:{(idx * 7 + 3) % 60:02d}.516Z WARN  git_indexer: refresh skipped due to active lease workspace={1000 + idx}
2026-04-17T10:12:{(idx * 7 + 4) % 60:02d}.612Z ERROR editor: stale diagnostics after apply request=q-{90000 + idx}
```

Minimal failing test:
```ts
it("refreshes the workspace index after a streamed patch even if the websocket reconnects", async () => {{
    const ctx = createTestRequestContext({{
        tenantId: "{service}",
        workspaceId: {1000 + idx},
        flags: {{ eagerSymbols: true, maxChangedFiles: 64 }},
    }});

    wsTransport.injectReconnect(ctx.sessionId);
    await applyPatchAndRefresh(ctx, makePatchChunks({6 + (idx % 5)}));

    expect(repoIndexer.lastRefresh?.workspaceId).toBe({1000 + idx});
    expect(repoIndexer.lastRefresh?.requestId).toBe(ctx.requestId);
    expect(editorState(ctx.workspaceId).diagnostics.stale).toBe(false);
}});
```

Acceptance criteria:
1. Duplicate websocket acknowledgements do not suppress the repository refresh.
2. The build queue never clears the request state before the indexer observes the final patch result.
3. Editor diagnostics and symbol graph refresh become monotonic after reconnect.
4. The fix remains safe under 32 concurrent apply requests and partial log flushes.
5. Add tests for reconnect, retry, and lease handoff.

Review checklist:
- identify the likely ordering bug
- propose the state transition change
- list the exact files to edit
- suggest a regression test matrix
- call out any risk to streamed log ordering or latency
""".strip()


def build_corpus(tokenizer: Tokenizer, max_target: int) -> str:
    parts = [
        "You are debugging a large codebase. Read the repository notes, the bug tickets, and the failing tests. Focus on race conditions, ordering issues, incremental indexing, and streamed RPC behavior.",
        repo_inventory(),
    ]

    idx = 1
    while True:
        for _ in range(8):
            parts.append(make_ticket(idx))
            idx += 1
        corpus = "\n\n".join(parts) + "\n"
        if len(tokenizer.encode(corpus).ids) >= max_target + 1024:
            return corpus


def tokenize_with_llama(args: argparse.Namespace, text: str) -> list[int]:
    if not args.llama_model:
        raise RuntimeError("llama tokenization requested without --llama-model")
    cmd = [
        str(args.llama_tokenize_cmd),
        "--log-disable",
        "--ids",
        "--stdin",
        "-m",
        str(args.llama_model),
    ]
    proc = subprocess.run(
        cmd,
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-tokenize failed with exit code {proc.returncode}"
        )
    try:
        ids = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("failed to parse llama-tokenize --ids output") from exc
    if not isinstance(ids, list) or not ids or not all(isinstance(v, int) for v in ids):
        raise RuntimeError("llama-tokenize returned an invalid token id payload")
    return ids


def tokenize_text(args: argparse.Namespace, tokenizer: Tokenizer, text: str) -> tuple[list[int], str]:
    mode = args.tokenizer_mode
    if mode in ("auto", "llama") and args.llama_model:
        try:
            return tokenize_with_llama(args, text), "llama-tokenize"
        except RuntimeError as exc:
            if mode == "llama":
                raise
            print(f"warning: {exc}; falling back to tokenizer.json", file=sys.stderr)
    return tokenizer.encode(text).ids, "tokenizer.json"


def decode_exact_prefix(tokenizer: Tokenizer, ids: list[int], target: int) -> str:
    if target > len(ids):
        raise ValueError(f"target {target} exceeds available corpus tokens {len(ids)}")
    text = tokenizer.decode(ids[:target])
    actual = len(tokenizer.encode(text).ids)
    if actual == target:
        return text

    for window in range(1, 129):
        lo = max(1, target - window)
        hi = min(len(ids), target + window)
        for candidate in range(lo, hi + 1):
            text = tokenizer.decode(ids[:candidate])
            actual = len(tokenizer.encode(text).ids)
            if actual == target:
                return text
    raise RuntimeError(f"failed to recover an exact {target}-token prefix")


def emit_prompts(
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    ids: list[int],
    tokenizer_source: str,
) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [PromptSpec(label_for_target(target), target) for target in sorted(set(args.targets))]
    manifest: dict[str, object] = {
        "tokenizer": tokenizer_source,
        "tokenizer_json": str(args.tokenizer_json) if args.tokenizer_json else None,
        "tokenizer_repo": None if args.tokenizer_json else args.tokenizer_repo,
        "notes": [
            "Prompt files are exact token prefixes under the selected tokenizer.",
            "The 22k sample uses 22528 tokens (22 * 1024).",
            "Counts do not include any BOS or chat-template tokens that a runtime may inject separately.",
        ],
        "prompts": [],
    }

    for spec in specs:
        out_path = args.out_dir / f"coding_{spec.label}.txt"
        if out_path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing file without --force: {out_path}")
        text = decode_exact_prefix(tokenizer, ids, spec.target_tokens)
        actual = len(tokenizer.encode(text).ids)
        if actual != spec.target_tokens:
            raise RuntimeError(
                f"internal error: wrote {actual} tokens for target {spec.target_tokens}"
            )
        out_path.write_text(text, encoding="utf-8")
        try:
            relative_path = out_path.resolve().relative_to(SCRIPT_DIR.resolve())
        except ValueError:
            relative_path = out_path.resolve()

        entry = {
            "label": spec.label,
            "target_tokens": spec.target_tokens,
            "actual_tokens": actual,
            "bytes_utf8": out_path.stat().st_size,
            "path": str(relative_path),
        }
        manifest["prompts"].append(entry)
        print(f"wrote {spec.label:>4} -> {out_path} ({actual} tokens)")

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest -> {manifest_path}")


def main() -> None:
    args = parse_args()
    tokenizer, tokenizer_desc = load_tokenizer(args)
    corpus = build_corpus(tokenizer, max(args.targets))
    ids, tokenizer_source = tokenize_text(args, tokenizer, corpus)
    if len(ids) < max(args.targets):
        raise SystemExit(
            f"generated corpus was too small: {len(ids)} tokens < requested {max(args.targets)}"
        )
    emit_prompts(tokenizer, args, ids, f"{tokenizer_source} ({tokenizer_desc})")


if __name__ == "__main__":
    main()
