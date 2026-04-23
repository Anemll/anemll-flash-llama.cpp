#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SOURCE_PROMPT = SCRIPT_DIR / "prompts" / "coding" / "coding_16k.txt"
DEFAULT_TOKENIZER_REPO = "MiniMaxAI/MiniMax-M2"


try:
    sys.path.insert(0, str(SCRIPT_DIR))
    import make_coding_prompts as coding_prompts
except Exception as exc:  # pragma: no cover - import failure is user-facing
    print(
        f"error: failed to import make_coding_prompts.py helpers: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass(frozen=True)
class RunCase:
    name: str
    prompt_target: int
    ctx_size: int
    extra_env: Dict[str, str]


@dataclass(frozen=True)
class RunResult:
    case: RunCase
    returncode: int
    stdout_path: Path
    stderr_path: Path
    output_text_path: Path
    output_text: str
    prompt_tps: str | None
    generation_tps: str | None
    nwg_paths: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether auto non-vec 2-pass NWG diverges from forced NWG=4. "
            "The script generates exact 32K/48K repeated-prompt inputs from the "
            "existing coding_16k prompt, then runs the normal Flash-MoE wrapper "
            "in raw-completion mode and compares the generated text."
        )
    )
    parser.add_argument("package_dir", type=Path, help="MiniMax Flash-MoE package directory")
    parser.add_argument(
        "--source-prompt",
        type=Path,
        default=DEFAULT_SOURCE_PROMPT,
        help=f"base prompt to repeat before exact token trimming (default: {DEFAULT_SOURCE_PROMPT})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="directory for generated prompts and run artifacts (default: mktemp)",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="keep the generated work directory even if --work-dir was not provided",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only generate exact 32K/48K prompts and print their paths",
    )
    parser.add_argument(
        "--assert-expected",
        action="store_true",
        help=(
            "exit non-zero unless 32K auto==forced4 and 48K auto!=forced4; "
            "useful for quick regression checking"
        ),
    )
    parser.add_argument("--n-predict", type=int, default=1, help="tokens to generate per run (default: 1)")
    parser.add_argument("--seed", type=int, default=123, help="sampling seed (default: 123)")
    parser.add_argument("--temp", type=float, default=0.0, help="temperature (default: 0.0)")
    parser.add_argument("--ctx-32k", type=int, default=40000, help="context size for the 32K control run")
    parser.add_argument("--ctx-48k", type=int, default=64000, help="context size for the 48K repro run")
    parser.add_argument("--batch", type=int, default=2048, help="logical batch size")
    parser.add_argument("--ubatch", type=int, default=32, help="physical batch size")
    parser.add_argument("--slot-bank", type=int, default=128, help="slot-bank size")
    parser.add_argument("--moe-topk", type=int, default=4, help="MoE top-k")
    parser.add_argument("--cache-io-split", type=int, default=8, help="MoE cache io split")
    parser.add_argument("--prefill-batch", type=int, default=16384, help="prefill batch size")
    parser.add_argument("--prefill-banks", type=int, default=4, help="prefill banks")
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="optional local tokenizer.json used for exact prompt generation",
    )
    parser.add_argument(
        "--tokenizer-repo",
        default=DEFAULT_TOKENIZER_REPO,
        help=f"Hugging Face tokenizer repo for exact prompt generation (default: {DEFAULT_TOKENIZER_REPO})",
    )
    return parser.parse_args()


def make_tokenizer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        tokenizer_json=args.tokenizer_json,
        tokenizer_repo=args.tokenizer_repo,
        llama_model=None,
        llama_tokenize_cmd=REPO_ROOT / "build" / "bin" / "llama-tokenize",
        tokenizer_mode="hf",
    )


def ensure_exact_prompts(args: argparse.Namespace, work_dir: Path) -> dict[int, Path]:
    tokenizer_args = make_tokenizer_args(args)
    tokenizer, tokenizer_desc = coding_prompts.load_tokenizer(tokenizer_args)

    source_text = args.source_prompt.read_text(encoding="utf-8")
    targets = (32768, 49152)
    target_max = max(targets)

    corpus = source_text
    while len(tokenizer.encode(corpus).ids) < target_max + 1024:
        corpus += source_text

    ids = tokenizer.encode(corpus).ids
    if len(ids) < target_max:
        raise RuntimeError(
            f"repeated source prompt only produced {len(ids)} tokens; need at least {target_max}"
        )

    prompt_dir = work_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths: dict[int, Path] = {}
    for target in targets:
        text = coding_prompts.decode_exact_prefix(tokenizer, ids, target)
        actual = len(tokenizer.encode(text).ids)
        if actual != target:
            raise RuntimeError(f"internal error: prompt target {target} decoded to {actual} tokens")

        label = f"coding_repeat_{target // 1024}k_exact.txt"
        out_path = prompt_dir / label
        out_path.write_text(text, encoding="utf-8")
        prompt_paths[target] = out_path

    manifest = prompt_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"source_prompt={args.source_prompt.resolve()}",
                f"tokenizer={tokenizer_desc}",
                *(f"{target}={path}" for target, path in prompt_paths.items()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_paths


def run_case(args: argparse.Namespace, work_dir: Path, prompt_path: Path, case: RunCase) -> RunResult:
    run_dir = work_dir / "runs" / case.name
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    output_text_path = run_dir / "generated.txt"

    cmd = [
        "bash",
        str(SCRIPT_DIR / "run_minimax_m2_flash.sh"),
        str(args.package_dir.resolve()),
        "-f",
        str(prompt_path.resolve()),
        "--moe-predict-top1-prev",
        "--moe-prefill-layer-major",
        "--moe-prefill-batch",
        str(args.prefill_batch),
        "--moe-prefill-micro-batch",
        "auto",
        "--moe-prefill-io-split",
        str(args.cache_io_split),
        "--moe-prefill-banks",
        str(args.prefill_banks),
        "-n",
        str(args.n_predict),
        "-st",
    ]

    env = os.environ.copy()
    env.update(
        {
            "GGML_METAL_TENSOR_ENABLE": "1",
            "DISPLAY_PROMPT": "0",
            "GGML_METAL_FLASH_ATTN_DEBUG": "1",
            "GGML_METAL_FLASH_ATTN_NONVEC_M4_ENABLE": "1",
            "GGML_METAL_FLASH_ATTN_NONVEC_2PASS_ENABLE": case.extra_env.get(
                "GGML_METAL_FLASH_ATTN_NONVEC_2PASS_ENABLE",
                "1",
            ),
            "SIMPLE_IO": "1",
            "RAW_COMPLETION": "1",
            "MOE_TOPK": str(args.moe_topk),
            "MOE_SLOT_BANK": str(args.slot_bank),
            "MOE_CACHE_IO_SPLIT": str(args.cache_io_split),
            "BATCH": str(args.batch),
            "UBATCH": str(args.ubatch),
            "CTX": str(case.ctx_size),
            "SEED": str(args.seed),
            "TEMP": str(args.temp),
        }
    )
    env.update(case.extra_env)

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    output_text = proc.stdout
    if output_text.startswith("running:"):
        first_newline = output_text.find("\n")
        if first_newline >= 0:
            output_text = output_text[first_newline + 1 :]
        else:
            output_text = ""
    output_text_path.write_text(output_text, encoding="utf-8")

    combined = proc.stdout + "\n" + proc.stderr
    prompt_tps, generation_tps = extract_perf_summary(combined)
    nwg_paths = extract_path_lines(combined)

    return RunResult(
        case=case,
        returncode=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_text_path=output_text_path,
        output_text=output_text,
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        nwg_paths=nwg_paths,
    )


def extract_perf_summary(text: str) -> tuple[str | None, str | None]:
    matches = re.findall(r"\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]", text)
    if not matches:
        return None, None
    prompt_tps, generation_tps = matches[-1]
    return prompt_tps, generation_tps


def extract_path_lines(text: str) -> tuple[str, ...]:
    paths = []
    for line in text.splitlines():
        if "ggml_metal_flash_attn_log_path_once:" in line and "phase=prefill-like" in line:
            paths.append(line.strip())
    return tuple(paths)


def outputs_match(lhs: RunResult, rhs: RunResult) -> bool:
    return lhs.output_text == rhs.output_text


def summarize_result(result: RunResult) -> str:
    parts = [result.case.name]
    if result.prompt_tps is not None:
        parts.append(f"prompt={result.prompt_tps} t/s")
    if result.generation_tps is not None:
        parts.append(f"gen={result.generation_tps} t/s")
    parts.append(f"rc={result.returncode}")
    return ", ".join(parts)


def print_output_preview(label: str, text: str) -> None:
    preview = text[:120]
    print(f"  {label}: {preview!r}")


def main() -> int:
    args = parse_args()

    if not args.package_dir.exists():
        print(f"error: missing package directory: {args.package_dir}", file=sys.stderr)
        return 2
    if not args.source_prompt.is_file():
        print(f"error: missing source prompt: {args.source_prompt}", file=sys.stderr)
        return 2

    created_tmp = False
    if args.work_dir is None:
        args.work_dir = Path(tempfile.mkdtemp(prefix="flashmoe-nwg-divergence-"))
        created_tmp = True
    else:
        args.work_dir.mkdir(parents=True, exist_ok=True)

    try:
        prompt_paths = ensure_exact_prompts(args, args.work_dir)

        print(f"work_dir: {args.work_dir}")
        print(f"32K prompt: {prompt_paths[32768]}")
        print(f"48K prompt: {prompt_paths[49152]}")

        if args.prepare_only:
            return 0

        cases = [
            RunCase("32k_auto", 32768, args.ctx_32k, {}),
            RunCase("32k_forced4", 32768, args.ctx_32k, {"GGML_METAL_FLASH_ATTN_NONVEC_2PASS_NWG": "4"}),
            RunCase("48k_ref_1pass", 49152, args.ctx_48k, {"GGML_METAL_FLASH_ATTN_NONVEC_2PASS_ENABLE": "0"}),
            RunCase("48k_auto", 49152, args.ctx_48k, {}),
            RunCase("48k_forced4", 49152, args.ctx_48k, {"GGML_METAL_FLASH_ATTN_NONVEC_2PASS_NWG": "4"}),
        ]

        results: dict[str, RunResult] = {}
        for case in cases:
            prompt_path = prompt_paths[case.prompt_target]
            print(f"\nrunning {case.name} ...")
            result = run_case(args, args.work_dir, prompt_path, case)
            results[case.name] = result
            print(f"  {summarize_result(result)}")
            if result.nwg_paths:
                print(f"  first prefill path: {result.nwg_paths[0]}")
                if len(result.nwg_paths) > 1:
                    print(f"  last prefill path:  {result.nwg_paths[-1]}")
            print(f"  stdout: {result.stdout_path}")
            print(f"  stderr: {result.stderr_path}")

        same_32 = outputs_match(results["32k_auto"], results["32k_forced4"])
        same_48 = outputs_match(results["48k_auto"], results["48k_forced4"])
        ref_eq_auto = outputs_match(results["48k_ref_1pass"], results["48k_auto"])
        ref_eq_forced4 = outputs_match(results["48k_ref_1pass"], results["48k_forced4"])

        print("\ncomparison:")
        print(f"  32K auto vs forced4: {'MATCH' if same_32 else 'DIFFER'}")
        print(f"  48K auto vs forced4: {'MATCH' if same_48 else 'DIFFER'}")
        print(f"  48K 1pass vs auto:   {'MATCH' if ref_eq_auto else 'DIFFER'}")
        print(f"  48K 1pass vs forced4:{' MATCH' if ref_eq_forced4 else ' DIFFER'}")

        if not same_32:
            print_output_preview("32k_auto", results["32k_auto"].output_text)
            print_output_preview("32k_forced4", results["32k_forced4"].output_text)
        if not same_48:
            print_output_preview("48k_auto", results["48k_auto"].output_text)
            print_output_preview("48k_forced4", results["48k_forced4"].output_text)

        if args.assert_expected and (not same_32 or same_48):
            print(
                "\nassertion failed: expected 32K control to match and 48K repro to diverge",
                file=sys.stderr,
            )
            return 1

        return 0
    finally:
        if created_tmp and not args.keep_work_dir and args.work_dir.exists():
            shutil.rmtree(args.work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
