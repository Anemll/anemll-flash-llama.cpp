# Flash-MoE GGUF Sidecar Tools

These tools implement the first Flash-MoE GGUF workflow for the vendored `llama.cpp` tree:

- keep the original GGUF as the canonical model artifact
- extract routed MoE tensors into a sidecar manifest plus raw bank files
- preserve exact tensor payload sizes and exact GGUF quantized bytes
- reload those expert tensors through `--moe-sidecar`
- support both a resident packed-bank path and an experimental streamed slot-bank path
- estimate persistent-bank cost and coverage from exact sidecar bytes plus `--moe-trace` output

By default, the helper scripts keep generated sidecars outside the repo under `~/Models/flash`.
Override that root with `FLASH_ROOT=/some/other/path` or set `SIDECAR_DIR` directly.

Modeling workflow: [`docs/moe-bank-modeling-workflow.md`](../../docs/moe-bank-modeling-workflow.md)

Detailed MiniMax M2.7 note: [`tools/flashmoe-sidecar/MINIMAX_M2.md`](./MINIMAX_M2.md)

Layer-major prefill algorithm note: [`docs/flashmoe-layer-major-dedup-single-sidecar.md`](../../docs/flashmoe-layer-major-dedup-single-sidecar.md)

## Model index

Per-model extract + run recipes in this document:

| Model | Arch | Extract | Run |
|---|---|---|---|
| Qwen3.5-35B-A3B | qwen3moe | [Extract](#extract-a-qwen35-sidecar) | [Run](#run-with-the-sidecar) |
| Gemma4-26B-A4B | gemma4 | [Extract](#extract-a-gemma4-26b-a4b-sidecar) | [Run](#run-gemma4-26b-a4b-with-the-sidecar) |
| Kimi K2 / K2.5 | deepseek2 (MLA) | [Extract](#extract-only-selected-layers) | [Run](#estimate-persistent-bank-cost-and-coverage) |
| MiniMax-M2.7 | minimax-m2 | [Export](#export-a-minimax-m27-flash-package) | [Run](#run-minimax-m27-with-flash-moe) |
| **GLM-5.1** | **glm-dsa (MLA + DSA indexer)** | [**Extract**](#extract-a-glm-51-sidecar) | [**Run**](#run-glm-51-with-the-sidecar) |

## Current scope

- Supported runtime modes in this build:
  - `stock`
  - `resident`
  - `resident-bank`
  - `slot-bank`
  - `oracle-all-hit`
  - `oracle-prefetch`
- Manifest layout implemented here:
  - `layer_major_whole_tensor`
- Future work still not implemented end-to-end:
  - dynamic quant bank switching

## Extract a Qwen3.5 sidecar

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --out-dir ~/Models/flash/qwen35
```

## Extract a Gemma4-26B-A4B sidecar

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --out-dir ~/Models/gemma4/packed_experts \
  --force
```

## Extract a GLM-5.1 sidecar

GLM-5.1 (`glm-dsa` arch) is a 256×22B MoE with MLA attention + DeepSeek-style sparse-attention indexer.
The extractor walks the 6 GGUF shards automatically — point it at the first shard.

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --out-dir ~/Models/GLM/GLM-5.1-sidecar
```

The sidecar is roughly 177 GiB at IQ1_M/IQ2_XXS (76 MoE layers × 256 experts × gate/up/down). Make sure the target SSD has the room.

## Export a MiniMax M2.7 Flash package

MiniMax M2.7 (`minimax-m2` arch) currently uses a dense-package workflow in this fork:

- routed experts are extracted into `sidecar/`
- dense and shared tensors are exported into `model-dense.gguf`
- package defaults are written to `flashmoe-package.json`

For the checked `UD-IQ2_XXS` package in this workspace, the generated package is about:

- `2.58 GiB` dense GGUF
- `58.3 GiB` routed sidecar

Build the package in one step:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS/MiniMax-M2.7-UD-IQ2_XXS-00001-of-00003.gguf \
  --out-dir ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  --force
```

Generic form:

```bash
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model /path/to/MiniMax-M2.7-00001-of-00003.gguf \
  --out-dir /path/to/MiniMax-M2.7-Flash \
  --force
```

For more background and sizing notes, see [`MINIMAX_M2.md`](./MINIMAX_M2.md).

## Export a dense-only GGUF (experimental)

If you already have a verified sidecar, you can also export a compact dense/shared-only GGUF for faster loading and to avoid keeping both the original full GGUF shards and the sidecar on disk.

This path is currently experimental:

- it is tested for `slot-bank` runs
- it excludes routed expert weight tensors entirely and keeps only dense/shared tensors in the GGUF
- it still requires the sidecar manifest plus `layer_XXX.bin` files at runtime
- `flashmoe-package.json` is informational only; `llama-cli` currently loads `model-dense.gguf` directly

Keep the original GGUF shards until you have verified that the dense GGUF + sidecar pair loads correctly on your machine.

Use `--perf` when comparing the compact dense GGUF against the original full GGUF shards. In the current `llama-cli` path it prints the Flash-MoE routed profile table, cached expert hit rate, and Metal replay cache hit rate, plus the prompt/generation throughput summary. It does not currently print the standard libllama `load time` footer on exit, so use shell timing or `/usr/bin/time` if you want an exact startup comparison for `model-dense.gguf`.

GLM-5.1 example:

```bash
python3 ../local_tools/export_dense_gguf.py \
  --model ~/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --sidecar ~/Models/GLM/GLM-5.1-sidecar \
  --out-dir ~/Models/GLM/GLM-5.1-IQ1-Dense \
  --force
```

Output:

- `~/Models/GLM/GLM-5.1-IQ1-Dense/model-dense.gguf`
- `~/Models/GLM/GLM-5.1-IQ1-Dense/flashmoe-package.json`

Run the compact dense GGUF with the same sidecar:

```bash
./build/bin/llama-cli \
  -m ~/Models/GLM/GLM-5.1-IQ1-Dense/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/GLM/GLM-5.1-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 4096 \
  --seed 123 --temp 0 \
  -p "What is Apple Neural Engine?" \
  -n 128 -st --moe-cache-io-split 4
```

Compared with the full GLM shard set, the compact dense GGUF is about 14 GiB on disk while routed expert bytes continue to come from the sidecar.

## Run MiniMax M2.7 with Flash-MoE

Important: MiniMax routed-expert width is controlled here by `--moe-topk`, or `MOE_TOPK=...` when using the wrapper. That is different from the sampler flag `--top-k`.

The wrapper default is speed-oriented (`MOE_TOPK=4`). MiniMax M2.7 natively routes to `8` experts, so use `MOE_TOPK=8` when you want model-default behavior.

M5 Max, native MiniMax routing width:

```bash
MOE_TOPK=8 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st \
  -n 128 \
  -p "What is Apple Neural Engine? Answer in 3 sentences."
```

This path was smoke-tested locally on an M5 Max with:

- `MOE_TOPK=8`
- `MOE_SLOT_BANK=64`
- `CTX=4096`
- `BATCH=64`
- `UBATCH=1`
- `N_GPU_LAYERS=999`

All wrapper environment variables for `run_minimax_m2_flash.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MOE_TOPK` | 4 (package) | Routed experts per token (not sampler `--top-k`) |
| `MOE_SLOT_BANK` | 64 (package) | Host-memory expert cache size (slots) |
| `MOE_CACHE_IO_SPLIT` | 4 (package) | Split each expert pread into N page-aligned chunks; `1` = single read, higher = more I/O parallelism |
| `MOE_PREFETCH_TEMPORAL` | 1 (package) | Set `0` to disable temporal prefetch |
| `PREFETCH_SIDECAR` | (unset) | Alternate sidecar path for prefetch-only reads |
| `CTX` | 4096 | Context size |
| `BATCH` | 64 | Batch size |
| `UBATCH` | 1 | Micro-batch size |
| `N_GPU_LAYERS` | 999 | GPU layer count (`0` = CPU only) |
| `SEED` | (unset) | RNG seed for reproducibility |
| `TEMP` | (unset) | Sampling temperature |
| `LLAMA_BIN` | `./build/bin/llama-cli` | Path to the llama-cli binary |

Examples:

```bash
# Disable prefetch, single-read expert loads
MOE_TOPK=4 MOE_SLOT_BANK=64 MOE_CACHE_IO_SPLIT=1 MOE_PREFETCH_TEMPORAL=0 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 128 -p "Hello"

# Higher I/O parallelism on a fast SSD
MOE_TOPK=8 MOE_SLOT_BANK=64 MOE_CACHE_IO_SPLIT=8 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 128 -p "Hello"
```

## Generate coding prompt sweep samples

For prompt-throughput comparisons it helps to keep the content style fixed and only change prompt length.
The helper below emits coding-task prompts at exact tokenizer lengths using the MiniMax tokenizer.

Targets generated by default:

- `1k` = `1024` tokens
- `4k` = `4096` tokens
- `16k` = `16384` tokens
- `22k` = `22528` tokens

Generate the prompt set:

```bash
python3 ./tools/flashmoe-sidecar/make_coding_prompts.py \
  --out-dir ./tools/flashmoe-sidecar/prompts/coding \
  --force
```

If you want to prefer runtime tokenization, pass a GGUF that works with `llama-tokenize --ids`:

```bash
python3 ./tools/flashmoe-sidecar/make_coding_prompts.py \
  --llama-model ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash/model-dense.gguf \
  --out-dir ./tools/flashmoe-sidecar/prompts/coding \
  --force
```

Note: counts are plain tokenizer counts for the prompt text itself. If a runtime adds a BOS token or a chat wrapper separately, the evaluated prompt length may differ by a token or two.

The `run_minimax_m2_flash.sh` wrapper now canonicalizes `-f/--file` and
`-bf/--binary-file` inputs before launching `llama-cli`. That means copy-pasted
examples like `-f ./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt` work
both from the repo root and from other current directories.

If Terminal.app still mangles a multi-line paste, you can skip the `-f` line
entirely and use:

```bash
PROMPT_FILE=./tools/flashmoe-sidecar/prompts/coding/coding_4k.txt \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh ...
```

The wrapper resolves `PROMPT_FILE` the same way it resolves `-f`.

For the built-in coding prompt samples, you can avoid paths entirely:

```bash
PROMPT_LABEL=4k \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh ...
```

Supported labels are: `1k`, `4k`, `16k`, and `22k`.

Run the four prompt sizes through the MiniMax wrapper:

```bash
bash ./tools/flashmoe-sidecar/run_minimax_m2_coding_prompt_sweep.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  --moe-prefill-layer-major \
  --moe-prefill-stripe 3:2:2 \
  --moe-prefill-batch 2048 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4 \
  -n 1 -st
```

You can also sweep a subset:

```bash
PROMPT_LABELS="1k 4k 16k" \
bash ./tools/flashmoe-sidecar/run_minimax_m2_coding_prompt_sweep.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -n 1 -st
```

M1 Max 64 GB — `MOE_TOPK=4` for lower expert I/O with a large slot bank:

```bash
MOE_TOPK=4 MOE_SLOT_BANK=128 bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"
```

Lower-memory starting point for a 24 GB-class M5:

```bash
MOE_TOPK=8 \
MOE_SLOT_BANK=32 \
CTX=2048 \
BATCH=32 \
UBATCH=1 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st \
  -n 96 \
  -p "Summarize Apple Neural Engine in 3 sentences."
```

Practical MiniMax memory notes for this exact `UD-IQ2_XXS-Flash` package:

- `MOE_SLOT_BANK=64` reserves about `17.15 GiB` before context/runtime overhead
- `MOE_SLOT_BANK=32` reserves about `9.87 GiB`
- `MOE_SLOT_BANK=24` reserves about `8.04 GiB`
- `MOE_SLOT_BANK=16` reserves about `6.22 GiB`

If a smaller-memory system is tight, reduce in this order:

- `MOE_SLOT_BANK` from `32` to `24`, then `16`
- `CTX` from `2048` to `1024`
- `BATCH` from `32` to `16`
- `N_GPU_LAYERS=0` only as a last resort

Reproducible comparison runs:

- pin `LLAMA_BIN` so the wrapper uses the exact binary you want to compare
- pin `SEED` and `TEMP`
- keep prompt text and `-n` identical across runs
- prefer `TEMP=0.2` over `TEMP=0` for long code prompts, because greedy decode can fall into repeats

```bash
LLAMA_BIN=/absolute/path/to/build/bin/llama-cli \
MOE_TOPK=4 \
SEED=123 \
TEMP=0.2 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"

LLAMA_BIN=/absolute/path/to/build/bin/llama-cli \
MOE_TOPK=8 \
SEED=123 \
TEMP=0.2 \
bash ./tools/flashmoe-sidecar/run_minimax_m2_flash.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  -st -n 4096 \
  -p "Make a game of Space Invaders in pygame"
```

## Run GLM-5.1 with the sidecar

Portable slot-bank recipe:

```bash
./build/bin/llama-cli \
  -m ~/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/GLM/GLM-5.1-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --no-warmup \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 4096 \
  --seed 123 --temp 0 \
  -p "What is Apple Neural Engine?" \
  -n 128 -st
```

Fast path on this branch for the current best GLM decode throughput on Apple Silicon (M5 MAX):

- export `model-dense.gguf` first and run the dense-only GGUF plus sidecar pair
- keep `--moe-predict-prev-token` off
- enable Metal replay plus CPU-visible slot writes
- use a larger slot bank such as `90` or `96`
- use `--perf` so you can confirm routed source time, cached expert hit rate, and replay hit rate
- if you keep a second copy of the sidecar on another SSD, point `--moe-prefetch-sidecar` at that alternate location so prefetch reads can come from a different drive

The recipe below is the end-user path we used to reproduce about `6.5` to `6.7 tok/s` on an M5 Max 128 GB with GLM-5.1 IQ1_M:

```bash
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY=1 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT=65536 \
LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB=0 \
LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES=1 \
./build/bin/llama-cli --perf \
  -m ~/Models/GLM/GLM-5.1-IQ1-Dense/model-dense.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/GLM/GLM-5.1-sidecar \
  --moe-prefetch-sidecar /Volumes/PrefetchSSD/GLM-5.1-sidecar \
  --moe-slot-bank 90 \
  --moe-topk 4 \
  --moe-cache-io-split 4 \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 4096 \
  --seed 123 --temp 0 \
  -p "Create game of Space Invaders in Swift" \
  -n 600 -st
```

If you have the memory headroom, `--moe-slot-bank 96` can be slightly better than `90`, but the gain is small compared with the extra reserve.

If you do not have a second drive for prefetch, omit `--moe-prefetch-sidecar` and prefetch will continue to use `--moe-sidecar`.

GLM-5.1 specific notes:

- **`--moe-topk 4`** is a reduction-only override of the model's native K=8. On IQ1_M/IQ2_XXS quants the K=4 vs K=8 quality gap is within noise for general use, while halving per-token expert I/O for ~2× decode. Drop the flag to use native K=8 if you need maximum fidelity.
- **`--moe-slot-bank 64`** is the starting point. With native K=8 the bank has only 8× headroom; if you have RAM, try `128` or `256` for higher reuse on warm caches.
- **`--moe-prefetch-temporal`** is the single biggest knob — it overlaps next-layer expert `pread`s with current-layer GPU compute.
- **`--moe-prefetch-sidecar PATH`** is optional. When set, prefetch reads use that alternate sidecar directory or manifest path while demand misses continue to use `--moe-sidecar`. This is useful when you keep a second sidecar copy on another SSD.
- **`--perf`** is recommended for tuning. In this CLI it prints the Flash-MoE routed breakdown (`Expert I/O source`, `Expert upload`), cached expert hit rate, and Metal replay cache hit rate, plus the prompt/generation throughput summary. It does not currently print `load time` on exit.
- **The best-known fast path on this branch is higher than the older baseline.** With dense-only export, Metal replay, CPU-visible slot writes, predictor off, and a `90` to `96` slot bank, GLM-5.1 IQ1_M is currently landing around `6.5` to `6.7 tok/s` on M5 Max 128 GB steady-state runs. Older full-GGUF or smaller-bank recipes remain useful as simpler baselines.
- If you hit hangs or memory pressure (the DSV2/Kimi GPU-bank path is shared with GLM), fall back with:
  ```bash
  LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1 ./build/bin/llama-cli ...
  ```

## Benchmark weighted 3-drive stripe splits

If you keep identical GLM sidecar copies on three drives and want to test weighted striping before changing the runtime format, use:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_stripe_bench.py \
  --sidecar-a ~/Models/GLM/test \
  --sidecar-b /Volumes/SN1/test \
  --sidecar-c /Volumes/SN2/test \
  --mode expert \
  --layers 3-13 \
  --families ffn_gate_exps,ffn_up_exps,ffn_down_exps \
  --max-layers 8 \
  --max-experts 16 \
  --repeats 3 \
  --no-cache \
  --drop-cache-between-runs
```

What it does:

- samples routed GLM families from the manifest on drive `A`
- reads the same expert byte ranges from the three sidecar copies by path
- benchmarks candidate weighted splits such as `A-only`, `3:1:0`, `3:1:1`, `4:1:1`, and `5:1:1`
- reports `P50`, `P90`, `P99`, `CV%`, effective `GiB/s`, and the exact page split per drive

Useful overrides:

- `--strategy NAME=A:B:C` to try a custom split, for example `--strategy tuned=4:1:1`
- `--experts 0-31` to lock the benchmark to a specific expert subset
- `--mode family` to benchmark one family at a time instead of full expert payloads
- `--page-bytes 16384` to match the current Flash-MoE page granularity

This benchmark is the quickest way to answer whether your `12 GB/s + 4 GB/s + 4 GB/s` drive mix wants a simple `3:1:1` split, a more latency-biased `4:1:1` or `5:1:1`, or no striping at all.

## Estimate a 3-bin expert-placement oracle

If you want to estimate a per-`(layer, expert)` hot/warm/cold placement across three drives before repacking the sidecar, use:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_bin_oracle.py \
  --sidecar ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash/sidecar \
  --trace /tmp/minimax.trace.jsonl \
  --mode lru-misses \
  --bank-size 60 \
  --bin apple=2.05 \
  --bin t710a=1.95 \
  --bin t710b=1.95
```

What it does:

- replays a `--moe-trace` JSONL trace
- either uses the raw routed requests or approximates slot-bank miss traffic with an LRU model
- assigns each `(layer, expert)` to a storage bin
- estimates routed source time assuming reads that land on different bins can overlap
- compares simple baselines (`single:...`, hot bands, frequency balance) against a local-search oracle

Notes:

- calibrate each `--bin NAME=RATE_GIB_S` from a real single-drive run: `effective GiB/s = bytes / source_time`
- the `oracle-local` result is a planning ceiling, not a promise for the current runtime
- `--assignment-out /tmp/oracle-bins.json` writes the oracle per-layer expert assignment for later repacker experiments
- the prototype is most useful when you want to compare "expert binning" against weighted striping on the same trace

## Plan a cross-machine layer split

If you want to keep a fully resident prefix on one machine and run the remaining layers through Flash-MoE on another, use:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_layer_split.py \
  --model ~/Models/GLM/GLM-5.1-IQ1-Dense/model-dense.gguf \
  --sidecar ~/Models/GLM/GLM-5.1-sidecar \
  --ctx 200000 \
  --ubatch 1 \
  --node name=m3u,mode=resident,usable_gib=76 \
  --node name=m5max,mode=flash,usable_gib=100,slot_bank=96
```

What it does:

- reads the dense-only GGUF metadata and exact per-layer dense/shared bytes
- reads the sidecar manifest to recover full-resident expert bytes per routed layer
- budgets KV cache at the requested context length
- assigns contiguous layer ranges across the nodes in the order you pass them
- emits a plan with the layer spans, per-node memory use, and the boundary activation size

Notes:

- `mode=resident` means routed layers on that node are fully resident
- `mode=flash` means routed layers on that node are budgeted as dense/shared + KV + `slot_bank`
- `usable_gib` should be the memory budget you are actually willing to give that node after OS/headroom
- transport is recorded in the plan as a hint; the current recommendation is to start with TCP over a private Thunderbolt IP link and keep RDMA as a later pluggable transport

## Verify the sidecar

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --sidecar ~/Models/flash/qwen35
```

Gemma4 verification uses the same command shape:

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py verify \
  --model ~/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --sidecar ~/Models/gemma4/packed_experts
```

## Inspect a model or subset

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py inspect \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --layers 1-2 \
  --families routed \
  --include-shared
```

## Extract only selected layers

```bash
PYTHON=python3 \
./tools/flashmoe-sidecar/flashmoe_sidecar.py extract \
  --model ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --out-dir ~/Models/flash/kimi-layer1 \
  --layers 1 \
  --families all \
  --include-shared
```

## Estimate persistent-bank cost and coverage

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 8 --banks 16 --banks 32 --banks 64 \
  --byte-budget-gib 8 --byte-budget-gib 16 --byte-budget-gib 24 --byte-budget-gib 32
```

Live terminal dashboard:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --trace ~/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --watch 20
```

Optional dashboard export:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_cache_estimator.py \
  --sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --trace /tmp/kimi-k25-trace.jsonl \
  --banks 4 --banks 16 --banks 64 \
  --byte-budget-gib 64 --byte-budget-gib 72 --byte-budget-gib 96 \
  --svg-out ~/Models/flash/logs/kimi-k25-cache-dashboard.svg
```

Long Kimi trace run without the normal `llama-cli` chat-loop exit:

```bash
mkdir -p ~/Models/flash/logs

nohup ./build/bin/llama-cli \
  -m ~/Models/Kimi/Kimi-K2.5-UD-TQ1_0-00001-of-00005.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/flash/Kimi-K2.5-sidecar \
  --moe-slot-bank 64 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --moe-trace-harness \
  --no-warmup \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  -c 256 \
  --context-shift \
  --seed 123 --temp 0 \
  --ignore-eos \
  --moe-trace ~/Models/flash/logs/kimi-k25-1h-trace.jsonl \
  -p "What is Apple Neural Engine?" \
  -n 12000 \
  > ~/Models/flash/logs/kimi-k25-1h-trace.log 2>&1 &
```

The default build includes `-DLLAMA_FLASH_MOE_GPU_BANK=ON`, and Kimi/DeepSeek2 GPU-bank placement is enabled by default at runtime.
`-ngl 999` offloads dense/shared tensors to GPU; routed expert bytes come from the sidecar slot-bank path.
Keep `--fit` enabled so dense/shared offload is clamped against the routed slot-bank reserve.
Use `-ub 1` for correct Kimi output; multi-token routed prefill produces degraded results.
Set `LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1` to force the host-backed path if you hit hangs or memory pressure.

## Run with the sidecar

```bash
./build/bin/llama-cli \
  -m ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode resident-bank \
  --moe-sidecar ~/Models/flash/qwen35 \
  --moe-topk 4 \
  --moe-verify-sidecar \
  --seed 123 --temp 0 \
  -p "Summarize Flash-MoE in two sentences." \
  -n 48
```

## Run a streamed slot-bank smoke test

```bash
./build/bin/llama-cli \
  -m ~/Models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/flash/qwen35 \
  --moe-slot-bank 32 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --no-warmup \
  -fit on \
  -ub 1 -b 64 \
  -ngl 999 \
  --seed 123 --temp 0 \
  -p "What is Apple Neural Engine" \
  -n 32 -st
```

## Run Gemma4-26B-A4B with the sidecar

Resident-bank smoke test:

```bash
./build/bin/llama-cli \
  --color off --simple-io \
  -m ~/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --moe-mode resident-bank \
  --moe-sidecar ~/Models/gemma4/packed_experts \
  --moe-topk 4 \
  -cnv -st -fit on \
  -ub 1 -b 1 -ngl 0 -c 4096 --seed 0 --temp 0 \
  -p "Make a poem about Apple Neural Engine in 4 lines." \
  -n 64
```

Streamed slot-bank smoke test:

```bash
./build/bin/llama-cli \
  --color off --simple-io \
  -m ~/Models/gemma4/gemma-4-26B-A4B-it-UD-IQ1_M.gguf \
  --moe-mode slot-bank \
  --moe-sidecar ~/Models/gemma4/packed_experts \
  --moe-slot-bank 16 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --no-warmup \
  -cnv -st -fit on \
  -ub 1 -b 1 -ngl 0 -c 4096 --seed 0 --temp 0 \
  -p "Make a poem about Apple Neural Engine in 4 lines." \
  -n 64
```

Gemma4-specific notes:

- `gemma-4-26B-A4B-it` is instruction tuned. For quality comparisons, prefer normal chat mode (`-cnv -st`) over `--moe-trace-harness`, which uses raw completion.
- Gemma4's native `n_expert_used = 8`. The default examples above use `--moe-topk 4` to halve per-token expert I/O at minimal quality cost — consistent with the recommendation across models. Drop `--moe-topk 4` to run at native K=8 if you need maximum fidelity.
- On smaller-memory devices, Gemma4 is more sensitive to slot-bank size than Qwen3.5-35B because each selected expert payload is larger. A slot bank of `8` or `16` is a better starting point than desktop-style larger banks.

In the default build of this fork, Qwen `slot-bank` is expected to use `-ngl 999`.
You do not need to know how many layers are routed MoE versus dense/shared; routed expert tensors are virtualized out of the normal GGUF loader and continue to come from the sidecar path. Keep `--fit` on so the fork clamps dense/shared offload against the routed bank budget on unified-memory systems.

If you want a shared-expert-only control run with the same dense/shared placement, add `--moe-shared-only`. That bypasses routed experts at graph build time while leaving shared experts active, which is useful for MLX-style prefill and dense/shared diagnostic comparisons.

## Server workflow for layer-major prefill

Three helper scripts cover the server path:

- `tools/flashmoe-sidecar/run_flashmoe_server.sh`
  Launches `llama-server` with Flash-MoE defaults, slot-bank sizing, and any
  layer-major prefill flags you pass through.
- `tools/flashmoe-sidecar/flashmoe_server_synth_test.py`
  Sends one synthetic long-prompt request to a running server and prints prompt
  and decode throughput without involving tools or a follow-up turn.
- `tools/flashmoe-sidecar/flashmoe_server_turn_test.py`
  Sends a big first-turn `/completion` request, then a same-slot follow-up turn.
- `tools/flashmoe-sidecar/flashmoe_server_smoke.sh`
  Starts the server, waits for readiness, runs the turn test, shuts the server
  down, and prints the key prompt / dedup / decode summaries.

If you want a stable OpenAI-style model id for `/v1/models` and
`/v1/chat/completions`, set:

- `MODEL_ALIAS=minimax-m2`
- `MODEL_ALIAS=gemma4`
- `MODEL_ALIAS=glm-dsa`
- `MODEL_ALIAS=qwen3moe`
- `MODEL_ALIAS=deepseek2`

`run_flashmoe_server.sh` forwards that as `--alias`, and the alias becomes the
model id returned by `/v1/models`.

`run_flashmoe_server.sh` also accepts an optional server profile:

- `FLASHMOE_SERVER_PROFILE=default`
  Uses the package/default slot-bank sizing path.
- `FLASHMOE_SERVER_PROFILE=highmem-decode`
  Raises the auto-sized slot-bank floor to `128` so the server can spend more
  host memory to improve decode-side cache hit rate on larger-memory systems.
  `MOE_SLOT_BANK` still overrides the profile if you want an exact value.

High memory use to improve decoding cache hit rate:

```bash
HOST=0.0.0.0 \
PORT=8080 \
MODEL_ALIAS=minimax-m2 \
MOE_TOPK=4 \
MOE_SLOT_BANK=128 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=4096 \
UBATCH=16 \
CTX=96000 \
SEED=123 \
TEMP=0.0 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -ctk q8_0 \
  -ctv q8_0 \
  --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 \
  --no-warmup \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 16192 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

Equivalent profile-driven form:

```bash
HOST=0.0.0.0 \
PORT=8080 \
MODEL_ALIAS=minimax-m2 \
FLASHMOE_SERVER_PROFILE=highmem-decode \
MOE_TOPK=4 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=4096 \
UBATCH=16 \
CTX=96000 \
SEED=123 \
TEMP=0.0 \
bash ./tools/flashmoe-sidecar/run_flashmoe_server.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -ctk q8_0 \
  -ctv q8_0 \
  --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 \
  --no-warmup \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 16192 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

### MiniMax tool-calling note

MiniMax tool calling in this fork now uses a server-side fallback for the
documented MiniMax XML tool format:

- the sampler grammar is disabled for MiniMax tool requests
- the raw assistant reply is still captured
- if the generic PEG parser rejects mixed prose / `</think>` / XML output, the
  server extracts `<minimax:tool_call> ... <invoke ...> ... </invoke> ...`
  blocks manually and converts them into normal OpenAI-style `tool_calls`

That means a direct `/v1/chat/completions` request can now succeed even when the
raw MiniMax reply looks like:

```xml
<minimax:tool_call>
<invoke name="Read">
<parameter name="file_path">/tmp/demo.py</parameter>
<parameter name="offset">1</parameter>
<parameter name="limit">20</parameter>
</invoke>
</minimax:tool_call>
```

Recommended workflow for debugging tool use:

1. Start the server with `run_flashmoe_server.sh`.
2. Send a direct curl request to `/v1/chat/completions` before trying a UI like
   Factory AI or DroidAI.
3. Check the server log for:
   - `Tool-call request: ...`
   - `Tool-parse state: ...`
   - `MiniMax tool-call parse fallback recovered ...`

Example MiniMax tool-call probe:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax-m2",
    "temperature": 0,
    "messages": [
      {
        "role": "user",
        "content": "Read lines 1 to 20 from /tmp/demo.py using the Read tool."
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "Read",
          "description": "Read part of a file",
          "parameters": {
            "type": "object",
            "properties": {
              "file_path": { "type": "string" },
              "offset": { "type": "integer" },
              "limit": { "type": "integer" }
            },
            "required": ["file_path", "offset", "limit"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "parallel_tool_calls": false,
    "max_tokens": 128
  }' | jq
```

Gemma4 benchmark example:

```bash
HOST=127.0.0.1 \
PORT=8091 \
PROMPT_LABEL=16k \
N_PREDICT=64 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=2048 \
UBATCH=32 \
CTX=25000 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/flashmoe_server_smoke.sh \
  ~/Models/gemma4/gemma-4-UD-Q3_K_XL-dense \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-batch 8192 \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 4
```

This server flow is tested on:

- MiniMax-M2.7
- Gemma4 26B-A4B
- GLM-5.1
- Qwen 3.5 35B-A3B
- Kimi K2.5

Qwen 3.5 currently has one important server caveat: the second turn may force a
full prompt re-processing pass because the server reports a lack of cache data
for its SWA / hybrid-memory path. The dedicated layer-major prefill path still
works there; the limitation is turn-prefix reuse.

Single-request synthetic 16K probe against a running server:

```bash
python3 ./tools/flashmoe-sidecar/flashmoe_server_synth_test.py \
  --url http://127.0.0.1:8080/completion \
  --prompt-label 16k \
  --n-predict 128
```

### Safe large-context workflow

For very large prompt tests on Apple Silicon, do not jump straight to the
largest context. Use the server smoke wrapper with:

- persistent logs under `tools/flashmoe-sidecar/logs/...`
- a live memory guard
- quantized KV cache
- disabled server prompt checkpoints

Important env vars:

- `FREE_MEM_ABORT_PERCENT=20`
  Interrupt the run if system-wide free memory falls to `20%` or less.
- `MEMORY_CHECK_SEC=2`
  Poll system free memory every 2 seconds.
- `REQUEST_TIMEOUT_SEC=...`
  Increase the client HTTP timeout for long-prefill runs.

Important server flags for safer large-context runs:

- `-ctk q8_0 -ctv q8_0`
  Halve KV cache memory versus the default f16 cache.
- `--ctx-checkpoints 0 --checkpoint-every-n-tokens -1`
  Disable server prompt checkpoints, which otherwise add large memory overhead
  during long prompt processing.
- `--no-warmup`
  Skip the empty warmup run when you are only testing memory / throughput.

Conservative MiniMax starting point:

```bash
HOST=127.0.0.1 \
PORT=8096 \
PROMPT_LABEL=1k \
N_PREDICT=16 \
FREE_MEM_ABORT_PERCENT=20 \
MEMORY_CHECK_SEC=2 \
MOE_TOPK=4 \
MOE_SLOT_BANK=64 \
MOE_CACHE_IO_SPLIT=8 \
BATCH=1024 \
UBATCH=16 \
CTX=4096 \
SEED=123 \
TEMP=0.1 \
bash ./tools/flashmoe-sidecar/flashmoe_server_smoke.sh \
  ~/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
  -ctk q8_0 -ctv q8_0 \
  --ctx-checkpoints 0 \
  --checkpoint-every-n-tokens -1 \
  --no-warmup \
  --moe-predict-top1-prev \
  --moe-prefill-layer-major \
  --moe-prefill-micro-batch 32 \
  --moe-prefill-io-split 8 \
  --moe-prefill-banks 1
```

Observed on M5 Max:

- the 1k smoke test above completed safely
- a conservative 64k MiniMax server run with the same safety flags and
  `CTX=72000` dropped system free memory from about `63%` idle to about `22%`
  during startup / prefill preparation

Largest verified guarded Q8 server contexts on this M5 Max:

- MiniMax-M2.7: `196608`
- Qwen 3.5 35B-A3B: `262144`
- Kimi K2.5: `262144`
- GLM-5.1: `188160`
- Gemma4 26B-A4B: full Q8 KV is not currently supported in this server reserve path

That means `64k` is already close enough to the machine limit that `128k` and
`250k` should not be attempted without another reduction in memory pressure or a
higher-abort safety policy.

## Notes

- The extractor writes one raw bank file per sparse layer: `layer_XXX.bin`.
- Each file concatenates whole-tensor expert payloads in a stable layer/family order.
- `resident-bank` overrides expert tensors at load time and keeps the dense graph untouched.
- `slot-bank` reads experts on demand with `pread()` into a small resident slot bank. It is still experimental.
- The default build includes `-DLLAMA_FLASH_MOE_GPU_BANK=ON`. In slot-bank mode, routed experts stream from SSD via the sidecar path regardless of `-ngl`. Dense/shared weights are offloaded to GPU via `-ngl 99`.
- For sidecar runs with `-ngl > 0`, keep `--fit` enabled. The fitter is the supported end-user path for clamping dense/shared offload against the routed slot-bank reserve. If you need to deliberately bypass it for a supervised manual test, set `LLAMA_FLASH_MOE_ALLOW_UNFIT_OFFLOAD=1`.
- `--moe-topk N` is an experimental reduction-only runtime override for routed experts per token. It must be less than or equal to the GGUF model's native MoE top-k.
- `--moe-shared-only` is an experimental shared-expert-only diagnostic. It is most meaningful on MoE architectures that actually have shared experts, such as Qwen3.5 MoE and Kimi/DeepSeek2.
- For Qwen3.5 small-memory smoke tests, prefer `--moe-slot-bank 32` or `64`. A slot bank of `256` effectively reserves a full routed-expert bank for that model.
- In a GPU-bank build, DeepSeek2/Kimi routed GPU-bank placement is attempted by default. If a machine hits hangs or memory pressure, set `LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1` to fall back to the older host-backed routed path.
- Partial sidecars are supported: any tensor not present in the manifest continues to load from the original GGUF.
- `flashmoe_cache_estimator.py` models persistent-bank cost from the exact manifest and, when given a trace, reports static-bank, global-budget, and LRU coverage/miss estimates.
- The default estimator output is terminal-friendly text with ASCII bars; `--svg-out` adds a self-contained dashboard file.
- `--watch N` turns the estimator into a live terminal dashboard that refreshes every `N` seconds until interrupted.
