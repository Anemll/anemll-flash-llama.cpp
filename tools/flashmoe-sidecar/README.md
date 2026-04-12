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
  --out-dir ~/Models/flash/GLM-5.1-sidecar
```

The sidecar is roughly 177 GiB at IQ1_M/IQ2_XXS (76 MoE layers × 256 experts × gate/up/down). Make sure the target SSD has the room.

## Export a MiniMax M2.7 Flash package

MiniMax M2.7 (`minimax-m2` arch) currently uses a dense-package workflow in this fork:

- routed experts are extracted into `sidecar/`
- dense and shared tensors are exported into `model-dense.gguf`
- package defaults are written to `flashmoe-package.json`

Tested quantisations:

| Quant | Shards | Dense GGUF | Routed sidecar |
|-------|--------|-----------|----------------|
| UD-IQ2_XXS | 3 | ~2.58 GiB | ~58.3 GiB |
| UD-Q4_K_XL | 4 | (4-bit) | (4-bit routed) |

Build the package in one step:

```bash
# UD-IQ2_XXS (2-bit)
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS/MiniMax-M2.7-UD-IQ2_XXS-00001-of-00003.gguf \
  --out-dir ~/Models/MiniMax-M2.7-GGUF/UD-IQ2_XXS-Flash \
  --force

# UD-Q4_K_XL (4-bit)
python3 ./tools/flashmoe-sidecar/minimax_m2_prepare.py \
  --model /Volumes/X2T/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf \
  --out-dir /Volumes/X2T/Models/MiniMax-M2.7-GGUF/UD-Q4_K_XL-Flash \
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

Use `--perf` when comparing the compact dense GGUF against the original full GGUF shards. In this fork, `--perf` prints the standard libllama timing summary including `load time`, so it is the easiest way to confirm that `model-dense.gguf` reduces startup cost. On Flash-MoE `slot-bank` runs it also prints the routed profile table, cached expert hit rate, and Metal replay cache hit rate.

GLM-5.1 example:

```bash
python3 ../local_tools/export_dense_gguf.py \
  --model ~/Models/GLM/GLM-5.1-UD-IQ1_M-00001-of-00006.gguf \
  --sidecar ~/Models/flash/GLM-5.1-sidecar \
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
  --moe-sidecar ~/Models/flash/GLM-5.1-sidecar \
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

MiniMax M2.7 natively routes to `8` of its 256 experts per layer. The router still scores all 256 experts every layer — `--moe-topk 4` simply keeps only the top 4 scoring experts instead of 8, cutting per-token expert I/O roughly in half for ~2× decode speedup with minor quality degradation. Use `MOE_TOPK=8` when you want full native routing fidelity.

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
MOE_TOPK=2 \
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
  --moe-sidecar ~/Models/flash/GLM-5.1-sidecar \
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
- use `--perf` so you can confirm `load time`, routed source time, and cached expert hit rate

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
  --moe-sidecar ~/Models/flash/GLM-5.1-sidecar \
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

GLM-5.1 specific notes:

- **`--moe-topk 4`** is a reduction-only override of the model's native K=8. On IQ1_M/IQ2_XXS quants the K=4 vs K=8 quality gap is within noise for general use, while halving per-token expert I/O for ~2× decode. Drop the flag to use native K=8 if you need maximum fidelity.
- **`--moe-slot-bank 64`** is the starting point. With native K=8 the bank has only 8× headroom; if you have RAM, try `128` or `256` for higher reuse on warm caches.
- **`--moe-prefetch-temporal`** is the single biggest knob — it overlaps next-layer expert `pread`s with current-layer GPU compute. Always on for SSD-bound MoE.
- **`--perf`** is recommended for tuning. It prints `load time` for dense-vs-full-GGUF comparisons, and on `slot-bank` runs it also prints the Flash-MoE routed breakdown (`Expert I/O source`, `Expert upload`), cached expert hit rate, and Metal replay cache hit rate.
- **The best-known fast path on this branch is higher than the older baseline.** With dense-only export, Metal replay, CPU-visible slot writes, predictor off, and a `90` to `96` slot bank, GLM-5.1 IQ1_M is currently landing around `6.5` to `6.7 tok/s` on M5 Max 128 GB steady-state runs. Older full-GGUF or smaller-bank recipes remain useful as simpler baselines.
- If you hit hangs or memory pressure (the DSV2/Kimi GPU-bank path is shared with GLM), fall back with:
  ```bash
  LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK=1 ./build/bin/llama-cli ...
  ```

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
