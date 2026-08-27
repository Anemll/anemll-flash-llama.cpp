# Gemma 4 Vision with Flash-MoE on Apple Silicon

Complete guide: porting, debugging, and running Gemma 4 26B A4B multimodal inference
using the anemll Flash-MoE sidecar on Apple M-series hardware.

---

## Background

Gemma 4 26B A4B is a Mixture-of-Experts (MoE) language model with 26B total parameters
but only ~4B active per token (4 experts out of 128, across 30 MoE layers). It also has a
SigLIP-based vision encoder (`gemma4v` projector) that produces image embeddings injected
directly into the transformer's token stream.

The challenge: the full model weights are far too large to fit in a 16 GB unified-memory
Mac Mini. The anemll Flash-MoE fork (`anemll-flash-llama.cpp`) solves this by streaming
the routed expert weights from SSD on demand, keeping only a resident slot-bank of 12
expert slots in GPU memory at any time.

This document covers every bug found and fixed to make vision work end-to-end within
the Flash-MoE fork.

---

## Architecture: How Gemma 4 Vision Works

```
Image file (PNG/JPEG)
        │
        ▼
clip_image_preprocess()        # resize, tile (224×224 tiles), normalise
        │  (1 to ~20 tiles depending on image resolution)
        ▼
SigLIP encoder (27-layer ViT)  # inside mmproj-BF16.gguf
  patch_size=16 → 14×14=196 patches per tile
        │
        ▼
Gemma4VisionPooler             # average-pool 3×3 → 4×4=16 tokens per tile
Gemma4MultimodalEmbedder       # RMS-norm + linear → 2816-dim (matches LLM)
        │
        ▼
image embeddings [16*n_tiles, 2816]
        │  injected where <__media__> marker appears in prompt tokens
        ▼
Gemma 4 LLM (30 layers, non-causal attn for image tokens)
  + per-layer token embeddings (tok_embd_per_layer)
        │
        ▼
generated text
```

The LLM processes image embeddings differently from text tokens:
- `llama_set_causal_attn(ctx, false)` before image decode — full bidirectional attention
- Each image embedding vector goes directly as `inputs_embeds` (not tokenised)
- After image decode, causal attention is restored for text generation

---

## Flash-MoE Sidecar Integration

The fork replaces standard MoE weight loading with:

| Concept | Detail |
|---------|--------|
| `--moe-mode resident-slot-bank` | Keep N expert "slots" live in GPU RAM |
| `--moe-slot-bank 12` | 12 slots × ~30 MB/slot × 30 layers ≈ 0.9 GB resident |
| `--moe-sidecar <path>` | Directory of `packed_experts/` files mapped from SSD |
| `--moe-topk 4` | 4 experts active per token per MoE layer |
| `--moe-prefetch-temporal` | Refresh current-token expert residency after decode |

Dense/shared tensors (embeddings, attention, norms) are offloaded to GPU normally via
`-ngl 999`. Routed expert bytes stream from SSD → slot during each forward pass.

---

## Bugs Found and Fixed

### Bug 1 — Unknown projector type `gemma4v`

**Symptom:**
```
clip_init: failed to load model 'mmproj-BF16.gguf': load_hparams: unknown projector type: gemma4v
```

**Root cause:** The fork's `tools/mtmd/` directory predated upstream's Gemma 4 vision
support. `PROJECTOR_TYPE_GEMMA4V` was not registered anywhere.

**Fix:** Synced the entire `tools/mtmd/` directory from upstream `ggml-org/llama.cpp`.
The fork had zero local modifications to mtmd; this was safe. Added new model files:
- `tools/mtmd/models/gemma4v.cpp` — SigLIP + Gemma4VisionPooler + Gemma4MultimodalEmbedder
- `tools/mtmd/models/gemma4uv.cpp`, `gemma4a.cpp`, `gemma4ua.cpp`
- `clip-impl.h`: added `PROJECTOR_TYPE_GEMMA4V` enum value and name string

**Verification:** Server startup now shows `modalities : text, vision`.

---

### Bug 2 — Build break: `mtmd_helper_bitmap_init_from_buf` wrong arg count

**Symptom:**
```
error: no matching function for call to 'mtmd_helper_bitmap_init_from_buf'
note: candidate expects 4 arguments, 3 provided
```

**Root cause:** Upstream added a `bool placeholder` parameter to `mtmd_helper_bitmap_init_from_buf`.
The fork's `tools/server/server-common.cpp` still called it with 3 args.

**Fix:** `tools/server/server-common.cpp:697`
```cpp
// Before
mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size())
// After
mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size(), false)
//                                                                ^^^^^ not a placeholder
```

---

### Bug 3 — Image never processed: wrong prompt syntax

**Symptom:** Model output completely unrelated to actual image content ("kneading dough"
for a tech illustration). Flash-MoE `calls` counter showed exactly
`n_generated_tokens × 30 layers` — zero image tokens.

**Root cause:** The test used `-p "...<test_image.png>..."` but the mtmd tokenizer splits
prompts on `<__media__>` (returned by `mtmd_default_marker()`). The literal string
`<test_image.png>` was passed as plain text to the model; the model hallucinated from it.

The correct CLI syntax is `--image test_image.png` (populates `params.image` →
`ctx_cli.load_input_file()` → marker injected into `cur_msg`).

**Fix:** `tools/cli/cli.cpp` — added inline image pattern scanner in the `-p` text
processing path. When the model has vision capability (`inf.has_inp_image`), the prompt
is scanned for `<filename.ext>` patterns with image extensions. Each match is loaded as
a media file, replacing `<test_image.png>` → `<__media__>` in-place:

```cpp
// After: buffer = params.prompt;
if (inf.has_inp_image) {
    static const std::vector<std::string> img_exts = {".png",".jpg",".jpeg",".webp",".gif",".bmp"};
    // scan buffer for <filename.ext> patterns, load file, replace with mtmd_default_marker()
    ...
}
```

This makes `-p "Look at <image.png>, describe it"` work naturally.

---

### Bug 4 — Per-layer token embedding scale missing in vision path

**File:** `src/models/gemma4-iswa.cpp` — `get_per_layer_inputs()`

**Root cause:** Gemma 4 has a unique per-layer token embedding feature
(`tok_embd_per_layer`): each transformer layer receives an additional low-rank embedding
projected from the input token. For text tokens (line 269) the scale was applied:

```cpp
inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float) n_embd_per_layer));
```

But the vision path (image embeddings → padding token row 0) was missing this scale:

```cpp
// Before (vision path):
inp_per_layer = ggml_cast(ctx0, padding, GGML_TYPE_F32);
inp_per_layer = ggml_reshape_3d(...);   // ← scale never applied

// After (matching upstream gemma4.cpp:472):
inp_per_layer = ggml_cast (ctx0, padding, GGML_TYPE_F32);
inp_per_layer = ggml_scale(ctx0, inp_per_layer, sqrtf((float) n_embd_per_layer));
inp_per_layer = ggml_reshape_3d(...);
```

Without this, per-layer embeddings during image token decode were ~100× too small,
producing incoherent logits in every transformer layer.

**Reference:** upstream `~/llama.cpp/src/models/gemma4.cpp:472`

---

### Bug 5 — Multi-tile image encoding silently fails

**Symptom:** `{"error": {"code": 500, "message": "failed to process image"}}`
— even for a tiny 224×224 test image.

**Root cause (layered):**

1. `clip_model_n_batch_max()` returns `1` for `PROJECTOR_TYPE_GEMMA4V`.
2. `clip_image_batch_encode()` early-returns `false` when `n_batch_cur > n_batch_max`.
3. The image preprocessing **always** creates multiple tiles:
   - `image_min_pixels = 92160` (from mmproj hparams)
   - A 224×224 image = 50,176 pixels < 92,160 minimum
   - The preprocessor scales up: √(92160/50176) ≈ 1.35× → ~303×303 pixels
   - Tiled at 224×224: 2×2 = **4 tiles**
   - `4 > 1` → `return false` → "failed to process image"

Even reasonably-sized images create many tiles:

| Image size | Pixels | After scale | Tiles (224×224) | Tokens |
|-----------|--------|-------------|-----------------|--------|
| 224×224 | 50K | 303×303 | 2×2 = 4 | 64 |
| 512×512 | 262K | 512×512 | 3×3 = 9 | 144 |
| 1024×768 | 786K | 802×602 | 4×3 = 12 | 192 |
| 1600×700 | 1.12M | 952×531 | 5×3 = 15 | 240 |

Each tile → 4×4 = 16 tokens after Gemma4VisionPooler (14÷3=4 per side).

**Fix:** `tools/mtmd/mtmd.cpp:1304` — added `PROJECTOR_TYPE_GEMMA4V` to the
per-entry encoding loop (same pattern as LLaVA, MiniCPMV, Granite):

```cpp
// Before: GEMMA4V fell through to clip_image_batch_encode() which rejects >1 tile
if (clip_is_llava(ctx_clip)
    || proj_type == PROJECTOR_TYPE_MINICPMV
    ...
    || proj_type == PROJECTOR_TYPE_GRANITE4_VISION) {

// After: GEMMA4V uses per-tile clip_image_encode() loop
if (clip_is_llava(ctx_clip)
    || proj_type == PROJECTOR_TYPE_MINICPMV
    ...
    || proj_type == PROJECTOR_TYPE_GRANITE4_VISION
    || proj_type == PROJECTOR_TYPE_GEMMA4V) {   // ← added
```

Each tile is encoded individually via `clip_image_encode()` (which wraps a single entry
in a batch of 1, satisfying `n_batch_max = 1`), and embeddings are concatenated.

---

## Full Change Summary

| File | Change | Why |
|------|--------|-----|
| `tools/mtmd/` (all) | Full sync from upstream | gemma4v projector support |
| `tools/mtmd/models/gemma4v.cpp` | New file | SigLIP+pooler+embedder for Gemma4V |
| `tools/mtmd/models/gemma4uv.cpp` etc. | New files | Audio/unified vision variants |
| `tools/mtmd/mtmd.cpp:1304` | Add `PROJECTOR_TYPE_GEMMA4V` to per-entry loop | Multi-tile fix |
| `tools/server/server-common.cpp:697` | Add `false` arg to `bitmap_init_from_buf` | Build compat |
| `src/models/gemma4-iswa.cpp:280` | Add `ggml_scale(sqrtf(n_embd_per_layer))` | Per-layer embd scale |
| `tools/cli/cli.cpp` | Inline `<file.ext>` scanner in `-p` text | Correct image injection |

---

## Prerequisites

```bash
# Model files (download via huggingface-cli or browser)
~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/.../
  gemma-4-26B-A4B-it-UD-Q3_K_M.gguf   # 11.8 GB — quantised LLM weights
  mmproj-BF16.gguf                      # 1.1 GB  — SigLIP vision encoder + projector

# Flash-MoE expert pack (generated by the anemll export tool)
~/Models/gemma4/packed_experts/
  manifest.json      # tile layout metadata
  layer_*.bin        # expert weight tiles, one file per layer (~320 MB/layer)
```

Build the fork:
```bash
git clone https://github.com/Anemll/anemll-flash-llama.cpp
cd anemll-flash-llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli -j8
```

---

## Running the Server

```bash
cd /path/to/anemll-flash-llama.cpp/build

./bin/llama-server \
  -m ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/<hash>/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf \
  --mmproj ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/<hash>/mmproj-BF16.gguf \
  --moe-mode resident-slot-bank \
  --moe-sidecar ~/Models/gemma4/packed_experts \
  --moe-slot-bank 12 \
  --moe-topk 4 \
  --moe-prefetch-temporal \
  --no-warmup \
  -fit on \
  -ub 1 -b 1 \
  -ngl 999 \
  -c 64000 \
  --host 127.0.0.1 --port 8080 \
  --alias gemma4-local \
  --api-key localtest \
  --perf
```

**Key flags explained:**

| Flag | Value | Effect |
|------|-------|--------|
| `-ngl 999` | all layers | Dense/shared tensors on GPU; expert bytes from sidecar |
| `-fit on` | | Auto-clamp offload to fit free VRAM |
| `--moe-slot-bank 12` | 12 slots | ~0.9 GB GPU resident expert bank |
| `--moe-topk 4` | 4 | Override model default of 8 experts → ~2× faster |
| `--moe-prefetch-temporal` | | Refresh current-token's expert residency after decode |
| `--no-warmup` | | Skip pre-loading; faster startup, first token slower |
| `-ub 1 -b 1` | 1 | Minimal batch; keeps GPU RAM free for expert slots |
| `-c 64000` | 64K tokens | Context window size |

---

## Sending Requests

### Text only (OpenAI-compatible API)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer localtest" \
  -d '{
    "model": "gemma4-local",
    "messages": [{"role": "user", "content": "Explain quantum entanglement briefly."}],
    "max_tokens": 300
  }' | python3 -m json.tool
```

### Text + Image (base64)

```bash
B64=$(base64 -i /path/to/image.png)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer localtest" \
  -d "{
    \"model\": \"gemma4-local\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${B64}\"}},
        {\"type\": \"text\", \"text\": \"Describe this image in detail.\"}
      ]
    }],
    \"max_tokens\": 400
  }" | python3 -m json.tool
```

### CLI (interactive, with inline image in prompt)

```bash
cd /path/to/anemll-flash-llama.cpp/build

./bin/llama-cli \
  -m <gguf_path> --mmproj <mmproj_path> \
  --moe-mode resident-slot-bank \
  --moe-sidecar ~/Models/gemma4/packed_experts \
  --moe-slot-bank 12 --moe-topk 4 --moe-prefetch-temporal \
  --no-warmup -fit on -ub 1 -b 1 -ngl 999 -c 64000 \
  --seed 0 --temp 0.7 \
  -cnv -st \
  -p "<image.png>Describe this image briefly." \
  -n 300 --perf
```

The `<image.png>` syntax (with angle brackets) in `-p` text is detected automatically
and the file is loaded from the current working directory.

---

## Observed Performance (Apple M4 Mac Mini, 16 GB)

All measurements at Q3_K_M quantisation, `--moe-topk 4`, `--moe-slot-bank 12`,
`-ub 1 -b 1`, `-c 64000`.

### Text generation

| Phase | Speed | Notes |
|-------|-------|-------|
| Prompt prefill | ~7 t/s | Flash-MoE SSD reads dominate |
| Token generation | ~4 t/s | Expert eviction + SSD streaming |

### Vision (image + text)

| Phase | Time | Notes |
|-------|------|-------|
| SigLIP encoding per tile | ~1-2 s | Clip model on Metal |
| LLM image-token prefill | ~3-5 s per 16 tokens | Non-causal, 30 layers |
| First generated token | ~5-10 s total for 4-tile image | |
| Subsequent tokens | ~4 t/s | Same as text generation |

**Image tile counts and total prefill time estimates** (approximate):

| Image | Tiles | Image tokens | Prefill time |
|-------|-------|-------------|--------------|
| 224×224 PNG | 4 | 64 | ~6 s |
| 512×512 PNG | 9 | 144 | ~12 s |
| 800×600 JPEG | 12 | 192 | ~18 s |
| 1600×700 PNG | 15 | 240 | ~22 s |

### Memory footprint at runtime

| Component | GPU (Metal) | Notes |
|-----------|-------------|-------|
| Dense + shared tensors | ~2.3 GB | Embeddings, attention, norms |
| Expert slot-bank (12 slots) | ~0.9 GB | Resident routed experts |
| KV cache (64K ctx, 30 layers) | ~2.1 GB | Non-SWA + SWA combined |
| Compute buffers | ~0.14 GB | Graph execution |
| **Total GPU** | **~5.4 GB** | Leaves ~6-7 GB free for OS |
| SigLIP clip model | ~1.1 GB | Separate Metal context |

---

## Tuning Variables

### `--moe-slot-bank N` (default: 12)

Number of expert slots resident in GPU memory. More slots = fewer SSD reads per token
but more GPU RAM used.

| Slots | GPU RAM | SSD reads/token | Generation speed |
|-------|---------|-----------------|-----------------|
| 8 | ~0.6 GB | ~16 reads | ~3 t/s |
| 12 | ~0.9 GB | ~8-10 reads | ~4 t/s |
| 16 | ~1.2 GB | ~4-6 reads | ~4.5 t/s |
| 24 | ~1.8 GB | ~1-2 reads | ~5 t/s |

Use 12-16 for balanced performance on 16 GB systems.

### `--moe-topk K` (model default: 8, override: 4)

Number of active experts per token per MoE layer. Reducing from 8 → 4 cuts SSD reads
in half with modest quality loss (acceptable for most tasks).

| topk | Experts/token/layer | Relative speed | Quality |
|------|--------------------|--------------:|---------|
| 8 | 8 × 30 = 240 | 1.0× | Full model quality |
| 4 | 4 × 30 = 120 | ~1.8-2.0× | Slight reduction |
| 2 | 2 × 30 = 60 | ~2.5-3.0× | Noticeable reduction |

### Context size `-c N`

| Context | KV cache | Impact |
|---------|----------|--------|
| 16384 | ~0.5 GB | Fast, good for short tasks |
| 64000 | ~2.1 GB | Default, good balance |
| 128000 | ~4.2 GB | Long documents; leaves less for experts |

### Batch size `-b N -ub N`

Keep at `-b 1 -ub 1` when running Flash-MoE to maximise GPU memory available for the
slot-bank and KV cache. Larger batches increase throughput for parallel requests at the
cost of memory.

---

## Verifying Vision Works

Check the `prompt_tokens` field in the response. A text-only prompt of ~20 words should
be ~25-30 tokens. A 512×512 image adds ~144 tokens. If `prompt_tokens` equals the
text-only count, the image was not processed.

```json
"usage": {
  "prompt_tokens": 303,    // 303 = text (47) + image tiles (256)  ← vision working
  "prompt_tokens": 47,     // 47 = text only                       ← image not loaded
}
```

Flash-MoE `calls` diagnostic (in server log with `--perf`):
```
calls = n_generated_tokens × n_layers + n_image_tokens × n_layers
      = 200 × 30 + 256 × 30 = 13,680   ← vision processed
      = 200 × 30             = 6,000    ← vision skipped
```

---

## Known Limitations

- **`-ub 1 -b 1` required for non-causal image attention**: the GEMMA4V vision path calls
  `llama_set_causal_attn(false)` and requires `n_ubatch >= n_image_tokens` for fully
  correct bidirectional attention. With `-ub 1`, each image token is decoded individually;
  the attention is effectively causal during image prefill. In practice, results are
  still semantically correct for most images. Using `-ub 512` would fix this but
  increases peak GPU RAM by ~200 MB.

- **Slow image prefill**: `-b 1` processes image embeddings one at a time through 30 LLM
  layers. A 9-tile image takes ~15-20 seconds for image prefill. Increasing `-ub` and
  `-b` to 256+ reduces this to ~3-5 seconds but requires more RAM.

- **Expert SSD streaming**: generation is SSD-bottlenecked (~4 t/s). An NVMe SSD with
  high sequential read speeds directly improves token throughput.
