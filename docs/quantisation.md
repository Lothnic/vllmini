# 4-Bit Quantization in vLLMini

> **Status:** merged with main
> **Dependencies:** `bitsandbytes`, `safetensors`  
> **Supported models:** Llama, Mistral, Qwen2/2.5/3

---

## Overview

vLLMini implements **4-bit NF4 (NormalFloat4) quantization** using `bitsandbytes` to reduce VRAM usage by ~4× for large language models. This enables running 7B-parameter models (like Mistral-7B) on consumer GPUs with as little as 8GB VRAM.

**Memory savings example — Mistral-7B:**

| Precision | VRAM (approx.) |
|-----------|---------------|
| FP32 | ~28 GB |
| BF16 | ~14 GB |
| **NF4 (4-bit)** | **~4 GB** |

## Architecture

The quantization system spans four components:

```
┌───────────────┐        ┌───────────────┐        ┌────────────────────┐        ┌───────────────┐
│   main.py     │────▶   │ weight_loader │────▶   │     models/base    │────▶   │ models/llama  │
│   (CLI)       │        │    .py        │        │  get_linear_layer  │        │  attention.py │
│  --quantize   │        │  _load_quant  │        │    (factory)       │        │    qwen3.py   │
└───────────────┘        └───────────────┘        └────────────────────┘        └───────────────┘
```

### Component Responsibilities

| File | Role |
|------|------|
| `main.py` | CLI flag `--quantize` / `-q`, passes to `load_hf_model()` |
| `models/base.py` | `get_linear_layer()` factory — returns `nn.Linear` or `bnb.nn.Linear4bit` |
| `models/llama.py` | Model architecture — MLP and Attention layers use `get_linear_layer()` |
| `models/attention.py` | Q/K/V/O projections use `get_linear_layer()` |
| `models/weight_loader.py` | Dual-path loading: `_load_standard()` vs `_load_quantized()` |

---

## How It Works

### Step 1: Layer Construction — The Linear Layer Factory

When `quantize=True` is passed via the config, every linear layer in the model is constructed as a `bitsandbytes.nn.Linear4bit` instead of a standard `torch.nn.Linear`.

```python
# models/base.py
def get_linear_layer(in_features, out_features, bias, quantize=False):
    if quantize:
        import bitsandbytes as bnb
        return bnb.nn.Linear4bit(
            in_features, out_features, bias=bias,
            compute_dtype=torch.bfloat16,   # Compute in BF16 for speed
            quant_type="nf4",               # NormalFloat4 quantization
        )
    return nn.Linear(in_features, out_features, bias=bias)
```

**Key design decision:** `compute_dtype=torch.bfloat16` means the weights are *stored* in 4-bit, but *dequantized to BF16* on-the-fly during matrix multiplications. This preserves numerical stability while cutting memory by ~4×.

**Which layers are quantized:**
- Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP projections: `gate_proj`, `up_proj`, `down_proj`

**Which layers are not quantized**
- Embeddings: `embed_tokens` (kept at full precision)
- Normalization: `RMSNorm` (kept at full precision)
- LM head: `lm_head` (kept at full precision)

This is standard practice — embeddings and the output head contain fewer parameters and are sensitive to quantization, while the bulk of parameters live in attention/MLP projections.

#### Interesting fact 
A key insight is 4-bit is a compressed encoding of the weights, we don't just truncate the weights to 4-bit.

### Step 2: Meta Device Initialization

The model is first instantiated on PyTorch's `meta` device, which creates the model graph with zero memory allocation:

```python
with torch.device("meta"):
    model = model_class(config)  # config.quantize = True
```

At this point, `Linear4bit` layers exist with the correct shapes and `quant_type="nf4"` attribute, but their parameters are placeholder meta tensors — no actual memory is consumed.

### Step 3: Sharded Weight Loading

This is where quantization actually happens. The `_load_quantized()` function processes safetensor shard files **one at a time** to minimize peak memory:

```
┌───────────────────────────────────────────────────────────────┐
│                   _load_quantized() Flow                      │
│                                                               │
│  For each shard file:                                         │
│    ┌─────────────┐                                            │
│    │ Load to CPU  │  safetensors → CPU tensors (FP32/BF16)    │
│    └──────┬──────┘                                            │
│           │                                                   │
│    For each weight in shard:                                  │
│           │                                                   │
│    ┌──────▼──────┐                                            │
│    │  Remap key   │  "model.layers.0.self_attn.q_proj.weight" │
│    │              │  → "layers.0.attn.q_proj.weight"          │
│    └──────┬──────┘                                            │
│           │                                                   │
│    ┌──────▼──────┐     ┌──────────────────┐                   │
│    │ Cast to BF16 │────▶│ Delete original  │ (free FP32 copy) │
│    └──────┬──────┘     └──────────────────┘                   │
│           │                                                   │
│    ┌──────▼──────────────────────────────────────┐            │
│    │ Is it a quantizable layer? (has quant_type) │            │
│    └──────┬────────────────────────┬─────────────┘            │
│       YES │                        │ NO                       │
│    ┌──────▼──────┐          ┌──────▼──────┐                   │
│    │ Params4bit  │          │  nn.Param   │                   │
│    │ → .to(GPU)  │          │  → .to(GPU) │                   │
│    │ (quantizes) │          │ (full prec) │                   │
│    └─────────────┘          └─────────────┘                   │
│                                                               │
│    Free shard → gc.collect() → cuda.empty_cache()             │
└───────────────────────────────────────────────────────────────┘
```

#### Per-Parameter Quantization with `Params4bit`

For layers that should be quantized (detected via `hasattr(param, "quant_type")`), the weight goes through `bitsandbytes.nn.Params4bit`:

```python
new_param = Params4bit(
    v_typed,                    # BF16 weight on CPU
    requires_grad=False,
    quant_type="nf4",           # NormalFloat4
)
del v_typed                     # Free CPU copy BEFORE GPU transfer
new_param = new_param.to(device)  # Quantization happens here!
setattr(target, attr_name, new_param)
```

The `.to(device)` call is where the actual quantization occurs:
1. The full-precision weight is sent to GPU
2. `bitsandbytes` computes the NF4 representation (absmax scaling + 4-bit encoding)
3. The result is stored as a packed 4-bit tensor with quantization state metadata

#### Memory Management

Aggressive memory management is critical when loading 7B models on 8GB GPUs:

1. **`shard.pop(k)`** — Weights are popped from the shard dict as they're processed, freeing memory incrementally
2. **`del v`** — The original tensor is deleted immediately after casting to BF16
3. **`del v_typed`** — The BF16 copy is deleted before GPU allocation during quantization
4. **`gc.collect()` + `cuda.empty_cache()`** — Forced cleanup after each shard

Without these optimizations, intermediate copies (FP32 original + BF16 cast + GPU copy) can exceed available VRAM.

### Step 4: Post-Load Fixups

After quantized loading, several fixups are applied:

1. **Weight tying:** Re-tie `lm_head.weight = embed_tokens.weight` if configured (per-parameter loading breaks Python object references).
2. **Rotary embedding buffers:** Re-materialize `cos_cached`/`sin_cached` on the target device (these are computed buffers, not saved in checkpoints).
3. **Strict Validation:** Collect all parameters and buffers still on the `meta` device. If any are missing (excluding correctly tied weights), a `ValueError` is raised. This prevents the model from running with random garbage memory if the checkpoint is incomplete.
4. **Deterministic Materialization:** Any remaining meta tensors are materialized as **zeros** (using `torch.zeros_like`) on the target device for safety.
5. **`model.to(device, dtype)`:** Moves non-quantized parameters; `bitsandbytes` parameters automatically skip this operation.

---

## Robustness & Error Handling

### Specific File Discovery
The shard discovery logic (`_find_shard_paths`) uses specific exception handling to distinguish between different types of failures:
- **`EntryNotFoundError`**: Specifically caught to determine if a model is sharded (e.g., if `model.safetensors` is missing, it looks for `model.safetensors.index.json`).
- **Other Exceptions**: Critical errors (network timeout, disk full, permission denied) are logged via a module-level logger and re-raised immediately to prevent confusing failure states.

### Streaming Generation
The `Generator` class implements **incremental full-sequence decoding** to ensure correctness with SentencePiece tokenizers (used by Mistral and Llama). 
- **The Problem**: Decoding single tokens individually loses the `▁` space prefix information, resulting in `"Helloworld"` instead of `"Hello world"`.
- **The Fix**: Each step decodes the entire generated sequence so far and extracts only the text diff. This correctly preserves spaces, multi-byte Unicode characters, and sub-word joins.

---

## NF4 — What Is NormalFloat4?

NF4 is a data type optimized for normally-distributed neural network weights (introduced in the [QLoRA paper](https://arxiv.org/abs/2305.14314)). Rather than using uniform quantization levels, NF4 places quantization bins at the quantiles of a standard normal distribution, where most weight values cluster.

```
Uniform 4-bit:    ████████████████  (evenly spaced levels)
NF4:              ███████████   █████   ███  ██ █  (more levels near 0)
                  ◄── dense near center ──►  ◄ sparse at tails ►
```

**Per-block quantization:** Weights are divided into blocks (typically 64 elements). Each block has its own absmax scale factor, allowing the 4-bit levels to adapt to local weight magnitudes. This is more accurate than per-tensor quantization.

### Inference Path

During inference, when a `Linear4bit` layer's `forward()` is called:
1. The 4-bit weights are **dequantized to BF16** using the stored scale factors
2. A standard BF16 matrix multiplication is performed
3. No 4-bit weights persist in the compute path — it's purely a storage optimization

This dequantization is fused into the CUDA kernel by `bitsandbytes`, so the overhead is minimal.

---

## File-by-File Reference

### `models/base.py` — `get_linear_layer()`

The factory function that bridges the model architecture and quantization:

```python
get_linear_layer(in_features, out_features, bias, quantize=False)
```

- When `quantize=False`: returns `nn.Linear`
- When `quantize=True`: returns `bnb.nn.Linear4bit` with NF4 and BF16 compute dtype

### `models/weight_loader.py` — Dual-Path Loading

The `load_hf_model()` function selects between two loading strategies:

| | `_load_standard()` | `_load_quantized()` |
|---|---|---|
| **When** | `quantize=False` | `quantize=True` |
| **Method** | `load_state_dict(assign=True)` | Per-parameter `Params4bit` |
| **Memory** | Loads all shards at once | Shard-by-shard, aggressive cleanup |
| **Speed** | Faster | Slower (per-parameter quantization) |

### `models/llama.py` — Model Architecture

The `LlamaConfig` dataclass carries the `quantize` flag:

```python
@dataclass
class LlamaConfig:
    # ... other fields ...
    quantize: bool = False
```

This flag is read by `MLP.__init__()` and propagated through `get_linear_layer()` calls. The same config is shared by Llama, Mistral (identical architecture), and Qwen (via inheritance).

### `models/attention.py` — Attention Projections

All four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) use `get_linear_layer()`:

```python
self.q_proj = get_linear_layer(hidden_size, num_heads * head_dim, 
                                bias=config.attention_bias, 
                                quantize=config.quantize)
```

---

## Limitations & Known Issues

| Issue | Details |
|-------|---------|
| **No gradient support** | All quantized parameters have `requires_grad=False`. Fine-tuning/LoRA not yet supported. |
| **CPU not supported** | `bitsandbytes` 4-bit quantization requires CUDA. CPU inference must use full precision. |
| **Loading speed** | Per-parameter quantization is slower than standard loading (~2-3× longer). |
| **Dequantization overhead** | Each forward pass dequantizes weights on-the-fly. Throughput is lower than FP16 at small batch sizes. |
| **No quantized KV cache** | The KV cache remains in BF16. For very long sequences, this can still consume significant VRAM. |

---

## Future Work

- **GPTQ/AWQ support**: Pre-quantized model formats that skip runtime quantization
- **Quantized KV cache**: Reduce KV cache memory for long-context inference
- maybe **Mixed quantization**: Different precision for different layers (e.g., keep first/last layers at higher precision)
