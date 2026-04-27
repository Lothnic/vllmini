# vLLMini

My own implementation of vLLM.

A lightweight inference engine for Large Language Models built from scratch to understand the internals of high performance serving.


## Models supported

- [x] Llama 2/3 and their derivatives
- [x] Qwen 2/3 and their derivatives
- [x] Mistral (succesfully tested on Mistral 7B Instruct v0.3 but using bnb for 4 bit quantisation)
- [ ] Mixtral [MOE]
- [ ] Gemma



## Installation

uv is recommended for dependency management.

```bash
git clone https://github.com/lothnic/vllmini.git
cd vllmini
uv sync
```

## Usage

- change the model id in the main.py file
- run `uv run python main.py`

## Project Architecture (as-is)

```
vllmini/
├── main.py                  # CLI chat loop (multi-turn, streaming, argparse)
├── benchmark.py             # Perf harness (TTFT, ITL, tok/s, VRAM)
├── engine/
│   ├── generator.py         # Single-seq generation loop (yield-based)
│   ├── sampler.py           # Stateless sampler (temperature, top-k, top-p, greedy)
│   └── sampling_params.py   # SamplingParams dataclass — per-request sampling config
├── models/
│   ├── base.py              # CausalLM ABC (forward interface)
│   ├── attention.py         # Attention + FlashAttention + RoPE utils
│   ├── llama.py             # LlamaConfig, RMSNorm, RotaryEmb, MLP, TransformerBlock, LlamaForCausalLM
│   ├── qwen3.py             # QwenAttention (+ QK-norm), QwenTransformerBlock, QwenForCausalLM
│   └── weight_loader.py     # HF download, config parse, meta-init, weight mapping, model registry
├── tests/
│   ├── conftest.py
│   ├── test_sampler.py      # 17 unit tests (SamplingParams validation + stateless Sampler)
│   └── test_main.py         # 22 unit tests (strip_thinking, parse_args, chat loop commands)
├── docs/
│   └── sampler.md           # Sampler internals + SamplingParams design rationale
├── archive/main.py          # Old non-streaming main
└── .github/workflows/ci.yml # CI (pytest + smoke)
```

## Interesting Things I learnt while building this (will be adding more to this list) 

### not currently ranked in any order of importance or anything

- there is a seperate inference mode in torch other that eval() and no_grad() called **torch.inference_mode()** which is more efficient than both. might deep dive into these someday later.
- we can initalise the model on a **meta device** and then load the weights directly to the target device/dtype to avoid cpu copies. still don't understand this but this prevented OOM due to double loading on cpu and gpu.
- the model architecture don't differ that much from llama to qwen to mistral. they all use the same basic building blocks just with some tweaks here and there. but still have more model to check out and integrate.
- for streaming ouput i am now using `yield` instead of `return` in the generation loop. this turns `generate()` into a **Python generator** — it lazily produces one token at a time and suspends between each one. the caller just does `for token in gen.generate(...)` and gets real-time streaming for free, no buffering the entire response in memory. this is what is used in the production LLM servers for SSE (Server-Sent Events) streaming over HTTP.
- the **sampler** is now **stateless** and the **sampling params** are passed as a **dataclass** to the sampler. this is done to support **continuous batching** which i intend to implement in the future (hopefully).
- **SDPA** or flashattention-2 is a game changer. it is a drop in replacement for **scaled_dot_product_attention** and it is much faster and more memory efficient. 

## Features and Branching

- Benchmarking script added and model performance compared to LMstudio. Next step is to compare with vLLM.
- Implemented **Quantization** using `bitsandbytes` for 4-bit NF4 quantisation.   ~~Still experimenting so it has a seperate branch~~ (it has been merged with main now).
    - **Key insight**: 4-bit is a compressed encoding of the weights, we don't just truncate the weights to 4-bit (Turns out this is not correct, more on this later.) 
    - **Dequantisation Formula**: 
    `` dequantized weight = codebook[4 bit index] × scale + zero point ``
    - **Basic idea**: so instead of storing weights as FP16/BF16 (2 bytes per parameter) or FP32 (4 bytes) which is full precision, we store each weight as a 4-bit index (0–15). This 4-bit value points to a specific float in a shared codebook (typically 16 values). So 16 weights share the same 16-entry codebook, meaning you need only 0.25 bytes per weight + 16 bytes per block for scale/zero-point. This reduces memory by ~8x vs FP16 and ~16x vs FP32.
    
- **RoPE Sharing** : Optimised rotary embedding buffers to share vram across 32+ layers.

## Benchmarking

- **Warmup**: we run one short generation first to "*warm up*" the GPU and JIT kernels (like SDPA - scaled_dot_product_attention ).
- **Prefil Timing**: measuring how long it takes for the first token ID to be generated from the first `model.forward()` call.
- **Decode Loop Timing**: Collect timestamps for every subsequent token to calculate the average Inter-Token Latency (ITL).
- **Throughput**: it is just the total no of token / total gen time.
- **VRAM Tracking**: using torch.cuda.max_memory_allocated() to find the peak VRAM usage.

