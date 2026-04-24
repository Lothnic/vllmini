# vLLMini

My own implementation of vLLM.

A lightweight inference engine for Large Language Models built from scratch to understand the internals of high performance serving.


## Models supported

- [x] Llama 2/3 and their derivatives
- [x] Qwen 2/3 and their derivatives
- [x] Mistral (succesfully tested on Mistral 7B Instruct v0.3 but using bnb for 4 bit quantisation)
- [ ] Mixtral
- [ ] Gemma
- [ ] Phi



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

## Interesting Things I learnt while building this (will be adding more to this list) 

### not currently ranked in any order of importance or anything

- there is a seperate inference mode in torch other that eval() and no_grad() called torch.inference_mode() which is more efficient than both. might deep dive into these someday later.
- we can initalise the model on a meta device and then load the weights directly to the target device/dtype to avoid cpu copies. still don't understand this but this prevented OOM due to double loading on cpu and gpu.
- the model architecture don't differ that much from llama to qwen to mistral. they all use the same basic building blocks just with some tweaks here and there. but still have more model to check out and integrate.

## Features and Branching

- Benchmarking script added and model performance compared to LMstudio. Next step is to compare with vLLM.
- Implemented **Quantisation** using `bitsandbytes` for 4-bit NF4 quantisation. Still experimenting so it has a seperate branch.
- **RoPE Sharing** : Optimised rotary embedding buffers to share vram across 32+ layers.

## Benchmarking

- **Warmup**: we run one short generation first to "*warm up*" the GPU and JIT kernels (like SDPA - scaled_dot_product_attention ).
- **Prefil Timing**: measuring how long it takes for the first token ID to be generated from the first `model.forward()` call.
- **Decode Loop Timing**: Collect timestamps for every subsequent token to calculate the average Inter-Token Latency (ITL).
- **Throughput**: it is just the total no of token / total gen time.
- **VRAM Tracking**: using torch.cuda.max_memory_allocated() to find the peak VRAM usage.

