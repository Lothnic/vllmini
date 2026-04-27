import json
import os
import gc
import torch
import torch.nn as nn
from safetensors.torch import load_file
from models.llama import LlamaConfig, LlamaForCausalLM
from models.qwen3 import QwenForCausalLM
from huggingface_hub import hf_hub_download

# REGISTRY

MODEL_REGISTRY = {
        "llama": LlamaForCausalLM,
        "mistral": LlamaForCausalLM, # Logic is identical
        "qwen2": QwenForCausalLM,    # Qwen2+ all use QK-norm
        "qwen2_5": QwenForCausalLM,
        "qwen3": QwenForCausalLM,
    }


def _remap_key(k: str) -> str:
    """Map HuggingFace weight names -> our names."""
    if k.startswith("model."):
        k = k[6:]
    k = k.replace("self_attn.", "attn.")
    k = k.replace("input_layernorm", "input_norm")
    k = k.replace("post_attention_layernorm", "post_norm")
    return k


def _resolve_parameter(model: nn.Module, key: str):
    """Walk dot-separated key to find (parent_module, attr_name)."""
    parts = key.split(".")
    target = model
    for part in parts[:-1]:
        target = getattr(target, part)
    return target, parts[-1]


def _find_shard_paths(model_id: str, local_only: bool) -> list[str]:
    """Return list of safetensors shard paths (single file or multi-shard).

    Tries local cache first.  If the file isn't cached *and* ``local_only``
    is ``False``, transparently falls back to downloading from the Hub.
    """

    def _download(filename: str, *, must_be_local: bool) -> str:
        """Try local first, then online if allowed."""
        try:
            return hf_hub_download(repo_id=model_id, filename=filename, local_files_only=True)
        except Exception:
            if must_be_local:
                raise
            return hf_hub_download(repo_id=model_id, filename=filename, local_files_only=False)

    # 1. Try single-file model
    try:
        return [_download("model.safetensors", must_be_local=local_only)]
    except Exception:
        pass

    # 2. Multi-shard model — get the index first, then each shard
    index_path = _download("model.safetensors.index.json", must_be_local=local_only)
    with open(index_path, "r") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    return [_download(f, must_be_local=local_only) for f in shard_files]


def _load_standard(model, shard_paths, device, dtype):
    """Fast path: load all weights at once via load_state_dict(assign=True)."""
    state_dict = {}
    for path in shard_paths:
        state_dict.update(load_file(path, device=device))

    mapped = {_remap_key(k): v.to(dtype) for k, v in state_dict.items()}
    del state_dict

    missing, unexpected = model.load_state_dict(mapped, strict=False, assign=True)
    del mapped

    return missing, unexpected


def _load_quantized(model, shard_paths, device, dtype):
    """Quantized path: load shard-by-shard, quantize per-parameter via Params4bit."""
    from bitsandbytes.nn import Params4bit

    for path in shard_paths:
        shard = load_file(path, device="cpu")  # Always load to CPU first

        keys = list(shard.keys())
        for k in keys:
            v = shard.pop(k)  # Pop to free memory as we iterate
            new_k = _remap_key(k)

            try:
                target, attr_name = _resolve_parameter(model, new_k)
                param = getattr(target, attr_name)
            except AttributeError:
                del v
                continue  # Skip unmapped keys (e.g. keys we don't use)

            v_typed = v.to(dtype=dtype)
            del v  # Free original tensor immediately

            if hasattr(param, "quant_type"):
                # This is a Linear4bit parameter — quantize and place on device
                new_param = Params4bit(
                    v_typed,
                    requires_grad=False,
                    quant_type=getattr(param, "quant_type", "nf4"),
                )
                del v_typed  # Free CPU copy before GPU allocation
                new_param = new_param.to(device)
                setattr(target, attr_name, new_param)
            else:
                # Normal parameter (embeddings, norms, lm_head, etc.)
                target.register_parameter(
                    attr_name,
                    nn.Parameter(v_typed.to(device), requires_grad=False),
                )
                del v_typed

        del shard
        gc.collect()
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_hf_model(model_id: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, quantize: bool = False):
    print(f"Loading model {model_id} to {device} with dtype {dtype} (quantize={quantize})")

    local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    config_path = hf_hub_download(repo_id=model_id, filename="config.json", local_files_only=local_only)

    with open(config_path, "r") as f:
        hf = json.load(f)

    config = LlamaConfig(
        vocab_size=hf["vocab_size"],
        hidden_size=hf["hidden_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        intermediate_size=hf["intermediate_size"],
        max_position_embeddings=hf.get("max_position_embeddings", 2048),
        rms_norm_eps=hf.get("rms_norm_eps", 1e-6),
        rope_theta=hf.get("rope_theta", 10000.0),
        attention_bias=hf.get("attention_bias", False),
        tie_word_embeddings=hf.get("tie_word_embeddings", False),
        head_dim=hf.get("head_dim"),
        quantize=quantize,
    )

    print(hf['architectures'][0])

    model_type = hf.get("model_type", "llama")
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model type '{model_type}' not supported. Supported types: {supported}")

    model_class = MODEL_REGISTRY[model_type]

    # 1. Initialize model on meta device (no actual memory allocation)
    with torch.device("meta"):
        model = model_class(config)

    # 2. Find shard files
    shard_paths = _find_shard_paths(model_id, local_only)

    # 3. Load weights — dual path strategy
    if quantize:
        _load_quantized(model, shard_paths, device, dtype)
    else:
        _load_standard(model, shard_paths, device, dtype)

    # 4. Re-tie weights if they were tied in config
    # assign=True / per-param loading breaks existing tying because it replaces Parameter objects
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    # 5. Re-materialize RotaryEmbedding buffers on the target device.
    # These are computed buffers (not saved in checkpoints) that remain as
    # meta tensors after meta-device init + assign=True loading.
    from models.llama import RotaryEmbedding
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, device=device).float() / config.head_dim))
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype, device=device)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            module.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            module.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 6. Ensure all remaining meta tensors are materialized on the correct device
    for param in model.parameters():
        if param.is_meta:
            param.data = torch.empty_like(param, device=device)
    for buffer in model.buffers():
        if buffer.is_meta:
            buffer.data = torch.empty_like(buffer, device=device)

    # 7. Move model to target device/dtype
    # Note: for quantized models, bnb parameters handle their own dtype,
    # model.to() will skip them automatically
    model.to(device, dtype=dtype)

    # 8. Check for missing parameters
    missing_keys = [name for name, param in model.named_parameters() if param.is_meta]
    if missing_keys:
        real_missing = [k for k in missing_keys if "lm_head" not in k]
        if real_missing:
            print(f"Missing: {real_missing}")

    config.device = device
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, config


# For testing
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading model {model_id}")
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")

    with open(config_path, "r") as f:
        hf = json.load(f)

    if hf['architectures'][0]=="LlamaForCausalLM":
        print(True)
    else:
        print(False)

    