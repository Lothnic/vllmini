import json
import os
import torch
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

def load_hf_model(model_id:str, device:str = "cuda", dtype:torch.dtype = torch.bfloat16):
    print(f"Loading model {model_id} to {device} with dtype {dtype}")

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
    )

    print(hf['architectures'][0])

    model_type = hf.get("model_type", "llama")
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model type '{model_type}' not supported. Supported types: {supported}")

    model_class = MODEL_REGISTRY[model_type]

    # Initialize model on meta device (no actual memory allocation)
    with torch.device("meta"):
        model = model_class(config)

    # Load weights directly to target device/dtype to avoid CPU copies
    try:
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors", local_files_only=local_only)
        state_dict = load_file(weights_path, device=device)
    except Exception:
        index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", local_files_only=local_only)
        with open(index_path, "r") as f:
            index = json.load(f)
        state_dict = {}
        for shard in set(index["weight_map"].values()):
            state_dict.update(load_file(hf_hub_download(repo_id=model_id, filename=shard, local_files_only=local_only), device=device))
        
    # Map HF names -> our names
    mapped = {}
    for k,v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[6:]
        new_k = new_k.replace("self_attn.", "attn.")
        new_k = new_k.replace("input_layernorm", "input_norm")
        new_k = new_k.replace("post_attention_layernorm", "post_norm")
        mapped[new_k] = v.to(dtype)

    del state_dict

    # assign=True replaces meta tensors with real ones (no double allocation)
    # Note: Tensors in 'mapped' are already on the target device and dtype.
    missing, unexpected = model.load_state_dict(mapped, strict=False, assign=True)

    # 1. Re-tie weights if they were tied in config.
    # Using assign=True breaks existing tying because it replaces Parameter objects.
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    # 2. Re-materialize RotaryEmbedding buffers on the target device.
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

    # 3. Ensure all remaining tensors (e.g. missing params, other buffers) are on the correct device/dtype
    # We use to_empty() for any remaining meta tensors, then to() for the rest.
    for param in model.parameters():
        if param.is_meta:
            param.data = torch.empty_like(param, device=device)
    for buffer in model.buffers():
        if buffer.is_meta:
            buffer.data = torch.empty_like(buffer, device=device)
            
    model.to(device, dtype=dtype)

    if missing:
        # Filter out expected missing buffers (RoPE caches computed at runtime)
        real_missing = [k for k in missing if "rotary_emb" not in k]
        if real_missing:
            print(f"Missing: {real_missing}")
    if unexpected:
        print(f"Unexpected: {unexpected}")

    del mapped

    config.device = device
    model.eval()
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

    