from safetensors import torch
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from models.llama import LlamaConfig, LlamaForCausalLM
from models.qwen3 import QwenForCausalLM
from huggingface_hub import hf_hub_download
import gc

# REGISTRY

MODEL_REGISTRY = {
        "llama": LlamaForCausalLM,
        "mistral": LlamaForCausalLM, # Logic is identical
        "qwen2": QwenForCausalLM,    # Qwen2+ all use QK-norm
        "qwen2_5": QwenForCausalLM,
        "qwen3": QwenForCausalLM,
    }

def load_hf_model(model_id:str, device:str = "cuda", dtype:torch.dtype = torch.bfloat16, quantize: bool = False):
    print(f"Loading model {model_id} to {device} with dtype {dtype} (quantize={quantize})")

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")

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

    # 1. Initialize model on meta device
    with torch.device("meta"):
        model = model_class(config)

    # 2. Find shard files
    try:
        # Check for single file
        shard_paths = [hf_hub_download(repo_id=model_id, filename="model.safetensors")]
    except Exception:
        # Multi-shard model
        index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        shard_paths = [hf_hub_download(repo_id=model_id, filename=f) for f in shard_files]

    # 3. Load shards one by one
    for path in shard_paths:
        shard = load_file(path, device="cpu")
        for k, v in shard.items():
            new_k = k
            if new_k.startswith("model."):
                new_k = new_k[6:]
            new_k = new_k.replace("self_attn.", "attn.")
            new_k = new_k.replace("input_layernorm", "input_norm")
            new_k = new_k.replace("post_attention_layernorm", "post_norm")
            
            # Find the parameter in the model
            try:
                # Use recursive getattr to find the parameter
                target = model
                target_name = new_k
                if "." in new_k:
                    parts = new_k.split(".")
                    for part in parts[:-1]:
                        target = getattr(target, part)
                    target_name = parts[-1]
                
                param = getattr(target, target_name)
                
                # Prepare weight on CPU
                v_cpu = v.to(dtype=dtype)
                
                if hasattr(param, "quant_type"):
                    # For bitsandbytes 4-bit parameters
                    from bitsandbytes.nn import Params4bit
                    new_param = Params4bit(
                        v_cpu, 
                        requires_grad=False, 
                        quant_type=getattr(param, "quant_type", "nf4")
                    ).to(device)
                    setattr(target, target_name, new_param)
                else:
                    # For normal parameters
                    target.register_parameter(target_name, nn.Parameter(v_cpu.to(device), requires_grad=False))
                
            except AttributeError:
                # Might be a buffer or something we handle later (like lm_head tying)
                pass
        
        del shard
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 5. Re-tie weights (Standard for Llama/Qwen)
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

    # Check for missing parameters (excluding RoPE buffers)
    missing_keys = []
    for name, param in model.named_parameters():
        if param.is_meta:
            missing_keys.append(name)
    
    if missing_keys:
        real_missing = [k for k in missing_keys if "lm_head" not in k] # lm_head handled by re-tie
        if real_missing:
            print(f"Missing: {real_missing}")

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

    