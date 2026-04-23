from safetensors import torch
import json
import torch
from safetensors.torch import load_file
from models.llama import LlamaConfig, LlamaForCausalLM
from huggingface_hub import hf_hub_download
import joblib

#CONFIG TO BE FILED LATER
# REPO_ID = ""
# FILE_NAME = ""

# model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME))
#  to download the model

def load_hf_model(model_id:str, device:str = "cuda", dtype:torch.dtype = torch.bfloat16):
    print(f"Loading model {model_id} to {device} with dtype {dtype}")

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")

    with open(config_path, "r") as f:
        hf = json.load(f)

    config = LlamaConfig(**hf)
    model = LlamaForCausalLM(config).to(device, dtype=dtype)

    # Load weights

    try:
        weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
    except Exception:
        index_path = hf_hub_download(repo_id = model_id, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        state_dict = {}
        for shard in set(index["weight_map"].values()):
            state_dict.update(load_file(hf_hub_download(repo_id=model_id, filename=shard)))
        
    # Map HF names -> our names
    mapped = {}
    for k,v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[6:]
        new_k = new_k.replace("self_attn.", "attn.")
        new_k = new_k.replace("input_layernorm", "input_norm")
        new_k = new_k.replace("post_attention_layernorm", "post_norm")
        mapped[new_k] = v

    missing, unexpected = model.load_state_dict(mapped,strict=False)
    if missing:
        print(f"Missing: {missing}")
    if unexpected:
        print(f"Unexpected: {unexpected}")

    model.eval()
    del state_dict,mapped
    torch.cude.empty_cache()
    return model, config

    