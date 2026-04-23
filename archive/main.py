"""
Minimal LLM Inference Engine in Pure PyTorch for TinyLlama-1.1B-Chat-v1.0
"""
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
# Only using transformers for the tokenizer
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# 1. Architecture Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)   # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        position_ids: (batch, seq_len) — absolute positions of each token
        Returns cos, sin: (batch, 1, seq_len, dim)
        """
        cos = self.cos_cached[position_ids].unsqueeze(1)  # (batch, 1, seq_len, dim)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.get("max_position_embeddings", 2048)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=self.max_position_embeddings,
                                          base=config.get("rope_theta", 10000.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # KV cache length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Append to KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states)

        # GQA: repeat KV heads to match Q heads for broadcasting
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Causal mask (only needed during prefill when q_len > 1)
        if q_len > 1:
            causal_mask = torch.triu(
                torch.ones(q_len, kv_seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up, then down-project
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_kv = self.self_attn(hidden_states, past_key_value=past_key_value, position_ids=position_ids)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_kv


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.embed_tokens = nn.Embedding(self.vocab_size, config["hidden_size"])
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config["num_hidden_layers"])])
        self.norm = RMSNorm(config["hidden_size"], eps=config.get("rms_norm_eps", 1e-6))
        self.lm_head = nn.Linear(config["hidden_size"], self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if position_ids is None:
            # Infer position_ids from cache length
            past_length = 0 if past_key_values[0] is None else past_key_values[0][0].shape[-2]
            position_ids = torch.arange(
                past_length, past_length + input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        present_key_values = []
        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(
                hidden_states,
                past_key_value=past_key_values[i],
                position_ids=position_ids,
            )
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, present_key_values


# -----------------------------------------------------------------------------
# 2. Weight Loading (Manual)
# -----------------------------------------------------------------------------

def normalize_hf_llama_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HF LLaMA keys (`model.*`) to this implementation's module layout."""
    normalized = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("model.")] = value
    return normalized


def load_model_and_config(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = "cuda"):
    print(f"Downloading config for {model_id}...")
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("Initializing model architecture...")
    model = LlamaForCausalLM(config)
    model = model.to(device).to(torch.float16)  # FP16 for RTX 4060

    print(f"Downloading weights for {model_id}...")
    # TinyLlama is a single safetensors file. For sharded models, you'd load the index first.
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    state_dict = normalize_hf_llama_state_dict(load_file(weights_path))

    print("Loading weights into model...")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Cleanup
    del state_dict
    torch.cuda.empty_cache()

    print("Model ready.")
    return model, config


# -----------------------------------------------------------------------------
# 3. Generation
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: LlamaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    past_key_values = None

    generated_tokens = []

    for i in range(max_new_tokens):
        # Prefill on first step (full prompt), decode on subsequent steps (last token only)
        if past_key_values is None:
            logits, past_key_values = model(input_ids)
        else:
            logits, past_key_values = model(input_ids[:, -1:], past_key_values=past_key_values)

        next_token_logits = logits[:, -1, :] / temperature

        # Top-p (nucleus) sampling
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 0] = False  # Keep at least one token
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output_text


# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model, config = load_model_and_config(MODEL_ID, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prompt = "Who is Sam Altman?"
    print(f"\nPrompt: {prompt}")
    output = generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1)
    print(f"Output: {output}")
