"""Llama-2/3, TinyLlama, Mistral, Qwen2.5 — all use this."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from models.base import CausalLM
from models.attention import Attention, FlashAttention


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    head_dim: int | None = None
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    quantize: bool = False
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)




class MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = get_linear_layer(config.hidden_size, config.intermediate_size, bias=False, quantize=config.quantize)
        self.up_proj = get_linear_layer(config.hidden_size, config.intermediate_size, bias=False, quantize=config.quantize)
        self.down_proj = get_linear_layer(config.intermediate_size, config.hidden_size, bias=False, quantize=config.quantize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = FlashAttention(config, rotary_emb)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, hidden_states, past_kv=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, present_kv = self.attn(hidden_states, past_kv, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present_kv


class LlamaForCausalLM(CausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList([TransformerBlock(config, self.rotary_emb) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, position_ids, past_key_values=None):
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        if position_ids is None:
            past_len = 0 if past_key_values[0] is None else past_key_values[0][0].shape[-2]
            position_ids = torch.arange(past_len, past_len + input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        present_key_values = []

        for i, layer in enumerate(self.layers):
            hidden_states, present_kv = layer(hidden_states, past_key_values[i], position_ids)
            present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, present_key_values