import torch
import torch.nn as nn
import math

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class Attention(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states, past_kv=None, position_ids=None):
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        kv_len = k.shape[-2] + (0 if past_kv is None else past_kv[0].shape[-2])
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if q_len > 1:
            mask = torch.triu(torch.ones(q_len, kv_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out, present_kv


