"""For Qwen3 -> llama + QK_norm"""
import math
import torch
import torch.nn.functional as F
from models.llama import LlamaConfig, RMSNorm, Attention as LlamaAttention, apply_rotary

class QwenAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, past_kv=None, position_ids=None):
        bsz, q_len, _ = hidden_states.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape for multi-head
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply QK-Norm (Qwen specific)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply Rotary Embeddings
        kv_len = k.shape[-2] + (0 if past_kv is None else past_kv[0].shape[-2])
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary(q, k, cos, sin)

        # 5. KV Caching
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present_kv = (k, v)

        # 6. Grouped Query Attention (repeat KV heads if needed)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # 7. Scaled Dot-Product Attention
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if q_len > 1:
            mask = torch.triu(torch.ones(q_len, kv_len, device=hidden_states.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        
        # 8. Output projection
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o_proj(out)
        
        return out, present_kv