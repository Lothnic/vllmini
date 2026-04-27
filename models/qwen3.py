"""For Qwen3 -> llama + QK_norm"""
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.llama import LlamaConfig, MLP, LlamaForCausalLM, TransformerBlock, RMSNorm, RotaryEmbedding
from models.attention import FlashAttention as LlamaFlashAttention, apply_rotary

class QwenAttention(LlamaFlashAttention):
    def __init__(self, config: LlamaConfig, rotary_emb: RotaryEmbedding):
        super().__init__(config, rotary_emb)
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

        # 7. Scaled Dot-Product Attention (Inherited from FlashAttention)
        out = self.core_attention(q, k, v, q_len, kv_len)
        
        # 8. Output projection
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        
        return out, present_kv

class QwenTransformerBlock(TransformerBlock):
    def __init__(self, config: LlamaConfig, rotary_emb: RotaryEmbedding):
        super().__init__(config, rotary_emb)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = QwenAttention(config, rotary_emb)
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
        

class QwenForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QwenTransformerBlock(config, self.rotary_emb) for _ in range(config.num_hidden_layers)])
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
        
    