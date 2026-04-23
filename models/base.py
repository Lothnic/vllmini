from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import bitsandbytes as bnb

def get_linear_layer(in_features: int, out_features: int, bias: bool, quantize: bool = False):
    if quantize:
        return bnb.nn.Linear4bit(in_features, out_features, bias=bias, compute_dtype=torch.bfloat16, quant_type="nf4")
    return nn.Linear(in_features, out_features, bias=bias)

class CausalLM(ABC, nn.Module):
    """Every model must implement this interface. The engine never looks inside."""

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list | None = None
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: (batch, seq_len)
            position_ids: (batch, seq_len)
            past_key_values: list of layer caches or None
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            present_key_values: list of layer caches for next step
        """
        pass