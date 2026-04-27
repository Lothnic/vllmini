import torch
import torch.nn.functional as F
from engine.sampling_params import SamplingParams

# Basically just sampling which means taking logits and returning tokens
# There are different sampling strategies:
# 1. Greedy sampling: always take the token with the highest probability
# 2. Temperature sampling: scale the logits by temperature
# 3. Top-p sampling: sample from the top p tokens
# 4. Top-k sampling: sample from the top k tokens
# 5. Nucleus sampling: sample from the smallest set of tokens whose cumulative probability exceeds p

class Sampler:

    def apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature

    def apply_top_k(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    def apply_top_p(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        
        # Mark tokens to remove
        remove = cumsum > p
        # Shift to the right to keep the first token that crosses the threshold
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        
        # Scatter back to original order
        remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        return logits.masked_fill(remove, float("-inf"))

    def sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """
        logits: (batch, vocab_size) — last token logits only
        Returns: (batch, 1) sampled token ids
        """

        if params.temperature > 0:
            logits = self.apply_temperature(logits, params.temperature)

        if params.top_k > 0:
            logits = self.apply_top_k(logits, params.top_k)

        if params.top_p < 1.0:
            logits = self.apply_top_p(logits, params.top_p)

        if params.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)