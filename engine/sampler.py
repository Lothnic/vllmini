import torch
import torch.nn.functional as F

# Basically just sampling which means taking logits and returning tokens
# There are different sampling strategies:
# 1. Greedy sampling: always take the token with the highest probability
# 2. Temperature sampling: scale the logits by temperature
# 3. Top-p sampling: sample from the top p tokens
# 4. Top-k sampling: sample from the top k tokens
# 5. Nucleus sampling: sample from the smallest set of tokens whose cumulative probability exceeds p

class Sampler:
    def __init__(self, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

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
        remove[..., 0] = False  # Always keep at least one
        
        # Scatter back to original order
        remove = remove.scatter(-1, sorted_indices, remove)
        return logits.masked_fill(remove, float("-inf"))

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch, vocab_size) — last token logits only
        Returns: (batch, 1) sampled token ids
        """

        if self.temperature > 0:
            logits = self.apply_temperature(logits, self.temperature)

        if self.top_k > 0:
            logits = self.apply_top_k(logits, self.top_k)

        if self.top_p < 1.0:
            logits = self.apply_top_p(logits, self.top_p)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)