# engine/sampler.py

what we are doing is **next-token** selection from **model logits**.

Think of logits as raw scores per token, e.g. for one step:
- token A: 0.1
- token B: 2.0
- token C: 0.3

Higher score = more likely token.

In your Sampler.sample() this happens in order:

## 1. Temperature (optional)

- Code: if self.temperature > 0: logits = logits / temperature
- Effect:
  - lower temp (<1): sharper, more confident choices
  - higher temp (>1): flatter, more random
  - temperature == 0: skip this and do greedy later

## 2. Top-k (optional)

- Code: keep only top k logits, set rest to -inf
- -inf means probability becomes zero after softmax.
- Example top_k=1: only the best token survives.

## 3. Top-p / nucleus (optional)

- Convert logits → probabilities via softmax
- Sort probabilities descending
- Keep the smallest set whose cumulative probability reaches p
- Mask everything else to -inf

## 4. Final choice

- If temperature <= 0: greedy (argmax) — deterministic.
- Else: sample from softmax distribution (torch.multinomial) — stochastic.