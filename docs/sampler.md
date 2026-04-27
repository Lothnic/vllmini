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

---

## Design: Stateless Sampler + SamplingParams

### Why the change was made

Originally, `Sampler` held `temperature`, `top_p`, and `top_k` as constructor arguments baked into the object:

```python
# Old design
sampler = Sampler(temperature=0.7, top_p=0.9)
sampler.sample(logits)  # always uses the same config
```

This breaks down with **continuous batching**, where multiple requests run through the engine simultaneously. Each request can have different sampling requirements:

```
Request A: "Write me a poem"   → temperature=1.5  (creative)
Request B: "What is 2+2?"      → temperature=0.0  (precise)
Request C: "Write Python code" → temperature=0.3  (structured)
```

A batched forward pass produces logits for all requests at once — shape `(3, vocab_size)`. With a stateful `Sampler`, you'd need a separate instance per request with no clean way to apply different configs to each row.

### The new design

`SamplingParams` is a plain dataclass that travels **with the request**, not with the sampler:

```python
# engine/sampling_params.py
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
```

`Sampler` is now stateless — config is passed at **call time**:

```python
# New design
sampler = Sampler()  # one instance, shared across all requests
sampler.sample(logits[0], params_A)  # temperature=1.5
sampler.sample(logits[1], params_B)  # temperature=0.0
sampler.sample(logits[2], params_C)  # temperature=0.3
```

### How this enables continuous batching (Phase 2)

Each `SequenceGroup` (future scheduler concept) will carry its own `SamplingParams`:

```python
@dataclass
class SequenceGroup:
    request_id: str
    prompt_tokens: list[int]
    sampling_params: SamplingParams  # per-request config
    ...
```

The scheduler can loop over all active requests and call `sampler.sample(logits[i], req.sampling_params)` with a single shared `Sampler` instance — no object juggling, no re-instantiation per request.

### Validation

Input validation (e.g. `temperature >= 0`, `top_p in (0, 1]`) lives in `SamplingParams.__post_init__`, so bad configs are caught at construction time, before they ever reach the sampler:

```python
SamplingParams(temperature=-1)  # raises ValueError immediately
```