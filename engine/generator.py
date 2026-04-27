"""Single-sequence generation loop. Will become the scheduler later."""
import torch
from engine.sampler import Sampler
from engine.sampling_params import SamplingParams

class Generator:
    def __init__(self, model, tokenizer, sampler: Sampler | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler or Sampler()

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, params: SamplingParams | None = None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.config.device)
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                logits, past_key_values = self.model(input_ids, position_ids=None)
            else:
                logits, past_key_values = self.model(input_ids[:, -1:], position_ids=None, past_key_values=past_key_values)

            next_token = self.sampler.sample(logits[:,-1,:], params or SamplingParams())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            yield self.tokenizer.decode(next_token[0], skip_special_tokens=True)