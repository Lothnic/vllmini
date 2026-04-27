import pytest
import torch
from engine.sampler import Sampler
from engine.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# SamplingParams — dataclass + validation
# ---------------------------------------------------------------------------

class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 1.0
        assert p.top_p == 1.0
        assert p.top_k == 0

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1)

    def test_top_p_above_one_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.1)

    def test_top_p_zero_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=0.0)

    def test_negative_top_k_raises(self):
        with pytest.raises(ValueError):
            SamplingParams(top_k=-1)

    def test_top_k_zero_is_valid(self):
        p = SamplingParams(top_k=0)  # 0 means "disabled", not an error
        assert p.top_k == 0

    def test_top_p_one_is_valid(self):
        p = SamplingParams(top_p=1.0)
        assert p.top_p == 1.0

    def test_temperature_zero_is_valid(self):
        p = SamplingParams(temperature=0.0)  # greedy
        assert p.temperature == 0.0


# ---------------------------------------------------------------------------
# Sampler — stateless, takes SamplingParams per call
# ---------------------------------------------------------------------------

class TestSampler:
    def setup_method(self):
        self.s = Sampler()  # no args — stateless
        self.logits = torch.tensor([[0.1, 2.0, 0.3]])

    def test_greedy_picks_argmax(self):
        params = SamplingParams(temperature=0.0)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert out.item() == 1  # index 1 has highest logit

    def test_top_k_1_picks_argmax(self):
        params = SamplingParams(temperature=0.0, top_k=1)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert out.item() == 1

    def test_top_p_filtering(self):
        params = SamplingParams(temperature=0.0, top_p=0.5)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert out.item() == 1

    def test_temperature_sampling_returns_valid_token(self):
        params = SamplingParams(temperature=1.0)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert 0 <= out.item() < self.logits.shape[-1]

    def test_batch_shape_preserved(self):
        params = SamplingParams(temperature=0.0)
        logits = torch.tensor([
            [0.1, 2.0, 0.3],
            [1.2, 0.1, 0.7],
        ])
        out = self.s.sample(logits, params)
        assert out.shape == (2, 1)

    def test_top_k_zero_disabled(self):
        """top_k=0 means no top-k filtering — should still return a valid token."""
        params = SamplingParams(temperature=0.0, top_k=0)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert out.item() == 1

    def test_top_p_one_no_filtering(self):
        """top_p=1.0 means no nucleus filtering — all tokens eligible, greedy picks argmax."""
        params = SamplingParams(temperature=0.0, top_p=1.0)
        out = self.s.sample(self.logits, params)
        assert out.shape == (1, 1)
        assert out.item() == 1  # highest logit in [[0.1, 2.0, 0.3]] is index 1

    def test_each_call_can_use_different_params(self):
        """Stateless: same Sampler instance, different params each call."""
        greedy = SamplingParams(temperature=0.0)
        creative = SamplingParams(temperature=2.0)

        out_greedy = self.s.sample(self.logits, greedy)
        out_creative = self.s.sample(self.logits, creative)

        # greedy must always pick index 1
        assert out_greedy.item() == 1
        # creative just needs to return a valid token id
        assert 0 <= out_creative.item() < self.logits.shape[-1]
