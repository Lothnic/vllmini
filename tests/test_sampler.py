import pytest
import torch
from engine.sampler import Sampler

def test_sampler_validation():
    with pytest.raises(ValueError):
        Sampler(temperature=-1)

    with pytest.raises(ValueError):
        Sampler(top_p=0)

    with pytest.raises(ValueError):
        Sampler(top_k=-1)

def test_temperature_zero_is_greedy():
    s = Sampler(temperature=0.0)
    logits = torch.tensor([[0.1, 2.0, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)
    assert out.item() == 1

def test_top_k_filtering():
    s = Sampler(temperature=0.0, top_k=1)

    logits = torch.tensor([[0.1, 2.0, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)
    assert out.item() == 1

def test_top_p_filtering():
    s = Sampler(temperature=0.0, top_p=0.5)

    logits = torch.tensor([[0.1, 2.0, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)
    assert out.item() == 1

def test_temperature_sampling():
    s = Sampler(temperature=1.0)

    logits = torch.tensor([[0.1, 2.0, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)


def test_batch_shape_is_preserved():
    s = Sampler(temperature=0.0)

    logits = torch.tensor(
        [
            [0.1, 2.0, 0.3],
            [1.2, 0.1, 0.7],
        ]
    )
    out = s.sample(logits)

    assert out.shape == (2, 1)

def test_top_p_above_one_raises():
    with pytest.raises(ValueError):
        Sampler(top_p=1.1)

def test_top_p_zero_raises():
    with pytest.raises(ValueError):
        Sampler(top_p=0.0)

def test_top_k_zero_is_valid():
    s = Sampler(temperature=0.0, top_k=0)

    logits = torch.tensor([[0.1, 2.0, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)
    assert out.item() == 1

def test_top_p_one_is_valid():
    s = Sampler(temperature=0.0, top_p=1.0)
    
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    out = s.sample(logits)
    
    assert out.shape == (1, 1)
    assert out.item() == 2
