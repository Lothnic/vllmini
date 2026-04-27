"""Tests for the benchmarking script (benchmark.py)."""
import pytest
from unittest.mock import MagicMock, patch
import torch
from benchmark import parse_args, benchmark

class TestParseArgs:
    def _parse(self, argv: list[str]):
        """Helper: patch sys.argv and call parse_args()."""
        with patch("sys.argv", ["benchmark.py"] + argv):
            return parse_args()

    def test_defaults_are_applied(self):
        args = self._parse([])
        # Default model is Mistral-7B now as updated by user
        assert args.device == ("cuda" if torch.cuda.is_available() else "cpu")
        assert args.quantize is False

    def test_model_id_short_flag(self):
        args = self._parse(["-m", "test-model"])
        assert args.model_id == "test-model"

    def test_model_id_long_flag(self):
        args = self._parse(["--model-id", "test-model-long"])
        assert args.model_id == "test-model-long"

    def test_quantize_short_flag(self):
        args = self._parse(["-q"])
        assert args.quantize is True

    def test_quantize_long_flag(self):
        args = self._parse(["--quantize"])
        assert args.quantize is True

    def test_unknown_flag_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["--unknown-flag"])

class TestBenchmarkFunction:
    @patch("benchmark.load_hf_model")
    @patch("benchmark.AutoTokenizer.from_pretrained")
    @patch("benchmark.Sampler")
    @patch("benchmark.SamplingParams")
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.max_memory_allocated", return_value=4 * 1024**3)
    def test_benchmark_runs_end_to_end(
        self,
        mock_vram,
        mock_empty,
        mock_reset,
        mock_sync,
        mock_params,
        mock_sampler_class,
        mock_tokenizer_class,
        mock_load_hf
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_config = MagicMock()
        mock_load_hf.return_value = (mock_model, mock_config)
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.eos_token_id = 50256
        
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        # sampler.sample returns a tensor
        mock_sampler.sample.return_value = torch.tensor([10])
        
        # model() returns (logits, past_key_values)
        mock_model.return_value = (torch.randn(1, 3, 32000), [MagicMock()])

        # Run benchmark with very few tokens to speed up test
        with patch("benchmark.MAX_NEW_TOKENS", 2):
            benchmark("test-model", "test-prompt", "cpu", quantize=True)

        # Verify calls
        mock_load_hf.assert_called_once_with("test-model", device="cpu", quantize=True)
        mock_tokenizer_class.assert_called_once_with("test-model")
        
        # Verify model was called (warmup + prefill + 1 decode step)
        assert mock_model.call_count >= 3
        
        # Verify sampler was called
        assert mock_sampler.sample.call_count >= 2

    @patch("benchmark.load_hf_model")
    @patch("benchmark.AutoTokenizer.from_pretrained")
    @patch("benchmark.Sampler")
    @patch("benchmark.SamplingParams")
    def test_benchmark_handles_eos(
        self,
        mock_params,
        mock_sampler_class,
        mock_tokenizer_class,
        mock_load_hf
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_load_hf.return_value = (mock_model, MagicMock())
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.eos_token_id = 999
        
        mock_sampler = MagicMock()
        mock_sampler_class.return_value = mock_sampler
        # Return EOS token on first sample
        mock_sampler.sample.return_value = torch.tensor([999])
        
        mock_model.return_value = (torch.randn(1, 3, 32000), [MagicMock()])

        benchmark("test-model", "test-prompt", "cpu", quantize=False)

        # Should break after prefill because first token was EOS
        assert mock_sampler.sample.call_count == 1
