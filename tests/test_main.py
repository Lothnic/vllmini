"""Tests for the CLI entry point (main.py)."""
import sys
import pytest
from unittest.mock import MagicMock, call, patch
from io import StringIO

from main import strip_thinking, parse_args

# For strip thinking

class TestStripThinking:
    def test_strips_content_before_closing_tag(self):
        output = "<think>lots of reasoning</think>Here is the answer."
        assert strip_thinking(output) == "Here is the answer."

    def test_returns_original_when_no_closing_tag(self):
        output = "Plain response without a think tag."
        assert strip_thinking(output) == output

    def test_strips_multiline_thinking_block(self):
        output = "<think>\nReasoning line 1\nReasoning line 2\n</think>\nFinal answer."
        assert strip_thinking(output) == "Final answer."

    def test_returns_empty_string_when_nothing_after_tag(self):
        output = "<think>some reasoning</think>"
        assert strip_thinking(output) == ""

    def test_uses_last_closing_tag_when_multiple(self):
        # Should split on the last </think> effectively by using [-1]
        output = "<think>first</think>middle<think>second</think>last"
        assert strip_thinking(output) == "last"

    def test_empty_input_returns_empty(self):
        assert strip_thinking("") == ""

    def test_only_opening_tag_no_close_returns_original(self):
        output = "<think>still thinking..."
        assert strip_thinking(output) == output

# for arg parse

class TestParseArgs:
    def _parse(self, argv: list[str]):
        """Helper: patch sys.argv and call parse_args()."""
        with patch("sys.argv", ["main.py"] + argv):
            return parse_args()

    def test_defaults_are_applied(self):
        args = self._parse([])
        assert args.temperature == 0.7
        assert args.top_p == 0.9
        assert args.max_tokens == 2048

    def test_model_id_long_flag(self):
        args = self._parse(["--model-id", "meta-llama/Llama-3.2-1B-Instruct"])
        assert args.model_id == "meta-llama/Llama-3.2-1B-Instruct"

    def test_model_id_short_flag(self):
        args = self._parse(["-m", "Qwen/Qwen3-4B"])
        assert args.model_id == "Qwen/Qwen3-4B"

    def test_device_flag(self):
        args = self._parse(["--device", "cpu"])
        assert args.device == "cpu"

    def test_device_short_flag(self):
        args = self._parse(["-d", "cpu"])
        assert args.device == "cpu"

    def test_hide_thinking_flag(self):
        args = self._parse(["--hide-thinking"])
        assert args.hide_thinking is True

    def test_hide_thinking_short_flag(self):
        args = self._parse(["-t"])
        assert args.hide_thinking is True

    def test_temperature_flag(self):
        args = self._parse(["--temperature", "1.5"])
        assert args.temperature == 1.5

    def test_top_p_flag(self):
        args = self._parse(["--top-p", "0.95"])
        assert args.top_p == 0.95

    def test_max_tokens_flag(self):
        args = self._parse(["--max-tokens", "512"])
        assert args.max_tokens == 512

    def test_unknown_flag_exits(self):
        with pytest.raises(SystemExit):
            self._parse(["--unknown-flag"])

# Chat loop command handling
# These tests exercise the command-dispatch logic isolated from the real model.

def _make_args(**kwargs):
    """Build a minimal namespace that mirrors parse_args() output."""
    defaults = dict(
        model_id="test-model",
        hide_thinking=False,
        device="cpu",
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )
    defaults.update(kwargs)
    return MagicMock(**defaults)


class TestChatLoopCommands:
    """
    Test the chat loop command handlers without touching the model.
    We import _process_command (a thin extraction of the switch logic) — if
    that doesn't exist yet, we test the behaviours inline via subprocess / mock.
    """

    def test_exit_command_breaks_loop(self, capsys):
        """Simulate one /exit input and ensure the loop terminates cleanly."""
        from main import main

        args = _make_args()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"

        with (
            patch("main.parse_args", return_value=args),
            patch("main.load_hf_model", return_value=(mock_model, MagicMock())),
            patch("main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("main.Generator"),
            patch("main.Sampler"),
            patch("builtins.input", return_value="/exit"),
        ):
            main()  # should return without hanging

    def test_reset_command_clears_history(self, capsys):
        """After /reset, subsequent /history should report 'No history.'"""
        from main import main

        args = _make_args()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"

        inputs = iter(["/reset", "/history", "/exit"])

        with (
            patch("main.parse_args", return_value=args),
            patch("main.load_hf_model", return_value=(MagicMock(), MagicMock())),
            patch("main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("main.Generator"),
            patch("main.Sampler"),
            patch("builtins.input", side_effect=inputs),
        ):
            main()

        captured = capsys.readouterr()
        assert "Chat reset." in captured.out
        assert "No history." in captured.out

    def test_empty_input_is_ignored(self, capsys):
        """An empty string should not be added to messages."""
        from main import main

        args = _make_args()
        mock_tokenizer = MagicMock()

        inputs = iter(["", "/exit"])

        with (
            patch("main.parse_args", return_value=args),
            patch("main.load_hf_model", return_value=(MagicMock(), MagicMock())),
            patch("main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("main.Generator"),
            patch("main.Sampler"),
            patch("builtins.input", side_effect=inputs),
        ):
            main()

        # If history is printed it would have content; since we /exit immediately
        # after a blank line, history is untouched — the test just verifies no crash.

    def test_eof_exits_gracefully(self, capsys):
        """EOFError (e.g. piped stdin exhausted) should exit without traceback."""
        from main import main

        args = _make_args()
        mock_tokenizer = MagicMock()

        with (
            patch("main.parse_args", return_value=args),
            patch("main.load_hf_model", return_value=(MagicMock(), MagicMock())),
            patch("main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("main.Generator"),
            patch("main.Sampler"),
            patch("builtins.input", side_effect=EOFError),
        ):
            main()

        captured = capsys.readouterr()
        assert "Exiting" in captured.out
