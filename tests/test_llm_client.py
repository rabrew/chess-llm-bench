"""Tests for LLM client response parsing."""

import pytest
from unittest.mock import patch, MagicMock

from src.llm_client import (
    parse_response, build_prompt, OllamaClient,
    build_eval_prompt, build_explanation_prompt, build_move_prompt,
    parse_eval_response, parse_explanation_response,
    MOVE_SYSTEM_PROMPT, EVAL_SYSTEM_PROMPT, EXPLANATION_SYSTEM_PROMPT,
)


class TestParseResponse:
    def test_complete_response(self, sample_llm_response):
        result = parse_response(sample_llm_response)
        assert result["eval"] == 45
        assert result["move"] == "Nf6"
        assert "Equal" in result["explanation"]
        assert result["side_claimed"] == "Equal"
        assert len(result["parse_errors"]) == 0

    def test_missing_eval(self):
        response = """Move: Nf6
Explanation: White is better — material advantage."""
        result = parse_response(response)
        assert result["eval"] is None
        assert result["move"] == "Nf6"
        assert "Missing Eval field" in result["parse_errors"]

    def test_missing_move(self):
        response = """Eval: 100
Explanation: White is better — material advantage."""
        result = parse_response(response)
        assert result["eval"] == 100
        assert result["move"] is None
        assert "Missing Move field" in result["parse_errors"]

    def test_missing_explanation(self):
        response = """Eval: 100
Move: Nf6"""
        result = parse_response(response)
        assert result["eval"] == 100
        assert result["move"] == "Nf6"
        assert result["explanation"] is None
        assert "Missing Explanation field" in result["parse_errors"]

    def test_negative_eval(self):
        response = """Eval: -150
Move: e5
Explanation: Black is better — White has weak pawns."""
        result = parse_response(response)
        assert result["eval"] == -150
        assert result["side_claimed"] == "Black"

    def test_eval_with_extra_text(self):
        response = """Eval: 100 centipawns
Move: Nf6
Explanation: White is better."""
        result = parse_response(response)
        assert result["eval"] == 100


class TestOllamaClient:
    def test_init(self):
        client = OllamaClient(base_url="http://localhost:11434", timeout=60, max_retries=2)
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 60
        assert client.max_retries == 2

    def test_base_url_strips_trailing_slash(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert not client.base_url.endswith("/")

    @patch("src.llm_client.requests.get")
    def test_is_available_true(self, mock_get):
        mock_get.return_value.status_code = 200
        client = OllamaClient()
        assert client.is_available() is True

    @patch("src.llm_client.requests.get")
    def test_is_available_false_on_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("connection refused")
        client = OllamaClient()
        assert client.is_available() is False

    @patch("src.llm_client.requests.get")
    def test_list_models_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.2:3b"}, {"name": "qwen2.5:7b"}]
        }
        mock_get.return_value.raise_for_status = MagicMock()
        client = OllamaClient()
        models = client.list_models()
        assert "llama3.2:3b" in models
        assert "qwen2.5:7b" in models

    @patch("src.llm_client.requests.get")
    def test_list_models_on_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("error")
        client = OllamaClient()
        models = client.list_models()
        assert models == []

    @patch("src.llm_client.requests.post")
    def test_pull_model_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.iter_lines.return_value = [b"pulling model"]
        client = OllamaClient()
        result = client.pull_model("llama3.2:3b")
        assert result is True

    @patch("src.llm_client.requests.post")
    def test_pull_model_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.RequestException("error")
        client = OllamaClient()
        result = client.pull_model("llama3.2:3b")
        assert result is False

    def test_get_options_72b(self):
        opts = OllamaClient._get_options("qwen2.5:72b")
        assert opts["num_gpu"] == 26
        assert "num_ctx" in opts

    def test_get_options_70b(self):
        opts = OllamaClient._get_options("llama3.3:70b")
        assert opts["num_gpu"] == 26

    def test_get_options_small_model(self):
        opts = OllamaClient._get_options("llama3.2:3b")
        assert opts == {}

    @patch("src.llm_client.requests.post")
    def test_chat_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "message": {"content": "Eval: 50\nMove: e4\nExplanation: Equal — test"}
        }
        client = OllamaClient(max_retries=1)
        result = client.chat("llama3.2:3b", "test prompt")
        assert result["success"] is True
        assert "Eval: 50" in result["response"]

    @patch("src.llm_client.requests.post")
    def test_chat_with_system_prompt(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"message": {"content": "e4"}}
        client = OllamaClient(max_retries=1)
        result = client.chat("llama3.2:3b", "prompt", system_prompt="You are a chess engine.")
        assert result["success"] is True

    @patch("src.llm_client.requests.post")
    @patch("src.llm_client.time.sleep")
    def test_chat_timeout_retries(self, mock_sleep, mock_post):
        import requests
        mock_post.side_effect = requests.Timeout("timed out")
        client = OllamaClient(max_retries=2)
        result = client.chat("llama3.2:3b", "test")
        assert result["success"] is False
        assert "timed out" in result["error"].lower() or result["error"] is not None

    @patch("src.llm_client.requests.post")
    @patch("src.llm_client.time.sleep")
    def test_chat_request_exception_retries(self, mock_sleep, mock_post):
        import requests
        mock_post.side_effect = requests.RequestException("connection refused")
        client = OllamaClient(max_retries=2)
        result = client.chat("llama3.2:3b", "test")
        assert result["success"] is False

    @patch("src.llm_client.requests.post")
    def test_chat_deepseek_extended_timeout(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"message": {"content": "e4"}}
        client = OllamaClient(timeout=180, max_retries=1)
        result = client.chat("deepseek-r1:7b", "test")
        assert result["success"] is True
        # deepseek-r1 should use timeout=600
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("timeout") == 600

    @patch("src.llm_client.requests.post")
    def test_chat_72b_with_options(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"message": {"content": "e4"}}
        client = OllamaClient(max_retries=1)
        result = client.chat("qwen2.5:72b", "test")
        assert result["success"] is True
        # Should have included options in payload
        payload = mock_post.call_args[1]["json"]
        assert "options" in payload


class TestBuildPrompt:
    def test_fen_only(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            prompt_format="fen_only",
        )
        assert "Current position (FEN):" in prompt
        assert "Moves played so far:" not in prompt
        assert "Think step by step" not in prompt

    def test_pgn_fen(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            pgn_moves="1. e4",
            prompt_format="pgn+fen",
        )
        assert "Moves played so far:" in prompt
        assert "1. e4" in prompt
        assert "Think step by step" not in prompt

    def test_cot_format(self):
        prompt = build_prompt(
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            pgn_moves="1. e4",
            prompt_format="cot",
        )
        assert "Think step by step" in prompt
        assert "Moves played so far:" in prompt


class TestBuildIsolatedPrompts:
    FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_build_eval_prompt_no_pgn(self):
        prompt = build_eval_prompt(self.FEN)
        assert self.FEN in prompt
        assert "centipawn" in prompt
        assert "Moves played so far:" not in prompt

    def test_build_eval_prompt_with_pgn(self):
        prompt = build_eval_prompt(self.FEN, pgn_moves="1. e4")
        assert "Moves played so far:" in prompt
        assert "1. e4" in prompt

    def test_build_explanation_prompt_no_pgn(self):
        prompt = build_explanation_prompt(self.FEN)
        assert self.FEN in prompt
        assert "Explanation:" in prompt
        assert "Moves played so far:" not in prompt

    def test_build_explanation_prompt_with_pgn(self):
        prompt = build_explanation_prompt(self.FEN, pgn_moves="1. e4")
        assert "Moves played so far:" in prompt

    def test_build_move_prompt_white(self):
        prompt = build_move_prompt(self.FEN)  # White to move
        assert "White" in prompt
        assert self.FEN in prompt

    def test_build_move_prompt_black(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        prompt = build_move_prompt(fen)
        assert "Black" in prompt


class TestParseEvalResponse:
    def test_plain_integer(self):
        result = parse_eval_response("45")
        assert result["eval"] == 45

    def test_negative_integer(self):
        result = parse_eval_response("-150")
        assert result["eval"] == -150

    def test_no_integer(self):
        result = parse_eval_response("equal position")
        assert result["eval"] is None
        assert "No integer found" in result["parse_errors"][0]

    def test_other_fields_are_none(self):
        result = parse_eval_response("100")
        assert result["move"] is None
        assert result["explanation"] is None
        assert result["side_claimed"] is None


class TestParseExplanationResponse:
    def test_white_explanation(self):
        result = parse_explanation_response("Explanation: White — White has a better position.")
        assert result["explanation"] is not None
        assert result["side_claimed"] == "White"
        assert result["eval"] is None
        assert result["move"] is None

    def test_equal_explanation(self):
        result = parse_explanation_response("Explanation: Equal — Both sides are equal.")
        assert result["side_claimed"] == "Equal"


class TestParseResponseEdgeCases:
    def test_empty_move_field(self):
        # "Move:  " (empty after label)
        response = "Eval: 50\nMove:  \nExplanation: Equal — test."
        result = parse_response(response)
        assert result["move"] is None
        assert "Empty move field" in result["parse_errors"]

    def test_unlabeled_starts_with_white(self):
        response = "Eval: 50\nMove: e4\nWhite is clearly better due to pawn structure."
        result = parse_response(response)
        assert result["side_claimed"] == "White"

    def test_unlabeled_starts_with_black(self):
        response = "Eval: -50\nMove: e5\nBlack is winning with active pieces."
        result = parse_response(response)
        assert result["side_claimed"] == "Black"

    def test_unlabeled_starts_with_equal(self):
        response = "Eval: 0\nMove: e4\nEqual — both sides have equal chances."
        result = parse_response(response)
        assert result["side_claimed"] == "Equal"

    def test_fallback_white_better_pattern(self):
        response = "Eval: 100\nMove: e4\nIn this position white stands better overall."
        result = parse_response(response)
        assert result["side_claimed"] == "White"

    def test_fallback_draw_pattern(self):
        response = "Eval: 0\nMove: e4\nThis game is heading for a draw position."
        result = parse_response(response)
        assert result["side_claimed"] == "Equal"

    def test_fallback_black_better_pattern(self):
        response = "Eval: -100\nMove: e5\nBlack is winning this endgame position."
        result = parse_response(response)
        assert result["side_claimed"] == "Black"

    def test_numbered_list_format(self):
        response = "1. 45\n2. Nf6\n3. White is better because of bishop pair."
        result = parse_response(response)
        assert result["eval"] == 45
        assert result["move"] == "Nf6"
        assert result["side_claimed"] == "White"

    def test_numbered_list_black(self):
        response = "1. -45\n2. e5\n3. Black is better in this endgame."
        result = parse_response(response)
        assert result["eval"] == -45
        assert result["side_claimed"] == "Black"

    def test_numbered_list_equal(self):
        response = "1. 0\n2. e4\n3. Equal — draw is likely here."
        result = parse_response(response)
        assert result["side_claimed"] == "Equal"

    def test_explanation_side_black_only(self):
        response = "Explanation: Black — Black has the advantage."
        result = parse_response(response)
        assert result["side_claimed"] == "Black"

    def test_explanation_neither_side(self):
        response = "Explanation: The position is complex."
        result = parse_response(response)
        assert result["side_claimed"] is None

    def test_markdown_bold_stripped(self):
        response = "**Eval:** 50\n**Move:** e4\n**Explanation:** Equal — test."
        result = parse_response(response)
        assert result["eval"] == 50
        assert result["move"] == "e4"

    def test_move_with_move_number_prefix(self):
        response = "Eval: 0\nMove: 21.Rxd4\nExplanation: Equal — test."
        result = parse_response(response)
        assert result["move"] == "Rxd4"

    def test_invalid_eval_string(self):
        response = "Eval: abc\nMove: e4\nExplanation: Equal — test."
        result = parse_response(response)
        assert result["eval"] is None
