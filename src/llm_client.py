"""Ollama LLM client for chess position analysis."""

import logging
import re
import time
from typing import Any

import requests

logger = logging.getLogger("chess_llm_bench")


class OllamaClient:
    """HTTP client for Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
        max_retries: int = 3,
    ):
        """Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def is_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List available models on the Ollama server.

        Returns:
            List of model tags
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model: Model tag to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=3600,  # Long timeout for model downloads
                stream=True,
            )
            for line in response.iter_lines():
                if line:
                    logger.debug(line.decode())
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False

    @staticmethod
    def _get_options(model: str) -> dict[str, Any]:
        """Build Ollama options for a model.

        num_predict caps output tokens to prevent runaway generation filling the
        KV cache and crashing the Ollama runner. Chess responses are short
        (eval + move + explanation), so 400 tokens is generous.
        num_ctx is capped to reduce KV cache VRAM pressure and allow more
        parallel slots.
        """
        model_lower = model.lower()
        if "72b" in model_lower:
            return {"num_gpu": 26, "num_ctx": 512, "num_predict": 400}
        if "70b" in model_lower:
            return {"num_gpu": 26, "num_ctx": 512, "num_predict": 400}
        if "deepseek-r1" in model_lower:
            # CoT disabled via think:false — no long reasoning chains, normal budget.
            return {"num_ctx": 1024, "num_predict": 400}
        if "gemma4" in model_lower:
            # Gemma4 has built-in thinking/CoT disabled via think:false payload flag.
            # Without disabling it, thinking tokens consume num_predict budget and
            # responses come back blank. Options are kept light since no CoT overhead.
            return {"num_ctx": 1024, "num_predict": 400}
        # All other models: cap context and output to prevent KV cache OOM
        return {"num_ctx": 1024, "num_predict": 400}

    def chat(self, model: str, prompt: str, system_prompt: str | None = None, temperature: float | None = None) -> dict[str, Any]:
        """Send a chat request to Ollama.

        Args:
            model: Model tag to use
            prompt: User prompt
            system_prompt: Optional system prompt prepended as a system message

        Returns:
            Dictionary with 'response' (raw text), 'inference_ms' (duration),
            and 'success' (boolean)
        """
        start_time = time.time()
        last_error = None
        options = self._get_options(model)
        # Large offloaded models (70B+) need more time
        if options.get("num_gpu") is not None:
            timeout = 600
        else:
            timeout = self.timeout

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                merged_options = dict(options)
                if temperature is not None:
                    merged_options["temperature"] = temperature
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                }
                if "gemma4" in model.lower() or "deepseek-r1" in model.lower():
                    payload["think"] = False
                if merged_options:
                    payload["options"] = merged_options
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()

                inference_ms = int((time.time() - start_time) * 1000)

                return {
                    "response": data.get("message", {}).get("content", ""),
                    "inference_ms": inference_ms,
                    "success": True,
                    "model": model,
                }

            except requests.Timeout:
                last_error = "Request timed out"
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} timed out")
            except requests.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        inference_ms = int((time.time() - start_time) * 1000)
        return {
            "response": "",
            "inference_ms": inference_ms,
            "success": False,
            "error": last_error,
            "model": model,
        }


MOVE_SYSTEM_PROMPT = (
    "You are a chess engine. When given a FEN position, output exactly one legal move "
    "in Standard Algebraic Notation (SAN).\n\n"
    "The prompt will tell you which side is to move. Only move that side's pieces.\n\n"
    "A legal move must satisfy ALL of the following:\n"
    "- Only move pieces belonging to the side that is to move.\n"
    "- The piece you are moving must actually exist on the square you name.\n"
    "- The piece must be able to reach the destination square by its movement rules "
    "(e.g. a bishop moves diagonally, a knight in an L-shape, a pawn one square forward "
    "or diagonally to capture).\n"
    "- The move must not leave your own king in check.\n"
    "- Castling (O-O or O-O-O) is only legal if the king and rook have not moved, "
    "there are no pieces between them, and the king does not pass through check.\n"
    "- En passant is only legal if the FEN en-passant square is set.\n\n"
    "Output the move only — no explanation, no punctuation, no move numbers.\n"
    "Examples of correct output: e4, Nf3, O-O, Bxc6, exd5, Qh5+"
)

EVAL_SYSTEM_PROMPT = (
    "You are a chess position evaluator. When given a chess position, output a single "
    "integer representing the centipawn evaluation from White's perspective.\n\n"
    "Positive = White is better. Negative = Black is better. 0 = equal.\n"
    "Typical range is -2000 to +2000.\n\n"
    "Output only the integer — no text, no explanation, no sign if positive."
)

EXPLANATION_SYSTEM_PROMPT = (
    "You are a chess analyst. When given a chess position, identify who stands better "
    "and give a one-sentence explanation of the key reason.\n\n"
    "Respond using this exact format:\n"
    "Explanation: <White / Black / Equal> — <one sentence reason>\n\n"
    "Do not include anything else in your response."
)


def build_eval_prompt(fen: str, pgn_moves: str | None = None) -> str:
    """Build an isolated prompt asking only for the centipawn evaluation."""
    moves_section = f"\nMoves played so far:\n{pgn_moves}\n" if pgn_moves else ""
    return (
        f"You are analysing a chess position.{moves_section}\n"
        f"Current position (FEN):\n{fen}\n\n"
        "What is the centipawn evaluation of this position from White's perspective?\n"
        "Positive = White is better. Negative = Black is better. 0 = equal.\n"
        "Respond with a single integer only."
    )


def build_explanation_prompt(fen: str, pgn_moves: str | None = None) -> str:
    """Build an isolated prompt asking only for a positional explanation."""
    moves_section = f"\nMoves played so far:\n{pgn_moves}\n" if pgn_moves else ""
    return (
        f"You are analysing a chess position.{moves_section}\n"
        f"Current position (FEN):\n{fen}\n\n"
        "Who stands better in this position — White, Black, or is it equal?\n"
        "Give a one-sentence explanation of the key reason.\n\n"
        "Respond using this exact format:\n"
        "Explanation: <White / Black / Equal> — <one sentence reason>"
    )


def extract_move_from_text(fen: str, text: str) -> str | None:
    """Try to find a legal move anywhere in the model's response text.

    Attempts, in order:
    1. Each whitespace token cleaned of punctuation/backticks, tried as SAN.
    2. Each token tried as UCI (e.g. e2e4, g1f3, e7e8q) and converted to SAN.

    Returns the first legal SAN move found, or None.
    """
    import chess as _chess

    try:
        board = _chess.Board(fen)
    except Exception:
        return None

    UCI_RE = re.compile(r'^[a-h][1-8][a-h][1-8][qrbnQRBN]?$')

    candidates = text.strip().split()
    for raw in candidates:
        token = raw.strip(".,;:!?()[]{}'\"`*#").rstrip("+#")
        token = re.sub(r"^\d+\.+", "", token)  # strip move numbers like "1." / "21..."
        if not token:
            continue

        # Try as SAN
        try:
            move = board.parse_san(token)
            if move in board.legal_moves:
                return board.san(move)
        except Exception:
            pass

        # Try as UCI
        uci_token = token.lower()
        if UCI_RE.match(uci_token):
            try:
                move = _chess.Move.from_uci(uci_token)
                if move in board.legal_moves:
                    return board.san(move)
            except Exception:
                pass

    return None


def build_move_prompt(fen: str) -> str:
    """Build an isolated prompt asking only for the best move.

    Used alongside the main three-task prompt so T2 gets its own focused call
    with a chess-engine system prompt that explains legality rules.
    """
    import chess as _chess
    board = _chess.Board(fen)
    side = "White" if board.turn == _chess.WHITE else "Black"
    legal_moves = ", ".join(board.san(m) for m in board.legal_moves)
    return (
        f"Position (FEN): {fen}\n"
        f"It is {side} to move.\n\n"
        f"Legal moves: {legal_moves}\n\n"
        f"What is the best move for {side}? You MUST pick one of the listed legal moves. Respond with the SAN move only."
    )


def build_prompt(
    fen: str,
    pgn_moves: str | None = None,
    prompt_format: str = "pgn+fen",
) -> str:
    """Build the standard three-task prompt for a chess position.

    Args:
        fen: FEN string of the position
        pgn_moves: PGN move history (optional)
        prompt_format: One of "fen_only", "pgn+fen", or "cot"

    Returns:
        Formatted prompt string
    """
    # Base prompt parts
    intro = "You are analysing a chess position."

    moves_section = ""
    if prompt_format in ("pgn+fen", "cot") and pgn_moves:
        moves_section = f"\nMoves played so far:\n{pgn_moves}\n"

    fen_section = f"\nCurrent position (FEN):\n{fen}"

    questions = """
Answer all three questions below.

1. What is the centipawn evaluation of this position from White's perspective?
   A positive number means White is better. A negative number means Black is better.
   0 means equal. Give only a number, no explanation.

2. What is the best move for the side to play? Give only the move in SAN notation.

3. Who stands better in this position — White, Black, or is it equal?
   Give a one-sentence explanation of the key reason."""

    cot_section = ""
    if prompt_format == "cot":
        cot_section = """

Think step by step before answering:
- What are the key features of this position?
- Who controls more space? Who has more active pieces?
- What is the most forcing move available?"""

    response_format = """

Respond using this exact format:
Eval: <integer>
Move: <SAN move>
Explanation: <White / Black / Equal> — <one sentence reason>"""

    return intro + moves_section + fen_section + questions + cot_section + response_format


def parse_eval_response(response_text: str) -> dict[str, Any]:
    """Parse an eval-only response — extract the first integer found."""
    result: dict[str, Any] = {
        "eval": None,
        "move": None,
        "explanation": None,
        "side_claimed": None,
        "parse_errors": [],
    }
    match = re.search(r"-?\d+", response_text.strip())
    if match:
        result["eval"] = int(match.group())
    else:
        result["parse_errors"].append("No integer found in eval-only response")
    return result


def parse_explanation_response(response_text: str) -> dict[str, Any]:
    """Parse an explanation-only response — extract explanation and side_claimed."""
    # Reuse the full parser and discard eval/move fields
    full = parse_response(response_text)
    return {
        "eval": None,
        "move": None,
        "explanation": full["explanation"],
        "side_claimed": full["side_claimed"],
        "parse_errors": full["parse_errors"],
    }


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM response to extract Eval, Move, and Explanation.

    Args:
        response_text: Raw response text from the LLM

    Returns:
        Dictionary with 'eval', 'move', 'explanation', 'side_claimed',
        and 'parse_errors' list
    """
    result = {
        "eval": None,
        "move": None,
        "explanation": None,
        "side_claimed": None,
        "parse_errors": [],
    }

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        # Strip markdown bold markers first so "**1. Eval:**" becomes "1. Eval:"
        line = line.replace("**", "")
        # Capture question number before stripping, to handle unlabeled answers
        # e.g. "2. d4" (no "Move:" label) — number tells us which field it is
        num_match = re.match(r"^(\d+)\.\s*(.+)$", line)
        question_num = int(num_match.group(1)) if num_match else None
        # Strip leading numbered-list prefix e.g. "1. ", "2. ", "3. "
        line = re.sub(r"^\d+\.\s*", "", line)

        # Parse Eval
        if line.lower().startswith("eval:"):
            eval_str = line[5:].strip()
            # Extract integer from the string
            match = re.search(r"-?\d+", eval_str)
            if match:
                try:
                    result["eval"] = int(match.group())
                except ValueError:
                    result["parse_errors"].append(f"Invalid eval value: {eval_str}")
            else:
                result["parse_errors"].append(f"Could not parse eval: {eval_str}")

        # Parse Move
        elif line.lower().startswith("move:"):
            move_str = line[5:].strip()
            # Clean up common formatting issues
            move_str = move_str.split()[0] if move_str.split() else ""
            move_str = move_str.rstrip(".,;")
            # Strip inline code backticks e.g. `Nf6`
            move_str = move_str.strip("`")
            # Strip PGN move-number prefixes e.g. "1...Nc5" -> "Nc5", "21.e4" -> "e4"
            move_str = re.sub(r"^\d+\.+", "", move_str)
            if move_str:
                result["move"] = move_str
            else:
                result["parse_errors"].append("Empty move field")

        # Parse Explanation (with or without "Explanation:" label)
        elif line.lower().startswith("explanation:"):
            explanation = line[12:].strip()
            result["explanation"] = explanation
            explanation_lower = explanation.lower()
            if explanation_lower.startswith("white"):
                result["side_claimed"] = "White"
            elif explanation_lower.startswith("black"):
                result["side_claimed"] = "Black"
            elif explanation_lower.startswith("equal"):
                result["side_claimed"] = "Equal"
            else:
                if "white" in explanation_lower and "black" not in explanation_lower:
                    result["side_claimed"] = "White"
                elif "black" in explanation_lower and "white" not in explanation_lower:
                    result["side_claimed"] = "Black"
                elif "equal" in explanation_lower or "draw" in explanation_lower:
                    result["side_claimed"] = "Equal"
        elif result["explanation"] is None and re.match(r"^(white|black|equal)\b", line, re.IGNORECASE):
            explanation = line
            result["explanation"] = explanation
            explanation_lower = explanation.lower()
            if explanation_lower.startswith("white"):
                result["side_claimed"] = "White"
            elif explanation_lower.startswith("black"):
                result["side_claimed"] = "Black"
            elif explanation_lower.startswith("equal"):
                result["side_claimed"] = "Equal"
        elif result["explanation"] is None and re.search(r"(white|black|equal).{0,30}(better|winning|advantage|stands)", line, re.IGNORECASE):
            # Catch lines like "Who stands better? — Black is slightly better..."
            explanation = re.sub(r"^.*?[—–-]\s*", "", line).strip() or line
            result["explanation"] = explanation
            explanation_lower = explanation.lower()
            if "white" in explanation_lower and "black" not in explanation_lower:
                result["side_claimed"] = "White"
            elif "black" in explanation_lower and "white" not in explanation_lower:
                result["side_claimed"] = "Black"
            elif "equal" in explanation_lower or "draw" in explanation_lower:
                result["side_claimed"] = "Equal"

        # Fallback: plain unlabeled explanation after eval and move are known.
        # Guard on non-empty line: blank separator lines between numbered items
        # must not set explanation to '' — that would block subsequent handlers.
        elif result["explanation"] is None and result["eval"] is not None and result["move"] is not None and line:
            result["explanation"] = line
            explanation_lower = line.lower()
            if "white" in explanation_lower and "black" not in explanation_lower:
                result["side_claimed"] = "White"
            elif "black" in explanation_lower and "white" not in explanation_lower:
                result["side_claimed"] = "Black"
            elif "equal" in explanation_lower or "draw" in explanation_lower:
                result["side_claimed"] = "Equal"
        # Fallback: unlabeled numbered answer — infer field from question number
        # Handles formats like "1. 25\n2. d4\n3. White is better because..."
        elif question_num == 1 and result["eval"] is None:
            match = re.search(r"-?\d+", line)
            if match:
                result["eval"] = int(match.group())
        elif question_num == 2 and result["move"] is None:
            move_str = line.split()[0] if line.split() else ""
            move_str = move_str.rstrip(".,;").strip("`")
            move_str = re.sub(r"^\d+\.+", "", move_str)
            if move_str:
                result["move"] = move_str
        elif question_num == 3 and result["explanation"] is None:
            result["explanation"] = line
            explanation_lower = line.lower()
            if "white" in explanation_lower and "black" not in explanation_lower:
                result["side_claimed"] = "White"
            elif "black" in explanation_lower and "white" not in explanation_lower:
                result["side_claimed"] = "Black"
            elif "equal" in explanation_lower or "draw" in explanation_lower:
                result["side_claimed"] = "Equal"

    # Check for missing fields
    if result["eval"] is None:
        result["parse_errors"].append("Missing Eval field")
    if result["move"] is None:
        result["parse_errors"].append("Missing Move field")
    if result["explanation"] is None:
        result["parse_errors"].append("Missing Explanation field")

    return result
