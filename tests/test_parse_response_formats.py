"""
Tests for parse_response against real-world LLM output formats.

Bigger models (deepseek-r1, mixtral, qwen2.5, command-r, llama3.3 70b, etc.)
produce a wide variety of formatting. Any response that contains valid content
must NOT be incorrectly flagged as having parse errors.
"""

import pytest

from src.llm_client import parse_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_fully_parsed(result, expected_eval=None, expected_move=None,
                        expected_side=None):
    """Assert no parse errors and optionally check field values."""
    assert result["parse_errors"] == [], (
        f"Unexpected parse errors: {result['parse_errors']}\n"
        f"Full result: {result}"
    )
    if expected_eval is not None:
        assert result["eval"] == expected_eval
    if expected_move is not None:
        assert result["move"] == expected_move
    if expected_side is not None:
        assert result["side_claimed"] == expected_side


# ---------------------------------------------------------------------------
# 1. Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_perfect_format(self):
        r = parse_response(
            "Eval: 45\nMove: Nf6\n"
            "Explanation: Equal — Both sides are developed normally."
        )
        assert_fully_parsed(r, 45, "Nf6", "Equal")

    def test_negative_eval(self):
        r = parse_response(
            "Eval: -200\nMove: Rxe5\n"
            "Explanation: Black is better — rook dominates the open file."
        )
        assert_fully_parsed(r, -200, "Rxe5", "Black")

    def test_zero_eval(self):
        r = parse_response(
            "Eval: 0\nMove: d4\n"
            "Explanation: Equal — The position is balanced."
        )
        assert_fully_parsed(r, 0, "d4", "Equal")


# ---------------------------------------------------------------------------
# 2. deepseek-r1 style — <think> reasoning block before answer
# ---------------------------------------------------------------------------

class TestDeepseekR1Style:
    def test_think_tag_before_answer(self):
        r = parse_response(
            "<think>\n"
            "Let me analyse this position step by step.\n"
            "White has a bishop pair and the center is contested.\n"
            "The best move is probably Nf6 to put pressure.\n"
            "</think>\n\n"
            "Eval: 30\n"
            "Move: Nf6\n"
            "Explanation: White is better — active bishop pair."
        )
        assert_fully_parsed(r, 30, "Nf6", "White")

    def test_think_tag_with_eval_inside_but_real_answer_after(self):
        """Parser must pick the answer section, not stray numbers inside <think>."""
        r = parse_response(
            "<think>\n"
            "Stockfish would give this maybe +50 or +60 centipawns.\n"
            "I think it's around +40.\n"
            "Best move is Bxf7+ for the attack.\n"
            "</think>\n\n"
            "Eval: 40\n"
            "Move: Bxf7+\n"
            "Explanation: White is better — sacrificial attack on f7 wins material."
        )
        assert_fully_parsed(r, 40, "Bxf7+", "White")

    def test_unclosed_think_tag(self):
        """Some models forget to close the tag."""
        r = parse_response(
            "<think>\n"
            "Thinking through the position...\n"
            "I will evaluate as +25 and play O-O.\n"
            "\n"
            "Eval: 25\n"
            "Move: O-O\n"
            "Explanation: White is better — king safety secured."
        )
        assert_fully_parsed(r, 25, "O-O", "White")


# ---------------------------------------------------------------------------
# 3. Markdown formatting (bold labels, code spans)
# ---------------------------------------------------------------------------

class TestMarkdownFormats:
    def test_bold_labels(self):
        r = parse_response(
            "**Eval:** 60\n"
            "**Move:** Qxd5\n"
            "**Explanation:** White is better — queen centralization wins a pawn."
        )
        assert_fully_parsed(r, 60, "Qxd5", "White")

    def test_bold_values(self):
        r = parse_response(
            "Eval: **80**\n"
            "Move: **Rxc6**\n"
            "Explanation: **White** is better — rook takes the hanging knight."
        )
        assert_fully_parsed(r, 80, "Rxc6", "White")

    def test_inline_code_move(self):
        r = parse_response(
            "Eval: 15\n"
            "Move: `e5`\n"
            "Explanation: Equal — pawn advances to contest the center."
        )
        assert_fully_parsed(r, 15, "e5", "Equal")

    def test_response_in_code_block(self):
        r = parse_response(
            "Here is my answer:\n\n"
            "```\n"
            "Eval: 55\n"
            "Move: Nd5\n"
            "Explanation: White is better — knight outpost on d5.\n"
            "```"
        )
        assert_fully_parsed(r, 55, "Nd5", "White")


# ---------------------------------------------------------------------------
# 4. Numbered list prefix (models that answer in numbered-list style)
# ---------------------------------------------------------------------------

class TestNumberedListFormats:
    def test_numbered_list(self):
        r = parse_response(
            "1. Eval: 20\n"
            "2. Move: Bf4\n"
            "3. Explanation: Equal — minor development advantage for White."
        )
        assert_fully_parsed(r, 20, "Bf4", "Equal")

    def test_numbered_with_dot_and_space(self):
        r = parse_response(
            "1. Eval: -35\n"
            "2. Move: Nc4\n"
            "3. Explanation: Black is better — better piece coordination."
        )
        assert_fully_parsed(r, -35, "Nc4", "Black")

    def test_unlabeled_numbered_answers(self):
        """llama3.2:3b cot format: just number + answer with no field label."""
        r = parse_response(
            "1. 25\n"
            "2. d4\n"
            "3. White is better — space advantage in the center."
        )
        assert_fully_parsed(r, 25, "d4", "White")

    def test_unlabeled_numbered_negative_eval(self):
        r = parse_response(
            "1. -50\n"
            "2. Nf6\n"
            "3. Black is better — knight outpost on f6."
        )
        assert_fully_parsed(r, -50, "Nf6", "Black")


# ---------------------------------------------------------------------------
# 5. Verbose preamble — model reasons before giving the formatted answer
# ---------------------------------------------------------------------------

class TestVerbosePreamble:
    def test_paragraph_before_answer(self):
        r = parse_response(
            "This is a complex middlegame position with both sides having "
            "castled. White has a slight space advantage but Black's pieces "
            "are well-coordinated.\n\n"
            "Eval: 25\n"
            "Move: Rfe1\n"
            "Explanation: White is better — rook centralisation increases pressure."
        )
        assert_fully_parsed(r, 25, "Rfe1", "White")

    def test_step_by_step_cot_then_answer(self):
        r = parse_response(
            "Key features:\n"
            "- White has open d-file\n"
            "- Black's king is slightly exposed\n"
            "- Material is equal\n\n"
            "Space: White controls more central squares.\n"
            "Active pieces: White rooks are more active.\n"
            "Forcing moves: Rd8+ wins material.\n\n"
            "Eval: 120\n"
            "Move: Rd8+\n"
            "Explanation: White is better — back-rank tactic wins the exchange."
        )
        assert_fully_parsed(r, 120, "Rd8+", "White")

    def test_conversational_answer_then_format(self):
        r = parse_response(
            "Let me evaluate this position.\n\n"
            "Based on my analysis, White has a clear advantage due to the "
            "passed pawn on d6 and better king safety.\n\n"
            "Eval: 90\n"
            "Move: d7\n"
            "Explanation: White is better — passed pawn one step from queening."
        )
        assert_fully_parsed(r, 90, "d7", "White")


# ---------------------------------------------------------------------------
# 6. Alternative label spellings and capitalisation
# ---------------------------------------------------------------------------

class TestLabelVariants:
    def test_lowercase_labels(self):
        r = parse_response(
            "eval: 10\nmove: Kf1\nexplanation: Equal — king sidesteps the check."
        )
        assert_fully_parsed(r, 10, "Kf1", "Equal")

    def test_uppercase_labels(self):
        r = parse_response(
            "EVAL: 50\nMOVE: Bg5\nEXPLANATION: White is better — pin wins material."
        )
        assert_fully_parsed(r, 50, "Bg5", "White")

    def test_mixed_case_labels(self):
        result = parse_response(
            "Eval: 75\nMove: Nxe5\nExplanation: White Is Better — pawn won."
        )
        assert result["eval"] == 75
        assert result["move"] == "Nxe5"
        assert result["parse_errors"] == []


# ---------------------------------------------------------------------------
# 7. Move notation variants
# ---------------------------------------------------------------------------

class TestMoveNotationVariants:
    def test_castles_kingside(self):
        r = parse_response(
            "Eval: 5\nMove: O-O\nExplanation: Equal — both sides have castled."
        )
        assert_fully_parsed(r, 5, "O-O", "Equal")

    def test_castles_queenside(self):
        r = parse_response(
            "Eval: 5\nMove: O-O-O\nExplanation: White is better — queen-side attack."
        )
        assert_fully_parsed(r, 5, "O-O-O", "White")

    def test_move_with_check(self):
        r = parse_response(
            "Eval: 300\nMove: Qxf7+\nExplanation: White is better — fork wins queen."
        )
        assert_fully_parsed(r, 300, "Qxf7+", "White")

    def test_move_with_checkmate(self):
        r = parse_response(
            "Eval: 10000\nMove: Qh7#\nExplanation: White is better — checkmate."
        )
        assert_fully_parsed(r, 10000, "Qh7#", "White")

    def test_promotion_move(self):
        r = parse_response(
            "Eval: 900\nMove: e8=Q\nExplanation: White is better — queen promotion."
        )
        assert_fully_parsed(r, 900, "e8=Q", "White")

    def test_move_with_trailing_punctuation(self):
        r = parse_response(
            "Eval: 50\nMove: Nf6.\nExplanation: Equal — solid knight development."
        )
        assert_fully_parsed(r, 50, "Nf6", "Equal")

    def test_move_with_trailing_comma(self):
        r = parse_response(
            "Eval: 50\nMove: d4,\nExplanation: Equal — central control."
        )
        assert_fully_parsed(r, 50, "d4", "Equal")


# ---------------------------------------------------------------------------
# 8. Eval value variants
# ---------------------------------------------------------------------------

class TestEvalVariants:
    def test_eval_with_plus_sign(self):
        r = parse_response(
            "Eval: +80\nMove: Bc4\nExplanation: White is better — active bishop."
        )
        assert_fully_parsed(r, 80, "Bc4", "White")

    def test_eval_with_word_centipawns(self):
        r = parse_response(
            "Eval: 120 centipawns\nMove: Rxd4\n"
            "Explanation: White is better — material won."
        )
        assert_fully_parsed(r, 120, "Rxd4", "White")

    def test_eval_with_cp_suffix(self):
        r = parse_response(
            "Eval: 60cp\nMove: Bb5\nExplanation: White is better — pin on knight."
        )
        assert_fully_parsed(r, 60, "Bb5", "White")

    def test_large_mate_eval(self):
        r = parse_response(
            "Eval: 10000\nMove: Qg7#\nExplanation: White is better — forced mate."
        )
        assert_fully_parsed(r, 10000, "Qg7#", "White")


# ---------------------------------------------------------------------------
# 9. Side-claimed detection edge cases
# ---------------------------------------------------------------------------

class TestSideClaimedDetection:
    def test_side_in_middle_of_explanation(self):
        r = parse_response(
            "Eval: 40\nMove: Nc3\n"
            "Explanation: The position favours White — better pawn structure."
        )
        # "White" appears in explanation, so side_claimed should be White
        assert r["parse_errors"] == []
        assert r["side_claimed"] == "White"

    def test_equal_with_draw_keyword(self):
        r = parse_response(
            "Eval: 0\nMove: h3\n"
            "Explanation: Equal — the position is a theoretical draw."
        )
        assert_fully_parsed(r, 0, "h3", "Equal")

    def test_black_better_various_phrasings(self):
        for phrase in [
            "Black is better — active rooks.",
            "Black stands better — weak white pawns.",
            "Black is winning — material advantage.",
        ]:
            r = parse_response(f"Eval: -100\nMove: Re8\nExplanation: {phrase}")
            assert r["side_claimed"] == "Black", f"Failed for: {phrase}"

    def test_white_better_various_phrasings(self):
        for phrase in [
            "White is better — space advantage.",
            "White stands better — active pieces.",
            "White is winning — passed pawn.",
        ]:
            r = parse_response(f"Eval: 100\nMove: d5\nExplanation: {phrase}")
            assert r["side_claimed"] == "White", f"Failed for: {phrase}"


# ---------------------------------------------------------------------------
# 10. Bigger model specific patterns
# ---------------------------------------------------------------------------

class TestBiggerModelPatterns:
    """Patterns observed in 14B+ models like qwen2.5:32b, llama3.3:70b,
    command-r:35b, mixtral:8x7b, deepseek-r1:14b etc."""

    def test_qwen_style_with_analysis_header(self):
        r = parse_response(
            "### Analysis\n\n"
            "The position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 is the Italian Game. "
            "White has a slight advantage due to better development.\n\n"
            "Eval: 35\n"
            "Move: Nf6\n"
            "Explanation: Equal — Black successfully counters with the Two Knights."
        )
        assert_fully_parsed(r, 35, "Nf6", "Equal")

    def test_llama3_70b_verbose_then_format(self):
        r = parse_response(
            "I'll analyze this chess position:\n\n"
            "**Position Assessment:**\n"
            "- White has a strong pawn center\n"
            "- Black's knight on c6 is well-placed\n"
            "- The f7 square looks weak\n\n"
            "**My Evaluation:**\n\n"
            "Eval: 45\n"
            "Move: Ng5\n"
            "Explanation: White is better — knight targets the weak f7 pawn."
        )
        assert_fully_parsed(r, 45, "Ng5", "White")

    def test_mixtral_numbered_steps(self):
        r = parse_response(
            "1. Evaluation of position:\n"
            "   The position is slightly in White's favour.\n\n"
            "2. Best move analysis:\n"
            "   White should play Nd5 to exploit the pin.\n\n"
            "Eval: 55\n"
            "Move: Nd5\n"
            "Explanation: White is better — knight fork threatens queen and rook."
        )
        assert_fully_parsed(r, 55, "Nd5", "White")

    def test_command_r_with_context_summary(self):
        r = parse_response(
            "Based on the FEN provided and the moves played so far, "
            "this is a Ruy Lopez position where White has a slight edge.\n\n"
            "Eval: 30\n"
            "Move: d4\n"
            "Explanation: White is better — central pawn break gains space."
        )
        assert_fully_parsed(r, 30, "d4", "White")

    def test_deepseek_r1_14b_long_think(self):
        r = parse_response(
            "<think>\n"
            "Okay, let me carefully analyze this position. The FEN shows...\n"
            "White has rooks on d1 and e1, which are very active.\n"
            "The key tactical motif is the pin along the d-file.\n"
            "Best move must be Rxd7 to win material.\n"
            "Centipawn eval: around +150 in White's favour.\n"
            "</think>\n\n"
            "Eval: 150\n"
            "Move: Rxd7\n"
            "Explanation: White is better — rook captures the pinned piece."
        )
        assert_fully_parsed(r, 150, "Rxd7", "White")

    def test_yi_34b_with_intro_sentence(self):
        r = parse_response(
            "Here's my analysis of the given chess position:\n\n"
            "Eval: -80\n"
            "Move: Qb6\n"
            "Explanation: Black is better — queen threatens back rank and f2."
        )
        assert_fully_parsed(r, -80, "Qb6", "Black")

    def test_codellama_34b_code_style_response(self):
        r = parse_response(
            "```\n"
            "Eval: 20\n"
            "Move: a4\n"
            "Explanation: Equal — pawn advance prepares queenside expansion.\n"
            "```"
        )
        assert_fully_parsed(r, 20, "a4", "Equal")

    def test_qwen25_72b_with_disclaimer(self):
        r = parse_response(
            "Note: I am an AI and chess analysis may not be perfect.\n\n"
            "Eval: 65\n"
            "Move: Bxh7+\n"
            "Explanation: White is better — Greek gift sacrifice gives mating attack."
        )
        assert_fully_parsed(r, 65, "Bxh7+", "White")

    def test_response_with_extra_blank_lines(self):
        r = parse_response(
            "\n\nEval: 10\n\n\nMove: Kg1\n\n\n"
            "Explanation: Equal — king moves to safety.\n\n"
        )
        assert_fully_parsed(r, 10, "Kg1", "Equal")

    def test_explanation_with_em_dash(self):
        r = parse_response(
            "Eval: 50\nMove: Nxf6\n"
            "Explanation: White is better\u2014the knight captures the defender."
        )
        assert r["parse_errors"] == []
        assert r["eval"] == 50
        assert r["move"] == "Nxf6"


# ---------------------------------------------------------------------------
# 11. All 19 models — explicit per-model coverage
# ---------------------------------------------------------------------------

class TestAllModelsExplicit:
    """One representative test per model using that model's typical output style."""

    def test_llama3_2_3b(self):
        """Small llama — tends to give clean minimal responses."""
        r = parse_response(
            "Eval: 20\n"
            "Move: e4\n"
            "Explanation: White is better — strong central pawn."
        )
        assert_fully_parsed(r, 20, "e4", "White")

    def test_gemma3_4b(self):
        """Gemma3 4b — wraps numbered labels in bold: **1. Eval:** 0"""
        r = parse_response(
            "Okay, let's analyze the chess position.\n\n"
            "**1. Eval:** 0\n\n"
            "**2. Move:** Nf3\n\n"
            "**3. Explanation:** Black — Black has better development."
        )
        assert_fully_parsed(r, 0, "Nf3", "Black")

    def test_gemma3_12b(self):
        """Gemma3 12b — same bold-numbered pattern as 4b."""
        r = parse_response(
            "## Chess Position Analysis\n\n"
            "**1. Eval:** -60\n"
            "**2. Move:** Qd4\n"
            "**3. Explanation:** Black — queen centralises with tempo."
        )
        assert_fully_parsed(r, -60, "Qd4", "Black")

    def test_qwen2_5_7b(self):
        """Qwen2.5 7b — may use bold labels."""
        r = parse_response(
            "**Eval:** 15\n"
            "**Move:** Nf3\n"
            "**Explanation:** Equal — solid development for both sides."
        )
        assert_fully_parsed(r, 15, "Nf3", "Equal")

    def test_mistral_7b(self):
        """Mistral 7b — typically follows the format but sometimes adds context."""
        r = parse_response(
            "Based on the position:\n"
            "Eval: 40\n"
            "Move: Bb5\n"
            "Explanation: White is better — Ruy Lopez pin on the knight."
        )
        assert_fully_parsed(r, 40, "Bb5", "White")

    def test_deepseek_r1_7b(self):
        """deepseek-r1 7b — uses <think> block like the 14b variant."""
        r = parse_response(
            "<think>\n"
            "I need to evaluate the position. White seems slightly better.\n"
            "Best move is probably Nc3 to develop.\n"
            "</think>\n\n"
            "Eval: 25\n"
            "Move: Nc3\n"
            "Explanation: White is better — active knight development."
        )
        assert_fully_parsed(r, 25, "Nc3", "White")

    def test_wizardlm2_7b(self):
        """WizardLM2 7b — PGN move-number prefix and 'Who stands better?' explanation."""
        r = parse_response(
            "1. Eval: -0.2 to -0.1\n\n"
            "2. Move: 1...Nc5\n\n"
            "3. Who stands better? — Black is slightly better due to active pieces."
        )
        assert r["parse_errors"] == []
        assert r["move"] == "Nc5"
        assert r["side_claimed"] == "Black"

    def test_llama3_1_8b(self):
        """llama3.1 8b — similar to 3.2 but occasionally wraps move in backticks."""
        r = parse_response(
            "Eval: 55\n"
            "Move: `Rxe5`\n"
            "Explanation: White is better — rook wins the central pawn."
        )
        assert_fully_parsed(r, 55, "Rxe5", "White")

    def test_gemma3_12b(self):
        """Gemma3 12b — may produce markdown headers then the format."""
        r = parse_response(
            "## Chess Position Analysis\n\n"
            "Eval: -60\n"
            "Move: Qd4\n"
            "Explanation: Black is better — queen centralises with tempo."
        )
        assert_fully_parsed(r, -60, "Qd4", "Black")

    def test_qwen2_5_14b(self):
        """Qwen2.5 14b — similar to 7b, bold labels common."""
        r = parse_response(
            "**Eval:** 70\n"
            "**Move:** Bxh7+\n"
            "**Explanation:** White is better — Greek gift sacrifice opens the king."
        )
        assert_fully_parsed(r, 70, "Bxh7+", "White")

    def test_phi4_14b(self):
        """Phi4 14b — tends to reason briefly then give structured output."""
        r = parse_response(
            "After reviewing the position:\n\n"
            "Eval: 35\n"
            "Move: Nd5\n"
            "Explanation: White is better — knight outpost dominates the centre."
        )
        assert_fully_parsed(r, 35, "Nd5", "White")

    def test_solar_10_7b(self):
        """Solar 10.7b — instruction-following model, typically clean format."""
        r = parse_response(
            "Eval: -45\n"
            "Move: Rb8\n"
            "Explanation: Black is better — rook enters the seventh rank."
        )
        assert_fully_parsed(r, -45, "Rb8", "Black")

    def test_qwen2_5_32b(self):
        """Qwen2.5 32b — larger model, may add analysis header and bold."""
        r = parse_response(
            "### Position Evaluation\n\n"
            "**Eval:** 90\n"
            "**Move:** d6\n"
            "**Explanation:** White is better — passed pawn on d6 is decisive."
        )
        assert_fully_parsed(r, 90, "d6", "White")

    def test_llama3_3_70b(self):
        """llama3.3 70b — large model, verbose with markdown bullets then format."""
        r = parse_response(
            "**Position Assessment:**\n"
            "- White has a space advantage\n"
            "- Black's pieces are passive\n"
            "- The a-file is half-open\n\n"
            "Eval: 80\n"
            "Move: Ra1\n"
            "Explanation: White is better — rook dominates the open a-file."
        )
        assert_fully_parsed(r, 80, "Ra1", "White")

    def test_qwen2_5_72b(self):
        """Qwen2.5 72b — disclaimer + bold labels, large context model."""
        r = parse_response(
            "Please note: this is AI analysis and may not be perfect.\n\n"
            "**Eval:** 110\n"
            "**Move:** Rxd7\n"
            "**Explanation:** White is better — rook captures the pinned piece."
        )
        assert_fully_parsed(r, 110, "Rxd7", "White")
