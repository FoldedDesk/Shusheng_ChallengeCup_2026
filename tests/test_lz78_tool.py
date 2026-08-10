from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


STANDARD_PROBLEM = r"""Consider the message

aabababcabcde.

Describe the decomposition into phrases that will be used by Lempel-Ziv, and give the encoded string obtained using Lempel-Ziv. When encoding a letter, use the mapping

\[
a\rightarrow000,\quad b\rightarrow001,\quad c\rightarrow010,\quad
d\rightarrow011,\quad e\rightarrow100.
\]
Remember to put your final answer within \boxed{}."""


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified LZ78 route unexpectedly called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def test_standard_lz78_phrase_pairs_and_bits_are_computed_from_the_prompt():
    hints = SympyTool().hints_for(STANDARD_PROBLEM)

    assert len(hints) == 1
    assert hints[0].startswith("本地LZ78编码答案: ")
    assert "Phrases: a, ab, aba, b, c, abc, d, e" in hints[0]
    assert "pairs: (0,a), (1,b), (2,a), (0,b), (0,c), (2,c), (0,d), (0,e)" in hints[0]
    assert "ceil(log2(8))=3 bits" in hints[0]
    assert (
        "encoded string: 000000 001001 010000 000001 000010 010010 000011 000100"
        in hints[0]
    )


def test_complete_lz78_contract_is_a_certified_multi_goal_route():
    evidence = _evidence(STANDARD_PROBLEM)

    assert len(evidence) == 1
    assert evidence[0].operation == "lz78_encoding"
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].verified

    result = ReasoningAgent(_NoModelClient()).solve(STANDARD_PROBLEM, {})
    assert result["final_response"].startswith(r"\boxed{Phrases: a, ab, aba")
    assert "encoded string: 000000 001001 010000 000001 000010 010010 000011 000100" in result["final_response"]
    selection = next(step for step in result["trace"] if step["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"


def test_explicit_wider_index_field_is_honoured():
    problem = (
        "Encode the message aabb using LZ78. Describe the decomposition into phrases and give "
        "the encoded string. The dictionary index uses 4 bits. "
        "When encoding a letter, use a->0, b->1."
    )

    hint = SympyTool().hints_for(problem)[0]
    assert hint.startswith("本地LZ78编码答案: ")
    assert "Phrases: a, ab, b" in hint
    assert "stated index width: 4 bits" in hint
    assert "encoded string: 00000 00011 00001" in hint
    assert _evidence(problem)[0].scope == "whole_goal"


def test_existing_terminal_phrase_requires_an_eof_convention_and_never_routes_whole():
    problem = (
        "Encode the message aa using LZ78. Describe the phrase decomposition and give "
        "the encoded string. When encoding a letter, use a->0."
    )

    hint = SympyTool().hints_for(problem)[0]
    evidence = _evidence(problem)
    assert hint.startswith("本地LZ78编码核验: ")
    assert "EOF convention is required" in hint
    assert evidence[0].operation == "lz78_encoding_check"
    assert evidence[0].scope == "subexpression"
    assert evidence[0].verified


def test_missing_mapping_and_insufficient_index_width_are_verification_only():
    missing_mapping = (
        "Encode the message abc using LZ78. Describe the phrase decomposition and give "
        "the encoded string. When encoding a letter, use a->00, b->01."
    )
    short_index = STANDARD_PROBLEM.replace(
        "When encoding a letter", "The dictionary index uses 2 bits. When encoding a letter"
    )

    missing_hint = SympyTool().hints_for(missing_mapping)[0]
    width_hint = SympyTool().hints_for(short_index)[0]
    assert "mapping omits c" in missing_hint
    assert "2-bit index cannot represent" in width_hint
    assert _evidence(missing_mapping)[0].scope == "subexpression"
    assert _evidence(short_index)[0].scope == "subexpression"


def test_extra_justification_and_unrelated_contract_parts_cannot_be_bypassed():
    proof_problem = STANDARD_PROBLEM.replace(
        "Remember to put", "Explain why the decomposition is valid. Remember to put"
    )
    extra_part = STANDARD_PROBLEM.replace(
        "Remember to put", "Also state the final dictionary size. Remember to put"
    )

    proof_evidence = _evidence(proof_problem)
    extra_evidence = _evidence(extra_part)
    assert proof_evidence[0].operation == "lz78_encoding_check"
    assert proof_evidence[0].scope == "subexpression"
    assert extra_evidence[0].operation == "lz78_encoding"
    assert extra_evidence[0].scope == "subexpression"


def test_lz77_and_problem_without_both_requested_outputs_do_not_trigger():
    lz77 = STANDARD_PROBLEM.replace("Lempel-Ziv", "LZ77")
    only_encoding = (
        "Encode the message abc using LZ78 and give the encoded string. "
        "When encoding a letter, use a->00, b->01, c->10."
    )

    assert not SympyTool().hints_for(lz77)
    assert not SympyTool().hints_for(only_encoding)
