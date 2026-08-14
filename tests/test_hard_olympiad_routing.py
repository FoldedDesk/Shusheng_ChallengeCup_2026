import json
from pathlib import Path

import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from rag.card_retriever import CardRetriever


DATASET = Path(__file__).parents[1] / "sample_data" / "judge1_style_112_hard_v1.jsonl"


def _rows():
    return {row["idx"]: row for row in map(json.loads, DATASET.read_text().splitlines())}


@pytest.mark.parametrize(
    "idx, topic, subject",
    [
        (5000, "olympiad_combinatorics", "离散数学"),
        (5001, "olympiad_combinatorics", "离散数学"),
        (5008, "olympiad_combinatorics", "离散数学"),
        (5028, "olympiad_combinatorics", "离散数学"),
        (5030, "olympiad_combinatorics", "离散数学"),
        (5049, "olympiad_number_theory", "数论"),
        (5061, "olympiad_number_theory", "数论"),
        (5067, "olympiad_polynomial", "高等代数"),
        (5073, "olympiad_sequence", "离散数学"),
        (5083, "olympiad_geometry", "初等几何"),
    ],
)
def test_english_hard_problem_routes_from_problem_text(idx, topic, subject):
    spec = build_problem_spec(_rows()[idx]["problem"])

    assert spec.profile.topic == topic
    assert spec.profile.subject == subject
    assert spec.profile.difficulty == "hard"
    assert SubmissionAgent._should_retrieve(spec)


@pytest.mark.parametrize("idx", [5024, 5025, 5027])
def test_numerical_method_retrieval_keeps_method_from_full_stem(idx):
    spec = build_problem_spec(_rows()[idx]["problem"])
    bundle = CardRetriever().retrieve(spec)

    assert spec.problem_text
    assert SubmissionAgent._should_retrieve(spec)
    assert bundle.language == "en"
    assert bundle.solve_cards
    assert bundle.solve_cards[0].id.startswith("note.数值分析.")


def test_specific_graph_cards_beat_generic_counting_advice():
    rows = _rows()
    deleted_edge = CardRetriever().retrieve(build_problem_spec(rows[5001]["problem"]))
    hypercube = CardRetriever().retrieve(build_problem_spec(rows[5010]["problem"]))

    assert "Deleting one edge" in deleted_edge.solve_cards[0].render("en")
    assert "hypercube" in hypercube.solve_cards[0].render("en").lower()


def test_stopwords_do_not_route_heap_game_to_multipartite_graph_formula():
    spec = build_problem_spec(_rows()[5039]["problem"])
    selected = CardRetriever().retrieve(spec).solve_cards[0]

    assert selected.id == "method.finite_game.minimax"


def test_basic_noncontest_tasks_are_not_promoted():
    for problem in (
        "Find the area of a circle of radius 2.",
        "Solve the inequality x^2<1.",
        "Divide the interval [0,2] into two pieces.",
    ):
        spec = build_problem_spec(problem)
        assert spec.profile.topic == "general"


@pytest.mark.parametrize(
    "problem, subject, topic",
    [
        (
            "Prove that the Fourier transform is a bijection on the Schwartz space.",
            "数学分析",
            "general",
        ),
        (
            "Prove that a linear transformation T: R^n to R^n is a bijection if and only if det(T) is nonzero.",
            "线性代数",
            "general",
        ),
        (
            "Prove that the finite-difference scheme is stable at all grid cells.",
            "数值分析",
            "general",
        ),
        (
            "Find all real roots of x^2=1 and verify that both sides are equal.",
            "高等代数",
            "olympiad_polynomial",
        ),
        (
            "Prove that the expected total winnings in n independent games equals n/2.",
            "概率论",
            "general",
        ),
    ],
)
def test_generic_cross_domain_words_do_not_override_explicit_subject(problem, subject, topic):
    profile = build_problem_spec(problem).profile

    assert profile.subject == subject
    assert profile.topic == topic


@pytest.mark.parametrize("problem", (
    (
        "Two players take turns rolling a fair die. The first to roll a 6 wins. "
        "Compute the probability that the first player wins."
    ),
    "两人轮流掷一枚公平骰子，先掷出6者获胜，求先手获胜的概率。",
))
def test_stochastic_turn_taking_stays_probability_and_rejects_minimax_card(problem):
    spec = build_problem_spec(problem)
    bundle = CardRetriever().retrieve(spec)

    assert spec.profile.subject == "概率论"
    assert spec.profile.topic == "general"
    assert spec.primary_method == "condition_on_events"
    assert "minimax" not in spec.alternative_method
    assert "method.finite_game.minimax" not in {
        card.id for card in (*bundle.solve_cards, *bundle.review_cards)
    }


def test_additive_representation_counting_and_chinese_matrix_keep_their_fields():
    counting = build_problem_spec(
        "How many ways can 20 be represented as a sum of three nonnegative integers?"
    ).profile
    matrix = build_problem_spec(
        "将矩阵 A 表示成对角形，并求特征值和特征向量。"
    ).profile

    assert counting.subject == "离散数学"
    assert counting.topic == "olympiad_combinatorics"
    assert matrix.subject == "线性代数"


@pytest.mark.parametrize(
    "problem, forbidden_card",
    [
        (
            "Determine the radius of convergence of the power series sum x^n/n.",
            "method.analysis.fourier",
        ),
        (
            "How many subsets of {1,2,3,4} have even cardinality?",
            "method.finite_game.minimax",
        ),
        (
            "Show that every ideal of Z is principal.",
            "method.algebra.splitting_field",
        ),
    ],
)
def test_specialized_rag_cards_require_operation_markers(problem, forbidden_card):
    bundle = CardRetriever().retrieve(build_problem_spec(problem))

    assert forbidden_card not in {
        card.id for card in (*bundle.solve_cards, *bundle.review_cards)
    }


def test_minimax_card_requires_decisions_legality_and_terminal_payoff_together():
    for problem in (
        "Two players take turns choosing integers. Prove that their choices form a finite set.",
        "A player may remove one stone on each move. Determine the generating function of the move counts.",
        "A game has a terminal state and a winner. Count the vertices in its state graph.",
    ):
        bundle = CardRetriever().retrieve(build_problem_spec(problem))
        assert "method.finite_game.minimax" not in {
            card.id for card in (*bundle.solve_cards, *bundle.review_cards)
        }


def test_loaded_note_requires_problem_text_semantics_not_shared_numbers_or_metadata():
    problem = (
        "Tile a 1008 by 1010 rectangle using rotatable dominoes and S-tetrominoes. "
        "Determine the minimum number of dominoes required."
    )
    bundle = CardRetriever().retrieve(build_problem_spec(problem))
    spec = build_problem_spec(problem)

    assert "note.图论进阶.13" not in {
        card.id for card in (*bundle.solve_cards, *bundle.review_cards)
    }
    assert spec.primary_method == "tiling_coloring_and_cut_invariant"
    assert SubmissionAgent._should_retrieve(spec)
    assert bundle.solve_cards[0].id == "method.tiling.invariant_profile"


def test_colored_cube_slice_problem_gets_the_slice_incidence_theorem_only():
    problem = (
        "A 9 by 9 by 9 cube consists of colored unit cubes. For every 9 by 9 by 1 "
        "rectangular prism, its set of distinct colors appears in a prism in each of "
        "the other two orientations. Find the maximum possible number of colors."
    )
    spec = build_problem_spec(problem)
    bundle = CardRetriever().retrieve(spec)

    assert spec.profile.subject == "离散数学"
    assert spec.primary_method == "slice_color_incidence_chain_bound"
    assert SubmissionAgent._should_retrieve(spec)
    assert bundle.solve_cards[0].id == "fact.combinatorics.colored_cube_slices"


def test_n_good_exotic_problem_gets_rank_theorem_with_exact_gate():
    problem = (
        "Call g: Z to Z an n-good function if g(1)=1 and g(a)-g(b) divides a^n-b^n for every "
        "distinct a,b. Call n exotic when the number of n-good functions is twice an "
        "odd integer. Find the third exotic integer."
    )
    spec = build_problem_spec(problem)
    bundle = CardRetriever().retrieve(spec)

    assert spec.primary_method == "classify_divisibility_preserving_integer_functions"
    assert bundle.solve_cards[0].id == "fact.number_theory.n_good_exotic"


def test_most_english_contest_section_problems_reach_hard_retrieval():
    rows = list(_rows().values())[:86]
    routed = 0
    for row in rows:
        spec = build_problem_spec(row["problem"])
        routed += int(spec.profile.difficulty == "hard" and SubmissionAgent._should_retrieve(spec))

    # Bare "positive integer" declarations are intentionally no longer
    # promoted to number theory, so a few generic grid/game stems stay out.
    assert routed >= 72
