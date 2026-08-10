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

    assert selected.id == "method.olympiad.combinatorics"


def test_basic_noncontest_tasks_are_not_promoted():
    for problem in (
        "Find the area of a circle of radius 2.",
        "Solve the inequality x^2<1.",
        "Divide the interval [0,2] into two pieces.",
    ):
        spec = build_problem_spec(problem)
        assert spec.profile.topic == "general"


def test_most_english_contest_section_problems_reach_hard_retrieval():
    rows = list(_rows().values())[:86]
    routed = 0
    for row in rows:
        spec = build_problem_spec(row["problem"])
        routed += int(spec.profile.difficulty == "hard" and SubmissionAgent._should_retrieve(spec))

    assert routed >= 75
