from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from reasoning.candidate_selector import assess_candidate
from tools.sympy_tool import SympyTool
from user_agent import ReasoningAgent


EULER_ZH = "一个连通平面简单图有10个顶点和16条边，若每个面边界长度至少为3，求其面数并验证欧拉公式。"
CURVATURE_ZH = "设曲面z=x^2+y^2，求原点处两条主曲率及高斯曲率，并写出二阶导数依据。"


class _NoModelClient:
    def chat(self, **kwargs):
        raise AssertionError(f"certified graph/geometry route called the model: {kwargs}")


def _evidence(problem: str):
    spec = build_problem_spec(problem)
    return SubmissionAgent._tool_evidence(SympyTool().results_for(problem), spec)


def test_euler_face_count_and_paraboloid_curvature_are_certified():
    cases = (
        (EULER_ZH, "planar_euler_faces", ("F=E-V+2=16-10+2=8", "10-16+8=2")),
        (
            CURVATURE_ZH,
            "paraboloid_curvature",
            ("f_{xx}=2", "f_{xy}=0", "f_{yy}=2", r"\kappa_1=\kappa_2=2", "K=", "=4"),
        ),
    )
    for problem, operation, terms in cases:
        evidence = _evidence(problem)
        result = ReasoningAgent(_NoModelClient()).solve(problem, {})

        assert evidence[0].operation == operation
        assert evidence[0].scope == "whole_goal"
        assert all(term in result["final_response"] for term in terms)
        assert assess_candidate(
            result["final_response"], "offline", build_problem_spec(problem), ()
        ).accepted
        assert next(
            step for step in result["trace"] if step["step"] == "selection"
        )["content"]["source"] == "sympy_verified"


def test_equivalent_english_graph_and_geometry_prompts_return_english():
    euler = (
        "A connected planar simple graph has 10 vertices and 16 edges. "
        "Find the number of faces and verify Euler's formula."
    )
    curvature = (
        "For the surface z=x^2+y^2, find both principal curvatures and the Gaussian "
        "curvature at the origin, and give the second partial derivatives used."
    )

    euler_evidence = _evidence(euler)
    curvature_evidence = _evidence(curvature)
    euler_answer = ReasoningAgent(_NoModelClient()).solve(euler, {})["final_response"]
    curvature_answer = ReasoningAgent(_NoModelClient()).solve(curvature, {})["final_response"]

    assert euler_evidence[0].operation == "planar_euler_faces"
    assert curvature_evidence[0].operation == "paraboloid_curvature"
    assert euler_answer.startswith("Euler's formula")
    assert curvature_answer.startswith("At the origin")


def test_extra_proofs_downgrade_graph_and_geometry_tools_to_checks():
    for problem in (EULER_ZH + "并证明欧拉公式。", CURVATURE_ZH + "并证明主曲率公式。"):
        evidence = _evidence(problem)

        assert len(evidence) == 1
        assert evidence[0].operation.endswith("_check")
        assert evidence[0].scope == "subexpression"


def test_nearby_graph_and_surface_problems_do_not_bypass_the_model():
    cases = (
        EULER_ZH.replace("连通", ""),
        EULER_ZH.replace("并验证欧拉公式", ""),
        CURVATURE_ZH.replace("x^2+y^2", "x^2-y^2"),
        CURVATURE_ZH.replace("原点", "点(1,0)"),
    )
    operations = {"planar_euler_faces", "paraboloid_curvature"}
    for problem in cases:
        assert not any(
            item.operation in operations and item.scope == "whole_goal"
            for item in _evidence(problem)
        )
