from __future__ import annotations

from pathlib import Path
from typing import Dict

from classifier import classify_difficulty, classify_problem_type, classify_subject
from core.client_adapter import ClientAdapter
from core.response_builder import ResponseBuilder
from core.state import MathState
from rag.retriever import LocalRetriever
from reasoning.finalizer import Finalizer
from reasoning.planner import Planner
from reasoning.critic import Critic
from reasoning.solver import Solver
from reasoning.verifier import Verifier
from tools.sympy_tool import SympyTool


class MathAgent:
    """State-isolated Chinese mathematics reasoning pipeline."""

    def __init__(self, client: ClientAdapter) -> None:
        self.client = client
        prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
        self.solver = Solver(client, self._prompt(prompt_dir / "solver.txt"))
        self.critic = Critic(client, self._prompt(prompt_dir / "critic.txt"))
        self.verifier = Verifier(client, self._prompt(prompt_dir / "verifier.txt"))
        self.planner = Planner()
        self.retriever = LocalRetriever()
        self.sympy = SympyTool()

    def solve(self, problem: str, metadata: Dict) -> Dict:
        state = MathState(problem=str(problem), metadata=dict(metadata or {}))
        state.subject = classify_subject(state.problem)
        state.problem_type = classify_problem_type(state.problem)
        state.difficulty = classify_difficulty(state.problem, state.problem_type)
        references = self.retriever.retrieve(state.problem, top_k=5)
        plan = self.planner.plan(state, references)
        sympy_hints = self.sympy.hints_for(state.problem)
        if sympy_hints:
            plan["sympy_hints"] = sympy_hints

        state.trace.append({
            "step": "classification",
            "content": {
                "subject": state.subject,
                "problem_type": state.problem_type,
                "difficulty": state.difficulty,
            },
        })
        state.trace.append({"step": "planning", "content": {"reference_count": len(references)}})
        state.trace.append({"step": "sympy", "content": {"hint_count": len(sympy_hints)}})
        candidates = self.solver.generate(state, plan)
        critic_reviews = [self.critic.review(state.problem, candidate) for candidate in candidates]
        state.trace.append({"step": "critic", "content": {"review_count": len(critic_reviews)}})
        verification = self.verifier.verify(state, candidates, critic_reviews, sympy_hints)
        state.verification = verification
        state.trace.append({
            "step": "verification",
            "content": {
                "correct": verification["correct"],
                "final_answer_provided": bool(verification.get("final_answer")),
            },
        })
        choice = verification.get("choice")
        selected = candidates[choice] if isinstance(choice, int) and choice < len(candidates) else ""
        final_source = verification.get("final_answer") or selected
        state.trace.append({"step": "response", "content": "built"})
        return ResponseBuilder.build(state, Finalizer.extract(final_source) if final_source else "TRUNCATED_ALL")

    @staticmethod
    def _prompt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return "请用中文解答数学题，并在最后给出明确答案。"
