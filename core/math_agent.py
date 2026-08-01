from __future__ import annotations

from pathlib import Path
from typing import Dict

from classifier import classify_difficulty, classify_problem_type, classify_subject
from core.client_adapter import ClientAdapter
from core.response_builder import ResponseBuilder
from core.state import MathState
from rag.retriever import LocalRetriever
from reasoning.finalizer import Finalizer
from reasoning.decomposer import Decomposer
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
        self.decomposer = Decomposer(client, self._prompt(prompt_dir / "decomposer.txt"))
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
        if str(state.metadata.get("source", "")).endswith("_hard"):
            state.difficulty = "hard"
        references = self.retriever.retrieve(state.problem, top_k=5)
        plan = self.planner.plan(state, references)
        decomposition = self.decomposer.decompose(state)
        decomposition_tools = self.decomposer.execute_tools(self.sympy, decomposition["tool_specs"])
        plan["decomposition"] = decomposition["steps"]
        plan["decomposition_tools"] = decomposition_tools
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
        state.trace.append({
            "step": "decomposition",
            "content": {"step_count": len(decomposition["steps"]), "tool_count": len(decomposition_tools)},
        })
        state.trace.append({"step": "sympy", "content": {"hint_count": len(sympy_hints)}})
        candidates = self.solver.generate(state, plan)
        needs_critic = state.difficulty == "hard" or state.problem_type in {"proof", "derivation"}
        critic_reviews = self.critic.review_all(
            state.problem,
            candidates,
            attempts=1 if state.difficulty == "hard" else 2,
        ) if needs_critic else []
        state.trace.append({"step": "critic", "content": {"review_count": len(critic_reviews)}})
        verification = self.verifier.verify(
            state,
            candidates,
            critic_reviews,
            sympy_hints,
            plan.get("response_contract", ""),
            plan.get("decomposition", []),
            plan.get("decomposition_tools", []),
        )
        state.verification = verification
        state.trace.append({
            "step": "verification",
            "content": {
                "correct": verification["correct"],
                "final_answer_provided": bool(verification.get("final_answer")),
                "error_count": len(verification.get("errors", [])),
            },
        })
        choice = verification.get("choice")
        selected = candidates[choice] if isinstance(choice, int) and choice < len(candidates) else ""
        final_source = verification.get("final_answer") or selected
        state.trace.append({"step": "response", "content": "built"})
        if state.problem_type in {"proof", "derivation", "explanation"}:
            answer = Finalizer.extract_solution(final_source) if final_source else "TRUNCATED_ALL"
        else:
            answer = Finalizer.extract(final_source) if final_source else "TRUNCATED_ALL"
        return ResponseBuilder.build(state, answer)

    @staticmethod
    def _prompt(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return "请用中文解答数学题，并在最后给出明确答案。"
