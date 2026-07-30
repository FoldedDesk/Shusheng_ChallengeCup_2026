from __future__ import annotations

import re
from typing import Dict, List

from core.client_adapter import ClientAdapter
from core.exception_handler import retry_once
from core.state import MathState
from reasoning.finalizer import Finalizer


class Verifier:
    def __init__(self, client: ClientAdapter, prompt: str) -> None:
        self.client_adapter = client
        self.prompt = prompt

    def verify(self, state: MathState, candidates: List[str]) -> Dict:
        if not candidates:
            return {
                "correct": False,
                "choice": None,
                "final_answer": "",
                "reason": "没有可用候选",
            }
        if len(candidates) == 1 and state.difficulty == "easy":
            return {
                "correct": True,
                "choice": 0,
                "final_answer": "",
                "reason": "简单题单次求解",
            }
        # Long candidate derivations make the remote model spend its entire
        # response budget restating analysis. The verifier needs conclusions
        # to compare, then independently recomputes from the original problem.
        joined = "\n\n".join(
            f"候选{i}的最终结论：{Finalizer.extract(text)}"
            for i, text in enumerate(candidates)
        )
        response = retry_once(lambda: self.client_adapter.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"题目：\n{state.problem}\n\n{joined}"},
            ],
            temperature=0.0,
            # Intern may emit a long hidden-style thinking preamble before its
            # structured three-line result. Leave room for FINAL to arrive.
            max_tokens=8192,
        ))
        raw = str(response or "")
        match = re.search(r"CHOICE\s*[:：]\s*(\d+)", raw, re.IGNORECASE)
        choice = int(match.group(1)) if match else 0
        if not 0 <= choice < len(candidates):
            choice = 0
        final_match = re.search(r"^FINAL\s*[:：]\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
        reason_match = re.search(r"REASON\s*[:：]\s*(.+)", raw, re.IGNORECASE)
        return {
            "correct": bool(response),
            "choice": choice,
            "final_answer": final_match.group(1).strip() if final_match else "",
            "reason": reason_match.group(1).strip() if reason_match else raw[:300] or "验证器不可用",
        }
