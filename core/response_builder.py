from __future__ import annotations

from typing import Dict

from core.serializer import safe_json
from core.state import MathState


class ResponseBuilder:
    """Build the only public agent response shape."""

    @staticmethod
    def build(state: MathState, final_response: str) -> Dict:
        answer = str(final_response or "").strip()
        if not answer:
            answer = "TRUNCATED_ALL"
        state.final_answer = answer
        return safe_json({"final_response": answer, "trace": state.trace})
