from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MathState:
    """Per-problem mutable state; a new instance is created for every solve call."""

    problem: str
    metadata: Dict[str, Any]
    subject: Optional[str] = None
    problem_type: Optional[str] = None
    difficulty: Optional[str] = None
    candidate_answers: List[str] = field(default_factory=list)
    verification: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def candidates(self) -> List[str]:
        """Compatibility alias for internal solver code."""
        return self.candidate_answers
