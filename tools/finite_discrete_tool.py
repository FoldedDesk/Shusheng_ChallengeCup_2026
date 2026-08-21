"""Compatibility shim for the retired narrow finite-problem recognizers.

Generic finite computation lives in ``core_textbook_tool`` and
``finite_structure_tool``.  This legacy class intentionally abstains so a
problem-specific recognizer cannot enter the formal submission path.
"""

from __future__ import annotations

from tools.tool_contract import ToolResult


class FiniteDiscreteTool:
    """Abstain; retained only for compatibility with older local scripts."""

    def results_for(self, problem: str) -> list[ToolResult]:
        del problem
        return []
