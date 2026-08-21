"""Compatibility shim for the retired narrow-pattern calculator.

Production uses only the generic symbolic and declarative whitelist tools.
This class remains importable for older local diagnostics, but deliberately
does not recognize or answer natural-language problem families.
"""

from __future__ import annotations

from tools.tool_contract import ToolResult


class DeterministicMathTool:
    """Abstain; retained only for compatibility with older local scripts."""

    def results_for(self, problem: str) -> list[ToolResult]:
        del problem
        return []
