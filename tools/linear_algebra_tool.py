"""Exact certificates for fully specified finite-dimensional linear algebra."""

from __future__ import annotations

import re
from typing import Optional

from tools.tool_contract import ToolResult, make_parameterized_tool_result


class LinearAlgebraTool:
    """Recover nilpotent Jordan data from a complete kernel/rank sequence."""

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if not text:
            return []
        try:
            result = self._nilpotent_jordan_partition(text)
        except Exception:
            result = None
        return [result] if result is not None and result.verified else []

    def _nilpotent_jordan_partition(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"幂零|nilpotent", text, re.I):
            return None
        if not re.search(r"Jordan\s*块|若尔当块|Jordan\s+blocks?|Jordan\s+partition", text, re.I):
            return None
        dimension = self._dimension(text)
        if dimension is None or not 1 <= dimension <= 10_000:
            return None

        kernel_matches = list(re.finditer(
            r"(?:\\dim|dim)\s*(?:\\ker|ker)\s*([A-Za-z])\s*"
            r"(?:\^\s*\{?\s*(\d+)\s*\}?)?\s*=\s*(\d+)",
            text,
            re.I,
        ))
        dimensions: dict[int, int] = {}
        symbols: set[str] = set()
        for match in kernel_matches:
            symbols.add(match.group(1).casefold())
            power = int(match.group(2) or 1)
            value = int(match.group(3))
            if power in dimensions and dimensions[power] != value:
                return None
            dimensions[power] = value

        if not dimensions:
            rank_matches = list(re.finditer(
                r"(?:\\operatorname\s*\{\s*rank\s*\}|rank)\s*([A-Za-z])\s*"
                r"(?:\^\s*\{?\s*(\d+)\s*\}?)?\s*=\s*(\d+)",
                text,
                re.I,
            ))
            ranks: dict[int, int] = {}
            for match in rank_matches:
                symbols.add(match.group(1).casefold())
                power = int(match.group(2) or 1)
                value = int(match.group(3))
                if power in ranks and ranks[power] != value:
                    return None
                ranks[power] = value
            zero_power = re.search(r"([A-Za-z])\s*\^\s*\{?\s*(\d+)\s*\}?\s*=\s*0", text)
            if not ranks or zero_power is None:
                return None
            symbols.add(zero_power.group(1).casefold())
            nilpotency_bound = int(zero_power.group(2))
            dimensions = {power: dimension - rank for power, rank in ranks.items()}
            dimensions[nilpotency_bound] = dimension

        if len(symbols) != 1 or not dimensions:
            return None
        maximum_power = max(dimensions)
        if set(dimensions) != set(range(1, maximum_power + 1)):
            return None
        if dimensions[maximum_power] != dimension:
            return None
        ordered_dimensions = [0] + [dimensions[power] for power in range(1, maximum_power + 1)]
        if any(not 0 <= value <= dimension for value in ordered_dimensions):
            return None

        # b_k = dim ker N^k - dim ker N^(k-1) is the number of blocks
        # of size at least k.  It must be nonnegative and nonincreasing.
        at_least = [
            ordered_dimensions[power] - ordered_dimensions[power - 1]
            for power in range(1, maximum_power + 1)
        ]
        if any(value < 0 for value in at_least) or any(
            at_least[index] < at_least[index + 1]
            for index in range(len(at_least) - 1)
        ):
            return None
        next_counts = at_least[1:] + [0]
        exact_size = [left - right for left, right in zip(at_least, next_counts)]
        partition = tuple(
            size
            for size in range(maximum_power, 0, -1)
            for _ in range(exact_size[size - 1])
        )
        if not partition or sum(partition) != dimension:
            return None

        reconstructed = {
            power: sum(min(power, size) for size in partition)
            for power in range(1, maximum_power + 1)
        }
        if reconstructed != dimensions:
            return None
        rank = dimension - dimensions[1]
        largest = partition[0]
        partition_text = "(" + ",".join(str(value) for value in partition) + ")"
        symbol = next(iter(symbols)).upper()
        zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        result = (
            rf"令 $d_k=\dim\ker {symbol}^k$。差分 $b_k=d_k-d_{{k-1}}$ 是大小至少为 $k$ 的 Jordan 块数；"
            rf"再次作差得到各块大小，因此全部块为 ${partition_text}$。由此 "
            rf"$\operatorname{{rank}}{symbol}={rank}$（因为 ${dimension}-d_1={rank}$），最大块大小为 {largest}，"
            rf"故最小多项式为 $t^{{{largest}}}$。"
            if zh else
            rf"Let $d_k=\dim\ker {symbol}^k$. The difference $b_k=d_k-d_{{k-1}}$ counts blocks of size "
            rf"at least $k$; differencing once more gives the partition ${partition_text}$. Hence "
            rf"$\operatorname{{rank}}{symbol}={rank}$ (since ${dimension}-d_1={rank}$), and the largest block has size {largest}, "
            rf"so the minimal polynomial is $t^{{{largest}}}$."
        )
        return make_parameterized_tool_result(
            problem=text,
            operation="nilpotent_jordan_partition",
            result=result,
            result_kind="jordan_partition_rank_minimal_polynomial",
            method="kernel_dimension_first_and_second_differences",
            whole=True,
            written_support=True,
            checks=(
                "ambient_dimension_parsed",
                "contiguous_kernel_or_rank_sequence",
                "kernel_dimensions_monotone",
                "block_at_least_counts_nonincreasing",
                "all_power_kernels_reconstructed",
                "partition_dimension_reconstructed",
            ),
            support=result,
            answer_shapes=("expression", "text"),
            requirements=(
                "result_present",
                "reasoning",
                "jordan_blocks",
                "operator_rank",
                "minimal_polynomial",
            ),
        )

    @staticmethod
    def _dimension(text: str) -> Optional[int]:
        words = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        match = re.search(
            r"(?:(\d+|一|二|两|三|四|五|六|七|八|九|十)\s*维(?:复|实)?(?:向量)?空间|"
            r"(?:on|of)\s+(?:an?\s+)?(\d+)[- ]dimensional\s+(?:vector\s+)?space)",
            text,
            re.I,
        )
        if match is None:
            return None
        token = next(group for group in match.groups() if group is not None)
        return int(token) if token.isdigit() else words.get(token)
