"""Exact certificates for fully specified finite Markov chains.

Only a complete numeric transition matrix in the current problem is accepted.
The tool deliberately abstains on symbolic, continuous-time, or ambiguous
requests; no model-provided result or executable expression is consumed.
"""

from __future__ import annotations

import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class StochasticMatrixTool:
    """Recompute bounded finite-state Markov-chain quantities exactly."""

    _MAX_STATES = 20
    _MAX_POWER = 10_000
    _MAX_BIRTH_DEATH_BOUNDARY = 500

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        if not self.sp:
            return []
        text = str(problem or "").strip()
        if not re.search(
            r"马尔可夫|Markov|转移矩阵|transition\s+matrix|"
            r"随机游走|random\s+walk|出生.?死亡|birth[- ]?death",
            text,
            re.IGNORECASE,
        ):
            return []
        results: list[ToolResult] = []
        try:
            birth_death = self._birth_death_hitting_probability(text)
        except Exception:
            birth_death = None
        if birth_death is not None and birth_death.verified:
            results.append(birth_death)
        parsed = self._transition_matrix(text)
        if parsed is None:
            return results
        matrix, labels = parsed
        for compiler in (
            self._transition_matrix_power,
            self._stationary_distribution,
            self._absorbing_states,
        ):
            try:
                result = compiler(text, matrix, labels)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _birth_death_hitting_probability(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"nearest[- ]neighbor|birth[- ]?death|random\s+walk|出生.?死亡|随机游走|"
            r"(?:马尔可夫|Markov)[^。.;\n]{0,100}(?:链|chain)",
            text,
            re.IGNORECASE,
        ):
            return None
        if not re.search(
            r"absorbed\s+at\s+\$?0\$?\s+and\s+\$?\d+\$?|"
            r"\$?0\$?\s+and\s+\$?\d+\$?\s+are\s+absorbing|"
            r"两端吸收|吸收(?:于|在)\s*0\s*(?:和|与|、)\s*\d+",
            text,
            re.IGNORECASE,
        ):
            return None
        boundary = re.search(
            r"(?:\\?\{|\{)?\s*0\s*,\s*1\s*,\s*"
            r"(?:\\(?:ldots|cdots|dots)|\.\.\.)\s*,\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        start = re.search(
            r"从(?:状态)?\s*\$?\s*(\d+)\s*\$?\s*出发|"
            r"starting\s+from(?:\s+(?:state|position))?\s*\$?\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        target = re.search(
            r"先(?:到达|到|击中)\s*\$?\s*(\d+)\s*\$?\s*(?:再|之)?前(?:到达|到|击中)?\s*\$?\s*0|"
            r"先(?:到达|到|击中)\s*\$?\s*(\d+)\s*\$?\s*的概率|"
            r"hitting\s+\$?\s*(\d+)\s*\$?\s+before\s+\$?\s*0",
            text,
            re.IGNORECASE,
        )
        probabilities = self._direction_probabilities(text)
        if not boundary or not start or not target or probabilities is None:
            return None
        upper = int(boundary.group(1))
        initial = int(next(group for group in start.groups() if group is not None))
        requested_upper = int(next(group for group in target.groups() if group is not None))
        if (
            requested_upper != upper
            or not 2 <= upper <= self._MAX_BIRTH_DEATH_BOUNDARY
            or not 0 < initial < upper
        ):
            return None

        state = self.sp.Symbol("i")
        up_expression = self._probability_formula(probabilities[0])
        down_expression = self._probability_formula(probabilities[1])
        if (
            up_expression.free_symbols - {state}
            or down_expression.free_symbols - {state}
        ):
            return None

        increments = [self.sp.Integer(1)]
        transitions = []
        for index in range(1, upper):
            up = self.sp.simplify(up_expression.subs(state, index))
            down = self.sp.simplify(down_expression.subs(state, index))
            if (
                up.free_symbols or down.free_symbols
                or up.is_real is not True or down.is_real is not True
                or up.is_positive is not True or down.is_nonnegative is not True
                or self.sp.simplify(up + down - 1) != 0
            ):
                return None
            transitions.append((up, down))
            increments.append(self.sp.simplify(increments[-1] * down / up))

        denominator = self.sp.simplify(sum(increments))
        if denominator.is_positive is not True:
            return None
        probability = self.sp.simplify(sum(increments[:initial]) / denominator)
        if probability.is_nonnegative is not True or (1 - probability).is_nonnegative is not True:
            return None
        values = [self.sp.Integer(0)]
        values.extend(
            self.sp.simplify(sum(increments[:index]) / denominator)
            for index in range(1, upper)
        )
        values.append(self.sp.Integer(1))
        if any(
            self.sp.simplify(
                values[index]
                - up * values[index + 1]
                - down * values[index - 1]
            ) != 0
            for index, (up, down) in enumerate(transitions, start=1)
        ):
            return None

        result = self.symbolic._format(probability)
        zh = self._is_chinese(text)
        support = (
            rf"令 $h_i=P_i(\tau_{{{upper}}}<\tau_0)$，则 $h_0=0,h_{{{upper}}}=1$，且 "
            rf"$h_i=p_i h_{{i+1}}+q_i h_{{i-1}}$。逐差尺度权重精确归一化并回代全部 "
            rf"{upper - 1} 个内部方程，得到 $h_{{{initial}}}={result}$。"
            if zh else
            rf"Let $h_i=P_i(\tau_{{{upper}}}<\tau_0)$. With $h_0=0$, "
            rf"$h_{{{upper}}}=1$ and $h_i=p_i h_{{i+1}}+q_i h_{{i-1}}$, exact "
            rf"successive-difference scale weights were normalized and substituted into all "
            rf"{upper - 1} interior equations, giving $h_{{{initial}}}={result}$."
        )
        return self._result(
            text,
            "birth_death_hitting_probability",
            result,
            "probability",
            "exact_scale_increments_and_recurrence_substitution",
            (
                "finite_state_interval_parsed",
                "transition_normalization_at_every_interior_state",
                "absorbing_boundaries_imposed",
                "all_interior_recurrences_substituted",
            ),
            support,
            ("probability", "number", "expression"),
            ("result_present", "numeric_result"),
        )

    @staticmethod
    def _direction_probabilities(text: str) -> Optional[tuple[str, str]]:
        patterns = (
            (
                r"(?:to\s+)?\$?\s*i\s*\+\s*1\s*\$?\s+with\s+probability\s+\$([^$]+)\$",
                r"(?:to\s+)?\$?\s*i\s*-\s*1\s*\$?\s+with\s+probability\s+\$([^$]+)\$",
            ),
            (
                r"(?:moves?|jumps?)\s+with\s+probability\s+\$([^$]+)\$\s+to\s+\$?\s*i\s*\+\s*1",
                r"(?:moves?|jumps?)\s+with\s+probability\s+\$([^$]+)\$\s+to\s+\$?\s*i\s*-\s*1",
            ),
            (
                r"以概率\s*\$([^$]+)\$\s*(?:加\s*1|向右|转移到\s*\$?i\s*\+\s*1)",
                r"以概率\s*\$([^$]+)\$\s*(?:减\s*1|向左|转移到\s*\$?i\s*-\s*1)",
            ),
        )
        for up_pattern, down_pattern in patterns:
            up = re.search(up_pattern, text, re.IGNORECASE)
            down = re.search(down_pattern, text, re.IGNORECASE)
            if up and down:
                return up.group(1), down.group(1)
        return None

    def _probability_formula(self, value: str):
        expression = str(value or "").strip().strip("$ ")
        if re.fullmatch(r"[A-Za-z]\s*=\s*.+", expression):
            expression = expression.split("=", 1)[1].strip()
        return self.sp.simplify(
            self.symbolic._parse(self.symbolic._latex_to_sympy(expression))
        )

    def _transition_matrix_power(self, text: str, matrix, labels) -> Optional[ToolResult]:
        del labels
        requests = []
        for pattern in (
            r"(?:求|计算|写出|find|compute)\s*(?:该)?\s*(\d+)\s*步转移矩阵",
            r"(?:find|compute)\s+(?:the\s+)?(\d+)[- ]step\s+transition\s+matrix",
            r"(?:求|计算|写出|find|compute)[^。.;\n]{0,30}?P\s*\^\s*\{?\s*(\d+)\s*\}?",
        ):
            requests.extend(int(match.group(1)) for match in re.finditer(pattern, text, re.IGNORECASE))
        powers = tuple(dict.fromkeys(requests))
        if len(powers) != 1:
            return None
        power = powers[0]
        if not 0 <= power <= self._MAX_POWER:
            return None
        if self._has_other_target(text, "power"):
            return None
        value = matrix ** power
        rendered = self.sp.latex(value)
        zh = self._is_chinese(text)
        answer = (
            rf"${{P}}^{{{power}}}={rendered}$。"
            if zh else
            rf"${{P}}^{{{power}}}={rendered}$."
        )
        support = (
            f"已精确验证每行和为 1、各元素非负，并以整数快速幂重算 {power} 步转移矩阵。"
            if zh else
            f"Every row was verified to sum to 1 with nonnegative entries, and the {power}-step matrix was recomputed by exact integer matrix powering."
        )
        return self._result(
            text,
            "finite_markov_transition_power",
            answer,
            "matrix",
            "exact_stochastic_matrix_power",
            ("single_complete_matrix", "row_stochastic_exact", "single_power_target", "integer_matrix_power_recomputed"),
            support,
            ("matrix", "expression"),
            ("result_present",),
        )

    def _stationary_distribution(self, text: str, matrix, labels) -> Optional[ToolResult]:
        del labels
        if not re.search(r"平稳分布|稳态分布|不变分布|stationary\s+distribution|invariant\s+distribution", text, re.IGNORECASE):
            return None
        if self._has_other_target(text, "stationary"):
            return None
        variables = self.sp.symbols(f"p0:{matrix.rows}", real=True)
        vector = self.sp.Matrix(1, matrix.rows, variables)
        equations = list(vector * matrix - vector)
        equations.append(sum(variables) - 1)
        solution_set = self.sp.linsolve(equations, variables)
        solutions = list(solution_set)
        if len(solutions) != 1:
            return None
        solution = tuple(self.sp.simplify(value) for value in solutions[0])
        if any(value.free_symbols or value.is_real is not True for value in solution):
            return None
        if any(value.is_nonnegative is not True for value in solution):
            return None
        row = self.sp.Matrix(1, matrix.rows, solution)
        if any(self.sp.simplify(value) != 0 for value in row * matrix - row):
            return None
        if self.sp.simplify(sum(solution) - 1) != 0:
            return None
        rendered = self.sp.latex(row)
        zh = self._is_chinese(text)
        answer = (
            rf"唯一平稳分布为 $\pi={rendered}$。"
            if zh else
            rf"The unique stationary distribution is $\pi={rendered}$."
        )
        support = (
            r"联立精确方程 $\pi P=\pi$ 与 $\sum_i\pi_i=1$，所得解非负；回代两式均成立。"
            if zh else
            r"The exact system $\pi P=\pi$, $\sum_i\pi_i=1$ has this nonnegative solution, and both equations were checked by substitution."
        )
        return self._result(
            text,
            "finite_markov_stationary_distribution",
            answer,
            "probability_vector",
            "exact_stationary_linear_system",
            ("single_complete_matrix", "row_stochastic_exact", "stationary_system_solved", "unique_nonnegative_solution", "stationarity_substituted"),
            support,
            ("vector", "probability", "expression"),
            ("result_present", "numeric_result", "stationary_distribution"),
        )

    def _absorbing_states(self, text: str, matrix, labels) -> Optional[ToolResult]:
        if not re.search(r"吸收态|吸收状态|absorbing\s+states?", text, re.IGNORECASE):
            return None
        if re.search(r"吸收概率|吸收时间|absorption\s+(?:probabilit|time)|expected\s+time", text, re.IGNORECASE):
            return None
        if self._has_other_target(text, "absorbing"):
            return None
        absorbing = []
        for row in range(matrix.rows):
            expected = [self.sp.Integer(int(row == column)) for column in range(matrix.cols)]
            if all(self.sp.simplify(matrix[row, column] - expected[column]) == 0 for column in range(matrix.cols)):
                absorbing.append(labels[row])
        zh = self._is_chinese(text)
        if absorbing:
            rendered = r"\{" + ",".join(absorbing) + r"\}"
            answer = f"吸收态为 ${rendered}$。" if zh else f"The absorbing states are ${rendered}$."
        else:
            answer = "不存在吸收态。" if zh else "There are no absorbing states."
        support = (
            r"逐行检查 $P_{ii}=1$ 且同一行其余元素全为 0。"
            if zh else
            r"Each row was checked for $P_{ii}=1$ with every other entry in that row equal to zero."
        )
        return self._result(
            text,
            "finite_markov_absorbing_states",
            answer,
            "state_set",
            "exhaustive_absorbing_row_check",
            ("single_complete_matrix", "row_stochastic_exact", "all_rows_enumerated", "absorbing_rows_recomputed"),
            support,
            ("set", "text", "expression"),
            ("result_present",),
        )

    def _transition_matrix(self, text: str):
        matches = list(re.finditer(
            r"\\begin\{[pbvBV]?matrix\}(.+?)\\end\{[pbvBV]?matrix\}",
            text,
            re.DOTALL,
        ))
        if len(matches) != 1:
            return None
        body = matches[0].group(1)
        rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if not cells or len(cells) > self._MAX_STATES:
            return None
        if len(cells) != len(cells[0]) or any(len(row) != len(cells) for row in cells):
            return None
        parsed_rows = []
        try:
            for row in cells:
                parsed = []
                for cell in row:
                    value = self.sp.simplify(self.symbolic._parse(self.symbolic._latex_to_sympy(cell)))
                    if value.free_symbols or value.is_real is not True or value.is_nonnegative is not True:
                        return None
                    parsed.append(value)
                if self.sp.simplify(sum(parsed) - 1) != 0:
                    return None
                parsed_rows.append(parsed)
            matrix = self.sp.Matrix(parsed_rows)
        except Exception:
            return None
        labels = self._state_labels(text, matrix.rows)
        return matrix, labels

    @staticmethod
    def _state_labels(text: str, count: int) -> tuple[str, ...]:
        normalized = str(text or "").replace(r"\{", "{").replace(r"\}", "}")
        match = re.search(
            r"(?:状态空间|state\s+space)\s*\$?\s*(?:S\s*)?=\s*\{([^{}]+)\}\s*\$?",
            normalized,
            re.IGNORECASE,
        )
        if match and not re.search(r"\\(?:ldots|cdots)|\.\.\.", match.group(1)):
            labels = tuple(part.strip(" $\t") for part in match.group(1).split(","))
            if len(labels) == count and all(re.fullmatch(r"[A-Za-z0-9_+-]+", item) for item in labels):
                return labels
        return tuple(str(index + 1) for index in range(count))

    @staticmethod
    def _has_other_target(text: str, selected: str) -> bool:
        targets = {
            "power": r"(?:\d+\s*步转移矩阵|\d+[- ]step\s+transition\s+matrix|P\s*\^\s*\{?\s*\d+)",
            "stationary": r"(?:平稳分布|稳态分布|不变分布|stationary\s+distribution|invariant\s+distribution)",
            "absorbing": r"(?:吸收态|吸收状态|absorbing\s+states?)",
        }
        present = {name for name, pattern in targets.items() if re.search(pattern, text, re.IGNORECASE)}
        return present != {selected}

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))

    @staticmethod
    def _result(
        problem: str,
        operation: str,
        result: str,
        result_kind: str,
        method: str,
        checks: tuple[str, ...],
        support: str,
        shapes: tuple[str, ...],
        requirements: tuple[str, ...],
    ) -> ToolResult:
        return make_parameterized_tool_result(
            problem=problem,
            operation=operation,
            result=result,
            result_kind=result_kind,
            method=method,
            whole=True,
            written_support=True,
            checks=checks,
            support=support,
            answer_shapes=shapes,
            requirements=requirements,
            assurance="exhaustive",
        )
