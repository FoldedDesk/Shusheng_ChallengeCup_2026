"""Exact, fail-closed certificates for small optimization problems."""

from __future__ import annotations

from itertools import combinations
import re
from typing import Optional

from tools.sympy_tool import SympyTool
from tools.tool_contract import ToolResult, make_parameterized_tool_result


class OptimizationTool:
    """Certify explicit two-variable covering or packing linear programs."""

    def __init__(self) -> None:
        self.symbolic = SympyTool()
        self.sp = self.symbolic.sympy

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        if self.sp is None or not text:
            return []
        try:
            result = self._two_variable_primal_dual(text)
        except Exception:
            result = None
        return [result] if result is not None and result.verified else []

    def _two_variable_primal_dual(self, text: str) -> Optional[ToolResult]:
        if not re.search(r"线性规划|linear\s+program", text, re.I):
            return None
        if not re.search(r"对偶|dual", text, re.I):
            return None
        parsed = self._parse_program(text)
        if parsed is None:
            return None
        sense, variables, objective, matrix, bounds = parsed
        x, y = variables
        coefficients = self.sp.Matrix([
            self.sp.diff(objective, x),
            self.sp.diff(objective, y),
        ])
        if self.sp.simplify(objective - coefficients.dot(self.sp.Matrix([x, y]))) != 0:
            return None
        if any(item.is_positive is not True for item in coefficients):
            return None
        if any(item.is_nonnegative is not True for item in (*matrix, *bounds)):
            return None

        relation = ">=" if sense == "min" else "<="
        primal_vertices = self._feasible_vertices(matrix, bounds, relation)
        primal = self._unique_optimum(primal_vertices, coefficients, sense)
        if primal is None:
            return None
        primal_point, primal_value = primal

        dual_matrix = matrix.T
        dual_bounds = coefficients
        dual_objective = bounds
        dual_relation = "<=" if sense == "min" else ">="
        dual_sense = "max" if sense == "min" else "min"
        dual_vertices = self._feasible_vertices(
            dual_matrix,
            dual_bounds,
            dual_relation,
        )
        dual = self._unique_optimum(dual_vertices, dual_objective, dual_sense)
        if dual is None:
            return None
        dual_point, dual_value = dual
        if self.sp.simplify(primal_value - dual_value) != 0:
            return None

        primal_slack = (
            matrix * primal_point - bounds
            if relation == ">="
            else bounds - matrix * primal_point
        )
        dual_slack = (
            dual_bounds - dual_matrix * dual_point
            if dual_relation == "<="
            else dual_matrix * dual_point - dual_bounds
        )
        if any(
            self.sp.simplify(dual_point[index] * primal_slack[index]) != 0
            for index in range(2)
        ) or any(
            self.sp.simplify(primal_point[index] * dual_slack[index]) != 0
            for index in range(2)
        ):
            return None

        u, v = self.sp.Symbol("u"), self.sp.Symbol("v")
        operation = (
            "covering_linear_program_primal_dual"
            if sense == "min"
            else "packing_linear_program_primal_dual"
        )
        primal_text = self._pair(primal_point)
        dual_text = self._pair(dual_point)
        value_text = self.symbolic._format(primal_value)
        dual_objective_text = self.symbolic._format(
            bounds[0] * u + bounds[1] * v
        )
        dual_constraints = (
            self.symbolic._format(matrix[0, column] * u + matrix[1, column] * v)
            + (r"\le " if dual_relation == "<=" else r"\ge ")
            + self.symbolic._format(coefficients[column])
            for column in range(2)
        )
        dual_constraint_text = r",\;".join(dual_constraints)
        zh = bool(re.search(r"[\u4e00-\u9fff]", text))
        result = (
            rf"逐一求两条约束边界与坐标轴的交点并作精确可行性检查，得到唯一最优解 "
            rf"$({x},{y})={primal_text}$，最优值为 ${value_text}$。其对偶为 "
            rf"${dual_sense}\ {dual_objective_text}$，约束为 ${dual_constraint_text}$ 且 $u,v\ge0$；"
            rf"对偶最优解为 $(u,v)={dual_text}$。该点对偶可行，目标值 "
            rf"$d^*={value_text}=p^*$，且四个互补松弛乘积均为零，故由强对偶得到最优性。"
            if zh else
            rf"Exact enumeration of all intersections of the two constraint boundaries and the axes gives "
            rf"the unique optimum $({x},{y})={primal_text}$ with optimal value ${value_text}$. The dual is "
            rf"${dual_sense}\ {dual_objective_text}$ subject to ${dual_constraint_text}$ and $u,v\ge0$, "
            rf"with dual optimum $(u,v)={dual_text}$. It is dual feasible, has "
            rf"$d^*={value_text}=p^*$, and all four complementary-slackness products vanish; strong duality "
            rf"therefore certifies optimality."
        )
        return make_parameterized_tool_result(
            problem=text,
            operation=operation,
            result=result,
            result_kind="primal_dual_optimum",
            method="exact_vertex_enumeration_primal_dual_crosscheck",
            whole=True,
            written_support=True,
            checks=(
                "two_variables_and_two_constraints_parsed",
                "nonnegative_standard_form_checked",
                "all_primal_vertices_enumerated",
                "unique_primal_optimum_checked",
                "all_dual_vertices_enumerated",
                "unique_dual_optimum_checked",
                "primal_dual_objectives_equal",
                "complementary_slackness_checked",
            ),
            support=result,
            answer_shapes=("number", "expression", "text", "tuple"),
            requirements=(
                "result_present",
                "numeric_result",
                "reasoning",
                "dual_certificate",
                "dual_optimality_check",
            ),
        )

    def _parse_program(self, text: str):
        clean = str(text)
        replacements = {
            r"\left": "",
            r"\right": "",
            r"\min": "min",
            r"\max": "max",
            r"\geq": ">=",
            r"\ge": ">=",
            "≥": ">=",
            r"\leq": "<=",
            r"\le": "<=",
            "≤": "<=",
            r"\{": "{",
            r"\}": "}",
            "$": "",
        }
        for source, target in replacements.items():
            clean = clean.replace(source, target)
        clean = re.sub(r"\\(?:quad|qquad|,|;|!)", " ", clean)
        clean = re.sub(r"\\+\s+", " ", clean)
        objective_match = re.search(
            r"(?<![A-Za-z])(?P<sense>min|max)(?:imize)?\s*\{?\s*"
            r"(?P<objective>[^:：{}]+?)\s*(?:[:：]|\bsubject\s+to\b|\bs\.?t\.?)",
            clean,
            re.I,
        )
        if objective_match is None:
            return None
        sense = objective_match.group("sense").lower()
        objective = self._expr(objective_match.group("objective"))
        if objective is None or len(objective.free_symbols) != 2:
            return None
        variables = tuple(sorted(objective.free_symbols, key=lambda item: item.name))
        if any(len(variable.name) != 1 for variable in variables):
            return None
        x, y = variables
        constraints = clean[objective_match.end():].split("}", 1)[0]

        pair_nonnegative = re.compile(
            rf"(?:{re.escape(x.name)}\s*[,，]\s*{re.escape(y.name)}|"
            rf"{re.escape(y.name)}\s*[,，]\s*{re.escape(x.name)})\s*>=\s*0",
            re.I,
        )
        has_pair = pair_nonnegative.search(constraints) is not None
        has_x = re.search(rf"(?<![A-Za-z]){re.escape(x.name)}\s*>=\s*0", constraints) is not None
        has_y = re.search(rf"(?<![A-Za-z]){re.escape(y.name)}\s*>=\s*0", constraints) is not None
        if not has_pair and not (has_x and has_y):
            return None
        relation_text = pair_nonnegative.sub("", constraints)
        relation_text = re.sub(
            rf"(?<![A-Za-z])(?:{re.escape(x.name)}|{re.escape(y.name)})\s*>=\s*0",
            "",
            relation_text,
        )
        pieces = [
            piece.strip(" .")
            for piece in re.split(r"[,，;；]|\band\b|且|并且", relation_text, flags=re.I)
            if piece.strip(" .")
        ]
        rows = []
        bounds = []
        expected_relation = ">=" if sense == "min" else "<="
        for piece in pieces:
            match = re.fullmatch(
                r"(?P<lhs>.+?)\s*(?P<relation>>=|<=)\s*"
                r"(?P<rhs>[-+]?\d+(?:\s*/\s*\d+)?|[-+]?\.\d+)",
                piece,
            )
            if match is None or match.group("relation") != expected_relation:
                return None
            lhs = self._expr(match.group("lhs"))
            rhs = self._expr(match.group("rhs"))
            if lhs is None or rhs is None or rhs.free_symbols or lhs.free_symbols - {x, y}:
                return None
            polynomial = self.sp.Poly(lhs, x, y)
            if polynomial.total_degree() > 1:
                return None
            constant = polynomial.coeff_monomial(1)
            row = (
                polynomial.coeff_monomial(x),
                polynomial.coeff_monomial(y),
            )
            rows.append(row)
            bounds.append(self.sp.simplify(rhs - constant))
        if len(rows) != 2:
            return None
        matrix = self.sp.Matrix(rows)
        bound_vector = self.sp.Matrix(bounds)
        if matrix.det() == 0:
            return None
        return sense, variables, objective, matrix, bound_vector

    def _feasible_vertices(self, matrix, bounds, relation: str):
        boundaries = [
            (matrix[row, 0], matrix[row, 1], bounds[row])
            for row in range(2)
        ]
        boundaries.extend(((self.sp.Integer(1), self.sp.Integer(0), self.sp.Integer(0)),
                           (self.sp.Integer(0), self.sp.Integer(1), self.sp.Integer(0))))
        vertices = set()
        for first, second in combinations(boundaries, 2):
            coefficient_matrix = self.sp.Matrix((first[:2], second[:2]))
            if self.sp.simplify(coefficient_matrix.det()) == 0:
                continue
            point = coefficient_matrix.inv() * self.sp.Matrix((first[2], second[2]))
            point = self.sp.Matrix([self.sp.simplify(item) for item in point])
            if any(item.is_nonnegative is not True for item in point):
                continue
            residual = matrix * point - bounds
            feasible = (
                all(item.is_nonnegative is True for item in residual)
                if relation == ">="
                else all((-item).is_nonnegative is True for item in residual)
            )
            if feasible:
                vertices.add(tuple(point))
        return tuple(self.sp.Matrix(point) for point in vertices)

    def _unique_optimum(self, vertices, objective, sense: str):
        if not vertices:
            return None
        values = [self.sp.simplify(objective.dot(point)) for point in vertices]
        if any(value.is_real is not True for value in values):
            return None
        best_value = min(values) if sense == "min" else max(values)
        winners = [
            point for point, value in zip(vertices, values)
            if self.sp.simplify(value - best_value) == 0
        ]
        return (winners[0], best_value) if len(winners) == 1 else None

    def _expr(self, value: str):
        try:
            return self.sp.simplify(self.symbolic._parse(value))
        except Exception:
            return None

    def _pair(self, vector) -> str:
        return r"\left(" + ",".join(
            self.symbolic._format(item) for item in vector
        ) + r"\right)"
