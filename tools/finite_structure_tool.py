"""Exhaustive solvers for explicit small logic, set, and graph objects.

The module accepts only fully enumerated finite inputs from the current
statement.  It does not infer missing elements, edges, or semantic conditions.
Unsupported or ambiguous syntax is an abstention.
"""

from __future__ import annotations

from collections import deque
from itertools import product
from math import comb
import re
from typing import Optional

from tools.tool_contract import ToolResult, make_parameterized_tool_result


class FiniteStructureTool:
    """Mechanically certify bounded finite structures from explicit data."""

    _MAX_LOGIC_VARIABLES = 10
    _LOGIC_TOKEN = re.compile(
        r"\s*(<->|<=>|->|=>|↔|→|¬|∧|∨|~|!|&|\||"
        r"\\(?:neg|lnot|land|wedge|lor|vee|to|rightarrow|implies|"
        r"leftrightarrow|iff)\b|\(|\)|\b(?:not|and|or|implies|iff)\b|"
        r"[A-Za-z][A-Za-z0-9_]*|[01])",
        re.IGNORECASE,
    )

    def results_for(self, problem: str) -> list[ToolResult]:
        text = str(problem or "").strip()
        results: list[ToolResult] = []
        for compiler in (
            self._hypercube_spanning_trees,
            self._propositional_classification,
            self._finite_set_operation,
            self._explicit_graph_property,
        ):
            try:
                result = compiler(text)
            except Exception:
                result = None
            if result is not None and result.verified:
                results.append(result)
        return results

    def _hypercube_spanning_trees(self, text: str) -> Optional[ToolResult]:
        if not re.search(
            r"超立方体|hypercube|Q\s*_\s*\{?\s*\d+\s*\}?",
            text,
            re.IGNORECASE,
        ) or not re.search(r"生成树|spanning\s+trees?", text, re.IGNORECASE):
            return None
        dimension = self._hypercube_dimension(text)
        if dimension is None or not 1 <= dimension <= 64:
            return None
        changed_coordinates = re.search(
            r"(?:相差|不同|differ(?:ing)?\s+in)\s*(?:恰好|exactly)?\s*"
            r"([一二两三四五六七八九十\d]+|one|two|three|four|five|six|"
            r"seven|eight|nine|ten)\s*(?:个)?(?:坐标|coordinates?)",
            text,
            re.IGNORECASE,
        )
        if changed_coordinates is not None:
            parsed = self._small_integer(changed_coordinates.group(1))
            if parsed != 1:
                return None

        prime_exponents: dict[int, int] = {}
        for eigen_index in range(1, dimension + 1):
            multiplicity = comb(dimension, eigen_index)
            for prime, exponent in self._factor_integer(2 * eigen_index).items():
                prime_exponents[prime] = prime_exponents.get(prime, 0) + multiplicity * exponent
        prime_exponents[2] = prime_exponents.get(2, 0) - dimension
        if any(exponent < 0 for exponent in prime_exponents.values()):
            return None
        factors = [
            str(prime) if exponent == 1 else rf"{prime}^{{{exponent}}}"
            for prime, exponent in sorted(prime_exponents.items())
            if exponent
        ]
        factorization = " \\cdot ".join(factors) if factors else "1"
        count = 1
        for prime, exponent in prime_exponents.items():
            count *= prime ** exponent
        zh = self._is_chinese(text)
        result = str(count)
        support = (
            rf"$Q_{{{dimension}}}$ 的拉普拉斯特征值为 $2k$，重数为 "
            rf"$\binom{{{dimension}}}{{k}}$（$0\le k\le {dimension}$）。根据矩阵树定理， "
            rf"$\tau(Q_{{{dimension}}})=2^{{-{dimension}}}"
            rf"\prod_{{k=1}}^{{{dimension}}}(2k)^{{\binom{{{dimension}}}{{k}}}}$；"
            rf"逐项分解质因数得 ${factorization}={count}$。"
            if zh else
            rf"The Laplacian eigenvalues of $Q_{{{dimension}}}$ are $2k$ with "
            rf"multiplicity $\binom{{{dimension}}}{{k}}$ for $0\le k\le {dimension}$. "
            rf"The matrix-tree theorem gives $\tau(Q_{{{dimension}}})=2^{{-{dimension}}}"
            rf"\prod_{{k=1}}^{{{dimension}}}(2k)^{{\binom{{{dimension}}}{{k}}}}$; "
            rf"factoring every term gives ${factorization}={count}$."
        )
        return self._result(
            text,
            "hypercube_spanning_trees",
            result,
            "integer_factorization",
            "laplacian_spectrum_matrix_tree_prime_exponents",
            (
                "hypercube_dimension_parsed",
                "standard_one_coordinate_adjacency_confirmed",
                "all_laplacian_multiplicities_enumerated",
                "zero_eigenvalue_omitted",
                "matrix_tree_vertex_divisor_applied",
                "prime_exponents_recomputed",
            ),
            support,
            ("count", "number", "expression"),
            ("result_present", "count_conclusion", "reasoning"),
        )

    @staticmethod
    def _hypercube_dimension(text: str) -> Optional[int]:
        patterns = (
            r"Q\s*_\s*\{?\s*(\d+)\s*\}?",
            r"([一二两三四五六七八九十\d]+)\s*维\s*超立方体",
            r"(\d+)\s*[- ]?dimensional\s+hypercube",
            r"hypercube\s+(?:graph\s+)?Q\s*_?\s*\{?\s*(\d+)\s*\}?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return FiniteStructureTool._small_integer(match.group(1))
        return None

    @staticmethod
    def _small_integer(value: str) -> Optional[int]:
        token = str(value or "").strip()
        if token.isdigit():
            return int(token)
        digits = {
            "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        if token in digits:
            return digits[token]
        if token.startswith("十") and token[1:] in digits:
            return 10 + digits[token[1:]]
        if token.endswith("十") and token[:-1] in digits:
            return 10 * digits[token[:-1]]
        if "十" in token:
            left, right = token.split("十", 1)
            if left in digits and right in digits:
                return 10 * digits[left] + digits[right]
        return None

    @staticmethod
    def _factor_integer(value: int) -> dict[int, int]:
        remaining = int(value)
        factors: dict[int, int] = {}
        divisor = 2
        while divisor * divisor <= remaining:
            while remaining % divisor == 0:
                factors[divisor] = factors.get(divisor, 0) + 1
                remaining //= divisor
            divisor += 1 if divisor == 2 else 2
        if remaining > 1:
            factors[remaining] = factors.get(remaining, 0) + 1
        return factors

    def _propositional_classification(self, text: str) -> Optional[ToolResult]:
        requested = self._logic_request(text)
        if not requested or re.search(
            r"谓词|量词|forall|exists|\\forall|\\exists|predicate",
            text,
            re.IGNORECASE,
        ):
            return None
        formulas = [
            segment for segment in self._math_segments(text)
            if self._contains_logic_operator(segment)
        ]
        if len(formulas) != 1:
            return None
        parsed = self._parse_logic(formulas[0])
        if parsed is None:
            return None
        ast, variables = parsed
        if not 1 <= len(variables) <= self._MAX_LOGIC_VARIABLES:
            return None

        rows: list[tuple[dict[str, bool], bool]] = []
        for bits in product((False, True), repeat=len(variables)):
            assignment = dict(zip(variables, bits))
            rows.append((assignment, self._eval_logic(ast, assignment)))
        all_true = all(value for _, value in rows)
        all_false = all(not value for _, value in rows)
        classification = (
            "tautology" if all_true else
            "contradiction" if all_false else
            "contingency"
        )
        truth = {
            "tautology": all_true,
            "contradiction": all_false,
            "contingency": not all_true and not all_false,
        }[requested] if requested != "classify" else None
        zh = self._is_chinese(text)
        if requested == "classify":
            names = {
                "tautology": "重言式",
                "contradiction": "矛盾式",
                "contingency": "可满足但非重言的偶然式",
            }
            result = names[classification] if zh else classification
        else:
            target_names = {
                "tautology": "重言式" if zh else "a tautology",
                "contradiction": "矛盾式" if zh else "a contradiction",
                "contingency": "偶然式" if zh else "a contingency",
            }
            if zh:
                result = f"该命题公式{'是' if truth else '不是'}{target_names[requested]}。"
            else:
                result = f"The formula is{' ' if truth else ' not '}{target_names[requested]}."

        witnesses = []
        true_row = next((row for row in rows if row[1]), None)
        false_row = next((row for row in rows if not row[1]), None)
        if true_row is not None and not all_true:
            witnesses.append(self._assignment_text(true_row[0], True))
        if false_row is not None and not all_false:
            witnesses.append(self._assignment_text(false_row[0], False))
        if zh:
            support = f"已穷举 {len(rows)} 个真值赋值。" + (
                " 见证：" + "；".join(witnesses) + "。" if witnesses else ""
            )
        else:
            support = f"All {len(rows)} truth assignments were enumerated." + (
                " Witnesses: " + "; ".join(witnesses) + "." if witnesses else ""
            )
        return self._result(
            text,
            "propositional_formula_classification",
            result,
            "truth_classification",
            "complete_truth_table_enumeration",
            (
                "single_formula_parsed",
                "all_variables_identified",
                "all_truth_assignments_enumerated",
                "classification_recomputed",
            ),
            support,
            ("truth", "text", "expression"),
            ("result_present", "judgement"),
        )

    @staticmethod
    def _logic_request(text: str) -> str:
        patterns = (
            ("tautology", r"重言式|永真式|tautolog"),
            ("contradiction", r"矛盾式|永假式|contradiction"),
            ("contingency", r"偶然式|contingen"),
        )
        matches = [name for name, pattern in patterns if re.search(pattern, text, re.IGNORECASE)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) >= 2 and re.search(
            r"判断.*(?:属于|是哪|类型)|分类|classify|which\s+(?:type|one)",
            text,
            re.IGNORECASE,
        ):
            return "classify"
        return ""

    @staticmethod
    def _contains_logic_operator(value: str) -> bool:
        return bool(re.search(
            r"<->|<=>|->|=>|[↔→¬∧∨~!&|]|"
            r"\\(?:neg|lnot|land|wedge|lor|vee|to|rightarrow|implies|"
            r"leftrightarrow|iff)\b|\b(?:not|and|or|implies|iff)\b",
            str(value or ""),
            re.IGNORECASE,
        ))

    def _parse_logic(self, source: str):
        tokens: list[str] = []
        cursor = 0
        value = str(source or "").strip()
        while cursor < len(value):
            match = self._LOGIC_TOKEN.match(value, cursor)
            if not match:
                return None
            tokens.append(self._normalize_logic_token(match.group(1)))
            cursor = match.end()
        if not tokens:
            return None
        position = 0

        def parse_expression(minimum_precedence: int = 1):
            nonlocal position
            if position >= len(tokens):
                raise ValueError("missing operand")
            token = tokens[position]
            if token == "not":
                position += 1
                left = ("not", parse_expression(5))
            elif token == "(":
                position += 1
                left = parse_expression(1)
                if position >= len(tokens) or tokens[position] != ")":
                    raise ValueError("missing close parenthesis")
                position += 1
            elif re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*|[01]", token):
                position += 1
                left = ("const", token == "1") if token in {"0", "1"} else ("var", token)
            else:
                raise ValueError("unexpected token")

            precedence = {"iff": 1, "implies": 2, "or": 3, "and": 4}
            while position < len(tokens):
                operator = tokens[position]
                level = precedence.get(operator, 0)
                if level < minimum_precedence:
                    break
                position += 1
                next_minimum = level if operator == "implies" else level + 1
                right = parse_expression(next_minimum)
                left = (operator, left, right)
            return left

        try:
            ast = parse_expression()
        except ValueError:
            return None
        if position != len(tokens):
            return None
        variables = tuple(sorted(self._logic_variables(ast)))
        return ast, variables

    @staticmethod
    def _normalize_logic_token(token: str) -> str:
        value = token.casefold()
        if value in {"¬", "~", "!", r"\neg", r"\lnot", "not"}:
            return "not"
        if value in {"∧", "&", r"\land", r"\wedge", "and"}:
            return "and"
        if value in {"∨", "|", r"\lor", r"\vee", "or"}:
            return "or"
        if value in {"→", "->", "=>", r"\to", r"\rightarrow", r"\implies", "implies"}:
            return "implies"
        if value in {"↔", "<->", "<=>", r"\leftrightarrow", r"\iff", "iff"}:
            return "iff"
        return token

    @classmethod
    def _logic_variables(cls, ast) -> set[str]:
        if ast[0] == "var":
            return {ast[1]}
        if ast[0] == "const":
            return set()
        if ast[0] == "not":
            return cls._logic_variables(ast[1])
        return cls._logic_variables(ast[1]) | cls._logic_variables(ast[2])

    @classmethod
    def _eval_logic(cls, ast, assignment: dict[str, bool]) -> bool:
        operation = ast[0]
        if operation == "var":
            return assignment[ast[1]]
        if operation == "const":
            return bool(ast[1])
        if operation == "not":
            return not cls._eval_logic(ast[1], assignment)
        left = cls._eval_logic(ast[1], assignment)
        right = cls._eval_logic(ast[2], assignment)
        if operation == "and":
            return left and right
        if operation == "or":
            return left or right
        if operation == "implies":
            return (not left) or right
        if operation == "iff":
            return left == right
        raise ValueError("unsupported logic operation")

    @staticmethod
    def _assignment_text(assignment: dict[str, bool], value: bool) -> str:
        rendered = ", ".join(
            f"{name}={'T' if state else 'F'}" for name, state in assignment.items()
        )
        return f"({rendered}) gives {'T' if value else 'F'}"

    def _finite_set_operation(self, text: str) -> Optional[ToolResult]:
        normalized = self._normalize_explicit_data(text)
        assignments: dict[str, tuple[str, ...]] = {}
        for match in re.finditer(
            r"(?<![A-Za-z0-9_])([A-Za-z])\s*=\s*\{([^{}]*)\}",
            normalized,
        ):
            name = match.group(1)
            body = match.group(2).strip()
            if re.search(r"\.\.\.|…|\\(?:ldots|cdots|dots)|\bto\b|至", body, re.IGNORECASE):
                return None
            values = [] if not body else [item.strip() for item in re.split(r"[,，]", body)]
            if any(not re.fullmatch(r"[-+]?\d+|[A-Za-z][A-Za-z0-9_]*", item) for item in values):
                return None
            assignments[name] = tuple(dict.fromkeys(values))
        if len(assignments) < 2 or any(len(values) > 50 for values in assignments.values()):
            return None

        symbolic_patterns = (
            ("finite_set_symmetric_difference", r"([A-Za-z])\s*(?:\\triangle|△)\s*([A-Za-z])"),
            ("finite_set_cartesian_product", r"([A-Za-z])\s*(?:\\times|×)\s*([A-Za-z])"),
            ("finite_set_union", r"([A-Za-z])\s*(?:\\cup|∪)\s*([A-Za-z])"),
            ("finite_set_intersection", r"([A-Za-z])\s*(?:\\cap|∩)\s*([A-Za-z])"),
            ("finite_set_difference", r"([A-Za-z])\s*(?:\\setminus|\\backslash|∖)\s*([A-Za-z])"),
        )
        requests: list[tuple[str, str, str]] = []
        for operation, pattern in symbolic_patterns:
            for match in re.finditer(pattern, normalized):
                if match.group(1) in assignments and match.group(2) in assignments:
                    requests.append((operation, match.group(1), match.group(2)))
        if not requests:
            word_patterns = (
                ("finite_set_union", r"(?:并集|union\s+of)\s*([A-Za-z])\s*(?:和|与|and)\s*([A-Za-z])"),
                ("finite_set_intersection", r"(?:交集|intersection\s+of)\s*([A-Za-z])\s*(?:和|与|and)\s*([A-Za-z])"),
                ("finite_set_difference", r"(?:差集|difference\s+of)\s*([A-Za-z])\s*(?:与|和|and)\s*([A-Za-z])"),
                ("finite_set_symmetric_difference", r"(?:对称差|symmetric\s+difference\s+of)\s*([A-Za-z])\s*(?:与|和|and)\s*([A-Za-z])"),
                ("finite_set_cartesian_product", r"(?:笛卡尔积|Cartesian\s+product\s+of)\s*([A-Za-z])\s*(?:与|和|and)\s*([A-Za-z])"),
            )
            for operation, pattern in word_patterns:
                match = re.search(pattern, normalized, re.IGNORECASE)
                if match and match.group(1) in assignments and match.group(2) in assignments:
                    requests.append((operation, match.group(1), match.group(2)))
        requests = list(dict.fromkeys(requests))
        if len(requests) != 1:
            return None

        operation, left_name, right_name = requests[0]
        left, right = set(assignments[left_name]), set(assignments[right_name])
        if operation == "finite_set_union":
            values = self._sorted_atoms(left | right)
            answer = self._render_set(values)
        elif operation == "finite_set_intersection":
            values = self._sorted_atoms(left & right)
            answer = self._render_set(values)
        elif operation == "finite_set_difference":
            values = self._sorted_atoms(left - right)
            answer = self._render_set(values)
        elif operation == "finite_set_symmetric_difference":
            values = self._sorted_atoms(left ^ right)
            answer = self._render_set(values)
        else:
            if len(left) * len(right) > 2_500:
                return None
            pairs = [
                f"({first},{second})"
                for first in self._sorted_atoms(left)
                for second in self._sorted_atoms(right)
            ]
            answer = self._render_set(pairs, preserve_order=True)
        symbol = {
            "finite_set_union": r"\cup",
            "finite_set_intersection": r"\cap",
            "finite_set_difference": r"\setminus",
            "finite_set_symmetric_difference": r"\triangle",
            "finite_set_cartesian_product": r"\times",
        }[operation]
        result = rf"${left_name}{symbol}{right_name}={answer}$"
        support = (
            "已逐个枚举两个显式有限集合的元素并按所求集合运算核对成员资格。"
            if self._is_chinese(text) else
            "Membership was checked element by element in the two explicitly enumerated finite sets."
        )
        return self._result(
            text,
            operation,
            result,
            "finite_set",
            "explicit_finite_set_enumeration",
            (
                "both_sets_fully_enumerated",
                "single_set_operation_parsed",
                "all_output_memberships_recomputed",
            ),
            support,
            ("expression", "text", "count"),
            ("result_present",),
        )

    def _explicit_graph_property(self, text: str) -> Optional[ToolResult]:
        if re.search(
            r"有向图|有向边|加权|权重|多重图|伪图|directed|digraph|weighted|"
            r"multigraph|pseudograph|->|→",
            text,
            re.IGNORECASE,
        ):
            return None
        normalized = self._normalize_explicit_data(text)
        atom = r"[-+]?\d+|[A-Za-z][A-Za-z0-9_]*"
        vertex_match = re.search(
            r"(?<![A-Za-z0-9_])V\s*=\s*\{([^{}]*)\}",
            normalized,
            re.IGNORECASE,
        )
        edge_match = re.search(
            r"(?<![A-Za-z0-9_])E\s*=\s*\{([^{}]*)\}",
            normalized,
            re.IGNORECASE,
        )
        if not vertex_match or not edge_match:
            return None
        vertex_body, edge_body = vertex_match.group(1), edge_match.group(1)
        if re.search(r"\.\.\.|…|\\(?:ldots|cdots|dots)", vertex_body + edge_body):
            return None
        vertices = [item.strip() for item in re.split(r"[,，]", vertex_body) if item.strip()]
        if (
            not 1 <= len(vertices) <= 1_000
            or len(set(vertices)) != len(vertices)
            or any(not re.fullmatch(atom, item) for item in vertices)
        ):
            return None
        edge_pattern = re.compile(rf"\(\s*({atom})\s*[,，]\s*({atom})\s*\)")
        edge_matches = list(edge_pattern.finditer(edge_body))
        remainder = edge_pattern.sub("", edge_body)
        if re.sub(r"[\s,，]", "", remainder):
            return None
        if len(edge_matches) > 5_000:
            return None
        vertex_set = set(vertices)
        edges: list[tuple[str, str]] = []
        canonical_edges: set[tuple[str, str]] = set()
        for match in edge_matches:
            left, right = match.group(1), match.group(2)
            if left not in vertex_set or right not in vertex_set or left == right:
                return None
            canonical = tuple(self._sorted_atoms((left, right)))
            if canonical in canonical_edges:
                return None
            canonical_edges.add(canonical)
            edges.append((left, right))

        requests: list[str] = []
        target_patterns = (
            ("explicit_graph_connected", r"是否(?:是)?连通|判断[^。.?]{0,30}连通|(?:determine|decide)\s+whether[^.?]{0,40}connected|is\s+(?:the\s+)?graph\s+connected"),
            ("explicit_graph_tree", r"是否(?:为|是)树|判断[^。.?]{0,30}(?:为|是)树|(?:determine|decide)\s+whether[^.?]{0,40}(?:a\s+)?tree|is\s+(?:the\s+)?graph\s+a\s+tree"),
            ("explicit_graph_bipartite", r"是否(?:为|是)?(?:二部图|二分图)|判断[^。.?]{0,30}(?:二部图|二分图)|(?:determine|decide)\s+whether[^.?]{0,40}bipartite|is\s+(?:the\s+)?graph\s+bipartite"),
            ("explicit_graph_degree_sequence", r"度数序列|度序列|各顶点(?:的)?度(?:数)?|degree\s+sequence|degrees?\s+of\s+(?:all|the)\s+vertices"),
            ("explicit_graph_shortest_path", r"最短(?:路|路径)|shortest\s+path"),
            ("explicit_graph_euler_circuit", r"欧拉回路|欧拉环游|Euler(?:ian)?\s+(?:circuit|cycle)"),
            ("explicit_graph_euler_path", r"欧拉路径|欧拉通路|Euler(?:ian)?\s+(?:path|trail)"),
        )
        for operation, pattern in target_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                requests.append(operation)
        if len(requests) != 1:
            return None
        operation = requests[0]

        adjacency = {vertex: set() for vertex in vertices}
        for left, right in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        connected = self._connected_vertices(adjacency, vertices[0]) == vertex_set
        zh = self._is_chinese(text)

        if operation == "explicit_graph_connected":
            result = "该图连通。" if connected and zh else (
                "该图不连通。" if zh else
                "The graph is connected." if connected else "The graph is not connected."
            )
            support = (
                f"从顶点 {vertices[0]} 作 BFS，访问到 {len(self._connected_vertices(adjacency, vertices[0]))}/{len(vertices)} 个顶点。"
                if zh else
                f"BFS from vertex {vertices[0]} reaches {len(self._connected_vertices(adjacency, vertices[0]))}/{len(vertices)} vertices."
            )
        elif operation == "explicit_graph_tree":
            is_tree = connected and len(edges) == len(vertices) - 1
            result = "该图是树。" if is_tree and zh else (
                "该图不是树。" if zh else
                "The graph is a tree." if is_tree else "The graph is not a tree."
            )
            support = (
                f"连通性为 {'真' if connected else '假'}，且 |E|={len(edges)}、|V|-1={len(vertices)-1}；有限简单图由此判定树性。"
                if zh else
                f"Connectivity is {connected}, and |E|={len(edges)} while |V|-1={len(vertices)-1}; this characterizes finite simple trees."
            )
        elif operation == "explicit_graph_bipartite":
            bipartite = self._is_bipartite(adjacency)
            result = "该图是二部图。" if bipartite and zh else (
                "该图不是二部图。" if zh else
                "The graph is bipartite." if bipartite else "The graph is not bipartite."
            )
            support = (
                "已对每个连通分支执行二着色并检查每条边的两端颜色。"
                if zh else
                "Every component was two-colored and both endpoints of every edge were checked."
            )
        elif operation == "explicit_graph_degree_sequence":
            mapping = r",\;".join(
                rf"d({vertex})={len(adjacency[vertex])}"
                for vertex in self._sorted_atoms(vertices)
            )
            sequence = sorted((len(neighbors) for neighbors in adjacency.values()), reverse=True)
            result = rf"${mapping}$；度数序列为 ${tuple(sequence)}$。" if zh else rf"${mapping}$; the degree sequence is ${tuple(sequence)}$."
            support = (
                f"逐边累计端点度数并核对握手恒等式：度数和 {sum(sequence)}=2|E|={2*len(edges)}。"
                if zh else
                f"Endpoint incidences were counted edge by edge; the degree sum {sum(sequence)} equals 2|E|={2*len(edges)}."
            )
        elif operation == "explicit_graph_shortest_path":
            endpoints = self._shortest_path_endpoints(text, vertex_set)
            if endpoints is None:
                return None
            start, goal = endpoints
            path = self._shortest_path(adjacency, start, goal)
            if path is None:
                result = f"从 {start} 到 {goal} 不存在路径。" if zh else f"There is no path from {start} to {goal}."
                distance = "infinite"
            else:
                arrow = r"\to"
                rendered_path = arrow.join(path)
                distance = str(len(path) - 1)
                result = (
                    rf"最短距离为 {distance}，一条最短路径是 ${rendered_path}$。"
                    if zh else
                    rf"The shortest distance is {distance}; one shortest path is ${rendered_path}$."
                )
            support = (
                "按无权图 BFS 的层次首次到达目标；该层号即最短距离。"
                if zh else
                "Breadth-first search reaches the target first at the displayed distance layer."
            )
        else:
            active = [vertex for vertex in vertices if adjacency[vertex]]
            edge_connected = not active or self._connected_vertices(adjacency, active[0]) >= set(active)
            odd = self._sorted_atoms(vertex for vertex in vertices if len(adjacency[vertex]) % 2)
            if operation == "explicit_graph_euler_circuit":
                holds = edge_connected and not odd
                name_zh, name_en = "欧拉回路", "an Euler circuit"
            else:
                holds = edge_connected and len(odd) in {0, 2}
                name_zh, name_en = "欧拉路径", "an Euler path"
            result = (
                f"该图{'存在' if holds else '不存在'}{name_zh}。"
                if zh else
                f"The graph {'has' if holds else 'does not have'} {name_en}."
            )
            support = (
                f"含边顶点连通性为 {'真' if edge_connected else '假'}，奇度顶点为 {odd}；应用无向图 Euler 判据。"
                if zh else
                f"The non-isolated vertices are connected: {edge_connected}; odd-degree vertices are {odd}. The undirected Euler criterion applies."
            )

        return self._result(
            text,
            operation,
            result,
            "finite_graph_property",
            "complete_explicit_graph_enumeration",
            (
                "all_vertices_and_edges_parsed",
                "simple_undirected_graph_validated",
                "single_graph_target_parsed",
                "bounded_graph_algorithm_completed",
            ),
            support,
            ("truth", "text", "expression", "number"),
            ("result_present", "judgement"),
        )

    @staticmethod
    def _connected_vertices(adjacency: dict[str, set[str]], start: str) -> set[str]:
        reached = {start}
        queue = deque((start,))
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return reached

    @staticmethod
    def _is_bipartite(adjacency: dict[str, set[str]]) -> bool:
        colors: dict[str, bool] = {}
        for start in adjacency:
            if start in colors:
                continue
            colors[start] = False
            queue = deque((start,))
            while queue:
                vertex = queue.popleft()
                for neighbor in adjacency[vertex]:
                    if neighbor not in colors:
                        colors[neighbor] = not colors[vertex]
                        queue.append(neighbor)
                    elif colors[neighbor] == colors[vertex]:
                        return False
        return True

    @staticmethod
    def _shortest_path(
        adjacency: dict[str, set[str]], start: str, goal: str
    ) -> Optional[list[str]]:
        parents: dict[str, Optional[str]] = {start: None}
        queue = deque((start,))
        while queue and goal not in parents:
            vertex = queue.popleft()
            for neighbor in sorted(adjacency[vertex]):
                if neighbor not in parents:
                    parents[neighbor] = vertex
                    queue.append(neighbor)
        if goal not in parents:
            return None
        path = []
        current: Optional[str] = goal
        while current is not None:
            path.append(current)
            current = parents[current]
        return list(reversed(path))

    @staticmethod
    def _shortest_path_endpoints(text: str, vertices: set[str]) -> Optional[tuple[str, str]]:
        atom = r"[-+]?\d+|[A-Za-z][A-Za-z0-9_]*"
        patterns = (
            rf"(?:从|由)\s*({atom})\s*(?:到|至)\s*({atom})",
            rf"from\s+({atom})\s+to\s+({atom})",
            rf"between\s+({atom})\s+and\s+({atom})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.group(1) in vertices and match.group(2) in vertices:
                return match.group(1), match.group(2)
        return None

    @staticmethod
    def _normalize_explicit_data(text: str) -> str:
        value = str(text or "").replace("$", "")
        value = value.replace(r"\(", "").replace(r"\)", "")
        value = value.replace(r"\[", "").replace(r"\]", "")
        value = value.replace(r"\{", "{").replace(r"\}", "}")
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _sorted_atoms(values) -> list[str]:
        def key(value: str):
            if re.fullmatch(r"[-+]?\d+", value):
                return 0, int(value), value
            return 1, value.casefold(), value
        return sorted((str(item) for item in values), key=key)

    @staticmethod
    def _render_set(values, *, preserve_order: bool = False) -> str:
        rendered = list(values) if preserve_order else FiniteStructureTool._sorted_atoms(values)
        return r"\varnothing" if not rendered else r"\{" + ",".join(rendered) + r"\}"

    @staticmethod
    def _math_segments(text: str) -> tuple[str, ...]:
        segments = []
        for match in re.finditer(
            r"\$(?P<dollar>[^$\n]+)\$|\\\((?P<paren>.*?)\\\)|\\\[(?P<bracket>.*?)\\\]",
            str(text or ""),
            re.DOTALL,
        ):
            segment = match.group("dollar") or match.group("paren") or match.group("bracket") or ""
            if segment.strip():
                segments.append(segment.strip())
        return tuple(segments)

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
        )
