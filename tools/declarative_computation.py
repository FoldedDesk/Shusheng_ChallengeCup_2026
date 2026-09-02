"""Fail-closed execution of small, model-described mathematical contracts.

The model supplies data, never executable code.  This module accepts a small
JSON schema, checks that quoted source fragments occur in the current problem,
interprets expressions through an AST whitelist, and recomputes the result
locally.  A valid witness certifies the JSON contract only; it does not by
itself certify that the model translated the whole problem correctly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Iterator, Mapping


_MAX_JSON_CHARS = 12_000
_MAX_AST_NODES = 240
_MAX_STATES = 250_000
_MAX_SEQUENCE_STEPS = 100_000
_MAX_INTEGER_BITS = 200_000
_MARKER = re.compile(r"(?i)\bCOMPUTE_JSON\b\s*:?\s*")


def _normalize_expression_source(source: str) -> str:
    """Normalize narrow mathematical spellings into the safe expression IR."""
    text = str(source or "").strip().replace("^", "**")
    return re.sub(
        r"\bint\(\s*(['\"])\1\.join\(\s*map\(\s*str\s*,\s*"
        r"([A-Za-z][A-Za-z0-9_]*)\s*\)\s*\)\s*\)",
        r"digits_to_int(\2)",
        text,
    )


@dataclass(frozen=True)
class DeclarativeEligibility:
    eligible: bool
    score: int = 0
    reasons: tuple[str, ...] = ()

    def trace_content(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DeclarativeWitness:
    status: str
    reason: str
    kind: str = "none"
    answer: str = ""
    states_examined: int = 0
    grounded: bool = False
    grounding_count: int = 0
    ir_hash: str = ""
    semantic_hash: str = ""

    @property
    def usable(self) -> bool:
        return bool(
            self.status == "certified"
            and self.reason == "ok"
            and self.answer
            and self.grounded
            and self.ir_hash
        )

    def trace_content(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "kind": self.kind,
            "ir_hash": self.ir_hash,
            "semantic_hash": self.semantic_hash,
            "states_examined": self.states_examined,
            "grounding_count": self.grounding_count,
            "grounded": self.grounded,
            "answer_present": bool(self.answer),
        }


@dataclass(frozen=True)
class DeclarativeAgreement:
    """Fail-closed agreement between two independent contract translations.

    Local execution certifies each submitted IR, but it cannot certify that a
    model translated the natural-language statement correctly. Requiring two
    independently sampled compilers to emit the same normalized IR prevents a
    single invented reduction from becoming mathematical evidence. This is a
    translation gate, not a proof that the shared translation is correct.
    """

    status: str
    reason: str
    answer: str = ""
    ir_hash: str = ""
    semantic_hash: str = ""

    @property
    def usable(self) -> bool:
        return bool(
            self.status == "certified"
            and self.reason == "ok"
            and self.answer
            and (self.ir_hash or self.semantic_hash)
        )

    def trace_content(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "ir_hash": self.ir_hash,
            "semantic_hash": self.semantic_hash,
            "answer_present": bool(self.answer),
            "independent_compilers": 2,
        }


def agree_declarative_witnesses(
    first: DeclarativeWitness,
    second: DeclarativeWitness,
) -> DeclarativeAgreement:
    """Admit only byte-equivalent normalized IR with identical execution."""
    if not first.usable or not second.usable:
        return DeclarativeAgreement("rejected", "branch_not_certified")
    if first.answer != second.answer:
        # Equal normalized IR should execute identically. Keep this explicit
        # check as a tripwire for interpreter nondeterminism or corruption.
        return DeclarativeAgreement("rejected", "execution_disagreement")
    semantic_match = bool(
        first.semantic_hash
        and second.semantic_hash
        and first.semantic_hash == second.semantic_hash
    )
    if not semantic_match and first.ir_hash != second.ir_hash:
        return DeclarativeAgreement("rejected", "independent_ir_disagreement")
    return DeclarativeAgreement(
        "certified",
        "ok",
        answer=first.answer,
        ir_hash=first.ir_hash if first.ir_hash == second.ir_hash else "",
        semantic_hash=(first.semantic_hash if semantic_match else ""),
    )


class _ContractError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _reject(reason: str, *, kind: str = "none", ir_hash: str = "") -> DeclarativeWitness:
    return DeclarativeWitness("rejected", reason, kind=kind, ir_hash=ir_hash)


class _SafeExpression:
    """A small expression interpreter with no attribute or code access."""

    _CALLS = frozenset({
        "abs", "all_distinct", "comb", "digits_to_int", "factorial",
        "gcd", "len", "max", "min", "pow", "prod", "sum",
    })
    _BINOPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: Fraction(a, b),
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
    }
    _CMPOPS = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source.strip() or len(source) > 1_500:
            raise _ContractError("expression_syntax")
        source = _normalize_expression_source(source)
        try:
            self.tree = ast.parse(source.strip(), mode="eval")
        except (SyntaxError, ValueError):
            raise _ContractError("expression_syntax") from None
        nodes = tuple(ast.walk(self.tree))
        if len(nodes) > _MAX_AST_NODES:
            raise _ContractError("expression_complexity")
        forbidden = (
            ast.Attribute, ast.Await, ast.DictComp, ast.GeneratorExp,
            ast.Lambda, ast.ListComp, ast.NamedExpr, ast.SetComp,
        )
        if any(isinstance(node, forbidden) for node in nodes):
            raise _ContractError("expression_forbidden")

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        value = self._eval(self.tree.body, dict(env))
        self._check_value(value)
        return value

    def validate_predicate_structure(self) -> None:
        """Reject precedence-ambiguous modular predicates.

        Model compilers frequently intend ``(a+b) % m == r`` but emit
        ``a+b % m == r``, which silently applies the modulus to only ``b``.
        For a boolean contract, every explicit modulo must therefore be a
        complete comparison operand. More involved modular arithmetic can be
        expressed with ``pow(..., modulus)`` or split into unambiguous
        predicates.
        """
        modulo_nodes = {
            id(node)
            for node in ast.walk(self.tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
        }
        if not modulo_nodes:
            return
        admitted: set[int] = set()
        for comparison in (
            node for node in ast.walk(self.tree) if isinstance(node, ast.Compare)
        ):
            for operand in (comparison.left, *comparison.comparators):
                if isinstance(operand, ast.BinOp) and isinstance(operand.op, ast.Mod):
                    admitted.add(id(operand))
        if modulo_nodes != admitted:
            raise _ContractError("ambiguous_modulo_scope")

    def _eval(self, node: ast.AST, env: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            if type(node.value) not in {int, bool}:
                raise _ContractError("expression_literal")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise _ContractError("unknown_name")
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(item, env) for item in node.elts]
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.Not):
                return not value
            raise _ContractError("expression_operator")
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            if isinstance(node.op, ast.Pow):
                return self._power(left, right)
            operation = self._BINOPS.get(type(node.op))
            if operation is None:
                raise _ContractError("expression_operator")
            try:
                return operation(left, right)
            except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
                raise _ContractError("expression_runtime") from None
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(self._eval(value, env)) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(self._eval(value, env)) for value in node.values)
            raise _ContractError("expression_operator")
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, env)
                operation = self._CMPOPS.get(type(operator))
                if operation is None:
                    raise _ContractError("expression_operator")
                try:
                    if not operation(left, right):
                        return False
                except (TypeError, ValueError):
                    raise _ContractError("expression_runtime") from None
                left = right
            return True
        if isinstance(node, ast.Subscript):
            container = self._eval(node.value, env)
            if isinstance(node.slice, ast.Slice):
                raise _ContractError("expression_forbidden")
            index = self._eval(node.slice, env)
            if type(index) is not int:
                raise _ContractError("expression_index")
            try:
                return container[index]
            except (IndexError, KeyError, TypeError):
                raise _ContractError("expression_runtime") from None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._CALLS:
                raise _ContractError("unknown_call")
            if node.keywords:
                raise _ContractError("expression_forbidden")
            args = [self._eval(item, env) for item in node.args]
            return self._call(node.func.id, args)
        raise _ContractError("expression_forbidden")

    def _call(self, name: str, args: list[Any]) -> Any:
        try:
            if name == "abs" and len(args) == 1:
                return abs(args[0])
            if name == "len" and len(args) == 1:
                return len(args[0])
            if name in {"min", "max"} and args:
                values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
                return min(values) if name == "min" else max(values)
            if name == "sum" and len(args) == 1:
                return sum(args[0])
            if name == "prod" and len(args) == 1:
                return math.prod(args[0])
            if name == "gcd" and len(args) == 2:
                return math.gcd(self._int(args[0]), self._int(args[1]))
            if name == "comb" and len(args) == 2:
                return math.comb(self._int(args[0]), self._int(args[1]))
            if name == "factorial" and len(args) == 1:
                value = self._int(args[0])
                if not 0 <= value <= 10_000:
                    raise _ContractError("expression_range")
                return math.factorial(value)
            if name == "all_distinct" and len(args) == 1:
                return len(args[0]) == len(set(args[0]))
            if name == "digits_to_int" and len(args) == 1:
                digits = list(args[0])
                if not digits or any(type(x) is not int or not 0 <= x <= 9 for x in digits):
                    raise _ContractError("expression_range")
                return int("".join(str(x) for x in digits))
            if name == "pow" and len(args) == 2:
                return self._power(args[0], args[1])
            if name == "pow" and len(args) == 3:
                base, exponent, modulus = map(self._int, args)
                if exponent < 0 or modulus == 0 or exponent.bit_length() > 24:
                    raise _ContractError("expression_range")
                return pow(base, exponent, modulus)
        except _ContractError:
            raise
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            raise _ContractError("expression_runtime") from None
        raise _ContractError("call_signature")

    def _power(self, left: Any, right: Any) -> Any:
        exponent = self._int(right)
        if exponent < 0 or exponent > 10_000:
            raise _ContractError("expression_range")
        try:
            value = left ** exponent
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            raise _ContractError("expression_runtime") from None
        self._check_value(value)
        return value

    @staticmethod
    def _int(value: Any) -> int:
        if type(value) is not int:
            raise _ContractError("expression_type")
        return value

    @staticmethod
    def _check_value(value: Any) -> None:
        if isinstance(value, int) and value.bit_length() > _MAX_INTEGER_BITS:
            raise _ContractError("expression_range")
        if isinstance(value, Fraction):
            if (
                value.numerator.bit_length() > _MAX_INTEGER_BITS
                or value.denominator.bit_length() > _MAX_INTEGER_BITS
            ):
                raise _ContractError("expression_range")


class DeclarativeComputationTool:
    """Validate and execute one grounded declarative computation block."""

    _TOP_LEVEL = {
        "finite_enumeration": {"kind", "axes", "constraints", "aggregate", "source_fragments"},
        "sequence_recurrence": {"kind", "start_index", "initial", "target_index", "formula", "result", "source_fragments"},
        "state_recurrence": {"kind", "steps", "initial", "updates", "result", "source_fragments"},
        "coefficient": {"kind", "expression", "variable", "degree", "source_fragments"},
    }

    @classmethod
    def tool_schemas(cls) -> list[dict[str, Any]]:
        """Return provider function schemas; the interpreter remains authoritative."""
        domain = {
            "type": "object",
            "description": (
                "One exact finite domain: range with inclusive lower/upper, values, "
                "permutations/combinations with values and length, or product with "
                "values and repeat."
            ),
            "additionalProperties": True,
        }
        submit = {
            "type": "function",
            "function": {
                "name": "submit_declarative_contract",
                "description": (
                    "Submit one exact finite computation contract. Never include a "
                    "computed answer or prose."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "finite_enumeration", "sequence_recurrence",
                                "state_recurrence", "coefficient",
                            ],
                        },
                        "axes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "domain": domain,
                                },
                                "required": ["name", "domain"],
                                "additionalProperties": False,
                            },
                        },
                        "constraints": {
                            "description": "One expression string or an array of them.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                        },
                        "aggregate": {
                            "type": "object",
                            "description": "count, sum with value, or probability with event.",
                            "additionalProperties": True,
                        },
                        "start_index": {"type": "integer"},
                        "initial": {
                            "description": "Integer array for a sequence or integer object for state recurrence."
                        },
                        "target_index": {"type": "integer"},
                        "formula": {"type": "string"},
                        "updates": {"type": "object", "additionalProperties": {"type": "string"}},
                        "result": {"type": "string"},
                        "steps": {"type": "integer"},
                        "expression": {"type": "string"},
                        "variable": {"type": "string"},
                        "degree": {"type": "integer"},
                        "source_fragments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 6,
                            "description": "Short verbatim fragments copied from the problem.",
                        },
                    },
                    "required": ["kind", "source_fragments"],
                    "additionalProperties": False,
                },
            },
        }
        decline = {
            "type": "function",
            "function": {
                "name": "decline_declarative_contract",
                "description": "Use only when no allowed contract exactly covers the problem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "enum": ["unsupported_domain", "unsupported_operation", "ambiguous_translation"],
                        }
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        }
        return [submit, decline]

    @staticmethod
    def response_from_tool_message(message: Any) -> str:
        """Convert exactly one provider tool call to the text parser protocol."""
        if not isinstance(message, Mapping):
            return ""
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
            return ""
        function = calls[0].get("function")
        if not isinstance(function, Mapping):
            return ""
        name = str(function.get("name", ""))
        arguments = function.get("arguments")
        if name == "decline_declarative_contract":
            return "DECLINE"
        if name != "submit_declarative_contract":
            return ""
        if isinstance(arguments, Mapping):
            payload = dict(arguments)
        elif isinstance(arguments, str) and len(arguments) <= _MAX_JSON_CHARS:
            try:
                payload = json.loads(DeclarativeComputationTool._repair_json_escapes(arguments))
            except (TypeError, ValueError, json.JSONDecodeError):
                return ""
        else:
            return ""
        if not isinstance(payload, dict):
            return ""
        return "COMPUTE_JSON: " + json.dumps(payload, ensure_ascii=False)

    def eligibility(self, problem: str, spec: Any = None) -> DeclarativeEligibility:
        text = str(problem or "")
        folded = text.casefold()
        task = str(getattr(getattr(spec, "profile", None), "task_kind", ""))
        if task in {"proof", "derivation", "explanation", "construction"} or re.search(
            r"证明|推导|说明为什么|解释|举例证明|\b(?:prove|show that|derive|justify|explain)\b",
            text,
            re.IGNORECASE,
        ):
            return DeclarativeEligibility(False, 0, ("proof_or_exposition",))

        reasons: list[str] = []
        score = 0
        if re.search(r"多少|个数|共有|计数|几种|\b(?:how many|count|number of)\b", folded):
            score += 3
            reasons.append("finite_result_shape")
        if re.search(
            r"子集|排列|组合|字符串|数码|数字|整数|同余|模\s*\d|骰子|有限|"
            r"\b(?:subset|permutation|combination|string|digit|integer|modulo|dice|finite)\b",
            folded,
        ):
            score += 2
            reasons.append("finite_structure_signal")
        if re.search(r"递推|递归|初值|\b(?:recurrence|recursive|initial values?)\b", folded):
            score += 4
            reasons.append("recurrence_signal")
        elif re.search(
            r"[A-Za-z]\s*_\s*\{?n\}?\s*=[^\n]{0,160}"
            r"[A-Za-z]\s*_\s*\{?n\s*[-−]\s*1\}?|"
            r"(?:重复|迭代)[^。！？\n]{0,80}(?:次|步)|同时更新|"
            r"\b(?:repeat|iterate)[^.!?\n]{0,120}(?:times?|steps?)\b|"
            r"\bsimultaneous\s+update\b",
            text,
            re.IGNORECASE,
        ):
            score += 4
            reasons.append("recurrence_signal")
        if re.search(r"系数|\bcoefficient\b", folded):
            score += 4
            reasons.append("coefficient_signal")
        if re.search(r"概率|\bprobability\b", folded) and re.search(
            r"骰子|硬币|有限|\b(?:dice|coin|finite)\b", folded
        ):
            score += 2
            reasons.append("finite_probability_signal")
        if re.search(r"\d", text):
            score += 1
            reasons.append("explicit_parameter")
        eligible = bool(score >= 4 and len(text) <= 2_500)
        return DeclarativeEligibility(eligible, score, tuple(reasons))

    @staticmethod
    def system_prompt(language: str = "zh") -> str:
        schema = (
            'FINITE: COMPUTE_JSON: {"kind":"finite_enumeration",'
            '"axes":[{"name":"x","domain":{"type":"range","lower":0,'
            '"upper":10}}],"constraints":["x%2==0"],'
            '"aggregate":{"op":"count"},"source_fragments":["verbatim quote"]}\n'
            'RECURRENCE: COMPUTE_JSON: {"kind":"sequence_recurrence",'
            '"start_index":0,"initial":[0,1],"target_index":10,'
            '"formula":"a[n-1]+a[n-2]","result":"a[10]",'
            '"source_fragments":["verbatim quote"]}\n'
            'COEFFICIENT: COMPUTE_JSON: {"kind":"coefficient",'
            '"expression":"(1+x)**5","variable":"x","degree":2,'
            '"source_fragments":["verbatim quote"]}'
        )
        if language == "zh":
            return (
                "你是有限数学计算合同编译器，不是解题器。只在题目能被下述白名单完整、"
                "无歧义地表达时输出恰好一段 COMPUTE_JSON；否则只输出 DECLINE。禁止输出答案、"
                "Python、伪代码或解释。允许 kind: finite_enumeration, sequence_recurrence, "
                "state_recurrence, coefficient。表达式只允许整数、变量、算术、比较、布尔、索引及 "
                "abs/min/max/sum/prod/len/gcd/comb/factorial/pow/all_distinct/digits_to_int。"
                "source_fragments 必须逐字复制题面中能约束合同的短片段。range 的上下界均包含；"
                "模 m 的剩余类用 0 到 m-1；互异排列优先使用单个 permutations 域；不要先在文本中"
                "解题，也不要因为可枚举而拒绝。固定模式如下（示例数字仅说明字段）：\n" + schema
            )
        return (
            "Compile one finite mathematical computation contract. Emit exactly one "
            "COMPUTE_JSON object only when the problem is completely and unambiguously "
            "covered by the whitelist; otherwise emit DECLINE. Never emit an answer, "
            "Python, pseudocode, or explanation. Allowed kinds: finite_enumeration, "
            "sequence_recurrence, state_recurrence, coefficient. source_fragments must "
            "be short verbatim fragments copied from the problem. Range endpoints are "
            "inclusive; residues modulo m use 0 through m-1; distinct arrangements should "
            "use one permutations domain. Do not solve in prose and do not decline merely "
            "because exhaustive enumeration is needed. Fixed schemas (numbers are field "
            "examples only):\n" + schema
        )

    @classmethod
    def verification_system_prompt(cls, language: str = "zh") -> str:
        """Independent branch restricted to literal statement transcription."""
        base = cls.system_prompt(language)
        if language == "zh":
            guard = (
                "\n你是独立的严格转录核验器。每个轴、定义域、约束和聚合表达式都必须能从"
                "题面逐项直接转录；禁止自行发明几何坐标模型、计数公式、容斥、对称化、"
                "状态压缩、递推或任何未在题面明确给出的数学归约。若需要先证明或推导某个"
                "归约才能建立合同，必须 DECLINE。不要参考其他编译器的输出。"
            )
        else:
            guard = (
                "\nAct as an independent strict transcription verifier. Every axis, domain, "
                "constraint, and aggregate expression must be mechanically transcribed from "
                "the statement. Do not invent a coordinate model, counting formula, "
                "inclusion-exclusion reduction, symmetry reduction, compressed state, "
                "recurrence, or any other unstated mathematical reduction. If a proof or "
                "derivation is needed before the contract is valid, DECLINE. Do not rely on "
                "another compiler's output."
            )
        return base + guard

    @staticmethod
    def request(problem: str) -> str:
        return (
            "Problem:\n" + str(problem or "").strip() + "\n\n"
            "For finite_enumeration use axes with range/values/permutations/combinations/"
            "product domains, constraints as expression strings, and aggregate count, "
            "sum, or probability. Ranges are inclusive. For recurrences provide all "
            "initial data and an exact target. For coefficient use one variable and a "
            "nonnegative integer degree. A permutations domain is "
            '{"type":"permutations","values":[0,1,2],"length":3}; a product domain is '
            '{"type":"product","values":[0,1],"repeat":4}. If any condition cannot be '
            "expressed exactly, output DECLINE."
        )

    def execute_response(self, problem: str, raw_response: str) -> DeclarativeWitness:
        if str(raw_response or "").strip() == "DECLINE":
            return _reject("compiler_declined")
        payload = self._payload(raw_response)
        if payload is None:
            return _reject("no_unique_payload")
        payload = self._normalize_payload(payload)
        kind = str(payload.get("kind", "")).strip()
        allowed = self._TOP_LEVEL.get(kind)
        if allowed is None:
            return _reject("unsupported_kind", kind=kind or "none")
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        ir_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if set(payload) != allowed:
            return _reject("unknown_field", kind=kind, ir_hash=ir_hash)

        fragments = payload.get("source_fragments")
        if not isinstance(fragments, list) or not 1 <= len(fragments) <= 6:
            return _reject("source_fragments", kind=kind, ir_hash=ir_hash)
        if not all(isinstance(item, str) and 2 <= len(item.strip()) <= 300 for item in fragments):
            return _reject("source_fragments", kind=kind, ir_hash=ir_hash)
        if not self._grounded(problem, fragments):
            return _reject("ungrounded_fragment", kind=kind, ir_hash=ir_hash)

        try:
            semantic_hash = ""
            if kind == "finite_enumeration":
                answer, states, semantic_hash = self._finite(payload)
            elif kind == "sequence_recurrence":
                answer, states = self._sequence(payload)
            elif kind == "state_recurrence":
                answer, states = self._state(payload)
            else:
                answer, states = self._coefficient(payload)
        except _ContractError as exc:
            return _reject(exc.reason, kind=kind, ir_hash=ir_hash)
        except Exception:
            return _reject("execution_failure", kind=kind, ir_hash=ir_hash)

        return DeclarativeWitness(
            "certified",
            "ok",
            kind=kind,
            answer=self._render(answer),
            states_examined=states,
            grounded=True,
            grounding_count=len(fragments),
            ir_hash=ir_hash,
            semantic_hash=semantic_hash or ir_hash,
        )

    @staticmethod
    def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Repair only representation-level aliases without changing semantics.

        Function-calling models occasionally rename ``source_fragments`` or
        serialize a JSON array into the ``constraints`` string field.  Both
        repairs remain fail-closed: grounding and the exact top-level schema are
        checked after normalization, and all other unknown fields are retained
        so that the strict schema check rejects them.
        """
        normalized = dict(payload)
        if (
            "source_fragments" not in normalized
            and "source_factors" in normalized
        ):
            normalized["source_fragments"] = normalized.pop("source_factors")

        fragments = normalized.get("source_fragments")
        if isinstance(fragments, str):
            try:
                decoded_fragments = json.loads(fragments.strip())
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_fragments = None
            if (
                isinstance(decoded_fragments, list)
                and decoded_fragments
                and all(isinstance(item, str) for item in decoded_fragments)
            ):
                normalized["source_fragments"] = decoded_fragments

        initial = normalized.get("initial")
        if isinstance(initial, str):
            try:
                decoded_initial = json.loads(initial.strip())
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_initial = None
            if isinstance(decoded_initial, (list, dict)) and decoded_initial:
                normalized["initial"] = decoded_initial

        constraints = normalized.get("constraints")
        if isinstance(constraints, str):
            candidate = constraints.strip()
            if candidate.startswith("[") and candidate.endswith("]"):
                try:
                    decoded = json.loads(candidate)
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded = None
                if (
                    isinstance(decoded, list)
                    and decoded
                    and all(isinstance(item, str) for item in decoded)
                ):
                    normalized["constraints"] = decoded
                elif decoded is None:
                    try:
                        expression_list = ast.parse(candidate, mode="eval").body
                    except (SyntaxError, ValueError):
                        expression_list = None
                    if (
                        isinstance(expression_list, (ast.List, ast.Tuple))
                        and expression_list.elts
                    ):
                        normalized["constraints"] = [
                            ast.unparse(item) for item in expression_list.elts
                        ]
        return normalized

    @classmethod
    def strip_blocks(cls, value: str) -> str:
        text = str(value or "")
        match = _MARKER.search(text)
        if match is None:
            return text.strip()
        brace = text.find("{", match.end())
        end = cls._balanced_end(text, brace) if brace >= 0 else None
        if end is None:
            return text.strip()
        return (text[:match.start()] + text[end:]).strip()

    @classmethod
    def _payload(cls, raw_response: str) -> dict[str, Any] | None:
        text = str(raw_response or "").strip()
        markers = list(_MARKER.finditer(text))
        if len(markers) != 1 or len(text) > _MAX_JSON_CHARS + 200:
            return None
        marker = markers[0]
        brace = text.find("{", marker.end())
        if brace < 0 or brace - marker.end() > 32:
            return None
        end = cls._balanced_end(text, brace)
        if end is None or end - brace > _MAX_JSON_CHARS:
            return None
        prefix = text[:marker.start()].strip().strip("`").strip()
        suffix = text[end:].strip().strip("`").strip()
        if prefix or suffix:
            return None
        raw_json = cls._repair_json_escapes(text[brace:end])
        try:
            payload = json.loads(raw_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _balanced_end(value: str, start: int) -> int | None:
        if start < 0 or start >= len(value) or value[start] != "{":
            return None
        depth = 0
        quoted = escaped = False
        for index in range(start, len(value)):
            char = value[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
        return None

    @staticmethod
    def _repair_json_escapes(value: str) -> str:
        out: list[str] = []
        quoted = False
        index = 0
        valid = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
        while index < len(value):
            char = value[index]
            if char == '"':
                quoted = not quoted
                out.append(char)
                index += 1
                continue
            if quoted and char == "\\" and index + 1 < len(value):
                nxt = value[index + 1]
                if nxt not in valid:
                    out.append("\\\\")
                    index += 1
                    continue
                out.extend((char, nxt))
                index += 2
                continue
            out.append(char)
            index += 1
        return "".join(out)

    @classmethod
    def _grounded(cls, problem: str, fragments: Iterable[str]) -> bool:
        normalized_problem = cls._ground_key(problem)
        return bool(normalized_problem) and all(
            bool(key := cls._ground_key(fragment)) and key in normalized_problem
            for fragment in fragments
        )

    @staticmethod
    def _ground_key(value: str) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).casefold()
        replacements = {
            r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
            r"\equiv": "≡", r"\ne": "≠", r"\neq": "≠", r"\pmod": "mod",
            r"\mod": "mod", r"\to": "to", r"\ldots": "...", r"\dots": "...",
            r"\{": "{", r"\}": "}", "…": "...", "ℓ": "l",
        }
        for source in sorted(replacements, key=len, reverse=True):
            text = text.replace(source, replacements[source])
        text = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", text)
        text = text.replace("$", "").replace("\\(", "").replace("\\)", "")
        return re.sub(r"[^0-9a-z\u4e00-\u9fff≤≥≡≠+*/^<>=.-]+", "", text)

    def _finite(self, payload: Mapping[str, Any]) -> tuple[Any, int, str]:
        axes_raw = payload.get("axes")
        if not isinstance(axes_raw, list) or not 1 <= len(axes_raw) <= 16:
            raise _ContractError("axes")
        axes: list[tuple[str, tuple[Any, ...]]] = []
        names: set[str] = set()
        for item in axes_raw:
            if not isinstance(item, dict) or set(item) != {"name", "domain"}:
                raise _ContractError("axes")
            name = item.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name):
                raise _ContractError("axis_name")
            if name in names:
                raise _ContractError("axis_name")
            names.add(name)
            axes.append((name, self._domain(item.get("domain"))))

        constraints = self._constraints(payload.get("constraints"))
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, dict):
            raise _ContractError("aggregate")
        op = aggregate.get("op")
        allowed_aggregate = {
            "count": {"op"},
            "sum": {"op", "value"},
            "probability": {"op", "event"},
        }
        if op not in allowed_aggregate or set(aggregate) != allowed_aggregate[op]:
            raise _ContractError("aggregate")
        value_expr = _SafeExpression(aggregate["value"]) if op == "sum" and isinstance(aggregate.get("value"), str) else None
        if op == "sum" and value_expr is None:
            raise _ContractError("aggregate")
        events = self._constraints(aggregate.get("event")) if op == "probability" else ()

        expressions = tuple(_SafeExpression(item) for item in constraints)
        event_expressions = tuple(_SafeExpression(item) for item in events)
        for expression in (*expressions, *event_expressions):
            expression.validate_predicate_structure()
        assignments, state_count = self._assignment_iterator(axes, constraints)
        if state_count > _MAX_STATES:
            raise _ContractError("state_limit")

        accepted = favorable = 0
        total: Any = 0
        examined = 0
        behavior = hashlib.sha256()
        behavior.update(json.dumps(
            {
                "domains": [list(values) for _, values in axes],
                "aggregate": op,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
        for env in assignments:
            examined += 1
            passed = all(self._predicate(expr, env) for expr in expressions)
            state_record: list[Any] = [passed]
            if not passed:
                behavior.update(b"[false]")
                continue
            accepted += 1
            if op == "sum":
                value = value_expr.evaluate(env)
                total += value
                state_record.append(self._render(value))
            elif op == "probability":
                event = all(self._predicate(expr, env) for expr in event_expressions)
                favorable += int(event)
                state_record.append(event)
            behavior.update(json.dumps(
                state_record,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8"))
        if op == "count":
            answer: Any = accepted
        elif op == "sum":
            answer = total
        else:
            if accepted == 0:
                raise _ContractError("empty_sample_space")
            answer = Fraction(favorable, accepted)

        # A second traversal with reversed constraints is an independent
        # postcondition against accidental order-sensitive evaluation.
        reverse_expressions = tuple(reversed(expressions))
        reverse_events = tuple(reversed(event_expressions))
        assignments2, _ = self._assignment_iterator(axes, constraints)
        accepted2 = favorable2 = 0
        total2: Any = 0
        for env in assignments2:
            if not all(self._predicate(expr, env) for expr in reverse_expressions):
                continue
            accepted2 += 1
            if op == "sum":
                total2 += value_expr.evaluate(env)
            elif op == "probability" and all(
                self._predicate(expr, env) for expr in reverse_events
            ):
                favorable2 += 1
        check = accepted2 if op == "count" else total2 if op == "sum" else Fraction(favorable2, accepted2)
        if check != answer:
            raise _ContractError("postcondition")
        return answer, examined, behavior.hexdigest()

    @staticmethod
    def _predicate(expression: _SafeExpression, env: Mapping[str, Any]) -> bool:
        value = expression.evaluate(env)
        if type(value) is not bool:
            raise _ContractError("constraint_type")
        return value

    @staticmethod
    def _constraints(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if isinstance(value, dict) and set(value) == {"type", "expression"} and value.get("type") == "expression" and isinstance(value.get("expression"), str):
            return (value["expression"],)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        raise _ContractError("constraints")

    def _domain(self, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, dict):
            raise _ContractError("domain")
        if set(value) == {"range"} and isinstance(value.get("range"), list) and len(value["range"]) == 2:
            value = {"type": "range", "lower": value["range"][0], "upper": value["range"][1]}
        kind = value.get("type")
        if kind == "range" and set(value) == {"type", "lower", "upper"}:
            lower, upper = value.get("lower"), value.get("upper")
            if type(lower) is not int or type(upper) is not int or lower > upper:
                raise _ContractError("domain")
            if upper - lower + 1 > _MAX_STATES:
                raise _ContractError("state_limit")
            return tuple(range(lower, upper + 1))
        if kind == "values" and set(value) == {"type", "values"}:
            values = value.get("values")
            if not isinstance(values, list) or not values or len(values) > _MAX_STATES:
                raise _ContractError("domain")
            if any(type(item) not in {int, bool} for item in values):
                raise _ContractError("domain")
            return tuple(values)
        if kind in {"permutations", "combinations"} and set(value) == {"type", "values", "length"}:
            values, length = value.get("values"), value.get("length")
            if not isinstance(values, list) or type(length) is not int or not 0 <= length <= len(values):
                raise _ContractError("domain")
            if len(set(values)) != len(values) or any(type(item) is not int for item in values):
                raise _ContractError("domain")
            state_count = (
                math.perm(len(values), length)
                if kind == "permutations"
                else math.comb(len(values), length)
            )
            if state_count > _MAX_STATES:
                raise _ContractError("state_limit")
            iterator = itertools.permutations(values, length) if kind == "permutations" else itertools.combinations(values, length)
            result = tuple(iterator)
            if len(result) != state_count:
                raise _ContractError("state_limit")
            return result
        if kind == "product" and set(value) == {"type", "values", "repeat"}:
            values, repeat = value.get("values"), value.get("repeat")
            if not isinstance(values, list) or not values or type(repeat) is not int or not 0 <= repeat <= 30:
                raise _ContractError("domain")
            if any(type(item) is not int for item in values):
                raise _ContractError("domain")
            size = len(values) ** repeat
            if size > _MAX_STATES:
                raise _ContractError("state_limit")
            return tuple(itertools.product(values, repeat=repeat))
        raise _ContractError("domain")

    def _assignment_iterator(
        self,
        axes: list[tuple[str, tuple[Any, ...]]],
        constraints: tuple[str, ...],
    ) -> tuple[Iterator[dict[str, Any]], int]:
        optimized = self._distinct_assignments(axes, constraints)
        if optimized is not None:
            return optimized
        size = math.prod(len(values) for _, values in axes)
        if size > _MAX_STATES:
            raise _ContractError("state_limit")
        names = tuple(name for name, _ in axes)
        domains = tuple(values for _, values in axes)
        return (dict(zip(names, values)) for values in itertools.product(*domains)), size

    @staticmethod
    def _distinct_assignments(
        axes: list[tuple[str, tuple[Any, ...]]],
        constraints: tuple[str, ...],
    ) -> tuple[Iterator[dict[str, Any]], int] | None:
        if len(axes) < 2:
            return None
        names = [name for name, _ in axes]
        pattern = re.compile(r"all_distinct\s*\(\s*\[([^\]]+)\]\s*\)")
        joined = " and ".join(constraints)
        match = pattern.search(joined)
        if match is None:
            return None
        listed = [item.strip() for item in match.group(1).split(",")]
        if len(listed) != len(names) or set(listed) != set(names):
            return None
        union = sorted(set().union(*(set(values) for _, values in axes)))
        if len(union) != len(names) or any(not set(values) <= set(union) for _, values in axes):
            return None
        count = math.factorial(len(union))
        if count > _MAX_STATES:
            raise _ContractError("state_limit")

        def generate() -> Iterator[dict[str, Any]]:
            for values in itertools.permutations(union):
                env = dict(zip(names, values))
                if all(env[name] in domain for name, domain in axes):
                    yield env

        return generate(), count

    def _sequence(self, payload: Mapping[str, Any]) -> tuple[Any, int]:
        start = payload.get("start_index")
        initial = payload.get("initial")
        target = payload.get("target_index")
        formula = payload.get("formula")
        result = payload.get("result")
        if type(start) is not int or type(target) is not int or target < start:
            raise _ContractError("recurrence_bounds")
        if not isinstance(initial, list) or not initial or any(type(item) is not int for item in initial):
            raise _ContractError("recurrence_initial")
        if not isinstance(formula, str) or not isinstance(result, str):
            raise _ContractError("recurrence_formula")
        steps = target - start + 1 - len(initial)
        if steps < 0 or steps > _MAX_SEQUENCE_STEPS:
            raise _ContractError("state_limit")
        values = {start + offset: value for offset, value in enumerate(initial)}
        formula_expr = _SafeExpression(formula)
        for n in range(start + len(initial), target + 1):
            value = formula_expr.evaluate({"a": values, "n": n})
            if type(value) not in {int, Fraction}:
                raise _ContractError("recurrence_type")
            values[n] = value
        answer = _SafeExpression(result).evaluate({"a": values, "n": target})
        return answer, max(0, steps)

    def _state(self, payload: Mapping[str, Any]) -> tuple[Any, int]:
        steps = payload.get("steps")
        initial = payload.get("initial")
        updates = payload.get("updates")
        result = payload.get("result")
        if type(steps) is not int or not 0 <= steps <= _MAX_SEQUENCE_STEPS:
            raise _ContractError("state_limit")
        if not isinstance(initial, dict) or not initial or not isinstance(updates, dict):
            raise _ContractError("state_schema")
        if set(initial) != set(updates) or len(initial) > 24:
            raise _ContractError("state_schema")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", str(name)) for name in initial):
            raise _ContractError("state_schema")
        if any(type(value) is not int for value in initial.values()) or any(not isinstance(value, str) for value in updates.values()):
            raise _ContractError("state_schema")
        if not isinstance(result, str):
            raise _ContractError("state_schema")
        expressions = {name: _SafeExpression(value) for name, value in updates.items()}
        state: dict[str, Any] = dict(initial)
        for _ in range(steps):
            old = dict(state)
            state = {name: expression.evaluate(old) for name, expression in expressions.items()}
            if any(type(value) not in {int, Fraction} for value in state.values()):
                raise _ContractError("state_type")
        return _SafeExpression(result).evaluate(state), steps

    def _coefficient(self, payload: Mapping[str, Any]) -> tuple[Any, int]:
        expression = payload.get("expression")
        variable = payload.get("variable")
        degree = payload.get("degree")
        if not isinstance(expression, str) or not isinstance(variable, str) or not re.fullmatch(r"[A-Za-z]", variable):
            raise _ContractError("coefficient_schema")
        if type(degree) is not int or not 0 <= degree <= 2_000:
            raise _ContractError("coefficient_schema")
        expression = self._normalize_polynomial_source(expression, variable)
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError):
            raise _ContractError("expression_syntax") from None
        if len(tuple(ast.walk(tree))) > _MAX_AST_NODES:
            raise _ContractError("expression_complexity")
        poly = self._poly(tree.body, variable, degree)
        return poly.get(degree, Fraction(0)), len(poly)

    @staticmethod
    def _normalize_polynomial_source(expression: str, variable: str) -> str:
        text = str(expression or "").replace(" ", "").replace("^", "**")
        escaped = re.escape(variable)
        text = re.sub(rf"(?<=\d)(?={escaped}\b)", "*", text)
        text = re.sub(r"(?<=\d)(?=\()", "*", text)
        text = re.sub(rf"(?<=\))(?={escaped}\b|\d|\()", "*", text)
        text = re.sub(rf"(?<={escaped})(?=\()", "*", text)
        return text

    def _poly(self, node: ast.AST, variable: str, cap: int) -> dict[int, Fraction]:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return {0: Fraction(node.value)}
        if isinstance(node, ast.Name) and node.id == variable:
            return {1: Fraction(1)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._poly(node.operand, variable, cap)
            return value if isinstance(node.op, ast.UAdd) else {k: -v for k, v in value.items()}
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left = self._poly(node.left, variable, cap)
            right = self._poly(node.right, variable, cap)
            sign = 1 if isinstance(node.op, ast.Add) else -1
            result = dict(left)
            for key, value in right.items():
                result[key] = result.get(key, Fraction(0)) + sign * value
            return {key: value for key, value in result.items() if value}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return self._poly_mul(self._poly(node.left, variable, cap), self._poly(node.right, variable, cap), cap)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int or not 0 <= node.right.value <= 2_000:
                raise _ContractError("coefficient_power")
            base = self._poly(node.left, variable, cap)
            result = {0: Fraction(1)}
            exponent = node.right.value
            while exponent:
                if exponent & 1:
                    result = self._poly_mul(result, base, cap)
                exponent //= 2
                if exponent:
                    base = self._poly_mul(base, base, cap)
            return result
        raise _ContractError("expression_forbidden")

    @staticmethod
    def _poly_mul(left: Mapping[int, Fraction], right: Mapping[int, Fraction], cap: int) -> dict[int, Fraction]:
        result: dict[int, Fraction] = {}
        for a_degree, a_value in left.items():
            for b_degree, b_value in right.items():
                degree = a_degree + b_degree
                if degree <= cap:
                    result[degree] = result.get(degree, Fraction(0)) + a_value * b_value
        return {key: value for key, value in result.items() if value}

    @staticmethod
    def _render(value: Any) -> str:
        if isinstance(value, Fraction):
            if value.denominator == 1:
                return str(value.numerator)
            return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"
        if type(value) in {int, bool}:
            return str(value)
        raise _ContractError("result_type")
