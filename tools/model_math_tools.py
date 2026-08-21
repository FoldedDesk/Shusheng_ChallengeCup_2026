"""Schema-constrained local mathematics tools for model-assisted solving.

The model may choose arguments, but it cannot execute code or provide the
computed result.  Every operation is parsed and recomputed locally with hard
size limits.  A tool result certifies only the submitted operation; deriving
that operation from the problem remains the solver's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
from math import factorial, gcd, log2
import re
from typing import Any, Iterable, Mapping

from tools.sympy_tool import SympyTool
from tools.tool_contract import CertificateStatus


@dataclass(frozen=True)
class ModelToolExecution:
    name: str
    ok: bool
    result: str = ""
    reason: str = ""
    preconditions: tuple[str, ...] = ()
    execution_checks: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()

    @property
    def local_certificate_status(self) -> CertificateStatus:
        if (
            self.ok
            and self.preconditions
            and self.execution_checks
            and self.postconditions
        ):
            return CertificateStatus.CERTIFIED_TRUE
        return CertificateStatus.NOT_CERTIFIED

    def message_content(self) -> str:
        payload = {
            "status": "ok" if self.ok else "rejected",
            "operation": self.name,
            "local_certificate_status": self.local_certificate_status.value,
            # The model chose the operation and its arguments.  Local code
            # certifies that operation only, never its correspondence to the
            # original problem.
            "problem_goal_status": CertificateStatus.NOT_CERTIFIED.value,
        }
        if self.ok:
            payload["result"] = self.result
            payload["scope"] = "submitted_operation_only"
            payload["contract_phases"] = {
                "precondition": list(self.preconditions),
                "execution": list(self.execution_checks),
                "postcondition": list(self.postconditions),
            }
        else:
            payload["reason"] = self.reason or "invalid_arguments"
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ModelMathTools:
    """Execute a deliberately small whitelist of deterministic operations."""

    _MAX_ARGUMENT_CHARS = 8_000
    _MAX_EXPRESSION_CHARS = 700
    _MAX_MATRIX_CELLS = 400
    _MAX_DIGIT_LENGTH = 10_000
    _MAX_MODULUS = 200_000
    _MAX_ENUMERATION_STATES = 1_000_000
    _MAX_RECURRENCE_INDEX = 100_000
    _MAX_FINITE_STATES = 30
    _MAX_FINITE_STATE_WORK = 3_000_000
    _MAX_PERMUTATION_SIZE = 200
    _MAX_POLYGON_VERTICES = 200
    _VARIABLE = re.compile(r"[A-Za-z]")

    def __init__(self, symbolic: SympyTool | None = None) -> None:
        self.symbolic = symbolic or SympyTool()

    @classmethod
    def schemas(cls, allowed_names: Iterable[str] | None = None) -> list[dict]:
        expression = {
            "type": "string",
            "minLength": 1,
            "maxLength": cls._MAX_EXPRESSION_CHARS,
            "description": "A mathematical expression using standard notation.",
        }
        variable = {
            "type": "string",
            "pattern": "^[A-Za-z]$",
        }
        schemas = [
            cls._schema(
                "calculate_expression",
                "Evaluate one explicit numeric expression exactly.",
                {"expression": expression},
                ["expression"],
            ),
            cls._schema(
                "simplify_expression",
                "Factor, expand, or simplify one algebraic expression.",
                {
                    "expression": expression,
                    "mode": {"type": "string", "enum": ["simplify", "factor", "expand"]},
                },
                ["expression", "mode"],
            ),
            cls._schema(
                "solve_equation",
                "Solve one explicit univariate equation locally.",
                {
                    "left": expression,
                    "right": expression,
                    "variable": variable,
                    "domain": {"type": "string", "enum": ["real", "complex"]},
                },
                ["left", "right", "variable", "domain"],
            ),
            cls._schema(
                "substitute_values",
                "Substitute explicit values into an algebraic expression and simplify.",
                {
                    "expression": expression,
                    "values": {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 12,
                        "additionalProperties": expression,
                    },
                },
                ["expression", "values"],
            ),
            cls._schema(
                "solve_polynomial_system",
                "Solve a small explicit polynomial system and return all isolated solutions.",
                {
                    "equations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "items": expression,
                        "description": "Expressions equal to zero.",
                    },
                    "variables": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 4,
                        "uniqueItems": True,
                        "items": variable,
                    },
                    "domain": {"type": "string", "enum": ["real", "complex"]},
                },
                ["equations", "variables", "domain"],
            ),
            cls._schema(
                "differentiate_expression",
                "Differentiate an explicit expression.",
                {"expression": expression, "variable": variable},
                ["expression", "variable"],
            ),
            cls._schema(
                "definite_integral",
                "Compute an explicit one-dimensional definite integral.",
                {
                    "expression": expression,
                    "variable": variable,
                    "lower": expression,
                    "upper": expression,
                },
                ["expression", "variable", "lower", "upper"],
            ),
            cls._schema(
                "finite_sum",
                "Compute an explicit finite symbolic sum.",
                {
                    "expression": expression,
                    "variable": variable,
                    "lower": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                    "upper": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                },
                ["expression", "variable", "lower", "upper"],
            ),
            cls._schema(
                "limit_expression",
                "Compute one explicit univariate limit, including a one-sided limit when requested.",
                {
                    "expression": expression,
                    "variable": variable,
                    "point": expression,
                    "direction": {"type": "string", "enum": ["two-sided", "left", "right"]},
                },
                ["expression", "variable", "point", "direction"],
            ),
            cls._schema(
                "polynomial_coefficient",
                "Extract one exact coefficient from an explicit univariate polynomial or generating expression.",
                {
                    "expression": expression,
                    "variable": variable,
                    "degree": {"type": "integer", "minimum": 0, "maximum": 100000},
                },
                ["expression", "variable", "degree"],
            ),
            cls._schema(
                "linear_recurrence_term",
                "Iterate an exact constant-coefficient affine recurrence a_n=sum(c_j*a_(n-j))+b.",
                {
                    "coefficients": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": expression,
                    },
                    "initial_values": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": expression,
                    },
                    "constant": expression,
                    "target_index": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": cls._MAX_RECURRENCE_INDEX,
                    },
                },
                ["coefficients", "initial_values", "constant", "target_index"],
            ),
            cls._schema(
                "finite_state_walk_count",
                "Count weighted walks in an explicitly supplied finite-state transition system.",
                {
                    "transition_rows": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": cls._MAX_FINITE_STATES,
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": cls._MAX_FINITE_STATES,
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 1_000_000,
                            },
                        },
                    },
                    "initial_counts": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": cls._MAX_FINITE_STATES,
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1_000_000_000_000,
                        },
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100_000,
                    },
                    "accepting_states": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": cls._MAX_FINITE_STATES,
                        "uniqueItems": True,
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": cls._MAX_FINITE_STATES - 1,
                        },
                    },
                },
                ["transition_rows", "initial_counts", "steps", "accepting_states"],
            ),
            cls._schema(
                "subtraction_game_outcome",
                "Solve a finite normal-play subtraction game from one heap and list the winning first moves.",
                {
                    "initial_heap": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100_000,
                    },
                    "moves": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 256,
                        "uniqueItems": True,
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100_000,
                        },
                    },
                },
                ["initial_heap", "moves"],
            ),
            cls._schema(
                "permutation_cycle_count",
                "Count labelled permutations with explicit allowed cycle lengths and optional exact/minimum/maximum counts for selected lengths.",
                {
                    "size": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": cls._MAX_PERMUTATION_SIZE,
                    },
                    "allowed_cycle_lengths": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": cls._MAX_PERMUTATION_SIZE,
                        "uniqueItems": True,
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": cls._MAX_PERMUTATION_SIZE,
                        },
                    },
                    "cycle_count_bounds": {
                        "type": "array",
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "length": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": cls._MAX_PERMUTATION_SIZE,
                                },
                                "minimum": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": cls._MAX_PERMUTATION_SIZE,
                                },
                                "maximum": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": cls._MAX_PERMUTATION_SIZE,
                                },
                            },
                            "required": ["length", "minimum", "maximum"],
                            "additionalProperties": False,
                        },
                    },
                },
                ["size", "allowed_cycle_lengths", "cycle_count_bounds"],
            ),
            cls._schema(
                "lattice_polygon_interior",
                "For a simple polygon with integer vertices, compute twice its area, boundary lattice points, and interior lattice points using Pick's theorem.",
                {
                    "vertices": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": cls._MAX_POLYGON_VERTICES,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "type": "integer",
                                "minimum": -1_000_000,
                                "maximum": 1_000_000,
                            },
                        },
                    },
                },
                ["vertices"],
            ),
            cls._schema(
                "factorial_ratio_prime_valuation",
                "Compute the exponent of a specified prime in a product of factorials divided by another product of factorials.",
                {
                    "prime": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 1_000_000,
                    },
                    "numerator_factorials": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 100,
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1_000_000_000,
                        },
                    },
                    "denominator_factorials": {
                        "type": "array",
                        "maxItems": 100,
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1_000_000_000,
                        },
                    },
                },
                ["prime", "numerator_factorials", "denominator_factorials"],
            ),
            cls._schema(
                "modular_power_sum",
                "Evaluate a finite sum of terms c*b^(u^v*m+q) modulo an explicit modulus without expanding the powers.",
                {
                    "terms": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "coefficient": {
                                    "type": "integer",
                                    "minimum": -1_000_000_000_000,
                                    "maximum": 1_000_000_000_000,
                                },
                                "base": {
                                    "type": "integer",
                                    "minimum": -1_000_000_000_000,
                                    "maximum": 1_000_000_000_000,
                                },
                                "exponent_base": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1_000_000_000,
                                },
                                "exponent_power": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100_000,
                                },
                                "exponent_multiplier": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 1_000_000_000,
                                },
                                "exponent_offset": {
                                    "type": "integer",
                                    "minimum": -1_000_000_000,
                                    "maximum": 1_000_000_000,
                                },
                            },
                            "required": [
                                "coefficient", "base", "exponent_base",
                                "exponent_power", "exponent_multiplier",
                                "exponent_offset"
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "modulus": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2_000_000_000,
                    },
                },
                ["terms", "modulus"],
            ),
            cls._schema(
                "bounded_integer_search",
                "Exhaustively count or optimize a small explicitly bounded integer domain using exact equations, inequalities, and congruences.",
                {
                    "variables": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "items": variable,
                    },
                    "lower_bounds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                    },
                    "upper_bounds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                    },
                    "equations": {
                        "type": "array",
                        "maxItems": 8,
                        "items": expression,
                        "description": "Expressions required to equal zero.",
                    },
                    "inequalities": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": expression,
                                "relation": {"type": "string", "enum": ["<", "<=", ">", ">="]},
                            },
                            "required": ["expression", "relation"],
                            "additionalProperties": False,
                        },
                    },
                    "congruences": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "properties": {
                                "expression": expression,
                                "modulus": {"type": "integer", "minimum": 1, "maximum": cls._MAX_MODULUS},
                                "remainder": {"type": "integer", "minimum": 0, "maximum": cls._MAX_MODULUS},
                            },
                            "required": ["expression", "modulus", "remainder"],
                            "additionalProperties": False,
                        },
                    },
                    "operation": {"type": "string", "enum": ["count", "list", "minimize", "maximize"]},
                    "objective": expression,
                },
                [
                    "variables", "lower_bounds", "upper_bounds", "equations",
                    "inequalities", "congruences", "operation", "objective",
                ],
            ),
            cls._schema(
                "matrix_operation",
                "Perform an exact operation on one explicit matrix; permanent is available for a square rational matrix of order at most 18.",
                {
                    "rows": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 20,
                            "items": {"oneOf": [{"type": "number"}, expression]},
                        },
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["determinant", "rank", "inverse", "eigenvalues", "permanent"],
                    },
                },
                ["rows", "operation"],
            ),
            cls._schema(
                "count_digit_strings",
                "Count positive base-10 integers with allowed digits and a modular condition using exact digit DP.",
                {
                    "minimum_length": {"type": "integer", "minimum": 1, "maximum": cls._MAX_DIGIT_LENGTH},
                    "maximum_length": {"type": "integer", "minimum": 1, "maximum": cls._MAX_DIGIT_LENGTH},
                    "digits": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 10,
                        "uniqueItems": True,
                        "items": {"type": "integer", "minimum": 0, "maximum": 9},
                    },
                    "modulus": {"type": "integer", "minimum": 1, "maximum": cls._MAX_MODULUS},
                    "remainder": {"type": "integer", "minimum": 0, "maximum": cls._MAX_MODULUS},
                    "leading_zero_allowed": {"type": "boolean"},
                },
                [
                    "minimum_length", "maximum_length", "digits", "modulus",
                    "remainder", "leading_zero_allowed",
                ],
            ),
            cls._schema(
                "count_modular_solutions",
                "Enumerate residues satisfying one explicit congruence.",
                {
                    "expression": expression,
                    "variable": variable,
                    "modulus": {"type": "integer", "minimum": 1, "maximum": cls._MAX_MODULUS},
                    "remainder": {"type": "integer", "minimum": 0, "maximum": cls._MAX_MODULUS},
                },
                ["expression", "variable", "modulus", "remainder"],
            ),
        ]
        if allowed_names is None:
            return schemas
        allowed = {str(name) for name in allowed_names}
        return [
            schema for schema in schemas
            if schema.get("function", {}).get("name") in allowed
        ]

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def execute_call(self, call: Mapping[str, Any]) -> ModelToolExecution:
        function = call.get("function", {}) if isinstance(call, Mapping) else {}
        if not isinstance(function, Mapping):
            return ModelToolExecution("unknown", False, reason="invalid_function")
        name = str(function.get("name", ""))
        raw = function.get("arguments", "")
        if not isinstance(raw, str) or len(raw) > self._MAX_ARGUMENT_CHARS:
            return ModelToolExecution(name or "unknown", False, reason="invalid_arguments")
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ModelToolExecution(name or "unknown", False, reason="invalid_json")
        if not isinstance(arguments, dict):
            return ModelToolExecution(name or "unknown", False, reason="invalid_arguments")
        handlers = {
            "calculate_expression": self._calculate,
            "simplify_expression": self._simplify,
            "solve_equation": self._solve_equation,
            "substitute_values": self._substitute_values,
            "solve_polynomial_system": self._solve_polynomial_system,
            "differentiate_expression": self._differentiate,
            "definite_integral": self._definite_integral,
            "finite_sum": self._finite_sum,
            "limit_expression": self._limit_expression,
            "polynomial_coefficient": self._polynomial_coefficient,
            "linear_recurrence_term": self._linear_recurrence_term,
            "finite_state_walk_count": self._finite_state_walk_count,
            "subtraction_game_outcome": self._subtraction_game_outcome,
            "permutation_cycle_count": self._permutation_cycle_count,
            "lattice_polygon_interior": self._lattice_polygon_interior,
            "factorial_ratio_prime_valuation": self._factorial_ratio_prime_valuation,
            "modular_power_sum": self._modular_power_sum,
            "bounded_integer_search": self._bounded_integer_search,
            "matrix_operation": self._matrix_operation,
            "count_digit_strings": self._count_digit_strings,
            "count_modular_solutions": self._count_modular_solutions,
        }
        handler = handlers.get(name)
        if handler is None:
            return ModelToolExecution(name or "unknown", False, reason="unknown_operation")
        try:
            result = handler(arguments)
        except Exception:
            return ModelToolExecution(name, False, reason="computation_failed")
        if result is None or not str(result).strip():
            return ModelToolExecution(name, False, reason="unsupported_or_unsolved")
        postconditions = self._postcheck(name, arguments, str(result).strip())
        return ModelToolExecution(
            name,
            True,
            str(result).strip(),
            preconditions=(
                "registered_operation",
                "schema_and_bounds_validated",
            ),
            execution_checks=("deterministic_handler_completed",),
            postconditions=postconditions,
        )

    def execute_contract(
        self,
        raw_contract: str,
        *,
        allowed_names: Iterable[str],
    ) -> ModelToolExecution:
        """Parse one strict JSON contract and execute its whitelisted operation."""
        text = str(raw_contract or "").strip()
        if not text or len(text) > self._MAX_ARGUMENT_CHARS * 2:
            return ModelToolExecution("unknown", False, reason="missing_contract")
        payload = self._extract_contract_object(text)
        if payload is None:
            return ModelToolExecution("unknown", False, reason="invalid_contract_json")
        if set(payload) == {"status", "reason"} and payload.get("status") == "ABSTAIN":
            return ModelToolExecution("abstain", False, reason="model_abstained")
        if set(payload) != {"status", "operation", "arguments"}:
            return ModelToolExecution("unknown", False, reason="invalid_contract_shape")
        if payload.get("status") != "CALL":
            return ModelToolExecution("unknown", False, reason="invalid_contract_status")
        operation = str(payload.get("operation", ""))
        allowed = {str(name) for name in allowed_names}
        if not operation or operation not in allowed:
            return ModelToolExecution(operation or "unknown", False, reason="operation_not_allowed")
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            return ModelToolExecution(operation, False, reason="invalid_contract_arguments")
        return self.execute_call({
            "id": "explicit-local-contract",
            "type": "function",
            "function": {
                "name": operation,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    @staticmethod
    def _extract_contract_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for position, character in enumerate(text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[position:])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "status" in payload:
                return payload
        return None

    def _postcheck(
        self,
        name: str,
        arguments: dict,
        result: str,
    ) -> tuple[str, ...]:
        """Independently validate the returned local fact when feasible.

        Unsupported postchecks intentionally return an empty tuple, which
        keeps the operation at NOT_CERTIFIED even though its deterministic
        result may still be shown to the solver as untrusted advisory data.
        """
        try:
            if name == "matrix_operation" and arguments.get("operation") == "permanent":
                values = self._rational_matrix(arguments.get("rows"), maximum_order=18)
                if values is None:
                    return ()
                observed = self.symbolic._parse(result)
                expected = self._permanent_glynn(values)
                if self.symbolic.sympy.simplify(observed - expected) == 0:
                    return (
                        "square_rational_matrix_reparsed",
                        "glynn_formula_crosscheck_passed",
                    )
                return ()
            if name == "linear_recurrence_term":
                repeated = self._linear_recurrence_term(arguments)
                if repeated is not None and self._equivalent_scalar(result, repeated):
                    return (
                        "recurrence_contract_reparsed",
                        "target_term_recomputed",
                    )
                return ()
            if name == "finite_state_walk_count":
                repeated = self._finite_state_walk_count_matrix(arguments)
                if repeated is not None and self._equivalent_json(result, repeated):
                    return (
                        "finite_transition_contract_reparsed",
                        "matrix_power_crosscheck_passed",
                    )
                return ()
            if name == "subtraction_game_outcome":
                repeated = self._subtraction_game_outcome_grundy(arguments)
                if repeated is not None and self._equivalent_json(result, repeated):
                    return (
                        "normal_play_move_contract_reparsed",
                        "grundy_crosscheck_passed",
                    )
                return ()
            if name == "permutation_cycle_count":
                repeated = self._permutation_cycle_count_reversed(arguments)
                if repeated is not None and self._equivalent_scalar(result, repeated):
                    return (
                        "cycle_inventory_contract_reparsed",
                        "reversed_egf_coefficient_crosscheck_passed",
                    )
                return ()
            if name == "lattice_polygon_interior":
                repeated = self._lattice_polygon_interior_reversed(arguments)
                if repeated is not None and self._equivalent_json(result, repeated):
                    return (
                        "simple_integer_polygon_reparsed",
                        "reverse_orientation_pick_crosscheck_passed",
                    )
                return ()
            if name == "factorial_ratio_prime_valuation":
                repeated = self._factorial_ratio_prime_valuation_digits(arguments)
                if repeated is not None and self._equivalent_scalar(result, repeated):
                    return (
                        "prime_and_factorial_contract_reparsed",
                        "base_p_digit_sum_crosscheck_passed",
                    )
                return ()
            if name == "modular_power_sum":
                repeated = self._modular_power_sum(arguments, reverse=True)
                if repeated is not None and str(repeated) == str(result):
                    return (
                        "modular_power_terms_reparsed",
                        "reverse_term_residue_crosscheck_passed",
                    )
                return ()
            if name == "bounded_integer_search":
                repeated = self._bounded_integer_search(arguments)
                if repeated is not None and self._equivalent_json(result, repeated):
                    return (
                        "finite_domain_reenumerated",
                        "count_and_optimizers_recomputed",
                    )
                return ()
            if name == "count_digit_strings":
                repeated = self._count_digit_strings_sparse(arguments)
                if repeated is not None and str(repeated) == str(result):
                    return (
                        "digit_automaton_contract_reparsed",
                        "sparse_state_dp_crosscheck_passed",
                    )
                return ()
            if name == "count_modular_solutions":
                repeated = self._count_modular_solutions(arguments)
                if repeated is not None and self._equivalent_json(result, repeated):
                    return (
                        "complete_residue_system_reenumerated",
                        "solution_count_and_residues_recomputed",
                    )
                return ()
            if name == "calculate_expression":
                expression = self._expression(arguments.get("expression"))
                if expression is not None and self._equivalent_scalar(expression, result):
                    return ("numeric_expression_reparsed", "exact_value_subtracted_to_zero")
                return ()
            if name == "simplify_expression":
                expression = self._expression(arguments.get("expression"))
                if expression is not None and self._equivalent_scalar(expression, result):
                    return ("source_expression_reparsed", "symbolic_difference_is_zero")
                return ()
        except Exception:
            return ()
        return ()

    def _equivalent_scalar(self, first: str, second: str) -> bool:
        if self.symbolic.sympy is None:
            return False
        left = self.symbolic._parse(str(first))
        right = self.symbolic._parse(str(second))
        return self.symbolic.sympy.simplify(left - right) == 0

    @staticmethod
    def _equivalent_json(first: str, second: str) -> bool:
        try:
            return json.loads(first) == json.loads(second)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False

    def _calculate(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression"}:
            return None
        expression = self._expression(arguments["expression"])
        if expression is None:
            return None
        value = self.symbolic.evaluate(expression)
        if value is None or self._free_symbols(expression):
            return None
        return value

    def _simplify(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "mode"} or self.symbolic.sympy is None:
            return None
        expression = self._expression(arguments["expression"])
        mode = arguments.get("mode")
        if expression is None or mode not in {"simplify", "factor", "expand"}:
            return None
        parsed = self.symbolic._parse(expression)
        operation = getattr(self.symbolic.sympy, mode)
        value = operation(parsed)
        if mode in {"factor", "expand"}:
            return self.symbolic.sympy.latex(value)
        return self.symbolic._format(value)

    def _solve_equation(self, arguments: dict) -> str | None:
        if set(arguments) != {"left", "right", "variable", "domain"}:
            return None
        left = self._expression(arguments["left"])
        right = self._expression(arguments["right"])
        variable = self._variable(arguments["variable"])
        domain = arguments.get("domain")
        if None in {left, right, variable} or domain not in {"real", "complex"}:
            return None
        if self.symbolic.sympy is None:
            return None
        symbol = self.symbolic.sympy.Symbol(variable)
        raw_solutions = self.symbolic.sympy.solve(
            self.symbolic._parse(f"({left})-({right})"), symbol
        )
        if domain == "real":
            raw_solutions = [
                solution for solution in raw_solutions
                if solution.is_real is not False
            ]
        solutions = [self.symbolic._format(solution) for solution in raw_solutions]
        return json.dumps(
            {"solutions": solutions, "count": len(solutions)},
            ensure_ascii=False,
        )

    def _substitute_values(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "values"} or self.symbolic.sympy is None:
            return None
        expression = self._expression(arguments["expression"])
        values = arguments.get("values")
        if expression is None or not isinstance(values, dict) or not 1 <= len(values) <= 12:
            return None
        substitutions = {}
        for raw_name, raw_value in values.items():
            name = self._variable(raw_name)
            value = self._expression(raw_value)
            if name is None or value is None:
                return None
            substitutions[self.symbolic.sympy.Symbol(name)] = self.symbolic._parse(value)
        result = self.symbolic._parse(expression).subs(substitutions)
        return self.symbolic._format(result)

    def _solve_polynomial_system(self, arguments: dict) -> str | None:
        if set(arguments) != {"equations", "variables", "domain"} or self.symbolic.sympy is None:
            return None
        equations = arguments.get("equations")
        variables = arguments.get("variables")
        domain = arguments.get("domain")
        if (
            not isinstance(equations, list) or not 1 <= len(equations) <= 4
            or not isinstance(variables, list) or not 1 <= len(variables) <= 4
            or len(set(variables)) != len(variables)
            or domain not in {"real", "complex"}
        ):
            return None
        names = [self._variable(item) for item in variables]
        expressions = [self._expression(item) for item in equations]
        if any(item is None for item in names) or any(item is None for item in expressions):
            return None
        symbols = [self.symbolic.sympy.Symbol(item) for item in names]
        parsed = [self.symbolic._parse(item) for item in expressions]
        if any(not item.is_polynomial(*symbols) for item in parsed):
            return None
        solutions = self.symbolic.sympy.solve_poly_system(parsed, *symbols)
        if solutions is None or len(solutions) > 2_000:
            return None
        rendered = []
        for solution in solutions:
            if domain == "real" and any(value.is_real is False for value in solution):
                continue
            rendered.append([self.symbolic._format(value) for value in solution])
        return json.dumps({"variables": names, "solutions": rendered, "count": len(rendered)}, ensure_ascii=False)

    def _differentiate(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "variable"}:
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        return None if expression is None or variable is None else self.symbolic.derivative(expression, variable)

    def _definite_integral(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "variable", "lower", "upper"}:
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        lower = self._expression(arguments["lower"])
        upper = self._expression(arguments["upper"])
        if None in {expression, variable, lower, upper}:
            return None
        return self.symbolic.definite_integral(expression, variable, lower, upper)

    def _finite_sum(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "variable", "lower", "upper"} or self.symbolic.sympy is None:
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        lower = arguments.get("lower")
        upper = arguments.get("upper")
        if (
            expression is None or variable is None
            or not self._bounded_integer(lower, -1_000_000, 1_000_000)
            or not self._bounded_integer(upper, -1_000_000, 1_000_000)
            or lower > upper or upper - lower > 2_000_000
        ):
            return None
        symbol = self.symbolic.sympy.Symbol(variable)
        value = self.symbolic.sympy.summation(
            self.symbolic._parse(expression), (symbol, lower, upper)
        )
        if value.has(self.symbolic.sympy.Sum):
            return None
        return self.symbolic._format(value)

    def _limit_expression(self, arguments: dict) -> str | None:
        if (
            set(arguments) != {"expression", "variable", "point", "direction"}
            or self.symbolic.sympy is None
        ):
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        point = self._expression(arguments["point"])
        direction = arguments.get("direction")
        if (
            None in {expression, variable, point}
            or direction not in {"two-sided", "left", "right"}
        ):
            return None
        symbol = self.symbolic.sympy.Symbol(variable)
        parsed = self.symbolic._parse(expression)
        target = self.symbolic._parse(point)
        if parsed.free_symbols - {symbol} or target.free_symbols:
            return None
        if direction == "two-sided":
            value = self.symbolic.sympy.limit(parsed, symbol, target)
        else:
            value = self.symbolic.sympy.limit(
                parsed,
                symbol,
                target,
                dir="-" if direction == "left" else "+",
            )
        if value.has(self.symbolic.sympy.Limit):
            return None
        return self.symbolic._format(value)

    def _polynomial_coefficient(self, arguments: dict) -> str | None:
        if (
            set(arguments) != {"expression", "variable", "degree"}
            or self.symbolic.sympy is None
        ):
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        degree = arguments.get("degree")
        if (
            expression is None
            or variable is None
            or not self._bounded_integer(degree, 0, 100_000)
        ):
            return None
        symbol = self.symbolic.sympy.Symbol(variable)
        parsed = self.symbolic.sympy.expand(self.symbolic._parse(expression))
        if parsed.free_symbols - {symbol} or not parsed.is_polynomial(symbol):
            return None
        value = self.symbolic.sympy.Poly(parsed, symbol).coeff_monomial(symbol ** degree)
        return self.symbolic._format(value)

    def _linear_recurrence_term(self, arguments: dict) -> str | None:
        expected = {"coefficients", "initial_values", "constant", "target_index"}
        if set(arguments) != expected or self.symbolic.sympy is None:
            return None
        coefficients = arguments.get("coefficients")
        initial_values = arguments.get("initial_values")
        constant = self._expression(arguments.get("constant"))
        target = arguments.get("target_index")
        if (
            not isinstance(coefficients, list)
            or not isinstance(initial_values, list)
            or not 1 <= len(coefficients) <= 20
            or len(coefficients) != len(initial_values)
            or constant is None
            or not self._bounded_integer(target, 0, self._MAX_RECURRENCE_INDEX)
        ):
            return None
        raw_coefficients = [self._expression(item) for item in coefficients]
        raw_initial = [self._expression(item) for item in initial_values]
        if any(item is None for item in (*raw_coefficients, *raw_initial)):
            return None
        parsed_coefficients = [self.symbolic._parse(item) for item in raw_coefficients]
        values = [self.symbolic._parse(item) for item in raw_initial]
        forcing = self.symbolic._parse(constant)
        if any(item.free_symbols for item in (*parsed_coefficients, *values, forcing)):
            return None
        if target < len(values):
            return self.symbolic._format(values[target])
        order = len(values)
        for _ in range(order, target + 1):
            next_value = forcing + sum(
                coefficient * values[-offset]
                for offset, coefficient in enumerate(parsed_coefficients, start=1)
            )
            values.append(self.symbolic.sympy.simplify(next_value))
            if len(values) > order:
                values.pop(0)
        return self.symbolic._format(values[-1])

    def _finite_state_arguments(
        self,
        arguments: dict,
    ) -> tuple[list[list[int]], list[int], int, list[int]] | None:
        expected = {
            "transition_rows", "initial_counts", "steps", "accepting_states"
        }
        if set(arguments) != expected:
            return None
        rows = arguments.get("transition_rows")
        initial = arguments.get("initial_counts")
        steps = arguments.get("steps")
        accepting = arguments.get("accepting_states")
        if (
            not isinstance(rows, list)
            or not 1 <= len(rows) <= self._MAX_FINITE_STATES
            or not isinstance(initial, list)
            or len(initial) != len(rows)
            or not isinstance(accepting, list)
            or not accepting
            or len(set(accepting)) != len(accepting)
            or not self._bounded_integer(steps, 0, 100_000)
        ):
            return None
        size = len(rows)
        if any(
            not isinstance(row, list)
            or len(row) != size
            or any(not self._bounded_integer(value, 0, 1_000_000) for value in row)
            for row in rows
        ):
            return None
        if any(
            not self._bounded_integer(value, 0, 1_000_000_000_000)
            for value in initial
        ):
            return None
        if any(not self._bounded_integer(value, 0, size - 1) for value in accepting):
            return None
        if steps * size * size > self._MAX_FINITE_STATE_WORK:
            return None
        return rows, initial, steps, accepting

    @staticmethod
    def _finite_state_payload(counts: list[int], accepting: list[int]) -> str:
        return json.dumps(
            {
                "accepting_count": sum(counts[index] for index in accepting),
                "final_counts": counts,
            },
            ensure_ascii=False,
        )

    def _finite_state_walk_count(self, arguments: dict) -> str | None:
        parsed = self._finite_state_arguments(arguments)
        if parsed is None:
            return None
        rows, counts, steps, accepting = parsed
        size = len(rows)
        current = list(counts)
        for _ in range(steps):
            following = [0] * size
            for source, source_count in enumerate(current):
                if not source_count:
                    continue
                for target, multiplicity in enumerate(rows[source]):
                    if multiplicity:
                        following[target] += source_count * multiplicity
            current = following
        return self._finite_state_payload(current, accepting)

    @staticmethod
    def _integer_matrix_product(
        left: list[list[int]],
        right: list[list[int]],
    ) -> list[list[int]]:
        size = len(left)
        result = [[0] * size for _ in range(size)]
        for row in range(size):
            for pivot, left_value in enumerate(left[row]):
                if not left_value:
                    continue
                for column, right_value in enumerate(right[pivot]):
                    if right_value:
                        result[row][column] += left_value * right_value
        return result

    @staticmethod
    def _integer_vector_matrix_product(
        vector: list[int],
        matrix: list[list[int]],
    ) -> list[int]:
        result = [0] * len(vector)
        for row, value in enumerate(vector):
            if not value:
                continue
            for column, multiplier in enumerate(matrix[row]):
                if multiplier:
                    result[column] += value * multiplier
        return result

    def _finite_state_walk_count_matrix(self, arguments: dict) -> str | None:
        parsed = self._finite_state_arguments(arguments)
        if parsed is None:
            return None
        matrix, vector, steps, accepting = parsed
        power = [list(row) for row in matrix]
        remaining = steps
        while remaining:
            if remaining & 1:
                vector = self._integer_vector_matrix_product(vector, power)
            remaining >>= 1
            if remaining:
                power = self._integer_matrix_product(power, power)
        return self._finite_state_payload(vector, accepting)

    def _subtraction_game_arguments(
        self,
        arguments: dict,
    ) -> tuple[int, list[int]] | None:
        if set(arguments) != {"initial_heap", "moves"}:
            return None
        heap = arguments.get("initial_heap")
        moves = arguments.get("moves")
        if (
            not self._bounded_integer(heap, 0, 100_000)
            or not isinstance(moves, list)
            or not 1 <= len(moves) <= 256
            or len(set(moves)) != len(moves)
            or any(not self._bounded_integer(move, 1, 100_000) for move in moves)
        ):
            return None
        return heap, sorted(move for move in moves if move <= heap)

    @staticmethod
    def _subtraction_game_payload(
        heap: int,
        winning: bool,
        winning_moves: list[int],
    ) -> str:
        return json.dumps(
            {
                "initial_heap": heap,
                "winning": winning,
                "winning_moves": winning_moves,
            },
            ensure_ascii=False,
        )

    def _subtraction_game_outcome(self, arguments: dict) -> str | None:
        parsed = self._subtraction_game_arguments(arguments)
        if parsed is None:
            return None
        heap, moves = parsed
        winning = [False] * (heap + 1)
        for stones in range(1, heap + 1):
            winning[stones] = any(
                move <= stones and not winning[stones - move] for move in moves
            )
        winning_moves = [
            move for move in moves if move <= heap and not winning[heap - move]
        ]
        return self._subtraction_game_payload(heap, winning[heap], winning_moves)

    def _subtraction_game_outcome_grundy(self, arguments: dict) -> str | None:
        parsed = self._subtraction_game_arguments(arguments)
        if parsed is None:
            return None
        heap, moves = parsed
        grundy = [0] * (heap + 1)
        for stones in range(1, heap + 1):
            reachable = {
                grundy[stones - move] for move in moves if move <= stones
            }
            value = 0
            while value in reachable:
                value += 1
            grundy[stones] = value
        winning_moves = [
            move for move in moves if move <= heap and grundy[heap - move] == 0
        ]
        return self._subtraction_game_payload(
            heap,
            grundy[heap] != 0,
            winning_moves,
        )

    def _permutation_cycle_arguments(
        self,
        arguments: dict,
    ) -> tuple[int, list[int], dict[int, tuple[int, int]]] | None:
        if set(arguments) != {
            "size", "allowed_cycle_lengths", "cycle_count_bounds"
        }:
            return None
        size = arguments.get("size")
        lengths = arguments.get("allowed_cycle_lengths")
        raw_bounds = arguments.get("cycle_count_bounds")
        if (
            not self._bounded_integer(size, 0, self._MAX_PERMUTATION_SIZE)
            or not isinstance(lengths, list)
            or not lengths
            or len(lengths) > self._MAX_PERMUTATION_SIZE
            or len(set(lengths)) != len(lengths)
            or any(
                not self._bounded_integer(length, 1, self._MAX_PERMUTATION_SIZE)
                for length in lengths
            )
            or not isinstance(raw_bounds, list)
            or len(raw_bounds) > 20
        ):
            return None
        allowed = sorted(length for length in lengths if length <= size)
        bounds: dict[int, tuple[int, int]] = {}
        for item in raw_bounds:
            if not isinstance(item, dict) or set(item) != {
                "length", "minimum", "maximum"
            }:
                return None
            length = item.get("length")
            minimum = item.get("minimum")
            maximum = item.get("maximum")
            if (
                not self._bounded_integer(length, 1, self._MAX_PERMUTATION_SIZE)
                or length not in lengths
                or length in bounds
                or not self._bounded_integer(minimum, 0, self._MAX_PERMUTATION_SIZE)
                or not self._bounded_integer(maximum, minimum, self._MAX_PERMUTATION_SIZE)
                or maximum > size // length
            ):
                return None
            bounds[length] = (minimum, maximum)
        return size, allowed, bounds

    @staticmethod
    def _cycle_inventory_coefficient(
        size: int,
        lengths: list[int],
        bounds: dict[int, tuple[int, int]],
    ) -> int | None:
        coefficients = [Fraction(0) for _ in range(size + 1)]
        coefficients[0] = Fraction(1)
        for length in lengths:
            minimum, maximum = bounds.get(length, (0, size // length))
            factor_terms: list[tuple[int, Fraction]] = []
            denominator = 1
            for count in range(0, maximum + 1):
                if count:
                    denominator *= length * count
                if count >= minimum:
                    factor_terms.append((length * count, Fraction(1, denominator)))
            following = [Fraction(0) for _ in range(size + 1)]
            for used, coefficient in enumerate(coefficients):
                if not coefficient:
                    continue
                for degree, factor in factor_terms:
                    if used + degree <= size:
                        following[used + degree] += coefficient * factor
            coefficients = following
        result = coefficients[size] * factorial(size)
        return int(result) if result.denominator == 1 else None

    def _permutation_cycle_count(self, arguments: dict) -> str | None:
        parsed = self._permutation_cycle_arguments(arguments)
        if parsed is None:
            return None
        size, lengths, bounds = parsed
        value = self._cycle_inventory_coefficient(size, lengths, bounds)
        return None if value is None else str(value)

    def _permutation_cycle_count_reversed(self, arguments: dict) -> str | None:
        parsed = self._permutation_cycle_arguments(arguments)
        if parsed is None:
            return None
        size, lengths, bounds = parsed
        value = self._cycle_inventory_coefficient(size, list(reversed(lengths)), bounds)
        return None if value is None else str(value)

    @staticmethod
    def _orientation(
        first: tuple[int, int],
        second: tuple[int, int],
        third: tuple[int, int],
    ) -> int:
        value = (
            (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
        )
        return (value > 0) - (value < 0)

    @staticmethod
    def _point_on_segment(
        point: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> bool:
        return bool(
            min(first[0], second[0]) <= point[0] <= max(first[0], second[0])
            and min(first[1], second[1]) <= point[1] <= max(first[1], second[1])
        )

    @classmethod
    def _segments_intersect(
        cls,
        first: tuple[int, int],
        second: tuple[int, int],
        third: tuple[int, int],
        fourth: tuple[int, int],
    ) -> bool:
        orientations = (
            cls._orientation(first, second, third),
            cls._orientation(first, second, fourth),
            cls._orientation(third, fourth, first),
            cls._orientation(third, fourth, second),
        )
        if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
            return True
        return bool(
            (orientations[0] == 0 and cls._point_on_segment(third, first, second))
            or (orientations[1] == 0 and cls._point_on_segment(fourth, first, second))
            or (orientations[2] == 0 and cls._point_on_segment(first, third, fourth))
            or (orientations[3] == 0 and cls._point_on_segment(second, third, fourth))
        )

    def _lattice_polygon_vertices(
        self,
        arguments: dict,
    ) -> list[tuple[int, int]] | None:
        if set(arguments) != {"vertices"}:
            return None
        raw_vertices = arguments.get("vertices")
        if (
            not isinstance(raw_vertices, list)
            or not 3 <= len(raw_vertices) <= self._MAX_POLYGON_VERTICES + 1
        ):
            return None
        vertices: list[tuple[int, int]] = []
        for item in raw_vertices:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or any(not self._bounded_integer(value, -1_000_000, 1_000_000) for value in item)
            ):
                return None
            vertices.append((item[0], item[1]))
        if len(vertices) > 3 and vertices[-1] == vertices[0]:
            vertices.pop()
        if (
            not 3 <= len(vertices) <= self._MAX_POLYGON_VERTICES
            or len(set(vertices)) != len(vertices)
        ):
            return None
        size = len(vertices)
        for first_index in range(size):
            first = vertices[first_index]
            second = vertices[(first_index + 1) % size]
            if first == second:
                return None
            for second_index in range(first_index + 1, size):
                if second_index in {
                    first_index,
                    (first_index + 1) % size,
                    (first_index - 1) % size,
                }:
                    continue
                third = vertices[second_index]
                fourth = vertices[(second_index + 1) % size]
                if self._segments_intersect(first, second, third, fourth):
                    return None
        return vertices

    @staticmethod
    def _lattice_polygon_payload(vertices: list[tuple[int, int]]) -> str | None:
        area_twice_signed = 0
        boundary = 0
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            area_twice_signed += first[0] * second[1] - first[1] * second[0]
            boundary += gcd(abs(second[0] - first[0]), abs(second[1] - first[1]))
        area_twice = abs(area_twice_signed)
        interior_twice = area_twice - boundary + 2
        if not area_twice or interior_twice < 0 or interior_twice % 2:
            return None
        return json.dumps(
            {
                "area_twice": area_twice,
                "boundary_points": boundary,
                "interior_points": interior_twice // 2,
            },
            ensure_ascii=False,
        )

    def _lattice_polygon_interior(self, arguments: dict) -> str | None:
        vertices = self._lattice_polygon_vertices(arguments)
        return None if vertices is None else self._lattice_polygon_payload(vertices)

    def _lattice_polygon_interior_reversed(self, arguments: dict) -> str | None:
        vertices = self._lattice_polygon_vertices(arguments)
        return (
            None
            if vertices is None
            else self._lattice_polygon_payload(list(reversed(vertices)))
        )

    def _factorial_valuation_arguments(
        self,
        arguments: dict,
    ) -> tuple[int, list[int], list[int]] | None:
        if set(arguments) != {
            "prime", "numerator_factorials", "denominator_factorials"
        } or self.symbolic.sympy is None:
            return None
        prime = arguments.get("prime")
        numerator = arguments.get("numerator_factorials")
        denominator = arguments.get("denominator_factorials")
        if (
            not self._bounded_integer(prime, 2, 1_000_000)
            or not bool(self.symbolic.sympy.isprime(prime))
            or not isinstance(numerator, list)
            or not 1 <= len(numerator) <= 100
            or not isinstance(denominator, list)
            or len(denominator) > 100
            or any(
                not self._bounded_integer(value, 0, 1_000_000_000)
                for value in (*numerator, *denominator)
            )
        ):
            return None
        return prime, numerator, denominator

    @staticmethod
    def _factorial_prime_valuation(value: int, prime: int) -> int:
        result = 0
        quotient = value
        while quotient:
            quotient //= prime
            result += quotient
        return result

    def _factorial_ratio_prime_valuation(self, arguments: dict) -> str | None:
        parsed = self._factorial_valuation_arguments(arguments)
        if parsed is None:
            return None
        prime, numerator, denominator = parsed
        value = sum(
            self._factorial_prime_valuation(item, prime) for item in numerator
        ) - sum(
            self._factorial_prime_valuation(item, prime) for item in denominator
        )
        return str(value)

    @staticmethod
    def _base_digit_sum(value: int, base: int) -> int:
        total = 0
        remaining = value
        while remaining:
            total += remaining % base
            remaining //= base
        return total

    def _factorial_ratio_prime_valuation_digits(self, arguments: dict) -> str | None:
        parsed = self._factorial_valuation_arguments(arguments)
        if parsed is None:
            return None
        prime, numerator, denominator = parsed

        def valuation(value: int) -> int:
            return (value - self._base_digit_sum(value, prime)) // (prime - 1)

        return str(sum(map(valuation, numerator)) - sum(map(valuation, denominator)))

    def _modular_power_arguments(
        self,
        arguments: dict,
    ) -> tuple[list[tuple[int, int, int]], int] | None:
        if set(arguments) != {"terms", "modulus"}:
            return None
        terms = arguments.get("terms")
        modulus = arguments.get("modulus")
        required = {
            "coefficient", "base", "exponent_base", "exponent_power",
            "exponent_multiplier", "exponent_offset",
        }
        if (
            not isinstance(terms, list)
            or not 1 <= len(terms) <= 16
            or not self._bounded_integer(modulus, 1, 2_000_000_000)
        ):
            return None
        parsed: list[tuple[int, int, int]] = []
        for item in terms:
            if not isinstance(item, dict) or set(item) != required:
                return None
            coefficient = item.get("coefficient")
            base = item.get("base")
            exponent_base = item.get("exponent_base")
            exponent_power = item.get("exponent_power")
            multiplier = item.get("exponent_multiplier")
            offset = item.get("exponent_offset")
            if (
                not self._bounded_integer(coefficient, -1_000_000_000_000, 1_000_000_000_000)
                or not self._bounded_integer(base, -1_000_000_000_000, 1_000_000_000_000)
                or not self._bounded_integer(exponent_base, 0, 1_000_000_000)
                or not self._bounded_integer(exponent_power, 0, 100_000)
                or not self._bounded_integer(multiplier, 0, 1_000_000_000)
                or not self._bounded_integer(offset, -1_000_000_000, 1_000_000_000)
            ):
                return None
            estimated_bits = 1.0
            if multiplier and exponent_base > 1:
                estimated_bits = (
                    exponent_power * log2(exponent_base)
                    + log2(multiplier)
                    + 2
                )
            if estimated_bits > 1_000_000:
                return None
            exponent = multiplier * pow(exponent_base, exponent_power) + offset
            if exponent < 0:
                return None
            parsed.append((coefficient, base, exponent))
        return parsed, modulus

    def _modular_power_sum(
        self,
        arguments: dict,
        *,
        reverse: bool = False,
    ) -> str | None:
        parsed = self._modular_power_arguments(arguments)
        if parsed is None:
            return None
        terms, modulus = parsed
        ordered = reversed(terms) if reverse else terms
        value = 0
        for coefficient, base, exponent in ordered:
            value = (value + coefficient * pow(base, exponent, modulus)) % modulus
        return str(value)

    def _bounded_integer_search(self, arguments: dict) -> str | None:
        expected = {
            "variables", "lower_bounds", "upper_bounds", "equations",
            "inequalities", "congruences", "operation", "objective",
        }
        if set(arguments) != expected or self.symbolic.sympy is None:
            return None
        variables = arguments.get("variables")
        lower = arguments.get("lower_bounds")
        upper = arguments.get("upper_bounds")
        equations = arguments.get("equations")
        inequalities = arguments.get("inequalities")
        congruences = arguments.get("congruences")
        operation = arguments.get("operation")
        objective = self._expression(arguments.get("objective"))
        if (
            not isinstance(variables, list)
            or not 1 <= len(variables) <= 3
            or len(set(variables)) != len(variables)
            or not isinstance(lower, list)
            or not isinstance(upper, list)
            or len(lower) != len(variables)
            or len(upper) != len(variables)
            or not isinstance(equations, list)
            or len(equations) > 8
            or not isinstance(inequalities, list)
            or len(inequalities) > 8
            or not isinstance(congruences, list)
            or len(congruences) > 8
            or operation not in {"count", "list", "minimize", "maximize"}
            or objective is None
        ):
            return None
        names = [self._variable(item) for item in variables]
        if any(item is None for item in names):
            return None
        if any(
            not self._bounded_integer(lo, -1_000_000, 1_000_000)
            or not self._bounded_integer(hi, -1_000_000, 1_000_000)
            or hi < lo
            for lo, hi in zip(lower, upper)
        ):
            return None
        state_count = 1
        for lo, hi in zip(lower, upper):
            state_count *= hi - lo + 1
            if state_count > self._MAX_ENUMERATION_STATES:
                return None
        symbols = [self.symbolic.sympy.Symbol(name) for name in names]
        allowed_symbols = set(symbols)

        def parsed_expression(raw: Any):
            value = self._expression(raw)
            if value is None:
                return None
            parsed = self.symbolic._parse(value)
            return parsed if parsed.free_symbols <= allowed_symbols else None

        parsed_equations = [parsed_expression(item) for item in equations]
        if any(item is None for item in parsed_equations):
            return None
        parsed_inequalities = []
        for item in inequalities:
            if not isinstance(item, dict) or set(item) != {"expression", "relation"}:
                return None
            parsed = parsed_expression(item.get("expression"))
            relation = item.get("relation")
            if parsed is None or relation not in {"<", "<=", ">", ">="}:
                return None
            parsed_inequalities.append((parsed, relation))
        parsed_congruences = []
        for item in congruences:
            if not isinstance(item, dict) or set(item) != {"expression", "modulus", "remainder"}:
                return None
            parsed = parsed_expression(item.get("expression"))
            modulus = item.get("modulus")
            remainder = item.get("remainder")
            if (
                parsed is None
                or not self._bounded_integer(modulus, 1, self._MAX_MODULUS)
                or not self._bounded_integer(remainder, 0, modulus - 1)
            ):
                return None
            parsed_congruences.append((parsed, modulus, remainder))
        parsed_objective = parsed_expression(objective)
        if parsed_objective is None:
            return None

        def inequality_holds(value, relation: str) -> bool:
            value = self.symbolic.sympy.simplify(value)
            if value.is_real is False or value.is_number is not True:
                return False
            if relation == "<":
                comparison = value < 0
            elif relation == "<=":
                comparison = value <= 0
            elif relation == ">":
                comparison = value > 0
            else:
                comparison = value >= 0
            return bool(comparison)

        count = 0
        listed: list[list[int]] = []
        best_value = None
        best_points: list[list[int]] = []
        ranges = [range(lo, hi + 1) for lo, hi in zip(lower, upper)]
        for point in product(*ranges):
            substitutions = dict(zip(symbols, point))
            if any(
                self.symbolic.sympy.simplify(item.subs(substitutions)) != 0
                for item in parsed_equations
            ):
                continue
            if any(
                not inequality_holds(item.subs(substitutions), relation)
                for item, relation in parsed_inequalities
            ):
                continue
            congruence_failed = False
            for item, modulus, remainder in parsed_congruences:
                value = self.symbolic.sympy.simplify(item.subs(substitutions))
                if value.is_integer is not True or int(value) % modulus != remainder:
                    congruence_failed = True
                    break
            if congruence_failed:
                continue
            count += 1
            if len(listed) < 200:
                listed.append(list(point))
            if operation in {"minimize", "maximize"}:
                value = self.symbolic.sympy.simplify(parsed_objective.subs(substitutions))
                if value.is_real is False or value.is_number is not True:
                    return None
                better = best_value is None or (
                    bool(value < best_value)
                    if operation == "minimize"
                    else bool(value > best_value)
                )
                if better:
                    best_value = value
                    best_points = [list(point)]
                elif self.symbolic.sympy.simplify(value - best_value) == 0 and len(best_points) < 200:
                    best_points.append(list(point))
        payload: dict[str, Any] = {"variables": names, "count": count}
        if operation == "list" or count <= 200:
            payload["solutions"] = listed
        if operation in {"minimize", "maximize"}:
            if best_value is None:
                payload["optimum"] = None
                payload["optimizers"] = []
            else:
                payload["optimum"] = self.symbolic._format(best_value)
                payload["optimizers"] = best_points
        return json.dumps(payload, ensure_ascii=False)

    def _matrix_operation(self, arguments: dict) -> str | None:
        if set(arguments) != {"rows", "operation"} or self.symbolic.sympy is None:
            return None
        rows = arguments.get("rows")
        operation = arguments.get("operation")
        if not isinstance(rows, list) or not rows or len(rows) > 20:
            return None
        width = len(rows[0]) if isinstance(rows[0], list) else 0
        if not width or width > 20 or len(rows) * width > self._MAX_MATRIX_CELLS:
            return None
        if any(not isinstance(row, list) or len(row) != width for row in rows):
            return None
        if operation != "permanent" and len(rows) * width > 100:
            return None
        matrix = self.symbolic.sympy.Matrix([
            [self.symbolic._parse(str(cell)) for cell in row] for row in rows
        ])
        if operation == "determinant" and matrix.rows == matrix.cols:
            return self.symbolic._format(matrix.det())
        if operation == "rank":
            return str(matrix.rank())
        if operation == "inverse" and matrix.rows == matrix.cols and matrix.det() != 0:
            return json.dumps(
                [[self.symbolic._format(cell) for cell in row] for row in matrix.inv().tolist()],
                ensure_ascii=False,
            )
        if operation == "eigenvalues" and matrix.rows == matrix.cols:
            if matrix.rows > 10:
                return None
            values = matrix.eigenvals()
            rendered = {
                self.symbolic._format(value): int(multiplicity)
                for value, multiplicity in values.items()
            }
            return json.dumps(rendered, ensure_ascii=False)
        if operation == "permanent" and matrix.rows == matrix.cols and matrix.rows <= 18:
            values = self._rational_matrix(rows, maximum_order=18)
            if values is None:
                return None
            return self.symbolic._format(self._permanent_ryser(values))
        return None

    def _rational_matrix(
        self,
        rows: Any,
        *,
        maximum_order: int,
    ) -> list[list[Any]] | None:
        if self.symbolic.sympy is None or not isinstance(rows, list) or not rows:
            return None
        if len(rows) > maximum_order or any(
            not isinstance(row, list) or len(row) != len(rows) for row in rows
        ):
            return None
        values: list[list[Any]] = []
        for row in rows:
            parsed_row = []
            for cell in row:
                value = self.symbolic._parse(str(cell))
                if value.free_symbols or value.is_rational is not True:
                    return None
                parsed_row.append(value)
            values.append(parsed_row)
        return values

    def _permanent_ryser(self, rows: list[list[Any]]):
        """Exact Ryser computation using Gray-code row-sum updates."""
        n = len(rows)
        if n == 0:
            return self.symbolic.sympy.Integer(1)
        row_sums = [self.symbolic.sympy.Integer(0) for _ in range(n)]
        total = self.symbolic.sympy.Integer(0)
        previous_gray = 0
        for counter in range(1, 1 << n):
            gray = counter ^ (counter >> 1)
            changed = gray ^ previous_gray
            column = changed.bit_length() - 1
            direction = 1 if gray & changed else -1
            for row_index in range(n):
                row_sums[row_index] += direction * rows[row_index][column]
            product_value = self.symbolic.sympy.Integer(1)
            for value in row_sums:
                product_value *= value
            subset_size = gray.bit_count()
            total += (-1 if (n - subset_size) % 2 else 1) * product_value
            previous_gray = gray
        return self.symbolic.sympy.simplify(total)

    def _permanent_glynn(self, rows: list[list[Any]]):
        """Independent exact Glynn-formula postcondition for a permanent."""
        n = len(rows)
        if n == 0:
            return self.symbolic.sympy.Integer(1)
        total = self.symbolic.sympy.Integer(0)
        for mask in range(1 << max(0, n - 1)):
            signs = [1]
            signs.extend(1 if mask & (1 << index) else -1 for index in range(n - 1))
            sign_product = 1
            for sign in signs:
                sign_product *= sign
            column_product = self.symbolic.sympy.Integer(1)
            for column in range(n):
                column_sum = sum(
                    signs[row] * rows[row][column] for row in range(n)
                )
                column_product *= column_sum
            total += sign_product * column_product
        return self.symbolic.sympy.simplify(total / (2 ** max(0, n - 1)))

    def _count_digit_strings_sparse(self, arguments: dict) -> str | None:
        """Dictionary-state implementation independent of the dense DP path."""
        expected = {
            "minimum_length", "maximum_length", "digits", "modulus",
            "remainder", "leading_zero_allowed",
        }
        if set(arguments) != expected:
            return None
        minimum = arguments.get("minimum_length")
        maximum = arguments.get("maximum_length")
        digits = arguments.get("digits")
        modulus = arguments.get("modulus")
        remainder = arguments.get("remainder")
        leading_zero_allowed = arguments.get("leading_zero_allowed")
        if (
            not self._bounded_integer(minimum, 1, self._MAX_DIGIT_LENGTH)
            or not self._bounded_integer(maximum, minimum, self._MAX_DIGIT_LENGTH)
            or not isinstance(digits, list) or not digits
            or any(not self._bounded_integer(item, 0, 9) for item in digits)
            or len(set(digits)) != len(digits)
            or not self._bounded_integer(modulus, 1, self._MAX_MODULUS)
            or not self._bounded_integer(remainder, 0, modulus - 1)
            or not isinstance(leading_zero_allowed, bool)
            or maximum * modulus > 5_000_000
        ):
            return None
        current: dict[int, int] = {}
        total = 0
        for length in range(1, maximum + 1):
            next_counts: dict[int, int] = {}
            if length == 1:
                allowed = digits if leading_zero_allowed else [digit for digit in digits if digit]
                for digit in allowed:
                    residue = digit % modulus
                    next_counts[residue] = next_counts.get(residue, 0) + 1
            else:
                for residue, count in current.items():
                    for digit in digits:
                        target = (10 * residue + digit) % modulus
                        next_counts[target] = next_counts.get(target, 0) + count
            current = next_counts
            if length >= minimum:
                total += current.get(remainder, 0)
        return str(total)

    def _count_digit_strings(self, arguments: dict) -> str | None:
        expected = {
            "minimum_length", "maximum_length", "digits", "modulus",
            "remainder", "leading_zero_allowed",
        }
        if set(arguments) != expected:
            return None
        minimum = arguments.get("minimum_length")
        maximum = arguments.get("maximum_length")
        digits = arguments.get("digits")
        modulus = arguments.get("modulus")
        remainder = arguments.get("remainder")
        leading_zero_allowed = arguments.get("leading_zero_allowed")
        if (
            not self._bounded_integer(minimum, 1, self._MAX_DIGIT_LENGTH)
            or not self._bounded_integer(maximum, minimum, self._MAX_DIGIT_LENGTH)
            or not isinstance(digits, list) or not digits
            or any(not self._bounded_integer(item, 0, 9) for item in digits)
            or len(set(digits)) != len(digits)
            or not self._bounded_integer(modulus, 1, self._MAX_MODULUS)
            or not self._bounded_integer(remainder, 0, modulus - 1)
            or not isinstance(leading_zero_allowed, bool)
            or maximum * modulus > 5_000_000
        ):
            return None
        current = [0] * modulus
        total = 0
        for length in range(1, maximum + 1):
            next_counts = [0] * modulus
            if length == 1:
                allowed = digits if leading_zero_allowed else [digit for digit in digits if digit]
                for digit in allowed:
                    next_counts[digit % modulus] += 1
            else:
                for residue, count in enumerate(current):
                    if not count:
                        continue
                    for digit in digits:
                        next_counts[(10 * residue + digit) % modulus] += count
            current = next_counts
            if length >= minimum:
                total += current[remainder]
        return str(total)

    def _count_modular_solutions(self, arguments: dict) -> str | None:
        if set(arguments) != {"expression", "variable", "modulus", "remainder"} or self.symbolic.sympy is None:
            return None
        expression = self._expression(arguments["expression"])
        variable = self._variable(arguments["variable"])
        modulus = arguments.get("modulus")
        remainder = arguments.get("remainder")
        if (
            expression is None or variable is None
            or not self._bounded_integer(modulus, 1, self._MAX_MODULUS)
            or not self._bounded_integer(remainder, 0, modulus - 1)
        ):
            return None
        symbol = self.symbolic.sympy.Symbol(variable)
        parsed = self.symbolic._parse(expression)
        if not parsed.is_polynomial(symbol):
            return None
        solutions = [
            value for value in range(modulus)
            if int(parsed.subs(symbol, value)) % modulus == remainder
        ]
        payload: dict[str, Any] = {"count": len(solutions)}
        if len(solutions) <= 200:
            payload["solutions"] = solutions
        return json.dumps(payload, ensure_ascii=False)

    def _expression(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text or len(text) > self._MAX_EXPRESSION_CHARS:
            return None
        if re.search(r"__|\b(?:eval|exec|import|open|lambda|compile|globals|locals)\b", text, re.IGNORECASE):
            return None
        try:
            self.symbolic._parse(text)
        except Exception:
            return None
        return text

    def _free_symbols(self, expression: str) -> bool:
        if self.symbolic.sympy is None:
            return True
        return bool(self.symbolic._parse(expression).free_symbols)

    @classmethod
    def _variable(cls, value: Any) -> str | None:
        return value if isinstance(value, str) and cls._VARIABLE.fullmatch(value) else None

    @staticmethod
    def _bounded_integer(value: Any, lower: int, upper: int) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper
