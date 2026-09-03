from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
from pathlib import Path


ROOT = Path(".")
SENSITIVE = re.compile(r"(?:api[_-]?key|x-access-token|ghp_[A-Za-z0-9_]+)", re.IGNORECASE)
ACTIVE_RUNTIME_PATHS = (
    Path("user_agent.py"),
    Path("core"),
    Path("classifier"),
    Path("reasoning"),
    Path("tools"),
    Path("rag"),
    Path("prompts"),
)
RUNTIME_TEXT_SUFFIXES = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml"}
METADATA_INDEX_KEYS = {
    "idx", "index", "item_idx", "item_index", "problem_idx", "problem_index",
    "question_idx", "question_index", "question_id", "sequence", "order",
}
LONG_PROBLEM_TEXT_MIN_CHARS = 48
FORBIDDEN_PATH = re.compile(
    r"(?:^|[/\\])(?:sample_data|tests?|validation_outputs)(?=[/\\])|"
    r"[/\\](?:sample_data|tests?|validation_outputs)$|"
    r"(?:^|[/\\])outputs[/\\][^\r\n]*\.jsonl(?:$|[?*])|"
    r"(?:^|[/\\])[^/\\\r\n]*\.jsonl(?:$|[?*])",
    re.IGNORECASE,
)
BARE_FORBIDDEN_DATA_ROOTS = {"sample_data", "validation_outputs"}
FORBIDDEN_REPLAY_MARKER = re.compile(
    r"\bjudge_replay\b|\bofficial_distribution_112\b|\bjudge1_style\b|"
    r"\bunseen_hard_(?:holdout|answer_key)\b",
    re.IGNORECASE,
)
PROBLEM_TEXT_SIGNAL = re.compile(
    r"(?:\b(?:find|determine|compute|calculate|prove|show|classify|let|given)\b|"
    r"求|计算|证明|确定|分类|设|已知|\\boxed|\$)",
    re.IGNORECASE,
)
PROBLEM_DIGEST_LITERAL = re.compile(r"^[0-9a-f]{40,128}$", re.IGNORECASE)
ANSWER_BEARING_KNOWLEDGE = re.compile(
    r"最终答案|标准答案|参考答案|题号|隐藏测试|评测题|"
    r"\b(?:final answer is|expected answer|answer key|hidden test|judge replay|question id)\b",
    re.IGNORECASE,
)


class ValidationClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        if "给出一条理由" in messages[-1]["content"]:
            return "FINAL: 正确。\n因为等式两边相同，所以命题成立。"
        if "CHOICE:" in messages[0]["content"]:
            return "CHOICE: 0\nREASON: 验证通过"
        return "【最终答案】验证答案"


class MetadataValidationClient:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.result_calls = 0

    def chat(self, **kwargs):
        del kwargs
        self.chat_calls += 1
        return "truncated text without metadata"

    def chat_result(self, *, tools=None, **kwargs):
        from core.model_response import ModelCallResult

        del kwargs
        self.result_calls += 1
        if not tools:
            raise AssertionError("model tools were not forwarded")
        return ModelCallResult(
            "truncated text with metadata",
            "length",
            {"completion_tokens": 128},
        )


def _active_runtime_files() -> list[Path]:
    files: list[Path] = []
    for relative in ACTIVE_RUNTIME_PATHS:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            continue
        files.extend(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in RUNTIME_TEXT_SUFFIXES
            and "__pycache__" not in candidate.parts
        )
    return sorted(set(files), key=lambda path: path.as_posix())


def _string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        literal_parts = [
            value.value
            for value in node.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        return "".join(literal_parts)
    return None


def _literal_strings(node: ast.AST | None) -> list[tuple[str, int]]:
    if node is None:
        return []
    return [
        (child.value, getattr(child, "lineno", getattr(node, "lineno", 1)))
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return {name for item in node.elts for name in _target_names(item)}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


def _subscript_key(node: ast.Subscript) -> str | None:
    return _string_literal(node.slice)


def _looks_like_long_problem(value: str) -> bool:
    compact = re.sub(r"\s+", " ", value).strip()
    return (
        len(compact) >= LONG_PROBLEM_TEXT_MIN_CHARS
        and bool(PROBLEM_TEXT_SIGNAL.search(compact))
    )


def _looks_like_problem_identifier(value: str) -> bool:
    compact = re.sub(r"\s+", " ", str(value or "")).strip()
    return bool(_looks_like_long_problem(compact) or PROBLEM_DIGEST_LITERAL.fullmatch(compact))


def _contains_forbidden_path(value: str) -> bool:
    normalized = value.strip().replace("\\", "/")
    if FORBIDDEN_PATH.search(normalized):
        return True
    lowered = normalized.casefold().rstrip("/")
    return lowered in BARE_FORBIDDEN_DATA_ROOTS


class _ComplianceScanner(ast.NodeVisitor):
    """Find answer-key coupling without rejecting ordinary exact algorithms."""

    def __init__(self, path: Path, tree: ast.AST) -> None:
        self.path = path
        self.violations: list[str] = []
        self.problem_aliases: list[set[str]] = [set()]
        self.metadata_aliases: list[set[str]] = [set()]
        self._docstring_nodes = self._find_docstrings(tree)

    @staticmethod
    def _find_docstrings(tree: ast.AST) -> set[int]:
        nodes: set[int] = set()
        for parent in ast.walk(tree):
            if not isinstance(
                parent,
                (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            body = getattr(parent, "body", None)
            if not body or not isinstance(body, list):
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                nodes.add(id(first.value))
        return nodes

    def _report(self, node: ast.AST, message: str) -> None:
        relative = self.path.relative_to(ROOT).as_posix()
        self.violations.append(f"{relative}:{getattr(node, 'lineno', 1)}: {message}")

    @property
    def aliases(self) -> set[str]:
        return self.problem_aliases[-1]

    @property
    def metadata_names(self) -> set[str]:
        return self.metadata_aliases[-1]

    def _uses_problem(self, node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id in self.aliases
            for child in ast.walk(node)
        )

    def _uses_metadata(self, node: ast.AST) -> bool:
        return any(
            (
                isinstance(child, ast.Name)
                and child.id in self.metadata_names
            )
            or (
                isinstance(child, ast.Attribute)
                and child.attr.lower() in {"metadata", "meta"}
            )
            for child in ast.walk(node)
        )

    def _is_metadata_alias(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Name)
            and node.id in self.metadata_names
        ) or (
            isinstance(node, ast.Attribute)
            and node.attr.lower() in {"metadata", "meta"}
        )

    def _long_problem_literal(self, node: ast.AST) -> bool:
        return any(_looks_like_long_problem(value) for value, _ in _literal_strings(node))

    def _problem_identifier_literal(self, node: ast.AST) -> bool:
        return any(_looks_like_problem_identifier(value) for value, _ in _literal_strings(node))

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        all_arguments = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        argument_names = {
            argument.arg
            for argument in all_arguments
            if re.search(r"problem|question|statement", argument.arg, re.IGNORECASE)
        }
        metadata_names = {
            argument.arg
            for argument in all_arguments
            if argument.arg.lower() in {"metadata", "meta"}
        }
        self.problem_aliases.append(set(self.aliases) | argument_names)
        self.metadata_aliases.append(set(self.metadata_names) | metadata_names)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for statement in node.body:
            self.visit(statement)
        self.metadata_aliases.pop()
        self.problem_aliases.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if id(node) in self._docstring_nodes or not isinstance(node.value, str):
            return
        if _contains_forbidden_path(node.value):
            self._report(node, "runtime reference to local dataset/test/output JSONL path")
        if FORBIDDEN_REPLAY_MARKER.search(node.value):
            self._report(node, "runtime source contains a local judge-replay marker")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if re.match(r"^(?:tests?|sample_data)(?:\.|$)", alias.name):
                self._report(node, f"runtime import from forbidden local data namespace {alias.name!r}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = str(node.module or "")
        if re.match(r"^(?:tests?|sample_data)(?:\.|$)", module):
            self._report(node, f"runtime import from forbidden local data namespace {module!r}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = (_subscript_key(node) or "").lower()
        if self._uses_metadata(node.value) and key in METADATA_INDEX_KEYS:
            self._report(node, f"runtime answer routing from metadata key {key!r}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.lower() in METADATA_INDEX_KEYS and self._uses_metadata(node.value):
            self._report(node, f"runtime answer routing from metadata attribute {node.attr!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        if call_name == "_closed_choice_hint":
            self._report(node, "closed-world prompt/option answer lookup is forbidden")
        path_call = (
            isinstance(node.func, ast.Name)
            and node.func.id in {"open", "Path"}
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"open", "Path"}
        )
        if path_call:
            values = (
                value
                for argument in node.args
                for value, _ in _literal_strings(argument)
            )
            for value in values:
                root = value.strip().replace("\\", "/").casefold().rstrip("/")
                if root in {"test", "tests", "outputs"}:
                    self._report(node, "runtime access to local dataset/test directory")
                    break
        if isinstance(node.func, ast.Attribute):
            key = _string_literal(node.args[0]) if node.args else None
            if (
                self._uses_metadata(node.func.value)
                and node.func.attr in {"get", "pop", "setdefault", "__getitem__"}
                and str(key or "").lower() in METADATA_INDEX_KEYS
            ):
                self._report(node, f"runtime answer routing from metadata key {key!r}")
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and self._uses_metadata(node.args[0])
            and str(_string_literal(node.args[1]) or "").lower() in METADATA_INDEX_KEYS
        ):
            key = _string_literal(node.args[1])
            self._report(node, f"runtime answer routing from metadata attribute {key!r}")
        for keyword in node.keywords:
            if str(keyword.arg or "").lower() == "selected_meanings":
                self._report(keyword.value, "preselected option meanings form an answer lookup table")
            if (
                str(keyword.arg or "").lower() == "expected"
                and self._long_problem_literal(keyword.value)
            ):
                self._report(keyword.value, "long problem statement supplied as expected= value")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        target_names = {
            name
            for target in node.targets
            for name in _target_names(target)
        }
        if self._uses_problem(node.value):
            self.aliases.update(target_names)
        else:
            self.aliases.difference_update(target_names)
        if self._is_metadata_alias(node.value):
            self.metadata_names.update(target_names)
        else:
            self.metadata_names.difference_update(target_names)
        assigned_names = {
            name.lower() for name in target_names
        }
        if "expected" in assigned_names and self._long_problem_literal(node.value):
            self._report(node, "long problem statement assigned to expected")
        self._check_long_problem_mapping(node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target_names = _target_names(node.target)
        if node.value is not None and self._uses_problem(node.value):
            self.aliases.update(target_names)
        else:
            self.aliases.difference_update(target_names)
        if node.value is not None and self._is_metadata_alias(node.value):
            self.metadata_names.update(target_names)
        else:
            self.metadata_names.difference_update(target_names)
        if (
            "expected" in {name.lower() for name in _target_names(node.target)}
            and self._long_problem_literal(node.value)
        ):
            self._report(node, "long problem statement assigned to expected")
        self._check_long_problem_mapping(node.value, node)
        self.generic_visit(node)

    def _check_long_problem_mapping(self, value: ast.AST | None, node: ast.AST) -> None:
        if not isinstance(value, ast.Dict):
            return
        for key in value.keys:
            key_text = _string_literal(key)
            if not key_text or not _looks_like_problem_identifier(key_text):
                continue
            self._report(node, "literal answer mapping keyed by a problem statement or digest")
            return

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for operator, left, right in zip(node.ops, operands, operands[1:]):
            equality = isinstance(operator, (ast.Eq, ast.NotEq, ast.In, ast.NotIn))
            if not equality:
                continue
            if (
                self._uses_problem(left) and self._problem_identifier_literal(right)
            ) or (
                self._uses_problem(right) and self._problem_identifier_literal(left)
            ):
                self._report(node, "problem-derived value compared against a statement or digest literal")
                break
        self.generic_visit(node)


def _scan_python(path: Path, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        relative = path.relative_to(ROOT).as_posix()
        return [f"{relative}:{error.lineno or 1}: Python syntax error: {error.msg}"]
    scanner = _ComplianceScanner(path, tree)
    scanner.visit(tree)
    return scanner.violations


def compliance_violations() -> list[str]:
    violations: list[str] = []
    for path in _active_runtime_files():
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            relative = path.relative_to(ROOT).as_posix()
            violations.append(f"{relative}:1: unreadable runtime source: {error}")
            continue
        if path.suffix.lower() == ".py":
            violations.extend(_scan_python(path, source))
        elif _contains_forbidden_path(source):
            relative = path.relative_to(ROOT).as_posix()
            violations.append(f"{relative}:1: runtime prompt references local dataset/test/output JSONL path")
        if SENSITIVE.search(source):
            relative = path.relative_to(ROOT).as_posix()
            violations.append(f"{relative}:1: credential-like text in active runtime source")
    for path in ROOT.glob("tools/exact_*.py"):
        violations.append(f"{path.as_posix()}:1: exact problem-route modules are forbidden")
    for path in (ROOT / "rag" / "knowledge").glob("*.txt"):
        violations.append(f"{path.as_posix()}:1: free-text runtime knowledge is forbidden; use audited cards.json")
    violations.extend(_knowledge_card_violations())
    return sorted(set(violations))


def _knowledge_card_violations() -> list[str]:
    path = ROOT / "rag" / "knowledge" / "cards.json"
    if not path.is_file():
        return ["rag/knowledge/cards.json:1: missing audited knowledge-card file"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        return [f"rag/knowledge/cards.json:1: unreadable card data: {type(error).__name__}"]
    cards = payload.get("cards") if isinstance(payload, dict) else None
    if not isinstance(cards, list):
        return ["rag/knowledge/cards.json:1: cards must be a JSON list"]
    violations: list[str] = []
    identifiers: set[str] = set()
    for index, card in enumerate(cards, start=1):
        prefix = f"rag/knowledge/cards.json:card[{index}]"
        if not isinstance(card, dict):
            violations.append(f"{prefix}: card must be an object")
            continue
        identifier = str(card.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            violations.append(f"{prefix}: missing or duplicate card id")
        identifiers.add(identifier)
        if card.get("kind") not in {"method", "theorem", "check"}:
            violations.append(f"{prefix}: invalid card kind")
        if not isinstance(card.get("domains"), list) or not card.get("domains"):
            violations.append(f"{prefix}: domains must be a non-empty list")
        if not isinstance(card.get("keywords"), list):
            violations.append(f"{prefix}: keywords must be a list")
        if not str(card.get("provenance", "")).strip():
            violations.append(f"{prefix}: provenance is required")
        texts = "\n".join(str(card.get(key, "")) for key in ("text_zh", "text_en"))
        if not str(card.get("text_zh", "")).strip():
            violations.append(f"{prefix}: text_zh is required")
        if ANSWER_BEARING_KNOWLEDGE.search(texts):
            violations.append(f"{prefix}: answer-bearing or evaluation-specific wording")
        if len(texts) > 1400:
            violations.append(f"{prefix}: card is too long to be a general method fact")
    return violations


def _answer_contract_violations() -> list[str]:
    """Guard explicit support requests without promoting bare short answers."""
    from classifier.problem_spec import build_problem_spec

    violations: list[str] = []
    support_cases = (
        "判断命题真假并给出一条理由。",
        "判断命题真假并给出简短理由。",
        "判断命题真假并简要说明理由。",
        "判断命题真假并说明依据。",
        "判断命题真假并简述理由。",
        "Decide whether the statement is true and state one reason.",
    )
    answer_only_cases = (
        "判断命题真假。",
        "选择下列说明正确的一项：A. 甲 B. 乙。",
        "Decide whether the statement is true.",
    )
    for problem in support_cases:
        contract = build_problem_spec(problem).answer_contract
        if contract.mode == "answer_only" or "reasoning" not in contract.support_requirements:
            violations.append(f"explicit support request lost its reasoning contract: {problem}")
    for problem in answer_only_cases:
        if build_problem_spec(problem).answer_contract.mode != "answer_only":
            violations.append(f"bare short task was over-promoted: {problem}")
    return violations


def _production_default_violations(agent: object) -> list[str]:
    """Keep rejected experiment switches out of the submitted default path."""
    internal = getattr(agent, "agent", agent)
    expected = {
        "primary_temperature": 0.2,
        "enable_subject_protocols": True,
        "enable_mog": False,
        "enable_candidate_audit": False,
        "enable_blind_consensus": False,
        "enable_quick_consensus": False,
        "enable_complex_subproblem_tools": False,
        "compact_primary_prompt": False,
    }
    violations: list[str] = []
    for name, expected_value in expected.items():
        actual = getattr(internal, name, None)
        if actual != expected_value:
            violations.append(
                f"production default {name}={actual!r}; expected {expected_value!r}"
            )
    return violations


def _transport_metadata_violations(agent_class) -> list[str]:
    client = MetadataValidationClient()
    internal = agent_class(client=client).agent
    trace: list[dict] = []
    _, result = internal._call(
        "compute",
        stage="primary",
        max_tokens=128,
        temperature=0.2,
        thinking_mode=True,
        trace=trace,
        model_tools=internal.model_math_tools,
        model_tool_names=("calculate_expression",),
        max_tool_rounds=1,
    )
    violations: list[str] = []
    if client.result_calls != 1 or client.chat_calls != 0:
        violations.append("tool-enabled call bypassed metadata-preserving adapter")
    if not result.provider_truncated:
        violations.append("provider length finish_reason was not preserved")
    if result.usage.get("completion_tokens") != 128:
        violations.append("provider usage was not preserved")
    return violations


def _response_structure_violations() -> list[str]:
    from reasoning.finalizer import Finalizer

    violations: list[str] = []
    truncated = (
        "We reduce the expression to the remaining boundary contribution, "
        "and the final simplification depends on"
    )
    if "truncated_sentence" not in Finalizer.validate_structure(truncated):
        violations.append("dangling English preposition was not detected as truncation")
    complete = "The sequence converges in probability."
    if Finalizer.validate_structure(complete):
        violations.append("complete mathematical sentence was rejected as truncation")
    return violations


def main() -> int:
    entry = ROOT / "user_agent.py"
    if not entry.is_file():
        print("missing user_agent.py", file=sys.stderr)
        return 1
    violations = compliance_violations()
    if violations:
        print("submission compliance scan failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    contract_violations = _answer_contract_violations()
    if contract_violations:
        print("answer-contract validation failed:", file=sys.stderr)
        for violation in contract_violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    module = importlib.import_module("user_agent")
    agent_class = module.ReasoningAgent
    init_parameters = list(inspect.signature(agent_class.__init__).parameters.values())
    if [parameter.name for parameter in init_parameters[:2]] != ["self", "client"]:
        print("ReasoningAgent.__init__ must accept client as its first argument", file=sys.stderr)
        return 1
    solve_parameters = list(inspect.signature(agent_class.solve).parameters.values())
    if [parameter.name for parameter in solve_parameters[:3]] != ["self", "problem", "metadata"]:
        print("ReasoningAgent.solve must accept problem and metadata", file=sys.stderr)
        return 1
    validation_agent = agent_class(client=ValidationClient())
    default_violations = _production_default_violations(validation_agent)
    if default_violations:
        print("production-default validation failed:", file=sys.stderr)
        for violation in default_violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    transport_violations = _transport_metadata_violations(agent_class)
    if transport_violations:
        print("transport-metadata validation failed:", file=sys.stderr)
        for violation in transport_violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    structure_violations = _response_structure_violations()
    if structure_violations:
        print("response-structure validation failed:", file=sys.stderr)
        for violation in structure_violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    result = validation_agent.solve("计算 1+1。", {"idx": 0})
    if not isinstance(result, dict) or not isinstance(result.get("final_response"), str) or not result["final_response"].strip():
        print("invalid response", file=sys.stderr)
        return 1
    json.dumps(result, ensure_ascii=False)
    model_result = agent_class(client=ValidationClient()).solve(
        "求函数 f(x)=x+1 在 x=2 的函数值。",
        {"idx": 999, "answer": "must be ignored"},
    )
    if not isinstance(model_result.get("final_response"), str) or not model_result["final_response"].strip():
        print("invalid model-path response", file=sys.stderr)
        return 1
    json.dumps(model_result, ensure_ascii=False)
    support_result = agent_class(client=ValidationClient()).solve(
        "判断命题真假并给出一条理由：1+1=2。",
        {"idx": 1000},
    )
    support_answer = str(support_result.get("final_response", ""))
    if "正确" not in support_answer or "因为" not in support_answer:
        print("explicit support was lost from final_response", file=sys.stderr)
        return 1
    json.dumps(support_result, ensure_ascii=False)
    print("submission validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
