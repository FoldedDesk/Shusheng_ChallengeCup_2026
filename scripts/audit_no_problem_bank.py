"""Audit the runtime import closure for hidden problem/answer coupling.

This is a development-only checker.  Corpus contents are read at audit time;
they are never copied into runtime modules or emitted in the report.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "user_agent.py"
RUNTIME_ASSETS = (
    ROOT / "prompts" / "submission.txt",
    ROOT / "rag" / "knowledge" / "cards.json",
)
FORBIDDEN_REACHABLE_MODULES = {
    "tools.exact_olympiad_tool",
    "tools.exact_statistics_tool",
    "tools.exact_textbook_tool",
    "tools.standard_textbook_tool",
    "tools.concept_fact_tool",
}
DIGEST = re.compile(r"^[0-9a-f]{40,128}$", re.IGNORECASE)


@dataclass(frozen=True)
class Literal:
    path: Path
    line: int
    value: str


@dataclass(frozen=True)
class CorpusValue:
    path: Path
    record_id: str
    kind: str
    value: str


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _module_path(module: str) -> Path | None:
    if not module:
        return None
    base = ROOT.joinpath(*module.split("."))
    file_path = base.with_suffix(".py")
    if file_path.is_file():
        return file_path
    init_path = base / "__init__.py"
    return init_path if init_path.is_file() else None


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def _package_initializers(path: Path) -> Iterator[Path]:
    relative = path.relative_to(ROOT)
    parents = relative.parent.parts
    for depth in range(1, len(parents) + 1):
        candidate = ROOT.joinpath(*parents[:depth], "__init__.py")
        if candidate.is_file():
            yield candidate


def _imported_modules(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _module_name(path)
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            parts = package.split(".") if package else []
            keep = max(0, len(parts) - node.level + 1)
            prefix = parts[:keep]
            if node.module:
                prefix.extend(node.module.split("."))
            base = ".".join(prefix)
        else:
            base = str(node.module or "")
        if base:
            yield base
        for alias in node.names:
            if alias.name != "*" and base:
                yield f"{base}.{alias.name}"


def runtime_import_closure() -> tuple[Path, ...]:
    pending = [ENTRY]
    reached: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in reached or not path.is_file():
            continue
        reached.add(path)
        pending.extend(_package_initializers(path))
        for module in _imported_modules(path):
            candidate = _module_path(module)
            if candidate is not None:
                pending.append(candidate)
    return tuple(sorted(reached, key=lambda item: item.as_posix()))


def _python_literals(path: Path) -> Iterator[Literal]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield Literal(path, getattr(node, "lineno", 1), node.value)


def _asset_literals(path: Path) -> Iterator[Literal]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))

        def walk(value: object) -> Iterator[str]:
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)

        for value in walk(payload):
            yield Literal(path, 1, value)
        return
    yield Literal(path, 1, path.read_text(encoding="utf-8"))


def runtime_literals(paths: Iterable[Path]) -> tuple[Literal, ...]:
    values: list[Literal] = []
    for path in paths:
        values.extend(_python_literals(path))
    for path in RUNTIME_ASSETS:
        if path.is_file():
            values.extend(_asset_literals(path))
    return tuple(values)


def _json_documents(path: Path) -> Iterator[object]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    yield json.loads(path.read_text(encoding="utf-8"))


def _field_values(value: object, keys: set[str]) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in keys and isinstance(item, (str, int, float)):
                yield str(item)
            yield from _field_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _field_values(item, keys)


def _record_id(value: object, fallback: str) -> str:
    if isinstance(value, dict):
        for key in ("idx", "id", "question_id"):
            if key in value:
                return str(value[key])
    return fallback


def corpus_values(inputs: Iterable[Path]) -> tuple[tuple[CorpusValue, ...], int]:
    problem_keys = {"problem", "question", "statement"}
    answer_keys = {"answer", "ground_truth", "reference_answer", "gold_answer"}
    values: list[CorpusValue] = []
    files: set[Path] = set()
    for supplied in inputs:
        candidates = (
            sorted(supplied.rglob("*.json")) + sorted(supplied.rglob("*.jsonl"))
            if supplied.is_dir()
            else [supplied]
        )
        for path in candidates:
            if path.suffix.lower() not in {".json", ".jsonl"} or not path.is_file():
                continue
            files.add(path)
            try:
                documents = tuple(_json_documents(path))
            except (OSError, UnicodeDecodeError, ValueError, TypeError):
                continue
            for number, document in enumerate(documents, start=1):
                identifier = _record_id(document, str(number))
                values.extend(
                    CorpusValue(path, identifier, "problem", item)
                    for item in _field_values(document, problem_keys)
                )
                values.extend(
                    CorpusValue(path, identifier, "answer", item)
                    for item in _field_values(document, answer_keys)
                )
    unique = {
        (item.path.resolve(), item.record_id, item.kind, _normalize(item.value)): item
        for item in values
        if _normalize(item.value)
    }
    return tuple(unique.values()), len(files)


def audit(corpora: Iterable[Path]) -> tuple[dict, list[dict]]:
    runtime = runtime_import_closure()
    literals = runtime_literals(runtime)
    corpus, corpus_file_count = corpus_values(corpora)
    problems = [item for item in corpus if item.kind == "problem"]
    answers = [item for item in corpus if item.kind == "answer"]
    findings: list[dict] = []

    reached_modules = {_module_name(path) for path in runtime}
    for module in sorted(reached_modules & FORBIDDEN_REACHABLE_MODULES):
        findings.append({"kind": "forbidden_reachable_module", "module": module})

    normalized_literals = [
        (item, _normalize(item.value)) for item in literals if _normalize(item.value)
    ]
    problem_hashes = {
        hashlib.sha256(_normalize(item.value).encode("utf-8")).hexdigest(): item
        for item in problems
    }
    for literal, value in normalized_literals:
        relative = literal.path.relative_to(ROOT).as_posix()
        if DIGEST.fullmatch(value) and value.casefold() in problem_hashes:
            match = problem_hashes[value.casefold()]
            findings.append({
                "kind": "problem_digest_literal",
                "source": f"{relative}:{literal.line}",
                "corpus": f"{match.path}:{match.record_id}",
            })
        if len(value) < 48:
            continue
        for item in problems:
            target = _normalize(item.value)
            if value == target or (len(value) >= 80 and value in target) or (len(target) >= 80 and target in value):
                findings.append({
                    "kind": "problem_text_overlap",
                    "source": f"{relative}:{literal.line}",
                    "corpus": f"{item.path}:{item.record_id}",
                    "chars": min(len(value), len(target)),
                })
        for item in answers:
            target = _normalize(item.value)
            if len(target) >= 24 and (value == target or target in value):
                findings.append({
                    "kind": "answer_text_overlap",
                    "source": f"{relative}:{literal.line}",
                    "corpus": f"{item.path}:{item.record_id}",
                    "chars": len(target),
                })

    deduplicated = list({json.dumps(item, sort_keys=True): item for item in findings}.values())
    summary = {
        "reachable_python_files": len(runtime),
        "runtime_literals": len(literals),
        "corpus_files": corpus_file_count,
        "corpus_problems": len(problems),
        "corpus_answers": len(answers),
        "findings": len(deduplicated),
        "reachable_modules": sorted(reached_modules),
    }
    return summary, sorted(deduplicated, key=lambda item: json.dumps(item, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="*", type=Path)
    args = parser.parse_args()
    summary, findings = audit(args.corpus)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if findings:
        print(json.dumps(findings, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print("no problem-bank coupling detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
