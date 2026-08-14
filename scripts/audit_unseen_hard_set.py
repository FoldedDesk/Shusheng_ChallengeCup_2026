"""Audit a genuinely held-out, high-difficulty local mathematics benchmark.

The answer and authoring notes in these files are offline-only.  ``main.py``
passes only ``problem``, ``idx``, and ``source`` to the submitted agent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier.difficulty import classify_difficulty
from classifier.problem_type import classify_problem_type
BOXED_INSTRUCTION = re.compile(
    r"(?:remember\s+to\s+put.*?\\boxed\s*\{\s*\}\s*\.?|"
    r"请(?:将|把)?.{0,24}(?:最终)?答案.{0,12}\\boxed\s*\{\s*\}.{0,3})\s*$",
    re.IGNORECASE | re.DOTALL,
)
TOKEN = re.compile(r"[a-z]+|[\u4e00-\u9fff]|\\[a-z]+|\d+", re.IGNORECASE)
STRUCTURAL_MARKERS = re.compile(
    r"证明|构造|分类|所有|最小|最大|至多|至少|当且仅当|唯一|"
    r"prove|construct|classify|determine\s+all|if\s+and\s+only\s+if|"
    r"least|greatest|minimum|maximum|exactly|at\s+most|at\s+least|"
    r"functional\s+equation|diophantine|irreducible|stopping\s+time|"
    r"eigenvalues?|generating\s+function|residue|uniformly|almost\s+surely",
    re.IGNORECASE,
)
CONSTRAINT_MARKERS = re.compile(
    r"(?:=|<=|>=|\\leq|\\geq|\\mid|\\equiv|\\sum|\\int|\\prod|"
    r"[<>]|for\s+(?:every|all|which)|such\s+that|对任意|满足|subject\s+to|使得)"
)
ADVANCED_MARKERS = re.compile(
    r"Jordan|nilpotent|minimal\s+polynomial|irreducible|holomorphic|improper\s+integral|"
    r"parameter\s+differentiation|L\^?p|P[oó]lya|prefix[- ]state|absorbing|"
    r"Vieta|Diophantine|Chinese\s+remainder|ring\s+homomorphism|bipartite\s+graph|"
    r"unicyclic|tangent.*ellipse|面积坐标|重心|环同态|幂等元|约旦|极小多项式|"
    r"全纯|广义积分|参数求导|吸收|状态递推|唯一环|丢番图",
    re.IGNORECASE,
)
PROOF_OBLIGATION = re.compile(
    r"证明|严格|无遗漏|完备|论证|证明.*唯一|"
    r"prove|justify|rigorous|completeness|complete\s+(?:finite\s+)?set|"
    r"both\s+necessity\s+and\s+sufficiency",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SimilarityMatch:
    path: str
    idx: str
    score: float


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    return rows


def semantic_body(problem: str) -> str:
    return BOXED_INSTRUCTION.sub("", str(problem or "")).strip()


def normalized_problem(problem: str) -> str:
    tokens = TOKEN.findall(semantic_body(problem).lower())
    return " ".join(tokens)


def _trigrams(value: str) -> set[str]:
    compact = value.replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[index:index + 3] for index in range(len(compact) - 2)}


def similarity(left: str, right: str) -> float:
    left_norm = normalized_problem(left)
    right_norm = normalized_problem(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()
    left_grams = _trigrams(left_norm)
    right_grams = _trigrams(right_norm)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(sequence, jaccard)


def existing_problems(candidate: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    for path in sorted((ROOT / "sample_data").glob("*.jsonl")):
        if path.resolve() == candidate.resolve():
            continue
        for row in load_jsonl(path):
            problem = row.get("problem")
            if isinstance(problem, str) and problem.strip():
                records.append((path, str(row.get("idx", "")), problem))
    return records


def closest_match(problem: str, corpus: list[tuple[Path, str, str]]) -> SimilarityMatch:
    best = SimilarityMatch("", "", 0.0)
    for path, idx, other in corpus:
        score = similarity(problem, other)
        if score > best.score:
            best = SimilarityMatch(str(path.relative_to(ROOT)), idx, score)
    return best


def hardness_signals(problem: str) -> tuple[str, ...]:
    """Return independent textual reasons that an item is not a one-step exercise."""
    body = semantic_body(problem)
    problem_type = classify_problem_type(problem)
    signals: list[str] = []
    if classify_difficulty(problem, problem_type) == "hard":
        signals.append("classifier_hard")
    if STRUCTURAL_MARKERS.search(problem):
        signals.append("structural_quantifier_or_extremum")
    if PROOF_OBLIGATION.search(problem):
        signals.append("proof_or_completeness_obligation")
    if ADVANCED_MARKERS.search(problem):
        signals.append("advanced_domain_object")
    if len(CONSTRAINT_MARKERS.findall(problem)) >= 1:
        signals.append("explicit_mathematical_constraint")
    if len(body) >= 140:
        signals.append("multi_clause_statement")
    return tuple(signals)


def validate_row(row: dict, *, closest: SimilarityMatch, threshold: float) -> list[str]:
    errors: list[str] = []
    required = {"idx", "problem", "answer", "subject", "source", "difficulty"}
    missing = sorted(required - set(row))
    if missing:
        return [f"missing fields: {', '.join(missing)}"]

    problem = str(row["problem"]).strip()
    answer = str(row["answer"]).strip()
    difficulty = row["difficulty"]
    if not problem or not answer:
        errors.append("problem and answer must be non-empty")
    if not isinstance(difficulty, dict):
        return [*errors, "difficulty must be an object"]
    if difficulty.get("level") != "hard":
        errors.append("difficulty.level must be hard")
    layers = difficulty.get("reasoning_layers")
    if not isinstance(layers, list) or len([item for item in layers if str(item).strip()]) < 2:
        errors.append("at least two non-empty reasoning_layers are required")
    traps = difficulty.get("common_traps")
    if not isinstance(traps, list) or not any(str(item).strip() for item in traps):
        errors.append("at least one common_traps entry is required")
    checks = difficulty.get("independent_checks")
    if not isinstance(checks, list) or len({str(item).strip() for item in checks if str(item).strip()}) < 2:
        errors.append("at least two distinct independent_checks are required")

    if len(semantic_body(problem)) < 90:
        errors.append("semantic problem body is too short for this high-difficulty benchmark")
    signals = hardness_signals(problem)
    if len(signals) < 3:
        errors.append(
            "fewer than three independent textual hardness signals found: "
            + ", ".join(signals)
        )
    if closest.score >= threshold:
        errors.append(
            f"near-duplicate score {closest.score:.3f} against {closest.path}#{closest.idx}"
        )
    return errors


def audit(path: Path, *, threshold: float = 0.72) -> dict:
    rows = load_jsonl(path)
    corpus = existing_problems(path)
    failures: list[dict] = []
    nearest: list[dict] = []
    seen: set[str] = set()
    languages = {"zh": 0, "en": 0}
    subjects: dict[str, int] = {}
    for row in rows:
        problem = str(row.get("problem", ""))
        key = normalized_problem(problem)
        duplicate_in_set = key in seen
        seen.add(key)
        match = closest_match(problem, corpus)
        errors = validate_row(row, closest=match, threshold=threshold)
        if duplicate_in_set:
            errors.append("duplicate normalized problem inside candidate set")
        if errors:
            failures.append({"idx": str(row.get("idx", "")), "errors": errors})
        nearest.append({
            "idx": str(row.get("idx", "")),
            "score": round(match.score, 4),
            "against": f"{match.path}#{match.idx}" if match.path else "",
            "hardness_signals": list(hardness_signals(problem)),
        })
        language = "zh" if re.search(r"[\u4e00-\u9fff]", semantic_body(problem)) else "en"
        languages[language] += 1
        subject = str(row.get("subject", "unknown"))
        subjects[subject] = subjects.get(subject, 0) + 1

    return {
        "path": str(path),
        "rows": len(rows),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "near_duplicate_threshold": threshold,
        "maximum_existing_similarity": max((item["score"] for item in nearest), default=0.0),
        "nearest_existing_items": nearest,
        "languages": languages,
        "subjects": subjects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    args = parser.parse_args()
    report = audit(args.input_file, threshold=args.similarity_threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
