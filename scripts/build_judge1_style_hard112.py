"""Build and statically audit the original Judge1-style hard regression set."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
SHARDS = tuple(
    SAMPLE_DIR / f"judge1_style_hard_shard_{name}.jsonl"
    for name in "abcd"
)
OUTPUT = SAMPLE_DIR / "judge1_style_112_hard_v1.jsonl"
SOURCE = "judge1_style_112_hard_v1"
BOXED_SUFFIX = "Remember to put your final answer within \\boxed{}."


def _load(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(row)
    return rows


def _normalized_problem(problem: str) -> str:
    return re.sub(r"\W+", "", problem, flags=re.UNICODE).lower()


def _is_english(problem: str) -> bool:
    body = problem.removesuffix(BOXED_SUFFIX)
    return not re.search(r"[\u4e00-\u9fff]", body)


def _validate_math_markup(idx: int, problem: str) -> None:
    if problem.count("$") % 2:
        raise ValueError(f"idx={idx}: unmatched dollar delimiter")
    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        if problem.count(opening) != problem.count(closing):
            raise ValueError(f"idx={idx}: unmatched {opening}/{closing}")
    if problem.count(r"\begin{") != problem.count(r"\end{"):
        raise ValueError(f"idx={idx}: unmatched LaTeX environment")


def validate(rows: list[dict]) -> dict:
    expected_indices = list(range(5000, 5112))
    actual_indices = [row.get("idx") for row in rows]
    if actual_indices != expected_indices:
        raise ValueError("indices must be exactly 5000..5111 in order")

    normalized = set()
    english = 0
    for row in rows:
        idx = row["idx"]
        if set(row) != {"idx", "problem", "answer", "subject", "source"}:
            raise ValueError(f"idx={idx}: unexpected JSON fields")
        if row["source"] != SOURCE:
            raise ValueError(f"idx={idx}: wrong source")
        if not all(str(row[key]).strip() for key in ("problem", "answer", "subject")):
            raise ValueError(f"idx={idx}: empty required field")

        problem = str(row["problem"])
        if not problem.endswith(BOXED_SUFFIX):
            raise ValueError(f"idx={idx}: missing stable boxed suffix")
        if any(ord(char) < 32 and char not in "\n\r\t" for char in problem):
            raise ValueError(f"idx={idx}: control character in problem")
        _validate_math_markup(idx, problem)

        key = _normalized_problem(problem.removesuffix(BOXED_SUFFIX))
        if key in normalized:
            raise ValueError(f"idx={idx}: duplicate normalized problem")
        normalized.add(key)
        english += int(_is_english(problem))

    if english != 87:
        raise ValueError(f"expected 87 English and 25 Chinese problems, got {english}/{112-english}")
    return {
        "rows": len(rows),
        "english": english,
        "chinese": len(rows) - english,
        "unique": len(normalized),
    }


def main() -> int:
    missing = [path.name for path in SHARDS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing shards: {', '.join(missing)}")
    rows = [row for path in SHARDS for row in _load(path)]
    summary = validate(rows)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": str(OUTPUT.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
