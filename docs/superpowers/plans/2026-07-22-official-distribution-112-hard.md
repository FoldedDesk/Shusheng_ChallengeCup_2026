# 高难度 112 题官方分布回归集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone 112-question Chinese mathematics regression set that preserves the existing subject distribution while requiring substantially deeper reasoning.

**Architecture:** Keep the JSONL schema and distribution constants shared with the existing set. Refactor the dataset test into a reusable assertion helper, then validate the new hard dataset at its own index range and source identifier. The dataset itself contains no runtime code and does not alter the agent or evaluator.

**Tech Stack:** JSONL, Python 3 standard library, `unittest`.

---

## File structure

- `sample_data/official_distribution_112_hard.jsonl`: 112 high-difficulty Chinese mathematics records, indices 4000–4111.
- `tests/test_official_distribution_dataset.py`: shared structural and distribution validation for both datasets.

### Task 1: Add a failing high-difficulty dataset contract

**Files:**
- Modify: `tests/test_official_distribution_dataset.py`
- Test: `tests/test_official_distribution_dataset.py`

- [ ] **Step 1: Write the failing test and reusable helper**

Replace the single `DATASET_PATH` constant with these constants and add the helper inside `OfficialDistributionDatasetTest`:

```python
BASE_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "official_distribution_112.jsonl"
HARD_DATASET_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "official_distribution_112_hard.jsonl"

def assert_dataset(self, path, start_idx, source, min_problem_length=1):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    self.assertEqual(len(rows), 112)
    self.assertEqual([row["idx"] for row in rows], list(range(start_idx, start_idx + 112)))
    self.assertEqual(Counter(row["subject"] for row in rows), EXPECTED_COUNTS)
    for row in rows:
        self.assertEqual(row["source"], source)
        self.assertTrue(all(str(row[key]).strip() for key in ("problem", "answer", "subject")))
        self.assertGreaterEqual(len(row["problem"]), min_problem_length)
```

Replace the original test with:

```python
def test_base_dataset_has_required_structure_and_distribution(self):
    self.assert_dataset(BASE_DATASET_PATH, 3000, "official_distribution_112")

def test_hard_dataset_has_required_structure_and_distribution(self):
    self.assert_dataset(HARD_DATASET_PATH, 4000, "official_distribution_112_hard", min_problem_length=30)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_official_distribution_dataset -v
```

Expected: `test_hard_dataset_has_required_structure_and_distribution` raises `FileNotFoundError` because the hard JSONL file does not exist.

### Task 2: Create the high-difficulty JSONL dataset

**Files:**
- Create: `sample_data/official_distribution_112_hard.jsonl`
- Test: `tests/test_official_distribution_dataset.py`

- [ ] **Step 1: Add 112 JSONL records using the established schema**

Create records with these non-negotiable properties:

```json
{"idx":4000,"problem":"设...（中文、多步推导题面）","answer":"...（唯一或明确等价答案）","subject":"离散数学","source":"official_distribution_112_hard"}
```

Use indices 4000–4023 for 24 离散数学题，4024–4036 for 13 数值分析题, then retain the exact order and counts from `EXPECTED_COUNTS` through index 4111. For each subject, make most questions require at least two derivation steps; distribute approximately 22 parameter-classification, construction, counterexample, or multi-lemma questions across the set.

- [ ] **Step 2: Verify the dataset contract passes**

Run:

```bash
python3 -m unittest tests.test_official_distribution_dataset -v
```

Expected: both base and hard dataset tests pass.

- [ ] **Step 3: Verify the complete focused regression suite**

Run:

```bash
python3 -m unittest tests.test_prompt_agent tests.test_official_distribution_dataset -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 4: Inspect the final dataset change**

Run:

```bash
git diff --check && git diff --stat -- sample_data/official_distribution_112_hard.jsonl tests/test_official_distribution_dataset.py
```

Expected: no whitespace errors; the dataset has 112 records and the test covers both data files.

- [ ] **Step 5: Commit the dataset**

```bash
git add sample_data/official_distribution_112_hard.jsonl tests/test_official_distribution_dataset.py
git commit -m "test: add harder 112-question regression set"
```
