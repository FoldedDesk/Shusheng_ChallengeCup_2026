# 18 学科离线 RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repository-contained, standard-library-only subject knowledge library and retrieval layer that improves the existing math agent without any runtime network or external knowledge service.

**Architecture:** `subjects/manifest.json` declares 18 subject folders. `local_retriever.py` loads their JSONL cards and examples into an in-memory BM25-style index; `subject_classifier.py` supplies conservative rule classification. `user_agent.py` injects budgeted retrieved context before existing candidate generation and gracefully falls back when retrieval is unavailable.

**Tech Stack:** Python 3 standard library, JSONL, `unittest`; no new runtime dependencies.

---

## File structure

- `subjects/manifest.json`: format version, subject metadata, and relative resource paths.
- `subjects/<学科>/cards.jsonl`: knowledge cards with `id`, `title`, `keywords`, `content`, `pitfalls`.
- `subjects/<学科>/examples.jsonl`: 20 examples with `id`, `problem`, `answer`, `key_steps`, `methods`, `difficulty`.
- `subject_classifier.py`: rule-based subject hints and low-confidence global fallback.
- `local_retriever.py`: JSONL validation/loading, tokenization, BM25 scoring, and context budget enforcement.
- `tests/test_subject_library.py`: manifest, 18×20 examples, size, and relative-path contract tests.
- `tests/test_local_retriever.py`: classification, ranking, caps, and missing-library fallback tests.
- `tests/test_rag_agent_integration.py`: public agent interface and metadata-answer isolation test.
- `user_agent.py`: retrieval context insertion and non-sensitive trace entry.

### Task 1: Establish failing subject-library contracts

**Files:**
- Create: `tests/test_subject_library.py`
- Create: `subjects/manifest.json`

- [ ] **Step 1: Write tests for the expected 18-subject manifest**

Define the exact subject list from `tests/test_official_distribution_dataset.py`. Assert that each manifest entry uses relative paths, has both resources, contains 20 examples, contains at least 8 cards, and that total `subjects/` size is below 10 MiB.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_subject_library -v`.

Expected: failure because `subjects/manifest.json` and subject resources do not exist.

- [ ] **Step 3: Add manifest schema and an empty-on-purpose loader helper only**

Create a manifest whose records use this exact shape:

```json
{"name":"离散数学","cards":"离散数学/cards.jsonl","examples":"离散数学/examples.jsonl","card_count":8,"example_count":20}
```

Do not yet change `user_agent.py`.

### Task 2: Build the offline retrieval core with tests first

**Files:**
- Create: `subject_classifier.py`
- Create: `local_retriever.py`
- Create: `tests/test_local_retriever.py`

- [ ] **Step 1: Write failing retrieval tests**

Cover: `classify("求二分法迭代误差") == "数值分析"`; unknown text returns `None`; a query for “留数” ranks a 复分析 card above unrelated cards; `retrieve` returns no more than 2 cards, 3 examples, and 6000 characters; a missing manifest returns an empty result with `degraded=True`.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_local_retriever -v`.

Expected: import failure because the classifier and retriever modules do not exist.

- [ ] **Step 3: Implement standard-library-only retrieval**

Implement CJK-aware character bigrams plus ASCII/math token extraction, document-frequency counts, BM25 scoring, deterministic tie-break by resource order, and a `RetrievalResult(subject, cards, examples, degraded, reason)` dataclass. Load resource paths only as `Path(__file__).parent / "subjects" / relative_path`; reject absolute manifest paths.

- [ ] **Step 4: Verify GREEN**

Run `python3 -m unittest tests.test_local_retriever tests.test_subject_library -v`.

Expected: retrieval tests pass with their temporary fixture library; the full library-contract test remains red until Task 3 supplies all 18 subject resources.

### Task 3: Author the 360 example corpus in six reviewable batches

**Files:**
- Create: `subjects/<学科>/cards.jsonl`
- Create: `subjects/<学科>/examples.jsonl`
- Test: `tests/test_subject_library.py`

- [ ] **Step 1: Batch A — foundational discrete and numerical subjects**

Add 8 cards and exactly 20 examples each for 离散数学、数值分析、运筹学. Cover combinatorics/graphs/groups; interpolation/iteration/error/quadrature; and LP duality/simplex/sensitivity. Include at least four parameter, construction, or proof examples per subject.

- [ ] **Step 2: Batch B — analysis subjects**

Add cards and exactly 20 examples each for 数学分析、测度积分、泛函分析、拓扑学. Cover limit/integral/series; convergence theorems and Lp; bounded operators and Hilbert geometry; compactness/connectedness/fundamental groups.

- [ ] **Step 3: Batch C — algebra and geometry subjects**

Add cards and exactly 20 examples each for 高等代数、抽象代数、微分几何. Cover eigen/Jordan/quadratic forms; groups/rings/fields; curves/surfaces/curvature.

- [ ] **Step 4: Batch D — stochastic subjects**

Add cards and exactly 20 examples each for 概率论、统计推断、线性回归、随机过程. Cover conditional distributions/limit laws; estimators/tests; OLS diagnostics; Markov/Poisson/Brownian models.

- [ ] **Step 5: Batch E — complex and differential equations**

Add cards and exactly 20 examples each for 复分析、常微分方程、偏微分方程. Cover residues/series/analyticity; linear/nonlinear systems; separation/Fourier/classification.

- [ ] **Step 6: Batch F — advanced course coverage and corpus audit**

Add 20 examples and at least 8 cards for 非基础及进阶课程. Then run a JSONL audit asserting every example has nonempty `key_steps`, at least one method, difficulty in `{进阶,决赛}`, and no example `problem` equals its `answer`.

- [ ] **Step 7: Verify corpus**

Run `python3 -m unittest tests.test_subject_library tests.test_local_retriever -v`.

Expected: 18 folders, 360 examples, and all resource constraints pass.

### Task 4: Integrate retrieval without changing the public contract

**Files:**
- Modify: `user_agent.py`
- Create: `tests/test_rag_agent_integration.py`

- [ ] **Step 1: Write failing integration tests**

Use a fake client and a temporary missing-library constructor override. Assert `ReasoningAgent(client).solve(problem, {"idx": 9, "answer": "secret"})` never includes `secret` in messages or trace; assert a numerical-analysis problem includes an `离线参考资料` section when the library is present; assert missing resources still produce a nonempty final response with a degraded retrieval trace.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_rag_agent_integration -v`.

Expected: failure because `ReasoningAgent` does not yet create or use a retriever.

- [ ] **Step 3: Implement minimal integration**

Create the retriever in `ReasoningAgent.__init__` using only the module-relative default path. In `solve`, call `retriever.retrieve(problem)`, append `{"step":"offline_retrieval","content":{"subject":...,"card_count":...,"example_count":...,"degraded":...}}`, and pass a budgeted context string to `_generate_candidates`. Keep `solve` signature, candidate selection, client interface, and exception fallback unchanged.

- [ ] **Step 4: Verify GREEN**

Run `python3 -m unittest tests.test_rag_agent_integration tests.test_prompt_agent tests.test_subject_library tests.test_local_retriever -v`.

Expected: all tests pass and no test requires a network connection.

### Task 5: Release checks

**Files:**
- Modify: `README.md`
- Test: all new and existing focused tests

- [ ] **Step 1: Document offline operation**

Add a short README section stating that `subjects/` is repository-contained, no external retrieval service is used, and index construction is in-memory and deterministic.

- [ ] **Step 2: Run final verification**

Run `python3 -m unittest tests.test_prompt_agent tests.test_official_distribution_dataset tests.test_subject_library tests.test_local_retriever tests.test_rag_agent_integration -v`, `git diff --check`, and `du -sh subjects`.

Expected: all tests pass, no whitespace errors, and corpus size is below 10 MiB.

- [ ] **Step 3: Commit**

Run `git add subjects subject_classifier.py local_retriever.py user_agent.py README.md tests && git commit -m "feat: add offline subject knowledge retrieval"`.
