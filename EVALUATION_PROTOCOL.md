# Evaluation Isolation Protocol

This project separates engineering regression tests from unseen-score evidence.
The separation is procedural as well as technical: a dataset stops being blind
after its first scored run or after any item-level inspection.

## Tiers

### DEVELOPMENT_ONLY

This tier includes every dataset, replay, or output already used for debugging,
prompt tuning, error attribution, or per-item review. In particular, all local
`sample_data/`, `judge/`, `tests/`, and `validation_outputs/` artifacts are
development material. They may verify imports, output validity, contracts,
metamorphic behavior, deterministic tools, and regressions. Their accuracy must
not be presented as an estimate of performance on new platform questions.

### BLIND_HOLDOUT

A blind holdout must be stored outside the repository and controlled by a person
or evaluation process that does not expose item text, answers, IDs, hashes, or
per-item outcomes during development. A major candidate may run it once. Only
aggregate accuracy, invalid count, call count, truncation count, and latency may
be returned. After that run, the set becomes `DEVELOPMENT_ONLY`.

### FINAL_FROZEN

The final frozen set remains outside the repository and is not run until the
candidate architecture and parameters are frozen. It is used once to estimate
generalization. Its questions and answers must never be copied into runtime
code, prompts, RAG resources, tests, scripts, or configuration.

## Production Change Gate

A production change must satisfy all of the following:

1. It is stated as a general mathematical or engineering capability, not an
   item-specific correction.
2. It passes synthetic and metamorphic tests that do not reproduce an evaluated
   problem or its distinctive constants.
3. It does not read an `answer` field at runtime and does not branch on item ID,
   order, hash, fingerprint, or a unique phrase from an evaluation item.
4. Development replay results are reported as regression evidence only.
5. A claimed generalization improvement requires a still-blind aggregate run.

## Required Reporting

Every experiment report labels its source tier and separately records:

- structural validity and obligation completeness;
- semantic accuracy when a legitimate offline answer key exists;
- model calls, provider truncations, elapsed time, and transport failures;
- paired wrong-to-right and right-to-wrong counts when a shared frozen solve is
  available;
- whether the set was blind before the run.

No result from `DEVELOPMENT_ONLY` is converted into a predicted platform score.

## Independent Silver References

A development-only silver reference may be generated only from a published
public problem, a self-authored problem, or a synthetic problem. Non-public
evaluation questions, Judge replays, hidden/private problem-only exports, and
platform evaluation logs must not be independently labeled or turned into an
answer bank, even when their official answers are absent. Such inputs are
restricted to aggregate runtime, validity, truncation, and cost checks.

The generator requires an explicit permitted source origin and rejects paths or
records marked as Judge, replay, hidden, private evaluation, or platform
evaluation data. It also rejects `answer`, `ground_truth`, `reward_model`,
judgement, and equivalent answer-bearing fields before any model call. Silver
references are confidence-graded hypotheses, not official ground truth. Only
high-confidence agreement is used for local regression, and it cannot justify
item-specific production code.

Silver records and merged benchmark files stay in ignored development output
directories. They must never be imported by `user_agent.py`, runtime tools,
prompts, or RAG resources, and they do not become evidence of unseen-platform
accuracy.
