# Garbage Detector Bracket Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept valid bracketed mathematical answers during extraction while continuing to reject quoted-list prompt pollution.

**Architecture:** Keep the shared `ReasoningAgent._looks_like_garbage` filter as the single decision point. Narrow its leading-character rule from “any `[`” to “a `[` followed by optional whitespace and a quote”, so all extraction paths consistently accept intervals without changing their control flow.

**Tech Stack:** Python 3, standard-library `re`, `unittest`.

---

## File structure

- `user_agent.py`: shared garbage-detection predicate used by fast regex, LLM extraction, and regex fallback.
- `tests/test_prompt_agent.py`: focused `unittest` regression coverage for accepted intervals and rejected quoted lists.

### Task 1: Specify the bracketed-answer regression

**Files:**
- Modify: `tests/test_prompt_agent.py`
- Test: `tests/test_prompt_agent.py`

- [ ] **Step 1: Write the failing test**

Add this test class after `PromptAgentTest`:

```python
class GarbageDetectionTest(unittest.TestCase):
    def test_accepts_bracketed_mathematical_intervals(self):
        self.assertFalse(ReasoningAgent._looks_like_garbage("[1, 1.5]"))
        self.assertFalse(ReasoningAgent._looks_like_garbage("[0, 1]"))
        self.assertFalse(ReasoningAgent._looks_like_garbage("[-1, 1]"))

    def test_rejects_quoted_bracketed_placeholder(self):
        self.assertTrue(ReasoningAgent._looks_like_garbage('["placeholder"]'))
        self.assertTrue(ReasoningAgent._looks_like_garbage("['placeholder']"))
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_prompt_agent.GarbageDetectionTest -v
```

Expected: `test_accepts_bracketed_mathematical_intervals` fails because the existing `^[\"'\[]` rule classifies every leading `[` as garbage.

### Task 2: Narrow the garbage rule

**Files:**
- Modify: `user_agent.py:577`
- Test: `tests/test_prompt_agent.py`

- [ ] **Step 1: Apply the minimal production change**

Replace the broad rule:

```python
if re.match(r"^[\"'\[]", text):
```

with two targeted checks:

```python
if re.match(r"^[\"']", text):
    return True
if re.match(r"^\[\s*[\"']", text):
```

Leave the existing `return True` after the rule in place. Do not change the existing placeholder, instruction, prose, or length checks.

- [ ] **Step 2: Run the focused regression test**

Run:

```bash
python3 -m unittest tests.test_prompt_agent.GarbageDetectionTest -v
```

Expected: both tests pass.

- [ ] **Step 3: Run the complete unit-test module**

Run:

```bash
python3 -m unittest tests.test_prompt_agent -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 4: Inspect the final change**

Run:

```bash
git diff --check && git diff -- user_agent.py tests/test_prompt_agent.py
```

Expected: no whitespace errors; diff contains only the targeted predicate and its regression tests.

- [ ] **Step 5: Commit the implementation**

```bash
git add user_agent.py tests/test_prompt_agent.py
git commit -m "fix: accept bracketed mathematical answers"
```
