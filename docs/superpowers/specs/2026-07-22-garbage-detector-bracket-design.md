# Garbage Detector Bracket Handling

## Goal

Prevent valid bracketed mathematical answers from being converted to `GARBAGE`, while retaining rejection of prompt-template pollution.

## Scope

Change only `ReasoningAgent._looks_like_garbage` and its focused tests.

## Design

Replace the broad leading-character rejection for `[` with a targeted rejection for a bracketed value whose first non-space character after `[` is a single or double quote. Existing checks for placeholders, instruction phrases, English reasoning, and excessive length remain unchanged.

This accepts mathematical interval, set, and vector forms such as `[1, 1.5]`, `[0, 1]`, and `[-1, 1]`. It continues to reject obvious quoted-list pollution such as `["placeholder"]` and `['placeholder']`.

## Data Flow

`_regex_fast_extract` and LLM extraction both call `_looks_like_garbage`. A valid bracketed answer must therefore pass that shared filter at every extraction path; otherwise regex fallback returns `GARBAGE` even when candidate generation was correct.

## Tests

Add direct unit assertions for valid interval answers and quoted-list garbage. The new interval assertion must fail against the existing broad `[` rule before the production change, then pass after it.

## Non-goals

Do not introduce a general mathematical-expression parser or modify candidate generation, LLM prompts, voting, or finalization.
