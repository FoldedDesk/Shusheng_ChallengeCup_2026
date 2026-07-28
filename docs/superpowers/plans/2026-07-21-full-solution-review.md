# 完整解答审核流程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让证明、推导和解释题返回可独立判分的完整中文解答，并在新增模型调用不可靠时安全降级。

**Architecture:** `ReasoningAgent.solve()` 在生成候选后按题干关键词分流。完整解答分支使用独立审核提示选择候选、用整理提示生成结构化解答，并在任一调用失效时保留最佳原候选；普通题维持原有提取与投票链路。

**Tech Stack:** Python 3、标准库 `unittest`、现有 OpenAI-compatible `client.chat` 包装器。

---

## File structure

- `user_agent.py`：定义审核与整理提示、完整解答分支及其降级规则。
- `tests/test_prompt_agent.py`：增加基于顺序假客户端的完整解答分支行为测试。

### Task 1: 建立完整解答成功路径测试

**Files:**
- Modify: `tests/test_prompt_agent.py`
- Modify: `user_agent.py`

- [ ] **Step 1: 写入会失败的审核与整理成功路径测试**

在测试文件中增加以下假客户端和测试（并补充 `from user_agent import AgentConfig, ReasoningAgent`）：

```python
class SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.responses.pop(0)


class FullSolutionFlowTest(unittest.TestCase):
    def test_uses_audited_candidate_and_returns_repaired_solution(self):
        client = SequencedClient([
            "候选零。\n【最终答案】0",
            "候选一。\n【最终答案】1",
            "CHOICE: 1\nISSUES: 补全证明结构",
            "【解答】\n由候选一可得关键推导。\n【结论】\n结论为 1。",
        ])
        agent = ReasoningAgent(
            client,
            AgentConfig(policy_sample_times=2, use_llm_extraction=False),
        )

        result = agent.solve("请证明结论为 1", {"idx": 7})

        self.assertEqual(result["final_response"], "【解答】\n由候选一可得关键推导。\n【结论】\n结论为 1。")
        audit = next(item for item in result["trace"] if item["step"] == "solution_audit")
        self.assertEqual(audit["content"]["selected_candidate"], 1)
        self.assertIn("候选一", client.calls[-1]["messages"][1]["content"])
```

- [ ] **Step 2: 运行测试并确认它因缺少完整解答分支而失败**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest.test_uses_audited_candidate_and_returns_repaired_solution -v`

Expected: FAIL；结果仍为普通题最终答案或没有 `solution_audit` trace。

- [ ] **Step 3: 实现最小完整解答成功路径**

在 `user_agent.py` 的提示词区新增：

```python
SOLUTION_AUDIT_PROMPT = """你是数学解答审核器。比较同一道题的候选完整解答，选择最适合提交给评测器的一份。

审核标准：是否回答题目、关键推导或证明步骤是否完整、结论是否由前文支持、是否存在明显矛盾。
不要自行重新解题，不要补造候选中不存在的结论。

严格输出两行：
CHOICE: <候选编号，从0开始>
ISSUES: <需要修复的简短中文说明；无问题写无>"""

SOLUTION_REPAIR_PROMPT = """你是数学解答整理器。只根据题目、候选解答和审核意见，写出可独立判分的完整中文解答。

规则：
1. 保留支撑结论所必需的定义、公式、推导或证明链；不要只写最终结论。
2. 不要引入候选解答没有支持的新结论。
3. 使用如下结构：先写【解答】，最后写【结论】和明确结论。
4. 只输出整理后的解答，不要解释审核过程。"""
```

向 `AgentConfig` 添加 `audit_max_tokens: int = 1024` 与 `repair_max_tokens: int = 4096`；在构造器中用上述提示分别初始化 `_PromptAgent`。在 `solve()` 的 `if not candidates` 后加入：

```python
if self._requires_full_solution(problem):
    return self._solve_full_solution(problem, candidates, idx, trace)
```

实现 `_requires_full_solution()`、`_audit_full_solutions()`、`_repair_full_solution()`、`_is_usable_full_solution()` 与 `_solve_full_solution()`：审核响应只能接受范围内的 `CHOICE`；成功整理结果必须至少 40 个字符且同时带有 `【解答】`、`【结论】`；每一步均追加 trace。将 `_generate_candidates_oneshot()` 的候选判定改为：完整解答题接受任何非空响应，其余题仍要求 `【最终答案】`。

- [ ] **Step 4: 运行成功路径测试**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest.test_uses_audited_candidate_and_returns_repaired_solution -v`

Expected: PASS，且输出显示运行 1 个测试、失败数为 0。

- [ ] **Step 5: 提交成功路径改动**

```bash
rtk git add user_agent.py tests/test_prompt_agent.py
rtk git commit -m "feat: review and repair full math solutions"
```

### Task 2: 建立降级行为测试并实现

**Files:**
- Modify: `tests/test_prompt_agent.py`
- Modify: `user_agent.py`

- [ ] **Step 1: 写入无效审核与无效整理结果的回退测试**

在 `FullSolutionFlowTest` 中加入：

```python
def test_falls_back_to_longest_candidate_when_audit_choice_is_invalid(self):
    long_candidate = "较长候选，包含完整推导。\n【最终答案】正确结论"
    client = SequencedClient([
        "短候选。\n【最终答案】错误结论",
        long_candidate,
        "CHOICE: 9\nISSUES: 无",
        "整理失败",
    ])
    agent = ReasoningAgent(
        client,
        AgentConfig(policy_sample_times=2, use_llm_extraction=False),
    )

    result = agent.solve("请推导该结论", {"idx": 8})

    self.assertEqual(result["final_response"], long_candidate)
    audit = next(item for item in result["trace"] if item["step"] == "solution_audit")
    repair = next(item for item in result["trace"] if item["step"] == "solution_repair")
    self.assertEqual(audit["content"]["selected_candidate"], 1)
    self.assertTrue(repair["content"]["used_fallback"])
```

- [ ] **Step 2: 运行回退测试并确认失败**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest.test_falls_back_to_longest_candidate_when_audit_choice_is_invalid -v`

Expected: FAIL，直到无效编号选择最长候选且无结构化整理结果回退原候选的规则实现完毕。

- [ ] **Step 3: 实现确定性的降级规则**

在 `_audit_full_solutions()` 中：仅在 `0 <= choice < len(candidates)` 时使用模型选择；其他情况返回 `_best_complete_candidate(candidates)`、`"审核不可用，保持原候选"` 与原始审核文本。实现：

```python
@staticmethod
def _best_complete_candidate(candidates: List[str]) -> int:
    return max(range(len(candidates)), key=lambda i: len(candidates[i]))
```

在 `_solve_full_solution()` 中，仅在 `_is_usable_full_solution(repaired)` 为真时使用整理文本；否则返回 `best.strip()`，并将 `used_fallback` 写为 `True`。所有 `client.chat` 异常必须在审核/整理私有方法中捕获，返回相同的降级路径，不得让 `solve()` 抛出异常。

- [ ] **Step 4: 运行回退测试**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest.test_falls_back_to_longest_candidate_when_audit_choice_is_invalid -v`

Expected: PASS，且输出显示运行 1 个测试、失败数为 0。

- [ ] **Step 5: 提交降级行为改动**

```bash
rtk git add user_agent.py tests/test_prompt_agent.py
rtk git commit -m "test: cover full solution fallback behavior"
```

### Task 3: 完整回归验证

**Files:**
- Verify: `user_agent.py`
- Verify: `tests/test_prompt_agent.py`

- [ ] **Step 1: 执行全部单元测试**

Run: `rtk python -m unittest discover -s tests -v`

Expected: 所有测试通过，失败数和错误数均为 0。

- [ ] **Step 2: 编译检查与差异检查**

Run: `rtk python -m py_compile user_agent.py && rtk git diff --check`

Expected: 两个命令均以状态码 0 结束，且无空白错误。

- [ ] **Step 3: 审查提交范围**

Run: `rtk git status --short && rtk git log --oneline -3`

Expected: 新提交只包含 `user_agent.py` 和 `tests/test_prompt_agent.py`，不包含用户已有的无关未提交文件。
