# 完整解答链路可靠性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 可靠识别中英文完整解答题，在不牺牲降级安全性的前提下跳过不必要的审核和整理模型调用。

**Architecture:** `ReasoningAgent` 在生成候选后，为完整解答题执行纯本地候选资格检查、质量评分和结论一致性检查。一致时直接返回最高分候选；冲突时才审核，且审核意见非“无”时才整理。所有模型调用失败和不合格候选均回退至本地最高分合格候选。

**Tech Stack:** Python 3、标准库 `re` 与 `unittest`、现有 OpenAI-compatible `client.chat` 包装器。

---

## File structure

- `user_agent.py`：完整题识别、候选质量门、调用决策和 trace。
- `tests/test_prompt_agent.py`：顺序假客户端与完整题的离线回归测试。

### Task 1: 写出完整题本地决策的失败测试

**Files:**
- Modify: `tests/test_prompt_agent.py`

- [ ] **Step 1: 扩展导入与顺序假客户端**

```python
from user_agent import AgentConfig, AgentMessage, ReasoningAgent, _PromptAgent


class SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.2, max_tokens=4096):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return self.responses.pop(0)
```

- [ ] **Step 2: 添加英文分流、一致跳过、只含答案回退和冲突整理测试**

```python
class FullSolutionFlowTest(unittest.TestCase):
    def _agent(self, responses):
        return ReasoningAgent(
            SequencedClient(responses),
            AgentConfig(policy_sample_times=2, use_llm_extraction=False),
        )

    def test_english_proof_with_agreeing_complete_candidates_skips_model_review(self):
        candidate = "设 x=1。由 x=x 可得结论。\n【最终答案】x=1"
        agent = self._agent([candidate, candidate])
        result = agent.solve("Prove that x = 1. Show all steps.", {"idx": 1})
        self.assertEqual(result["final_response"], candidate)
        self.assertEqual(len(agent.client.calls), 2)
        self.assertTrue(any(item["step"] == "full_solution_consensus" for item in result["trace"]))

    def test_answer_only_candidates_do_not_enter_full_solution_fallback(self):
        agent = self._agent(["【最终答案】1", "【最终答案】1"])
        result = agent.solve("请证明结论为 1", {"idx": 2})
        self.assertEqual(result["final_response"], "TRUNCATED_ALL")

    def test_conflicting_complete_candidates_are_audited_and_repaired(self):
        first = "由定义逐步计算得到 1。\n【最终答案】1"
        second = "由另一条完整推导得到 2。\n【最终答案】2"
        repaired = "【解答】\n由第二个候选的推导可知结论成立。\n【结论】\n答案为 2。"
        agent = self._agent([first, second, "CHOICE: 1\nISSUES: 统一表述", repaired])
        result = agent.solve("请推导结果", {"idx": 3})
        self.assertEqual(result["final_response"], repaired)
        self.assertEqual(len(agent.client.calls), 4)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest -v`

Expected: 至少英文分流、一致跳过和只含答案回退测试失败，因为当前实现会调用审核/整理器并接受只含答案的候选。

- [ ] **Step 4: 提交测试基线**

```bash
rtk git add tests/test_prompt_agent.py
rtk git commit -m "test: cover full solution routing decisions"
```

### Task 2: 实现候选质量门与按需调用

**Files:**
- Modify: `user_agent.py`
- Test: `tests/test_prompt_agent.py`

- [ ] **Step 1: 扩展完整题识别与生成提示**

将 `_requires_full_solution` 改为忽略大小写的中英文正则，匹配中文词和 `prove|proof|derive|derivation|show all steps|explain|justify`。将完整题的生成要求改为明确输出关键推导和 `【最终答案】`，并使 fallback prompt 在完整题场景下采用相同要求。

- [ ] **Step 2: 添加本地候选资格与评分方法**

在 `ReasoningAgent` 中添加：

```python
def _full_solution_answer(self, candidate: str) -> Optional[str]: ...
def _is_complete_solution_candidate(self, candidate: str) -> bool: ...
def _full_solution_quality_score(self, candidate: str) -> int: ...
def _select_best_full_solution(self, candidates: List[str]) -> int: ...
```

资格检查要求候选至少 40 个非空字符、有 `【最终答案】` 或 `【结论】` 后的非空答案，并在去除答案行后仍有至少 20 个字符的推导文本。评分由完整结论标识、推导关键词和文本长度组成；同分时保留较早候选。

- [ ] **Step 3: 重写完整题分支**

在 `solve()` 中将完整题交给 `_solve_full_solution`，后者先过滤不合格候选；没有合格候选时追加 `full_solution_no_usable_candidate` trace 并返回 `TRUNCATED_ALL`。对合格候选提取并规范化最终答案：所有答案一致时追加 `full_solution_consensus` trace，直接返回最高分候选；答案冲突时调用 `_audit_full_solutions`。

审核返回的 `ISSUES` 去除空白后为 `无`、`none` 或 `no issues` 时，不调用 `_repair_full_solution`，追加 `solution_repair` trace（`skipped=True`）并直接返回所选候选。其他审核意见才调用整理器；无效整理结果或异常时返回最高分候选。

- [ ] **Step 4: 加强整理输出验收**

```python
@staticmethod
def _is_usable_full_solution(solution: Optional[str]) -> bool:
    if not solution or len(solution.strip()) < 40:
        return False
    match = re.search(r"【结论】\s*(.+)", solution, re.DOTALL)
    return "【解答】" in solution and bool(match and match.group(1).strip())
```

- [ ] **Step 5: 运行完整题测试确认通过**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest -v`

Expected: 3 tests pass；一致候选只发起两次生成调用，冲突候选发起两次生成、一次审核和一次整理调用。

- [ ] **Step 6: 提交实现**

```bash
rtk git add user_agent.py tests/test_prompt_agent.py
rtk git commit -m "feat: gate and streamline full solution review"
```

### Task 3: 覆盖失败降级并完成回归验证

**Files:**
- Modify: `tests/test_prompt_agent.py`
- Verify: `user_agent.py`, `main.py`, `llm_client.py`

- [ ] **Step 1: 添加无效审核和无效整理的回退测试**

```python
def test_invalid_audit_and_repair_fall_back_to_highest_quality_candidate(self):
    weak = "有一些说明文字但没有完整推导。\n【最终答案】1"
    strong = "根据定义进行两步推导，先得到中间式，再代入化简得到结果。\n【最终答案】2"
    agent = self._agent([weak, strong, "CHOICE: 9\nISSUES: 无"])
    result = agent.solve("请证明该结论", {"idx": 4})
    self.assertEqual(result["final_response"], strong)
    audit = next(item for item in result["trace"] if item["step"] == "solution_audit")
    self.assertEqual(audit["content"]["selected_candidate"], 1)
```

- [ ] **Step 2: 运行降级测试确认通过**

Run: `rtk python -m unittest tests.test_prompt_agent.FullSolutionFlowTest.test_invalid_audit_and_repair_fall_back_to_highest_quality_candidate -v`

Expected: PASS；非法 `CHOICE` 不会越界，最终返回本地质量最高的候选。

- [ ] **Step 3: 运行完整回归与静态检查**

Run: `rtk python -m unittest discover -s tests -v && rtk python -m py_compile user_agent.py main.py llm_client.py && rtk git diff --check`

Expected: 所有单元测试通过、编译成功且无空白错误。

- [ ] **Step 4: 审查暂存范围并提交验证测试**

```bash
rtk git status --short
rtk git add tests/test_prompt_agent.py
rtk git commit -m "test: cover full solution failure fallbacks"
```

仅暂存本任务修改的 `tests/test_prompt_agent.py`，保留用户已有的未提交文件不变。
