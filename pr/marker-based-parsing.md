# FD08.2.0 — 基于标记的解析：防御 Intern-S Thinking Process 污染

## 问题诊断

Intern-S 模型在所有 API 调用中都强制输出 "Thinking Process" 文本（类似 DeepSeek-R1），且 thinking 与正式回复共用一个 token budget。这导致两个层面的污染：

1. **答案提取污染**：extraction agent 输出的 thinking 文本被当成答案（如 `*   $d=4$: $\mathbb{F}_{3^4} = \mathbb{F}_{81}$`、`NO-ANSWER.` 等）
2. **验证器解析污染**：verifier 的 thinking 中的 `Line 1: -1/8`、`Line 3: 10` 等被解析为 self_answer、confidence 等
3. **Policy 截断**：max_tokens=4096 下，thinking 吃掉 ~400 token，长推理问题（有限域、微积分）在写完答案前就被截断

## 修改内容

### 1. Prompt 改为标记格式

**提取 Prompt**：要求 `ANSWER: <答案>` 格式
```
从以下数学解答中提取最终答案。用 ANSWER: <答案> 的格式输出，不要输出任何其他内容。找不到答案则输出 ANSWER: NO_ANSWER
```

**验证 Prompt**：要求 `SELF_ANSWER:` / `MATCH:` / `CONFIDENCE:` 三标签格式
```
用以下格式输出（严格按此格式，每行一个标签）：
SELF_ANSWER: <你的答案>
MATCH: YES 或 NO
CONFIDENCE: <0-10的整数，10为非常确定>
```

### 2. 解析改为标签正则 + LAST match

**提取解析** (`_llm_extract_answer`)：
- `re.findall(r"ANSWER\s*[:：]\s*(.+?)(?:\n|$)", ..., re.I)[-1]` 取最后一个 ANSWER: 行
- NO_ANSWER 检测改用正则 `^NO[_\-\s]*ANSWER`（兼容 NO-ANSWER、NO_ANSWER 等变体）
- 答案尾部清理：`re.sub(r"[\`'\".,;:!?）\]】\s]+$", "", ans)`
- 失败时回退到原始文本的 regex 提取

**验证解析** (`_parse_verdict`)：
- 旧版 50 行逐行状态机 → 删除
- 改为三个独立 `findall(...)[-1]`：`SELF_ANSWER:`、`MATCH:`、`CONFIDENCE:`
- 每个标记取最后匹配（thinking 中可能出现同样的标签，但正式输出一定在最后）

### 3. Token 配额调整

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `max_tokens` (policy) | 4096 | 8192 | 长推理问题被截断，thinking+回复共用 budget |
| `extraction_max_tokens` | 512 | 1024 | extraction 回复短但 thinking 仍吃 ~400 |

### 4. Trace 增强

extract_answer trace 中新增 `raw_response` 字段，记录 extraction agent 的完整原始输出，方便后续排查。

## 改动范围

仅 `user_agent.py` DESIGN AREA 内。改动函数：
- `VERIFIER_PROMPT`、`EXTRACTION_PROMPT`
- `AgentConfig.max_tokens`、`AgentConfig.extraction_max_tokens`
- `_extract_answer`（返回值从 2-tuple 改为 3-tuple）
- `_llm_extract_answer`（返回值从 `str|None` 改为 `(str|None, str)`）
- `_parse_verdict`（完全重写）
- `solve()` 中 extract_answer trace 构建

## 验证结果

| 问题 | 答案 | final_response | 提取 | 选择方式 |
|------|------|---------------|------|----------|
| 0 (有限域) | 72 | 72 ✓ | 2/3 正确提取 | majority_vote (2票) |
| 1 (微积分) | -1 | -1 ✓ | 2/3 正确提取 | majority_vote (2票) |
| 2 (留数) | -1/8 | -1/8 ✓ | 2/3 正确提取 | majority_vote (2票) |

所有问题三题全过。每次有 1/3 的 extraction 调用仍输出 thinking 文本（LLM 未遵循 ANSWER: 格式），但多数投票机制成功兜底——只要 ≥2 个候选提取出正确答案即可。
