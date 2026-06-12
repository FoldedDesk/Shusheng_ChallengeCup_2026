# PR：分步推理

**版本：** FD08.1.0 → FD08.2.0
**日期：** 2026-06-12
**类型：** 功能增强

## 改动内容

新增可选的分步（多阶段）推理模式，作为一次性生成的替代方案：

1. **`STAGE_PROMPTS`** — 四个阶段专用提示词：
   - 第一步：分析题意（已知条件、求解目标、约束）
   - 第二步：提出解题思路（无需计算）
   - 第三步：逐步推导和计算
   - 第四步：用【最终答案】标记写出最终答案

2. **`_generate_candidates_staged()`** — 每个候选通过 4 次连续 LLM 调用生成，每次调用都接收之前所有阶段的上下文。

3. **`_generate_candidates_oneshot()`** — 原 `_generate_candidates()` 重命名，一次性生成逻辑不变。

4. **`_generate_candidates()`** 现在根据 `config.use_staged_reasoning` 分发。

5. **新增配置项：**
   - `use_staged_reasoning: bool = False` — 分步推理开关（默认关闭）
   - `staged_stages: int = 4` — 推理阶段数

## 原因

Intern-S 的 thinking 过程较冗长，在单次长生成中容易失去焦点。将问题拆分为阶段：
- 每阶段目标明确 → 减少干扰
- 每阶段接收之前所有阶段的上下文 → 推理连贯
- 第四阶段明确要求【最终答案】标记 → 答案提取更干净

代价：API 调用量 ×4（3 个候选 × 4 步 = 12 次，对比原本 3 次）。默认关闭。

## 何时启用

对一次性生成容易出错的复杂题目启用：
```python
config = AgentConfig(use_staged_reasoning=True)
```

## 预期效果

- 复杂题目的推理质量更好
- 答案格式更规范（第四阶段明确要求【最终答案】）
- API 成本更高（4 倍 policy 调用）
- 若多数投票成功，verifier 仍会被跳过 → 部分回收成本

## 验证方法

启用分步推理后运行测试（临时修改 AgentConfig 默认值或传入 config）。

检查 trace 中的 `policy_call_{id}_stage{0-3}` 步骤。每个候选应有 4 个阶段。最终答案应保持正确。
