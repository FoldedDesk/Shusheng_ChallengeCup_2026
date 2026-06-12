# PR：多数投票选择

**版本：** FD08.1.0 → FD08.2.0
**日期：** 2026-06-12
**类型：** 功能增强

## 改动内容

新增多数投票作为首选策略，verifier 评分作为回退：

1. **Phase 3：多数投票** — 提取并归一化所有候选答案后，统计出现次数：
   - 若 ≥2 个候选答案一致 且 占比 ≥ `majority_vote_threshold`（默认 0.5）：直接采用多数答案，**跳过 verifier**。
   - 否则：回退到 verifier 评分（Phase 4）。

2. **新增配置项：** `majority_vote_threshold: float = 0.5` — 达成多数共识所需的最低占比。

3. **Trace 增强：** 多数投票成功时，添加 `majority_vote` 步骤，记录答案、票数和所选候选。

## 原因

自一致性（self-consistency）是 LLM 推理中的强信号：如果多次独立采样产生相同答案，该答案更可能正确。verifier 即使经过改进仍是 LLM，仍然可能判断错误。当候选答案一致时，根本不需要 verifier，还能省 API 调用。

以 `policy_sample_times=3`、`threshold=0.5` 为例：
- 2/3 一致 → 多数胜出（66% ≥ 50%）
- 3/3 一致 → 多数胜出
- 各自不同（1/1/1）→ 回退到 verifier

## 预期效果

- 候选答案一致时（简单题常见），跳过 verifier → 每题节省 6 次 API 调用
- 选择基于答案共识而非不可靠的评判者
- 难题候选不一致时，verifier 仍提供备选方案

## 验证方法

运行测试命令。检查 trace：
- 若 ≥2 个候选归一化答案相同：应看到 `majority_vote` 步骤，而非 `verifier_call_*` 和 `score_summary`
- 最终答案应保持正确
