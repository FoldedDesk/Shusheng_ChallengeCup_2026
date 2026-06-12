# PR：Verifier 独立求解 + 答案匹配

**版本：** FD08.1.0 → FD08.2.0
**日期：** 2026-06-12
**类型：** 功能增强

## 改动内容

将 verifier 从主观判断（"这个解答看起来对不对"）改为独立求解 + 答案比对：

1. **重写 `VERIFIER_PROMPT`** — 指示 verifier：
   - 先独立求解题目，得出自己的答案
   - 再将自己的答案与候选答案进行对比
   - 输出格式从 `VERDICT: A/B` 改为 `SELF_ANSWER: ...` / `MATCH: YES/NO` / `CONFIDENCE: 0-10`

2. **重写 `_parse_verdict()`** — 现在返回 `(is_match, confidence, self_answer)` 三元组，而非之前的 `(is_correct, confidence)`。解析 `SELF_ANSWER`、`MATCH`、`CONFIDENCE` 三个字段。

3. **更新 `_verify_candidate()`** — 在 trace 中存储 `self_answer` 供后续使用。评分逻辑：`MATCH=YES` 时得 `confidence` 分，否则得 `0.0` 分。

## 原因

baseline verifier 不一致的原因：
- 它试图判断推理过程是否正确，而非判断答案是否正确
- 同一个模型既生成候选又评判候选，存在自评偏差
- 对同一正确答案的评判经常翻转（如 idx=0 的 candidate 0 两次都得 B，但它的推理和 candidate 1 几乎一样且答案也是 72）

独立求解 + 答案比对更客观：如果 verifier 独立算出的答案和候选的答案一致，候选大概率正确。同时 verifier 的自解答案可作为额外的共识信号。

## 预期效果

- 验证更一致（正确的候选不再被误判为错误）
- verifier 的自解答案为后续多数投票提供额外数据点
- verifier 的 token 消耗略有增加（需要写出自己的求解过程）

## 验证方法

运行测试命令。检查 trace：
- `verifier_call_*` 条目应包含 `self_answer` 字段
- `parsed_match` 应比旧的 `parsed_correct` 更一致
- 最终答案在已知正确答案的题目上应保持正确
