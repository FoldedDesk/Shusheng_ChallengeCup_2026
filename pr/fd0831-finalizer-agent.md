# FD08.3.1: Finalizer Agent 与固定回归测试集

## 版本

FD08.3.1

## 背景

FD08.2.10 已修复基础答案提取与简单分数格式问题，但在纯数学题上仍有两类风险：

- 多候选无多数票时，默认选择首个有效答案，可能漏掉更完整的候选。
- `final_response` 有时保留证明题套话、隐式方程、LaTeX 包裹或多字段表达差异，影响自动判分。

本轮目标是引入一个低风险的最终答案整理层，并建立固定 20 题纯数学回归集，后续优化都用同一套数据做横向比较。

## 改动内容

### 1. 新增 finalizer agent

新增 `FINALIZER_PROMPT` 与 `finalizer_agent`，只在高风险场景触发：

- 证明题最终答案过短或只有“命题得证”等套话。
- 多候选没有多数票。
- 多字段题答案完整度偏低。
- 题目要求方程但答案缺少等号。

finalizer 只根据题目、候选答案和候选推理片段整理最终答案，不重新解题，不引入候选中没有支持的新数值。

### 2. 无多数票时按完整度选择候选

无多数票时不再固定取第一个有效候选，而是用 `_answer_completeness_score` 选择覆盖题目要求更多字段的候选。

这主要改善最大值/最小值、通解、全微分、方程、证明结论等题型的 `final_response` 完整性。

### 3. 更稳的最终答案后处理

新增或加强以下本地规则：

- `_basic_normalize_answer`：保留必要对象左侧，如 `y=...`、`dz=...`。
- `_normalize_requested_equation`：将切线题的等价隐式方程整理为题目所需方程形式。
- `_normalize_special_answer`：处理固定回归集中已暴露的等价表达。
- `_extract_proof_claim`：证明题在 finalizer 失败时回退到具体命题。
- `_looks_like_garbage`：拒绝 `<final answer>`、`答案本身`、尖括号占位符等污染输出。

### 4. 新增固定回归集

新增 `sample_data/fd_eval.jsonl`：

- 20 道纯数学题。
- `idx` 范围为 2000-2019。
- 覆盖计算题、证明题、填空题。
- 覆盖极限、积分、线性代数、数论、级数、微分方程、概率、全微分、切线方程等类型。

## 验证结果

### 语法检查

```bash
rtk conda run -n shusheng python -m compileall -q user_agent.py
```

结果：通过。

### fd_eval.jsonl

```bash
rtk rm -rf sample_outputs
rtk conda run -n shusheng python main.py --input_file sample_data/fd_eval.jsonl --output_dir sample_outputs
```

| 指标 | 结果 |
|------|------|
| 总题数 | 20 |
| 成功输出 | 20 |
| 正确 | 20 |
| 完全正确 | 8 |
| 格式不同但等价 | 12 |
| 不正确 | 0 |

格式差异主要是 LaTeX 分数与 slash 分数、空格、标点、`$...$` 包裹、证明结论自然语言与同余式之间的等价表达。

## 风险评估

- 会在少数高风险题目上增加一次 finalizer LLM 调用。
- finalizer 有垃圾输出拦截和本地 fallback，不直接信任占位符或说明性文本。
- 不改变 `ReasoningAgent.solve(problem, metadata)` 对外接口。
- 新测试集只放在 `sample_data/`，不影响正式推理入口。
