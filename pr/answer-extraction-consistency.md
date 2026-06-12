# PR: 答案提取 + 一致性加权 + 验证器置信度

**版本**: FD08.1.0
**日期**: 2026-06-12
**状态**: 已完成

## 修复记录

### 修复 1（v1）：Intern-S 模型 Thinking 适配

**问题**：Intern-S 模型会在输出开头生成 "Thinking Process" 内部推理文本，导致：
- verifier 用 `max_tokens=256` 时 thinking 占满 tokens，VERDICT 被截断
- 答案/判定标记在文本开头的 thinking 中出现而非末尾的真实输出

**修复**：
1. verifier `max_tokens` 256 → 1024，给 thinking + 判定留足空间
2. `_extract_answer` 改为从文本**末尾**向前搜索标记（取最后一次匹配）
3. `_parse_verdict` 改为取最后匹配的 VERDICT 和 CONFIDENCE
4. user_message 中增加 `【最终答案】` 格式提示，双重强调

### 修复 2（v2）：答案提取改进 + 置信度可视化

**问题**：
- 模型输出如 `10. The size is $81 - 9 = 72$.` 时，提取到的是带步骤号的整行而非纯答案 `72`
- 置信度信息埋在 trace 嵌套 dict 里，不直观

**修复**：
1. `_extract_answer` 新增优先级 4：匹配行末 `= <value>` 模式，提取等号右侧数值
2. `_extract_answer` 优先级 5 增加步号排除：`^\d+[.)、]` 开头的行不再被当作答案
3. trace 新增 `score_summary` 步骤，一行展示所有 candidate 得分：
   ```
   #0 ans=72 score=1.000(v=1.000+c=0.000) | #1 ans=54 score=0.500(v=0.500+c=0.000) | selected=#0
   ```

## 改动概述

在 DESIGN AREA 约束内，优化 agent 的选择机制。零额外 API 调用，纯逻辑改进。

## 具体改动

### 1. POLICY_PROMPT — 结构化输出

- 要求模型用 `【最终答案】<答案>` 格式给出答案
- 答案必须单独一行，不含额外解释

### 2. VERIFIER_PROMPT — 置信度评分

- 从二值（A/B）改为 0~10 置信度
- 输出格式：`VERDICT: A` + `CONFIDENCE: 8`

### 3. 新增方法

| 方法 | 功能 |
|---|---|
| `_extract_answer(text)` | 从模型输出提取答案：`【最终答案】` > `\boxed{}` > `答案：` > 末行 |
| `_normalize_answer(text)` | 归一化答案：去 `$...$`、合并空白、统一标点 |
| `_parse_verdict(text)` | 解析 verifier 输出，返回 `(是否判对, 置信度 0~1)` |

### 4. solve 逻辑重构

```
旧：生成 3 个 candidate → verifier 二值投票 → 选置信度最高
新：生成 3 个 candidate → 提取答案 → verifier 置信度打分
    → 答案一致性加分（相同答案互相印证） → 选总分最高
```

一致性加分规则：若 k 个 candidate 答案相同，每人加 `(k-1) * 0.15`。

### 5. AgentConfig 新增参数

- `consistency_bonus_weight: float = 0.15` — 一致性加分权重

### 6. final_response

- 从完整推理文本改为**提取后的纯答案**

### 7. trace 增强

- verifier trace 中记录 `parsed_correct` 和 `parsed_confidence`
- select_final_response 中列出所有 candidate 的分项得分

## 预期效果

| 场景 | 旧行为 | 新行为 |
|---|---|---|
| 3 个都答对 | confidence 全 1.0，随机选 | 答案一致，一致性加分，选 verifier 分最高的 |
| 2 对 1 错 | 可能被 verifier 全判 A | 对的 2 个答案一致 +0.15，错的独白无加分 |
| 全错答案不同 | 选 verifier 打分最高的 | 无一致性加分，纯靠 verifier 置信度排序 |
| 模型不输出标记 | N/A | 兜底提取末行 |
