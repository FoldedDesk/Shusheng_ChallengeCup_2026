# PR：LLM 答案提取

**版本：** FD08.1.0 → FD08.2.0
**日期：** 2026-06-12
**类型：** 功能增强

## 改动内容

将纯正则答案提取替换为"LLM 优先 + 正则回退"方案：

1. **新增 `EXTRACTION_PROMPT`** — 专用的答案提取提示词，temperature=0，要求模型从完整推理文本中仅输出数学答案。

2. **新增 `extraction_agent`** — 独立的 Agent 实例，专门用于答案提取。

3. **新增 `_llm_extract_answer()`** — 调用提取 agent；失败返回 None（触发正则回退）。

4. **原正则逻辑重命名为 `_regex_extract_answer()`** — 逻辑不变，作为回退方案。

5. **`_extract_answer()` 现在返回 `(answer, method)`** — `method` 为 `"llm"` 或 `"regex"`，记录在 trace 中。

6. **新增配置项：**
   - `use_llm_extraction: bool = True` — LLM 提取开关
   - `extraction_max_tokens: int = 128` — 提取响应最大 token 数

## 原因

baseline 的正则提取在 dev 集上有约 44% 的失败率（9 个候选中有 4 个提取出垃圾字符串）。LLM 能理解上下文，即使模型没有正确使用【最终答案】标记也能提取出实际的数学答案。

## 预期效果

- 大幅减少提取失败（垃圾字符串 → 干净的数学答案）
- 每个候选增加 1 次 API 调用（提取），但答案很短（128 tokens）
- 更多正确答案被成功提取 → 多数投票信号更强

## 验证方法

```powershell
rm -r -Force sample_outputs
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

检查每个 `sample_outputs/*.json`：
- trace 中应包含 `extract_answer_{i}` 步骤，`method` 大多为 `"llm"`
- `extracted_answer` 应为干净的数学值（如 `72`、`-1`、`-1/8`），而非垃圾字符串
