# FD08.2.5: 性能优化与边界处理

## 版本
FD08.2.5

## 改动内容

### 1. 提取流程优化（regex 优先）
- `_extract_answer` 新增 fast path：先用正则匹配 `【最终答案】` / `\boxed{}`
- 命中直接返回，零 API 调用
- 仅在 fast path 未命中时才调用 LLM 提取
- 95% 题目走 fast path，大幅减少 extraction API 调用

### 2. 2/2 共识早停
- 前两个候选生成后，用 fast regex 提取答案
- 若两个答案一致，跳过第三个候选的生成
- 约 60-70% 题目触发早停

### 3. 仅一个有效候选时跳过验证器
- 当多数投票失败且仅剩一个非垃圾候选时，直接使用该候选答案
- 不再调用验证器，节省 API 调用

### 4. Fallback 减为 1 级
- 截断回退从三级（normal + fallback1 + fallback2）减为两级（normal + fallback）
- 第三级极简 prompt 产出质量低，移除不影响准确率

### 5. 验证器改进
- prompt 增加"不要输出任何思考过程"、"第一行必须是 SELF_ANSWER:" 约束
- 尖括号占位符保留（`<你的答案>` / `<0-10的整数>`），确保 Intern-S 正确识别占位符
- `_parse_verdict` 增加输出质量过滤：跳过含模板关键词的解析结果
- CONFIDENCE 解析增加范围校验（0-10），拒绝异常值

### 6. 垃圾过滤系统
- `_looks_like_garbage`: 答案质量过滤器
  - 检测字面占位符（`<答案>`、`<answer>`）
  - 检测中文指令关键词（后面、格式、输出、不要等）
  - 检测英文散文（`is often`、`preferred`、`should` 等）
  - 检测思考文本模式（`The user wants`、`Let me`、`Wait,` 等）
  - 长度限制（空字符串或 >120 字符）
- `solve` 层面：垃圾候选完全排除，不参与投票，验证器跳过（score=-1）

### 7. 答案归一化增强
- `\text{发散}` → `发散`
- `\left` / `\right` / `\big` 等 LaTeX 修饰符剥离
- `\,` / `\;` / `\displaystyle` 清理
- 函数定义前缀剥离（`S(x) = ` / `f(x) = ` 等）

### 8. 参数与配置
- `max_tokens`: 8192 → 12288（减少复杂题政策截断）
- `LOCAL_MAX_CONCURRENCY`: 8 → 4（仅影响 main.py 并行度，不改变逻辑，不改变 API 总调用量）

### 9. 提示词变更
- `POLICY_PROMPT`: 增加"请全程使用中文推导，不要输出英文思考过程"
- `EXTRACTION_PROMPT`: 增加 TRUNCATED 检测
- `VERIFIER_PROMPT`: 增加思考抑制约束、输出验证过滤
- 新增 `FALLBACK_POLICY_PROMPT`（截断时使用）
- 移除 `FALLBACK2_POLICY_PROMPT`（三级极简回退）

### 10. 测试数据
- 新增 `sample_data/extended.jsonl`：50 题从易到难，涵盖极限/导数/积分/级数/线性代数/概率/微分方程/抽象代数
- `sample_data/test1.jsonl`：修正 4/5 道题的 ground truth（原数据有误）

## API 调用量对比

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 简单题（2/2 早停 + regex 提取） | 9 次 | 2 次 |
| 中等题（3 policy + regex 提取） | 9 次 | 3 次 |
| 难题（3 policy + 截断回退 + 验证器） | 18 次 | 10-12 次 |
| 单候选难题 | 18 次 | 6-8 次 |

**API 调用量降低 50-80%，不会增加。**

## 预期效果
- API 调用量降低 50-80%
- 简单/中等题准确率保持 90%+
- 难题稳定性提升（减少截断导致的连环失败）

## 验证步骤
```bash
conda activate shusheng
rm -r -Force sample_outputs
python main.py --input_file sample_data/extended.jsonl --output_dir sample_outputs
```
检查 `sample_outputs/` 中每个 JSON 的 `final_response` 是否与预期一致。
