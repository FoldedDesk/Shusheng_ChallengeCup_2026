# FD08.2.6: final_response 格式化

## 版本
FD08.2.6

## 改动内容

`final_response` 返回前应用 `_normalize_answer`，剥离格式化噪声。

### 改动点（3 行代码）

1. `solve` 返回值中 `final_response` 改为 `self._normalize_answer(extracted_answers[best_id])`
2. 函数前缀剥离正则增加 `'` 支持（`f'(x) = ` / `f''(x) = `）
3. 新增变量赋值前缀剥离（`y = ` / `dz = `）

### 效果

| 指标 | FD08.2.5 | FD08.2.6 |
|------|----------|----------|
| 完全匹配 | 33/50 (66%) | 39/50 (78%) |
| 格式不匹配 | 17 | 11 |
| 真正答错 | 0 | 0 |

消除的 6 个格式差异：`$...$` 包裹、`y=`/`f'(x)=`/`dz=` 前缀、`\text{}` 包裹。

## 验证步骤
```bash
conda activate shusheng
rm -r -Force sample_outputs
python main.py --input_file sample_data/extended.jsonl --output_dir sample_outputs
```
