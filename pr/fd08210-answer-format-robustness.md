# FD08.2.10: 答案格式鲁棒性与平台兼容修复

## 版本
FD08.2.10

## 背景

README 强调正式评测主要依据 `final_response` 的答案正确性，同时要求 `user_agent.py` 能被平台稳定加载，并且输出必须是非空、可 JSON 序列化的字符串字段。

本轮本地测试中，`dev.jsonl` 第 3 题原先输出 `-\frac{1}{8}`，标准答案为 `-1/8`，数学值一致但字符串不完全一致。当前 `\boxed{...}` 提取也无法处理 `\boxed{\frac{1}{2}}` 这类嵌套大括号。

## 改动内容

### 1. 类型兼容性

将设计区内残留的 PEP 604 类型写法替换为 `Optional[...]`：

- `Tuple[str, str, str | None]` -> `Tuple[str, str, Optional[str]]`
- `Tuple[str | None, str]` -> `Tuple[Optional[str], str]`
- `str | None` -> `Optional[str]`

该改动不影响运行逻辑，只降低平台 Python 版本差异带来的加载风险。

### 2. `\boxed{...}` 嵌套提取

新增 `_extract_last_braced_latex`，使用括号深度扫描提取最后一个 LaTeX 命令参数。

修复前：

```text
\boxed{\frac{1}{2}} -> \frac{1
```

修复后：

```text
\boxed{\frac{1}{2}} -> \frac{1}{2}
```

### 3. 简单整数分数归一化

新增 `_normalize_numeric_frac`，在最终答案归一化中将简单整数 LaTeX 分数转换为 slash 格式：

```text
-\frac{1}{8} -> -1/8
\dfrac{3}{5} -> 3/5
\tfrac{12}{51} -> 12/51
```

只处理分子、分母均为整数的简单分数，避免误改含变量或含 `\pi` 的符号答案。

## 验证结果

### 语法与导入

```bash
rtk conda run -n shusheng python -m compileall -q main.py user_agent.py llm_client.py
rtk conda run -n shusheng python -c "from user_agent import ReasoningAgent, AgentConfig; print('import ok')"
```

结果：均通过。

### dev.jsonl

```bash
rtk conda run -n shusheng python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs_fd08210_live
```

| 指标 | 结果 |
|------|------|
| 总题数 | 3 |
| 成功输出 | 3 |
| 完全匹配 | 3 |
| 错误 | 0 |

第 3 题最终输出由 `-\frac{1}{8}` 规范为 `-1/8`。

### mixed.jsonl

```bash
rtk conda run -n shusheng python main.py --input_file sample_data/mixed.jsonl --output_dir sample_outputs
```

| 指标 | 结果 |
|------|------|
| 总题数 | 50 |
| 成功输出 | 50 |
| 运行错误 | 0 |
| 严格字符串完全一致 | 17 |
| 简单格式等价 | 5 |

主要剩余问题集中在应用题和多字段答案：模型往往算出核心数值，但 `final_response` 缺少标准答案要求的部分字段，或自然语言表达与标准答案差异较大。

## 风险评估

- 不增加模型调用次数。
- 不修改 prompt。
- 不依赖本地路径、标准答案或隐藏数据。
- 不改变 `ReasoningAgent` 入口接口。
- 只影响答案提取和最终格式清理，风险较低。
