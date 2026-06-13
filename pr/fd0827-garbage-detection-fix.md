# FD08.2.7: 垃圾检测修复 + 应用题支持 + 比赛合规

## 版本
FD08.2.7

## 改动内容

### 1. `_looks_like_garbage` 修复（3 处）

| 修复 | 问题 | 影响 |
|------|------|------|
| 字符集 → 完整短语匹配 | `[后面格式输出不要答案值跟]` 误伤含`格`字的答案（如`价格`） | 应用题全灭 |
| 长度限制 120→500 | 自然语言答案超 120 字符 | 应用题全灭 |
| `^\d+[.)]` → `^\d+[.)、]\s` | `0.5` 被误判为 `0. item` 列表项 | 小数答案误判 |

### 2. 比赛平台合规

| 改动 | 原因 |
|------|------|
| 删除 `from llm_client import InternChatClient` | 正式评测不提供选手的 `llm_client.py` |
| 删除 `FALLBACK2_POLICY_PROMPT` | 死代码 |
| `AgentConfig \| None` → `Optional[AgentConfig]` | 兼容 Python 3.9 |
| 添加 `from __future__ import annotations` | 兼容 3.7+ 的类型语法 |

### 3. 新增应用题测试集
- `sample_data/1.jsonl`：10 道计算应用题（880-889）
- 涵盖经济学、微积分应用、物理、概率

## 验证结果

| 测试集 | 题数 | 答错 | 调用/题 |
|--------|------|------|---------|
| extended.jsonl | 50 | 0 | ~2.1 |
| extended2.jsonl | 30 | 0 | 2.1 |
| 1.jsonl（应用题） | 10 | 0 | 2.5 |
| **合计** | **90** | **0** | |

## 比赛合规自查

| # | 要求 | 结果 |
|---|------|------|
| 1 | 不依赖绝对路径 | ✅ |
| 2 | 不依赖隐藏/缓存/临时文件 | ✅ |
| 3 | 不依赖 answer 字段 | ✅ |
| 4 | 不依赖 idx 连续/顺序 | ✅ |
| 5 | 不依赖同一进程/实例 | ✅ |
| 6 | 不依赖 llm_client.py | ✅ |
| 7 | 不依赖 client 私有字段 | ✅ |
| ⚠️ | PEP 604 类型语法 | ✅ 已修 |

## 验证步骤
```bash
conda activate shusheng
rm -r -Force sample_outputs
python main.py --input_file sample_data/extended2.jsonl --output_dir sample_outputs
```
