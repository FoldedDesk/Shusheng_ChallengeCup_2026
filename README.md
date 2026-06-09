# 推理智能体样例代码

本样例展示如何基于 [lagent](https://github.com/InternLM/lagent) 实现一个数学推理智能体。你需要主要修改 `user_agent.py`，让智能体读取题目并返回最终解答。

比赛主要依据 `final_response` 的答案正确性进行评测；`trace` 用于展示智能体的交互和推理过程，在结果分数接近或相同的情况下可作为设计质量参考。

## 你需要修改哪里

请主要修改：

```text
user_agent.py
```

文件中有显式标记：

```text
# ==================== PARTICIPANT DESIGN AREA START ====================
# ===================== PARTICIPANT DESIGN AREA END =====================
```

两条标记之间是你的设计区。你可以在这里实现：

- 智能体整体求解流程
- 提示词和消息组织方式
- 模型调用、工具调用或外部模块调用
- 规划、反思、验证、记忆、检索等推理策略
- `final_response` 的生成方式
- `trace` 的记录方式

`main.py` 和 `llm_client.py` 用于本地调试。正式评分时，平台会使用官方运行脚本和 API client，并加载你提交的 `user_agent.py`。

## 智能体接口

你的智能体需要提供：

```python
agent.solve(problem: str, metadata: dict) -> dict
```

返回值必须包含：

```python
{
  "final_response": "你的最终答案"
}
```

推荐同时返回 `trace`：

```python
{
  "final_response": "你的最终答案",
  "trace": [
    {"step": "plan", "content": "..."},
    {"step": "model_call", "content": "..."},
    {"step": "finalize", "content": "..."}
  ]
}
```

`trace` 用于记录智能体交互和推理过程。格式不需要复杂，建议保持简单、完整、可读。

## 输入格式

输入文件是 JSONL，每行一道题。每行至少包含：

```json
{"idx": 0, "problem": "题目文本"}
```

样例数据中可能额外包含 `answer`，用于对照参考；智能体接口只需要处理 `problem`。

## 输出格式

本地调试时，每道题会保存为一个独立 JSON 文件：

```text
outputs/
  0.json
  1.json
  2.json
```

成功样例：

```json
{
  "idx": 0,
  "status": "success",
  "final_response": "智能体系统对该题的最终输出",
  "trace": []
}
```

失败样例：

```json
{
  "idx": 0,
  "status": "error",
  "final_response": "",
  "error": {
    "type": "RuntimeError",
    "message": "错误信息"
  },
  "trace": []
}
```

如果某个 `idx.json` 已经存在且文件非空，本地 runner 会跳过该题，便于中断后继续运行。

## 本地试跑

安装依赖：

```bash
pip install -r requirements.txt
```

设置 API key：

```bash
export INTERN_API_KEY="sk-..."
```

运行样例：

```bash
python main.py --input_file sample_data/dev.jsonl --output_dir sample_outputs
```

本地 runner 默认并发数为 8。如需调整：

```bash
export LOCAL_MAX_CONCURRENCY=4
```

默认 API 和模型为：

```text
INTERN_API_BASE=https://chat.intern-ai.org.cn/api/v1/chat/completions
INTERN_MODEL=intern-s2-preview
```

如需本地覆盖：

```bash
export INTERN_API_BASE="https://chat.intern-ai.org.cn/api/v1/chat/completions"
export INTERN_MODEL="intern-s2-preview"
```

## 样例 Baseline

当前 `user_agent.py` 实现了一个简单 baseline：

1. 对同一道题生成多个候选解答。
2. 对每个候选解答做简单验证。
3. 选择置信度最高的候选作为 `final_response`。
4. 将候选和选择过程写入 `trace`。

这个 baseline 使用 lagent 的 `Agent` 和 `AgentMessage` 组织 policy agent 与 verifier agent 的消息传递。它只用于展示接口和基本组织方式，不限制你采用其它智能体设计。

## 注意事项

- 不要在代码里写死 API key。
- 不要依赖本地绝对路径。
- 保证 `final_response` 是可读的最终答案。
- 尽量记录清楚 `trace`，展示智能体的关键交互和推理过程。
