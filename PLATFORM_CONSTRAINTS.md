# 数学智能体平台约束

本文记录当前已知的官方平台接口、仓库、输入输出、评测及安全约束，供上下文重置后恢复使用。

本文不记录当前智能体自行设置的模型调用次数、Token 预算、时间预算、候选筛选策略等实现参数。

## 1. 固定入口

平台按以下方式加载智能体，调用格式不可更改：

```python
from user_agent import ReasoningAgent

agent = ReasoningAgent(client=official_client)
result = agent.solve(problem, metadata)
```

仓库根目录必须存在 `user_agent.py`，其中必须定义 `ReasoningAgent`。

构造函数必须兼容以下签名：

```python
def __init__(self, client, *args, **kwargs):
    pass
```

求解入口必须兼容以下签名：

```python
def solve(self, problem: str, metadata: dict) -> dict:
    pass
```

## 2. 输入约束

- `problem` 是原始数学题目的字符串。
- 正式评测中的每个 `problem` 都是一道独立题目，不包含一道题的多个小问。
- `metadata` 是字典，至少包含 `idx`；其他元信息基本与解题无关，不应依赖。
- 正式评测不会向选手代码提供参考答案。
- 正式评测不会可靠提供 `subject` 或题型标签，学科和题型需要根据 `problem` 自行判断。
- samples 中的 `answer` 和 `subject` 仅用于本地调试及格式说明。
- 当前公开说明称数据按中文题面准备，数学公式、LaTeX、变量和专有名词按题面保留；项目同时应兼容可能出现的英文题面。
- 题目输入是 `str`，当前无需处理直接上传的图片或 OCR。
- 题型可能包括计算、推导、证明、解释、选择题和填空题，也可能混合数学内部不同方向。

## 3. 输出约束

最低合法返回值：

```python
{
    "final_response": "非空的最终答案字符串"
}
```

推荐返回值：

```python
{
    "final_response": "72",
    "trace": [
        {"step": "plan", "content": "整体求解规划"},
        {"step": "model_call", "content": "模型调用摘要"},
        {"step": "finalize", "content": "答案提取和校验摘要"}
    ]
}
```

返回要求：

- `final_response` 必须是非空、可读字符串。
- 返回字典整体必须支持 JSON 序列化。
- `trace` 可选；提供时必须是数组结构。
- `trace` 可记录规划、调用、候选、校验和拒绝原因，但不得包含密钥、令牌或个人隐私。
- 不能以空字符串、内部错误文本或不可判分的占位内容代替最终答案。

## 4. 官方 Client 约束

- `client` 由评测平台在构造 `ReasoningAgent` 时注入。
- 禁止硬编码 API Key。
- 禁止依赖本地密钥配置文件。
- 模型调用、接口限流、Token 统计和超时管理由平台官方 Client 托管。
- Client 调用格式参考基线中的 `InternChatClient`。

标准调用示例：

```python
response = client.chat(
    messages=[{"role": "user", "content": problem}],
    temperature=0.2,
    max_tokens=4096,
)
```

## 5. 仓库规范

- 正式参赛代码必须托管到 AtomGit 队伍自建组织仓库。
- GitHub 官方仓库仅作为 Baseline 拉取和参考来源。
- 除根目录必须存在 `user_agent.py` 外，其他模块、目录、框架和依赖可以自行设计。
- 可以修改、重构或替换 Baseline，也可以新增模块和依赖。
- 所有文件读取必须使用相对路径，禁止依赖开发机器的绝对路径。
- Agent 可以维护运行状态，但不能依赖题目执行顺序。
- 不能假设多道题一定在同一进程中执行。

推荐但非强制的目录结构：

```text
项目根目录
├── user_agent.py
├── requirements.txt
├── prompts/
├── tools/
└── utils/
```

## 6. 评分依据

- 官方 Judger 结合标准答案对 `final_response` 评分。
- 核心评分依据是 `final_response` 的答案正确率。
- `trace` 主要用于问题排查；同分场景下，智能体设计和诊断信息可能作为加分参考。
- 测试集不公开，只提供 samples 供理解题目风格和运行格式。
- 不应假设 Judger 只接受与参考答案完全相同的原始字符串，但最终答案必须完整、明确且可判分。

## 7. 禁止事项

- 禁止读取或推断正式测试集标准答案。
- 禁止读取、反向分析或利用官方 Judger 逻辑核对答案。
- 禁止把本地样本的 `answer` 字段传入 Agent 的运行输入、提示词或推理上下文。
- 禁止编写破坏性或恶意执行代码来核对评测结果。
- 禁止在代码中保存或输出平台密钥、令牌及个人隐私。
- 禁止依赖本机绝对路径、题目执行顺序或跨题共享进程。

## 8. Baseline 的作用

官方 Baseline 只提供最小可运行模板，用于理解：

- 赛题逻辑和本地调试流程；
- 标准输入、输出数据结构；
- 固定智能体入口；
- 平台 Runner 与选手代码之间的调用关系。

是否继续使用 Baseline 中的 `lagent` 框架不受强制限制，只要严格满足固定入口和返回格式即可。
