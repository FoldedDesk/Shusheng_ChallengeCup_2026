# Math Agent Competition
# Codex Final Implementation PR
# 中文数学推理智能体正式评测版

---

# 0. Task Definition

请实现一个面向数学智能体比赛的中文数学解题 Agent。

目标：

构建一个可以通过正式评测的智能体系统：

- 输入：中文数学题字符串 problem
- 输出：符合规范的 final_response
- 支持：
  - 计算题
  - 证明题
  - 推导题
  - 解释题
  - 选择题
  - 填空题

系统必须运行在：

- Linux Docker
- 无 GPU
- 无运行时联网
- 官方 client 提供模型调用

---

# 1. Non-Negotiable Competition Contract

## 1.1 Entry File

提交根目录必须存在：

```
user_agent.py
```

必须提供：

```python
class ReasoningAgent:
```

---

## 1.2 Constructor

必须支持：

```python
ReasoningAgent(client=official_client)
```

因此：

```python
def __init__(self, client, *args, **kwargs):
```

要求：

- 保存 client
- 初始化轻量组件

禁止：

- 加载巨大数据
- 调用模型
- 读取隐藏文件


---

## 1.3 Solve Interface

必须提供：

```python
def solve(
    self,
    problem: str,
    metadata: dict
) -> dict:
```

返回：

```python
{
    "final_response": "answer"
}
```

推荐：

```python
{
    "final_response": "...",
    "trace": [...]
}
```

---

# 2. Absolute Restrictions

## 禁止

### 标准答案泄露

不能：

```python
metadata["answer"]
```

不能读取：

- sample answer
- hidden answer
- judge


---

### 禁止依赖 baseline

不能假设：

- main.py 会执行
- llm_client.py 会存在


---

### 禁止联网

不能：

- 在线搜索
- 外部 API
- 在线数据库


---

### 禁止个人环境依赖

禁止：

```
/Users/name/project
/home/user/file
```

所有资源：

使用：

```python
Path(__file__)
```

---

# 3. Final Architecture

采用：

```
user_agent.py

    |
    v

ReasoningAgent

    |
    v

MathReasoningSystem

    |
    +----------------+
    |                |
Classifier       Solver
    |                |
    +------Verifier-+
             |
             v

ResponseBuilder

             |
             v

final_response
```

---

# 4. Repository Structure

创建：

```
project/

├── user_agent.py
├── requirements.txt


├── core/

│   ├── math_agent.py
│   ├── state.py
│   ├── client_adapter.py
│   ├── response_builder.py
│   ├── serializer.py
│   └── exception_handler.py


├── reasoning/

│   ├── planner.py
│   ├── solver.py
│   ├── critic.py
│   ├── verifier.py
│   └── finalizer.py


├── classifier/

│   ├── subject.py
│   ├── problem_type.py
│   └── difficulty.py


├── tools/

│   ├── sympy_tool.py
│   └── latex_parser.py


├── rag/

│   ├── retriever.py
│   └── knowledge/


├── prompts/

│   ├── solver.txt
│   ├── verifier.txt
│   └── critic.txt


└── tests/

    ├── test_import.py
    ├── test_output.py
    └── test_sample.py
```

---

# 5. Implementation Requirements

# Phase 1
# Platform Adapter

实现：

## user_agent.py

只负责：

- 接收官方 client
- 调用内部 agent


示例：

```python
class ReasoningAgent:

    def __init__(self, client, *args, **kwargs):
        self.agent = MathAgent(
            ClientAdapter(client)
        )

    def solve(self, problem, metadata):
        return self.agent.solve(
            problem,
            metadata
        )
```

验收：

```bash
python tests/test_import.py
```

---

# Phase 2
# Core State System

实现：

```python
class MathState:
```

保存：

```python
problem
metadata
subject
problem_type
difficulty
candidate_answers
verification
```

要求：

每题创建新的 state。

禁止：

global memory。


---

# Phase 3
# Client Adapter

所有模型调用统一经过：

```
core/client_adapter.py
```

接口：

```python
chat(
 messages,
 temperature,
 max_tokens
)
```

内部：

调用：

```python
client.chat()
```

禁止：

访问：

- key
- token
- private fields


---

# Phase 4
# Problem Classifier

## Subject

识别：

- 离散数学
- 数值分析
- 测度积分
- 微分几何
- 概率论
- 抽象代数
- 随机过程
- 复分析
- 常微分方程
- 统计推断
- 泛函分析
- 线性回归
- 偏微分方程
- 高等代数
- 运筹学
- 数学分析
- 拓扑学
- 进阶数学


---

## Problem Type

识别：

- calculation
- proof
- derivation
- explanation
- choice
- fill_blank


---

# Phase 5
# Reasoning Pipeline

实现：

```
problem

↓

classification

↓

planning

↓

solution generation

↓

verification

↓

answer extraction

↓

response
```

---

# Phase 6
# Solver

## 简单题

一次生成：

```
solution
```

---

## 困难题

生成：

```
candidate1

candidate2

candidate3
```

然后：

进入 verifier。


---

# Phase 7
# Verifier

实现：

检查：

- 数学结论
- 计算正确性
- 条件完整性
- 证明逻辑


输出：

```python
{
"correct":True,
"reason":"..."
}
```

---

# Phase 8
# Mathematical Tools

## SymPy

支持：

- derivative
- integral
- equation
- matrix
- limit


所有输出：

转换：

```
str
list
dict
float
```

禁止：

直接返回 SymPy object。


---

# Phase 9
# Offline RAG

允许：

本地数学资料。

实现：

轻量：

- BM25
- TF-IDF


禁止：

大型数据库。


检索：

```
top_k <= 5
```

失败：

直接继续。


---

# Phase 10
# Response Builder

最终输出：

必须：

```json
{
"final_response":"数学答案",
"trace":[]
}
```

---

## final_response

要求：

- str
- 非空
- 可直接判分


不要：

输出：

- debug
- 日志
- Python对象


---

## trace

用于：

- 调试
- 记录步骤


禁止：

- key
- token
- 私密路径


---

# Phase 11
# Reliability Layer

处理：

## 模型失败

策略：

```
retry
 |
fallback
 |
return
```


## 工具失败

继续：

```
without tool
```


## 分类失败

进入：

```
general solver
```


---

# Phase 12
# Serialization

实现：

```python
safe_json()
```

处理：

- numpy
- pandas
- sympy
- datetime
- Path
- Exception


保证：

```python
json.dumps(result)
```

成功。


---

# Phase 13
# Testing

## Import Test

```bash
python tests/test_import.py
```

检查：

```python
from user_agent import ReasoningAgent
```


---

## Output Test

检查：

返回：

```python
dict
```

包含：

```python
final_response
```


---

## Sample Test

运行：

```bash
python tests/test_sample.py
```

读取：

```
sample_data/dev.jsonl
```


---

# Phase 14
# Submission Validation

实现：

```
scripts/validate_submission.py
```

检查：

- 文件存在
- import成功
- 返回格式正确
- 无敏感信息


---

# Phase 15
# Optimization

优化方向：

优先级：

1. final_response正确率
2. verifier质量
3. 题型分类
4. SymPy辅助
5. RAG增强
6. trace完善


不要为了复杂架构降低稳定性。


---

# Codex Execution Instructions

严格按照：

```
Phase 1
↓
测试
↓
Phase 2
↓
测试
↓
Phase 3
...
```

执行。

不要：

- 一次生成所有代码
- 引入大型框架
- 添加无必要依赖


每完成一个阶段：

运行测试。

最终目标：

生成一个可以直接提交数学智能体比赛的代码仓库。
