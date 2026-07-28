# 112 题官方分布回归集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一个严格符合指定 18 个学科计数的 112 题中文 JSONL 本地回归集。

**Architecture:** 一个静态 JSONL 数据文件承载题目和标准答案；一次性标准库校验在不修改文件的情况下检查行数、字段、索引、题型分布和中文题面约束。

**Tech Stack:** JSONL、Python 3 标准库 `json`、`collections.Counter`、`pathlib`。

---

## File structure

- Create: `sample_data/official_distribution_112.jsonl`：112 条原创中文数学题。
- Verify: `sample_data/official_distribution_112.jsonl`：使用标准库内联校验，不新增运行时代码。

### Task 1: 生成严格分布的题目数据

**Files:**
- Create: `sample_data/official_distribution_112.jsonl`

- [ ] **Step 1: 写入统一记录结构**

每一行使用如下 JSON 结构，`idx` 从 3000 连续到 3111，`source` 固定：

```json
{"idx": 3000, "problem": "设……，求……。", "answer": "……", "subject": "离散数学", "source": "official_distribution_112"}
```

- [ ] **Step 2: 按指定计数写入 112 道原创中文题**

按以下顺序写入：离散数学 24、数值分析 13、测度积分 11、微分几何 9、概率论 8、抽象代数 8、随机过程 7、复分析 7、常微分方程 5、统计推断 4、泛函分析 4、线性回归 3、偏微分方程 3、非基础及进阶课程 2、高等代数 1、运筹学 1、数学分析 1、拓扑学 1。

每题应有独立且明确的答案；题面只能使用中文自然语言和数学符号/LaTeX，不含图片依赖、未定义背景或英文自然语言。

- [ ] **Step 3: 校验 JSON、索引、字段与分布**

Run:

```bash
rtk python -c 'import json; from collections import Counter; from pathlib import Path; p=Path("sample_data/official_distribution_112.jsonl"); rows=[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]; expected={"离散数学":24,"数值分析":13,"测度积分":11,"微分几何":9,"概率论":8,"抽象代数":8,"随机过程":7,"复分析":7,"常微分方程":5,"统计推断":4,"泛函分析":4,"线性回归":3,"偏微分方程":3,"非基础及进阶课程":2,"高等代数":1,"运筹学":1,"数学分析":1,"拓扑学":1}; assert len(rows)==112; assert [r["idx"] for r in rows]==list(range(3000,3112)); assert all(set(("idx","problem","answer","subject","source"))<=set(r) and all(str(r[k]).strip() for k in ("problem","answer","subject","source")) and r["source"]=="official_distribution_112" for r in rows); assert Counter(r["subject"] for r in rows)==expected; print("validated", len(rows), Counter(r["subject"] for r in rows))'
```

Expected: 退出码为 0，输出 `validated 112` 及与设计一致的分布。

- [ ] **Step 4: 检查格式与变更范围**

Run: `rtk git diff --check -- sample_data/official_distribution_112.jsonl && rtk git status --short`

Expected: 无空白错误；新文件显示为未跟踪，用户已有的其它改动保持不变。

- [ ] **Step 5: 提交数据集**

```bash
rtk git add sample_data/official_distribution_112.jsonl
rtk git commit -m "test: add 112 question distribution regression set"
```

仅暂存新数据文件，不包含任何既有改动。
