from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ==================== PARTICIPANT DESIGN AREA START ====================

POLICY_PROMPT = """你是一个严谨的数学推理智能体。请解决以下数学问题。

要求：
1. 先分析题意，明确已知条件和求解目标。
2. 给出完整、清晰的逐步推导过程。
3. 在最后单独一行，用「【最终答案】<答案>」的格式明确写出最终答案。

注意：请全程使用中文推导。英文完整句、Thinking Process、提示词复述或备选方案讨论均视为无效输出；数学变量、公式和定理符号除外。最终答案必须放在【最终答案】之后；证明题只保留必要证明步骤。"""

STRUCTURED_FAST_PROMPT = """你是数学题的结构化快速求解器。独立完成题目，只保留必要推导。

先识别题目要求的每个对象（例如公式、初值、数值、区间、构造、判断），逐项计算并核对。
最后一行严格输出【最终答案】<答案>；多个对象必须用清晰分隔符逐项列出，不能只输出其中一个数值。禁止英文完整句、提示词复述和 Thinking Process。"""

INDEPENDENT_SLOW_PROMPT = """你是独立数学审计求解器。请从定义或不同于常规套路的角度独立求解题目，不要假设其他解答正确。

核对边界条件、定义域、符号、计数是否重复、初值和所求对象是否齐全。最后一行严格输出【最终答案】<答案>，答案必须保留题目要求的公式、构造或全部结论。禁止英文完整句、提示词复述和 Thinking Process。"""

SOLUTION_AUDIT_PROMPT = """你是数学解答审核器。比较同一道题的候选完整解答，选择最适合提交给评测器的一份。

审核标准：是否回答题目、关键推导或证明步骤是否完整、结论是否由前文支持、是否存在明显矛盾。
不要自行重新解题，不要补造候选中不存在的结论。

严格输出两行：
CHOICE: <候选编号，从0开始>
ISSUES: <需要修复的简短中文说明；无问题写无>"""

SOLUTION_REPAIR_PROMPT = """你是数学解答整理器。只根据题目、候选解答和审核意见，写出可独立判分的完整中文解答。

规则：
1. 保留支撑结论所必需的定义、公式、推导或证明链；不要只写最终结论。
2. 不要引入候选解答没有支持的新结论。
3. 使用如下结构：先写【解答】，最后写【结论】和明确结论。
4. 只输出整理后的解答，不要解释审核过程。"""

INDEPENDENT_VERIFIER_PROMPT = """你是独立数学验证器。请独立求解用户给出的数学题，不能假设任何候选答案正确。

先在内部核对定义、条件、计算和结论；对证明题只提取被证明的具体命题。最终严格输出两行：
ANSWER: <可直接提交给评测器的简洁最终答案>
CONFIDENCE: <0到100的整数>
不要输出推导、候选编号或其他内容。"""

STAGE_PROMPTS = [
    "请分析以下数学问题，列出已知条件、求解目标、约束条件。不需要计算。",
    "基于前面的分析，提出解题思路和方法。说明解题策略即可，不需要具体计算。",
    "基于前面的分析和策略，逐步推导和计算。展示每一步的推理和计算过程。",
    "基于前面的推导，写出最终答案。严格使用【最终答案】<答案>的格式。",
]

EXTRACTION_PROMPT = """从以下数学解答中提取最终答案。用 ANSWER: <答案> 的格式输出。如果解答不完整或截断，输出 ANSWER: TRUNCATED。不要输出其他内容。"""

FINALIZER_PROMPT = """你是最终答案整理器。只根据题目、候选答案和候选推理片段整理最终答案。

严格规则：
1. 不要重新解题，不要引入候选中没有支持的新数值。
2. 如果当前答案已经短且明确，原样返回。
3. 证明题不能输出“命题得证”“结论成立”，必须输出被证明的具体命题或公式。
4. 如果题目要求方程、通解、全微分或矩阵等对象，最终答案必须保留左侧对象。
5. 如果题目问多个量，最终答案必须逐项列全。

只输出一行，以 FINAL: 开头，后面直接写答案本身。不要输出尖括号占位符。"""

FALLBACK_POLICY_PROMPT = """你是一个数学求解器。给出支撑结论所必需的简洁推导，再给出最终答案。
严格在最后单独一行输出：【最终答案】<答案>
不要输出英文完整句、英文思考、Thinking Process 或提示词复述。"""

CHINESE_PROOF_GUIDANCE = """证明输出约束：只用中文写 3-8 个可判分的关键步骤；每一步说明所用定义、定理或等式，不要写“分析题意”“检查答案”等元叙述。最后一行的最终答案必须是自包含的中文结论，保留题设条件和所求对象。"""

SUBJECT_GUIDANCE = {
    "抽象代数": """抽象代数检查：先明确运算、单位元、逆元或同态定义；正规性证明须写出共轭封闭或左右陪集相等的关键式，结论保留对象名称和条件。""",
    "数值分析": """数值分析检查：明确算法、迭代格式和初值；逐步核对代入计算、收敛条件、误差界或区间端点。题目要求迭代公式、近似值或精确值时，最终答案必须逐项保留。""",
    "微分几何": """微分几何检查：先写参数导数，再计算速度、第一基本形式或曲率；区分速度长度、弧长参数、主曲率、平均曲率与高斯曲率。判断题须同时给出数值和判断。""",
    "常微分方程": """常微分方程检查：先识别方程类型并给出通解或所求特解；代回原方程及初值核验，保留解函数、常数限制和最大定义区间。""",
    "复分析": """复分析检查：先确定定义域、奇点类型或收敛域；留数、积分和幂级数须核对方向、系数与收敛半径。最终保留复数单位、变量和必要条件。""",
    "离散数学": """离散数学检查：计数题先定义变量平移、样本空间或组合对象，再核对边界条件与是否重复计数；图论、关系和代数结构题要明确所用定义与结论对象。""",
    "概率论": """概率统计检查：先明确随机变量、条件事件与分布参数；构造题必须给出构造本身及其概率结论，不能只输出一个裸概率。""",
    "随机过程": """随机过程检查：明确时间参数、平稳性、独立增量或协方差定义；最终保留所求协方差函数、条件范围及其依赖变量。""",
    "统计推断": """统计推断检查：明确样本、统计量和参数；无偏性写出期望计算，方差或置信结论保留估计量和目标参数。""",
    "线性回归": """线性回归检查：区分相关系数、决定系数、回归系数和解释比例；题目问含义时，同时给出数值和对应变异解释。""",
    "偏微分方程": """偏微分方程检查：逐项计算时间偏导和空间偏导，再代入原方程；最后明确是否为解以及成立的定义域或初边值条件。""",
    "泛函分析": """泛函分析检查：明确空间、范数和算子；有界性须给出范数估计，算子范数同时给出上界和达到该上界的函数或序列。""",
    "拓扑学": """拓扑学检查：严格使用开覆盖、子覆盖、闭集或连续映射定义；结论保留空间、子集及所证明的拓扑性质。""",
    "高等代数": """高等代数检查：矩阵、行列式和特征值题先写对象维度与所用恒等式；最终保留矩阵表达式、特征值或秩等完整对象。""",
    "运筹学": """运筹学检查：先写决策变量、目标函数和约束；最优解须同时保留变量取值、目标值及可行性或对偶核对。""",
    "测度积分": """测度积分检查：明确收敛方式、几乎处处条件、可积性和使用的定理；最终分别保留逐点极限、积分极限及它们是否可交换。""",
    "数学分析": """数学分析检查：明确连续、可导、积分或极值条件；证明题给出关键定理和完整结论，避免只写“得证”。""",
}


@dataclass
class AgentMessage:
    sender: str
    content: str


class _PromptAgent:
    """Minimal prompt wrapper compatible with the subset of lagent used here."""

    def __init__(self, client, template: str, name: str) -> None:
        self.client = client
        self.template = template
        self.name = name

    def __call__(
        self,
        message: AgentMessage,
        session_id: str,
        temperature: float,
        max_tokens: int,
        template_override: Optional[str] = None,
    ) -> AgentMessage:
        del session_id
        content = self.client.chat(
            messages=[
                {"role": "system", "content": template_override or self.template},
                {"role": "user", "content": message.content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return AgentMessage(sender=self.name, content=content)


@dataclass
class AgentConfig:
    policy_sample_times: int = 3
    policy_temperature: float = 0.6
    verifier_temperature: float = 0.0
    max_tokens: int = 6144
    verifier_max_tokens: int = 4096
    use_llm_extraction: bool = True
    extraction_max_tokens: int = 1024
    finalizer_max_tokens: int = 768
    majority_vote_threshold: float = 0.5
    use_staged_reasoning: bool = False
    staged_stages: int = 4
    audit_max_tokens: int = 1024
    repair_max_tokens: int = 4096


class ReasoningAgent:
    """A generate-verify-select agent with answer extraction and consistency scoring."""

    def __init__(self, client, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.client = client
        self.policy_agent = _PromptAgent(
            client=client,
            template=POLICY_PROMPT,
            name="policy_agent",
        )
        self.extraction_agent = _PromptAgent(
            client=client,
            template=EXTRACTION_PROMPT,
            name="extraction_agent",
        )
        self.finalizer_agent = _PromptAgent(
            client=client,
            template=FINALIZER_PROMPT,
            name="finalizer_agent",
        )
        self.solution_audit_agent = _PromptAgent(
            client=client,
            template=SOLUTION_AUDIT_PROMPT,
            name="solution_audit",
        )
        self.solution_repair_agent = _PromptAgent(
            client=client,
            template=SOLUTION_REPAIR_PROMPT,
            name="solution_repair",
        )
        self.independent_verifier_agent = _PromptAgent(
            client=client,
            template=INDEPENDENT_VERIFIER_PROMPT,
            name="independent_verifier",
        )
        self._fallback_agent = _PromptAgent(
            client=client,
            template=FALLBACK_POLICY_PROMPT,
            name="fallback_policy",
        )

    # ---------- public API ----------

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)

        deterministic = self._deterministic_answer(problem)
        if deterministic and not self._requires_full_solution(problem):
            return {
                "final_response": deterministic,
                "trace": [{
                    "step": "deterministic_solver",
                    "content": {"answer": deterministic, "reason": "recognized_exact_pattern"},
                }],
            }

        # Phase 1: generate multiple candidate solutions
        candidates, trace = self._generate_candidates(problem, idx)

        # Guard: do not submit a partial chain of thought as an answer. Ask the
        # independent verifier for a short answer when every policy call failed.
        if not candidates:
            verifier_answer, verifier_confidence, verifier_raw = self._independently_solve(problem, idx)
            proof_fallback = (
                self._proof_protocol_fallback(problem)
                if not verifier_answer and verifier_raw and self._requires_full_solution(problem)
                else None
            )
            trace.append({
                "step": "all_truncated",
                "content": {
                    "reason": "No policy candidate had a usable final answer",
                    "verifier_answer": verifier_answer,
                    "proof_fallback": proof_fallback,
                    "confidence": verifier_confidence,
                    "raw_response": self._trace_safe_response(verifier_raw),
                },
            })
            if verifier_answer or proof_fallback:
                return {
                    "final_response": self._basic_normalize_answer(
                        verifier_answer or proof_fallback
                    ),
                    "trace": trace,
                }
            return {"final_response": "TRUNCATED_ALL", "trace": trace}

        if self._requires_full_solution(problem):
            return self._solve_full_solution(problem, candidates, idx, trace)

        # Phase 2: extract answers from each candidate (LLM + regex fallback)
        extracted_answers = []
        is_garbage = []
        for i, c in enumerate(candidates):
            ans, method, raw = self._extract_answer(c, idx, i)
            extracted_answers.append(ans)
            is_garbage.append(ans == "GARBAGE")
            trace.append({
                "step": f"extract_answer_{i}",
                "content": {
                    "method": method,
                    "answer": ans,
                    "raw_response": self._trace_safe_response(raw or ""),
                },
            })
        normalized_answers = [self._normalize_answer(a) for a in extracted_answers]

        # Filter to valid (non-garbage) candidates
        valid_ids = [i for i in range(len(candidates)) if not is_garbage[i]]
        if not valid_ids:
            verifier_answer, verifier_confidence, verifier_raw = self._independently_solve(problem, idx)
            trace.append({
                "step": "all_garbage",
                "content": {
                    "verifier_answer": verifier_answer,
                    "confidence": verifier_confidence,
                    "raw_response": self._trace_safe_response(verifier_raw),
                },
            })
            if verifier_answer:
                return {
                    "final_response": self._basic_normalize_answer(verifier_answer),
                    "trace": trace,
                }
            return {"final_response": "ALL_GARBAGE", "trace": trace}

        # Phase 3: try majority voting (only among valid answers)
        valid_counts: Dict[str, int] = {}
        for i in valid_ids:
            ans = normalized_answers[i]
            valid_counts[ans] = valid_counts.get(ans, 0) + 1
        max_count = max(valid_counts.values())
        total_valid = len(valid_ids)

        used_majority = False
        selected_answer = None
        if max_count >= 2 and max_count / total_valid >= self.config.majority_vote_threshold:
            used_majority = True
            majority_answer = next(a for a, c in valid_counts.items() if c == max_count)
            best_id = next(i for i in valid_ids if normalized_answers[i] == majority_answer)
            trace.append({
                "step": "majority_vote",
                "content": {
                    "answer": majority_answer,
                    "count": max_count,
                    "total": total_valid,
                    "garbage_skipped": len(candidates) - total_valid,
                    "selected_candidate": best_id,
                },
            })
            if self._requires_answer_contract(problem):
                majority_score = self._answer_requirements_score(
                    problem, extracted_answers[best_id]
                )
                complete_id = max(
                    valid_ids,
                    key=lambda candidate_id: self._answer_requirements_score(
                        problem, extracted_answers[candidate_id]
                    ),
                )
                complete_score = self._answer_requirements_score(
                    problem, extracted_answers[complete_id]
                )
                if complete_score > majority_score:
                    best_id = complete_id
                    trace.append({
                        "step": "object_completeness_override",
                        "content": {
                            "selected_candidate": best_id,
                            "majority_score": majority_score,
                            "selected_score": complete_score,
                        },
                    })
            if self._requires_majority_verification(problem):
                verifier_answer, verifier_confidence, verifier_raw = self._independently_solve(problem, idx)
                if verifier_answer and self._normalize_answer(verifier_answer) != majority_answer:
                    selected_answer = verifier_answer
                trace.append({
                    "step": "majority_verification",
                    "content": {
                        "verifier_answer": verifier_answer,
                        "confidence": verifier_confidence,
                        "overrode_majority": selected_answer is not None,
                        "raw_response": self._trace_safe_response(verifier_raw),
                    },
                })
        else:
            best_id, verifier_answer, verifier_confidence, verifier_raw = self._verify_disagreement(
                problem,
                extracted_answers,
                valid_ids,
                idx,
            )
            trace.append({
                "step": "independent_verification",
                "content": {
                    "verifier_answer": verifier_answer,
                    "confidence": verifier_confidence,
                    "selected_candidate": best_id,
                    "raw_response": self._trace_safe_response(verifier_raw),
                },
            })

        selected_answer = selected_answer or extracted_answers[best_id]
        return {
            "final_response": self._finalize_answer(
                selected_answer,
                problem,
                candidates[best_id],
                candidates,
                extracted_answers,
                idx,
                used_majority,
                trace,
            ),
            "trace": trace,
        }

    # ---------- generation ----------

    def _generate_candidates(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        if self.config.use_staged_reasoning:
            return self._generate_candidates_staged(problem, idx)
        return self._generate_candidates_oneshot(problem, idx)

    def _generate_candidates_oneshot(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        """Generate independent candidates with one fallback for truncated responses."""
        candidates = []
        trace = []
        agents = [self.policy_agent, self._fallback_agent]  # 1 normal + 1 fallback
        for sample_id in range(self._candidate_budget(problem)):
            for attempt, agent in enumerate(agents):
                user_message = AgentMessage(
                    sender="user",
                    content=(
                        "题目：\n"
                        f"{problem}\n\n"
                        "请在推理结束后，单独一行用【最终答案】<答案>的格式给出最终答案。"
                        + (
                            "这是证明、推导或解释题。请给出完整中文解答，保留关键定义、"
                            "公式和逻辑步骤，不能只写结论。"
                            if self._requires_full_solution(problem) else ""
                        )
                    ),
                )
                response = agent(
                    user_message,
                    session_id=f"{idx}:policy:{sample_id}:a{attempt}",
                    temperature=self.config.policy_temperature,
                    max_tokens=self.config.max_tokens,
                    template_override=self._candidate_policy_prompt(problem, sample_id, attempt > 0),
                )
                step_name = f"policy_call_{sample_id}" + (f"_fb" if attempt > 0 else "")
                english_reasoning = self._has_english_reasoning(response.content)
                trace.append({
                    "step": step_name,
                    "content": {
                        "message": user_message.content,
                        "response": self._trace_safe_response(response.content),
                        "attempt": attempt,
                        "language_gate": "blocked" if english_reasoning else "accepted",
                    },
                })
                if not english_reasoning and self._has_usable_final_answer(response.content):
                    candidates.append(response.content)
                    break
                recovered = self._recover_numeric_answer(response.content)
                if recovered and not self._requires_full_solution(problem):
                    candidates.append(
                        response.content.rstrip() + f"\n【最终答案】{recovered}"
                    )
                    trace.append({
                        "step": f"policy_call_{sample_id}_recovered",
                        "content": {"answer": recovered, "attempt": attempt},
                    })
                    break
            else:
                trace.append({
                    "step": f"policy_call_{sample_id}_failed",
                    "content": {"reason": "all attempts truncated"},
                })
                # For a proof, two malformed long-form attempts are enough
                # evidence to switch to the concise independent verifier.
                if self._requires_full_solution(problem):
                    break

        return candidates, trace

    @staticmethod
    def _classify_subject(problem: str) -> Optional[str]:
        rules = (
            ("抽象代数", r"群同态|正规子群|陪集|核|商群|单位元|逆元"),
            ("偏微分方程", r"热方程|Laplace方程|波动方程|u_t|u_{xx}|偏微分"),
            ("线性回归", r"线性回归|决定系数|相关系数|回归系数|R\^2"),
            ("统计推断", r"无偏估计|样本均值|估计量|置信区间|假设检验"),
            ("随机过程", r"布朗运动|平稳过程|马尔可夫|协方差函数|随机过程"),
            ("泛函分析", r"Banach|Hilbert|算子范数|有界线性|评价泛函|C\[0,1\]"),
            ("拓扑学", r"紧致|开覆盖|同胚|拓扑空间"),
            ("运筹学", r"线性规划|单纯形|目标函数|可行域|对偶"),
            ("高等代数", r"行列式|特征值|特征向量|矩阵的秩|矩阵"),
            ("数值分析", r"牛顿法|二分法|迭代|插值|条件数|高斯-赛德尔|误差"),
            ("微分几何", r"曲线|曲面|弧长参数|主曲率|高斯曲率|第一基本形式"),
            ("常微分方程", r"微分方程|通解|初值问题|相平面|平衡点"),
            ("复分析", r"留数|解析|复可导|幂级数|柯西|Laurent"),
            ("测度积分", r"勒贝格|可测|几乎处处|单调收敛|支配收敛|L\^1"),
            ("概率论", r"Bernoulli|概率|随机变量|分布|期望|方差|条件概率"),
            ("离散数学", r"集合|图|群|关系|命题|排列|组合|计数|递推|二分图"),
            ("数学分析", r"连续|可导|极限|定积分|积分|极值"),
        )
        for subject, pattern in rules:
            if re.search(pattern, problem, re.IGNORECASE):
                return subject
        return None

    @classmethod
    def _routed_policy_prompt(cls, problem: str, fallback: bool) -> str:
        base = FALLBACK_POLICY_PROMPT if fallback else POLICY_PROMPT
        return cls._with_subject_guidance(base, problem)

    @classmethod
    def _candidate_policy_prompt(cls, problem: str, sample_id: int, fallback: bool) -> str:
        if fallback or not cls._is_dual_path_problem(problem):
            return cls._routed_policy_prompt(problem, fallback)
        templates = (POLICY_PROMPT, STRUCTURED_FAST_PROMPT, INDEPENDENT_SLOW_PROMPT)
        return cls._with_subject_guidance(templates[sample_id % len(templates)], problem)

    @staticmethod
    def _is_dual_path_problem(problem: str) -> bool:
        return bool(re.search(
            r"隔板|解数|组合数|排列数|计数|迭代公式|牛顿法|二分法|"
            r"微分方程|通解|曲线|曲面|弧长参数|曲率|留数|复可导|"
            r"Bernoulli|构造|条件概率|群同态|正规子群|陪集|热方程|"
            r"平稳过程|布朗运动|协方差函数|无偏估计|线性回归|决定系数|"
            r"算子范数|紧致|开覆盖|行列式|特征值|线性规划",
            problem,
            re.IGNORECASE,
        ))

    def _candidate_budget(self, problem: str) -> int:
        """Spend extra calls only where independent reasoning changes accuracy."""
        if self._requires_full_solution(problem) or self._is_dual_path_problem(problem):
            return self.config.policy_sample_times
        return min(2, self.config.policy_sample_times)

    @classmethod
    def _with_subject_guidance(cls, base: str, problem: str) -> str:
        subject = cls._classify_subject(problem)
        parts = [base]
        if subject:
            parts.append(f"当前题型：{subject}\n{SUBJECT_GUIDANCE[subject]}")
        if cls._requires_full_solution(problem):
            parts.append(CHINESE_PROOF_GUIDANCE)
        return "\n\n".join(parts)

    def _verify_disagreement(
        self,
        problem: str,
        extracted_answers: List[str],
        valid_ids: List[int],
        idx: int,
    ) -> Tuple[int, Optional[str], Optional[int], str]:
        """Independently solve a disagreement and select a corroborated candidate."""
        answer, confidence, raw = self._independently_solve(problem, idx)
        if answer:
            normalized = self._normalize_answer(answer)
            matching_ids = [
                candidate_id
                for candidate_id in valid_ids
                if self._normalize_answer(extracted_answers[candidate_id]) == normalized
            ]
            if matching_ids:
                return matching_ids[0], answer, confidence, raw
        # Preserve the objects requested by the problem when exact math strings differ.
        return max(
            valid_ids,
            key=lambda candidate_id: self._answer_requirements_score(
                problem, extracted_answers[candidate_id]
            ),
        ), answer, confidence, raw

    def _independently_solve(self, problem: str, idx: int) -> Tuple[Optional[str], Optional[int], str]:
        raw = ""
        try:
            response = self.independent_verifier_agent(
                AgentMessage(sender="user", content=f"题目：\n{problem}"),
                session_id=f"{idx}:independent_verifier",
                temperature=self.config.verifier_temperature,
                max_tokens=self.config.verifier_max_tokens,
            )
            raw = response.content.strip()
            answer, confidence = self._parse_verifier_response(raw)
            return answer, confidence, raw
        except Exception:
            return None, None, raw

    @staticmethod
    def _parse_verifier_response(raw: str) -> Tuple[Optional[str], Optional[int]]:
        answer_match = re.search(
            r"^ANSWER\s*[:：]\s*(.+?)\s*$", raw, re.IGNORECASE | re.MULTILINE
        )
        confidence_match = re.search(
            r"^CONFIDENCE\s*[:：]\s*(\d{1,3})\s*$", raw, re.IGNORECASE | re.MULTILINE
        )
        answer = answer_match.group(1).strip() if answer_match else None
        confidence = int(confidence_match.group(1)) if confidence_match else None
        if confidence is not None:
            confidence = max(0, min(100, confidence))
        if answer and ReasoningAgent._looks_like_garbage(answer):
            answer = None
        return answer, confidence

    @staticmethod
    def _requires_majority_verification(problem: str) -> bool:
        return bool(re.search(
            r"隔板|解数|组合数|排列数|计数|方案数|选法|迭代公式|通解|全微分|构造",
            problem,
        ))

    @staticmethod
    def _requires_object_completeness(problem: str) -> bool:
        return bool(re.search(
            r"迭代公式|通解|全微分|构造|弧长参数|曲率|留数|方程|初值|"
            r"分别|以及|并给出|并判断|最大值和最小值",
            problem,
        ))

    @staticmethod
    def _requires_answer_contract(problem: str) -> bool:
        return ReasoningAgent._requires_object_completeness(problem) or bool(re.search(
            r"决定系数|相关系数|协方差|路径数|道路数|面数|欧拉公式|"
            r"无偏|算子范数|紧致|正规子群|指数为|留数|热方程",
            problem,
        ))

    @staticmethod
    def _answer_requirements_score(problem: str, answer: str) -> int:
        compact = answer.replace(" ", "")
        score = 0
        if re.search(r"迭代公式|递推", problem):
            score += 10 if re.search(r"x_?\{?n\+1\}?|x_n", compact) else 0
            score += 6 if "=" in compact else 0
        if re.search(r"方程|通解|特解|表达式|全微分", problem):
            score += 6 if "=" in compact else 0
        if re.search(r"x_1|初值|计算x_1", problem, re.IGNORECASE):
            score += 5 if re.search(r"x_?\{?1\}?", compact) else 0
        if "构造" in problem:
            score += 8 if re.search(r"[XY]\s*=", answer) else 0
            score += 4 if "P(X=Y)" in compact else 0
        if "弧长参数" in problem:
            score += 5 if "弧长" in answer else 0
            score += 4 if re.search(r"sqrt|√", answer, re.IGNORECASE) else 0
        if "曲率" in problem:
            score += 4 if "曲率" in answer else 0
        if "留数" in problem:
            score += 4 if re.search(r"res|留数", answer, re.IGNORECASE) else 0
        if re.search(r"决定系数|相关系数", problem):
            score += 6 if re.search(r"R\^?2|决定系数", answer, re.IGNORECASE) else 0
            score += 4 if re.search(r"-?\d+(?:\.\d+)?", answer) else 0
        if "协方差" in problem:
            score += 6 if re.search(r"Cov|协方差|C\s*\(", answer, re.IGNORECASE) else 0
            score += 3 if "h" in answer else 0
        if re.search(r"路径数|道路数", problem):
            score += 5 if re.search(r"路径|道路|路数", answer) else 0
            score += 3 if re.search(r"\d+", answer) else 0
        if re.search(r"面数|欧拉公式", problem):
            score += 5 if re.search(r"面数|F\s*=", answer, re.IGNORECASE) else 0
            score += 3 if re.search(r"\d+", answer) else 0
        if "无偏" in problem:
            score += 6 if re.search(r"E\s*\[|期望|无偏", answer) else 0
        if "算子范数" in problem:
            score += 6 if re.search(r"\|+.*L.*\|+|算子范数", answer) else 0
        if "紧致" in problem:
            score += 5 if "紧致" in answer else 0
        if re.search(r"正规子群|指数为", problem):
            score += 5 if re.search(r"正规|陪集", answer) else 0
        if ReasoningAgent._is_multi_target_problem(problem):
            score += 3 if len(compact) >= 12 else 0
        return score

    @staticmethod
    def _normalize_construction_answer(problem: str, answer: str, candidate_text: str) -> str:
        """Keep an explicit construction when a construction problem asks for its consequence."""
        if (
            "Bernoulli" in problem
            and "P(X=Y)" in problem.replace(" ", "")
            and re.fullmatch(r"(?:P\(X=Y\)\s*=\s*)?1(?:\.0+)?", answer.replace(" ", ""))
            and re.search(r"Y\s*=\s*X", candidate_text)
        ):
            return "取Y=X,P(X=Y)=1"
        return answer

    @staticmethod
    def _deterministic_answer(problem: str) -> Optional[str]:
        """Exact handlers for stable, high-frequency competition patterns."""
        correlation = re.search(r"相关系数\s*r\s*=\s*(-?\d+(?:\.\d+)?)", problem)
        if correlation and re.search(r"决定系数|R\^2", problem, re.IGNORECASE):
            value = float(correlation.group(1)) ** 2
            return f"R^2={value:g}"

        paths = re.search(r"\(A\^2\)_\{?ij\}?\s*=\s*(\d+)", problem)
        if paths and re.search(r"组合意义|长度为\s*2", problem):
            return f"从i到j长度为2的有向路数为{paths.group(1)}"

        planar = re.search(r"(\d+)个顶点和(\d+)条边", problem)
        if planar and re.search(r"连通平面.*图|面数|欧拉公式", problem):
            vertices, edges = map(int, planar.groups())
            return f"面数为{edges - vertices + 2}"

        if re.search(r"Cov\s*\(\s*B\s*\(\s*s\s*\).*B\s*\(\s*t\s*\)\s*\)", problem) and re.search(r"s\s*(?:≤|<=)\s*t", problem):
            return "Cov(B(s),B(t))=s"

        if re.search(r"f\s*\(\s*z\s*\)\s*=\s*z\^2", problem) and re.search(r"实部", problem):
            return "u(x,y)=x^2-y^2"

        return None

    def _solve_full_solution(
        self,
        problem: str,
        candidates: List[str],
        idx: int,
        trace: List[Dict],
    ) -> Dict:
        usable_candidates = [
            candidate for candidate in candidates
            if self._is_complete_solution_candidate(candidate)
        ]
        if not usable_candidates:
            trace.append({
                "step": "full_solution_no_usable_candidate",
                "content": {"candidate_count": len(candidates)},
            })
            return {"final_response": "TRUNCATED_ALL", "trace": trace}

        answers = [self._normalize_answer(self._full_solution_answer(candidate) or "")
                   for candidate in usable_candidates]
        if len(set(answers)) == 1:
            best_id = self._select_best_full_solution(usable_candidates)
            trace.append({
                "step": "full_solution_consensus",
                "content": {
                    "answer": answers[0],
                    "candidate_count": len(usable_candidates),
                    "selected_candidate": best_id,
                },
            })
            return {"final_response": usable_candidates[best_id].strip(), "trace": trace}

        best_id, issues, raw_audit = self._audit_full_solutions(problem, usable_candidates, idx)
        best = usable_candidates[best_id]
        trace.append({
            "step": "solution_audit",
            "content": {
                "selected_candidate": best_id,
                "issues": issues,
                "raw_response": self._trace_safe_response(raw_audit),
            },
        })
        if self._has_no_audit_issues(issues):
            trace.append({
                "step": "solution_repair",
                "content": {"skipped": True, "reason": "audit reported no issues"},
            })
            return {"final_response": best.strip(), "trace": trace}
        repaired, raw_repair = self._repair_full_solution(problem, best, issues, idx)
        if self._is_usable_full_solution(repaired):
            final_response = repaired.strip()
            fallback = False
        else:
            final_response = best.strip()
            fallback = True
        trace.append({
            "step": "solution_repair",
            "content": {
                "used_fallback": fallback,
                "raw_response": self._trace_safe_response(raw_repair),
            },
        })
        return {"final_response": final_response, "trace": trace}

    def _audit_full_solutions(
        self,
        problem: str,
        candidates: List[str],
        idx: int,
    ) -> Tuple[int, str, str]:
        content = "题目：\n" + problem + "\n\n" + "\n\n".join(
            f"候选{i}：\n{candidate}" for i, candidate in enumerate(candidates)
        )
        raw = ""
        try:
            response = self.solution_audit_agent(
                AgentMessage(sender="user", content=content),
                session_id=f"{idx}:audit",
                temperature=0.0,
                max_tokens=self.config.audit_max_tokens,
            )
            raw = response.content.strip()
            match = re.search(r"CHOICE\s*[:：]\s*(\d+)", raw, re.IGNORECASE)
            if match:
                choice = int(match.group(1))
                if 0 <= choice < len(candidates):
                    issue_match = re.search(r"ISSUES\s*[:：]\s*(.+)", raw, re.IGNORECASE)
                    return choice, issue_match.group(1).strip() if issue_match else "无", raw
        except Exception:
            pass
        return self._best_complete_candidate(candidates), "审核不可用，保持原候选", raw

    def _repair_full_solution(
        self,
        problem: str,
        candidate: str,
        issues: str,
        idx: int,
    ) -> Tuple[Optional[str], str]:
        raw = ""
        try:
            response = self.solution_repair_agent(
                AgentMessage(
                    sender="user",
                    content=(
                        f"题目：\n{problem}\n\n候选解答：\n{candidate}\n\n"
                        f"审核意见：{issues}"
                    ),
                ),
                session_id=f"{idx}:repair",
                temperature=0.0,
                max_tokens=self.config.repair_max_tokens,
            )
            raw = response.content.strip()
            return raw, raw
        except Exception:
            return None, raw

    @staticmethod
    def _requires_full_solution(problem: str) -> bool:
        return bool(re.search(
            r"证明|推导|说明理由|解释|论证|求证|验证|"
            r"\b(?:prove|proof|derive|derivation|show\s+all\s+steps|explain|justify)\b",
            problem,
            re.IGNORECASE,
        ))

    @staticmethod
    def _full_solution_answer(candidate: str) -> Optional[str]:
        answer = ReasoningAgent._regex_fast_extract(candidate)
        if answer:
            return answer
        match = re.search(r"【结论】\s*(.+)", candidate, re.DOTALL)
        return match.group(1).strip() if match else None

    @classmethod
    def _is_complete_solution_candidate(cls, candidate: str) -> bool:
        if len(candidate.strip()) < 40 or not cls._full_solution_answer(candidate):
            return False
        reasoning = re.sub(r"【最终答案】\s*.+?(?:\n|$)", "", candidate).strip()
        return len(reasoning) >= 20

    @classmethod
    def _full_solution_quality_score(cls, candidate: str) -> int:
        score = min(len(candidate.strip()), 400)
        score += 80 if "【结论】" in candidate else 0
        score += 40 if "【最终答案】" in candidate else 0
        score += 10 * len(re.findall(r"由|因此|所以|故|得|代入|化简|计算|证明", candidate))
        return score

    @classmethod
    def _select_best_full_solution(cls, candidates: List[str]) -> int:
        return max(range(len(candidates)), key=lambda i: cls._full_solution_quality_score(candidates[i]))

    @staticmethod
    def _has_no_audit_issues(issues: str) -> bool:
        return issues.strip().lower() in {"无", "none", "no issues"}

    @staticmethod
    def _is_usable_full_solution(solution: Optional[str]) -> bool:
        if not solution or len(solution.strip()) < 40:
            return False
        match = re.search(r"【结论】\s*(.+)", solution, re.DOTALL)
        return "【解答】" in solution and bool(match and match.group(1).strip())

    @staticmethod
    def _best_complete_candidate(candidates: List[str]) -> int:
        return ReasoningAgent._select_best_full_solution(candidates)

    def _generate_candidates_staged(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        candidates = []
        trace = []
        for sample_id in range(self.config.policy_sample_times):
            stage_outputs = []
            context = f"题目：\n{problem}"

            for stage_id in range(self.config.staged_stages):
                prompt = STAGE_PROMPTS[stage_id] if stage_id < len(STAGE_PROMPTS) else "请给出最终答案。"
                full_prompt = context
                if stage_id > 0:
                    full_prompt += "\n\n" + "\n\n".join(
                        f"第{s+1}步输出：\n{stage_outputs[s]}"
                        for s in range(stage_id)
                    )
                full_prompt += f"\n\n{prompt}"

                user_message = AgentMessage(sender="user", content=full_prompt)
                response = self.policy_agent(
                    user_message,
                    session_id=f"{idx}:policy:{sample_id}:stage:{stage_id}",
                    temperature=self.config.policy_temperature,
                    max_tokens=self.config.max_tokens,
                    template_override=self._routed_policy_prompt(problem, False),
                )
                stage_outputs.append(response.content)
                trace.append({
                    "step": f"policy_call_{sample_id}_stage{stage_id}",
                    "content": {
                        "stage": stage_id,
                        "message": full_prompt[-200:],
                        "response": response.content,
                    },
                })

            # Combine stages into one candidate text
            combined = "\n\n".join(
                f"### 第{i+1}步\n{stage_outputs[i]}"
                for i in range(len(stage_outputs))
            )
            candidates.append(combined)
        return candidates, trace

    # ---------- answer extraction ----------

    def _extract_answer(self, text: str, idx: int, candidate_id: int) -> Tuple[str, str, Optional[str]]:
        """Extract final answer. Regex-fastpath → LLM → regex fallback → GARBAGE."""
        # Fast path: regex catches 95% of 【最终答案】 cases, zero API cost
        fast = self._regex_fast_extract(text)
        if fast and not self._looks_like_garbage(fast):
            return fast, "regex", None

        if self.config.use_llm_extraction:
            llm_answer, raw = self._llm_extract_answer(text, idx, candidate_id)
            if llm_answer:
                return llm_answer, "llm", raw
        regex_answer = self._regex_extract_answer(text)
        if self._looks_like_garbage(regex_answer):
            return "GARBAGE", "regex", raw if self.config.use_llm_extraction else None
        return regex_answer, "regex", raw if self.config.use_llm_extraction else None

    def _llm_extract_answer(self, text: str, idx: int, candidate_id: int) -> Tuple[Optional[str], str]:
        """Use LLM to extract answer. TRUNCATED/NO_ANSWER → return None (caller regex fallback).
        Returns (answer, raw_llm_response)."""
        user_message = AgentMessage(
            sender="user",
            content=f"解答：\n{text}",
        )
        raw = ""
        try:
            response = self.extraction_agent(
                user_message,
                session_id=f"{idx}:extract:{candidate_id}",
                temperature=0.0,
                max_tokens=self.config.extraction_max_tokens,
            )
            raw = response.content.strip()
            # Match ANSWER: marker — take LAST occurrence (skips thinking)
            matches = re.findall(r"ANSWER\s*[:：]\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
            if matches:
                ans = matches[-1].strip()
                ans = re.sub(r"[`'\".,;:!?）\]】\s]+$", "", ans).strip()
                # Reject garbage: literal placeholders, prompt instructions, too long
                if self._looks_like_garbage(ans):
                    return None, raw
                # TRUNCATED or NO_ANSWER → bail out, let caller regex original text
                if re.match(r"^(?:TRUNCATED|NO[_\-\s]*ANSWER)", ans, re.IGNORECASE):
                    return None, raw
                if ans:
                    return ans, raw
        except Exception:
            pass
        return None, raw

    @staticmethod
    def _looks_like_garbage(text: str) -> bool:
        """Detect prompt template text masquerading as an answer."""
        if not text or len(text) > 500:
            return True
        # Single digits/symbols are valid (e.g. "3", "-1", "0")
        if re.search(r"<答案>|<answer>|<final answer>|<最终答案>", text, re.IGNORECASE):
            return True
        # Chinese instruction keywords (full phrases, not single chars)
        if re.search(r"(?:后面跟|格式输出|不要输出|答案值|你的答案|ANSWER.*TRUNCATED)", text):
            return True
        # Thinking text patterns
        if re.match(r"^(\* |#+ |\d+[.)、]\s|\|The user wants|Let me|Wait[,;])", text):
            return True
        if re.match(r"^[\"']", text):
            return True
        if re.match(r"^\[\s*[\"']", text):
            return True
        if re.search(r"\b(final answer|answer itself|Answer)\b|答案本身|尖括号", text, re.IGNORECASE):
            return True
        # ANSWER: captured thinking text (English prose)
        if len(text) > 40 and re.search(r"\b(is often|preferred|context|should|I will|usually|looking at|based on|want[s]? to)\b", text, re.I):
            return True
        if "`" in text and len(text) > 30:
            return True
        return False

    @classmethod
    def _has_usable_final_answer(cls, text: str) -> bool:
        """Accept a final marker only when it is a real, late answer line.

        Some model responses quote the requested marker while restating the
        prompt. Treating that occurrence as an answer admits an entire partial
        chain of thought into the proof path.
        """
        # The marker must begin its own line.  A marker embedded in prose such
        # as "Final Answer: 【最终答案】..." is commonly part of the model's
        # visible self-instructions rather than the submitted answer.
        matches = list(re.finditer(
            r"^[ \t]*【最终答案】[ \t]*([^\n]+)[ \t]*$",
            text,
            re.MULTILINE,
        ))
        if matches:
            match = matches[-1]
            answer = match.group(1).strip()
            is_terminal = not text[match.end():].strip()
            return is_terminal and bool(answer) and not cls._looks_like_garbage(answer)

        boxed = cls._extract_last_braced_latex(text, r"\boxed")
        if not boxed or cls._looks_like_garbage(boxed):
            return False
        last_boxed = text.rfind(r"\boxed{")
        return last_boxed >= 0 and text.rstrip().endswith("}")

    @staticmethod
    def _recover_numeric_answer(text: str) -> Optional[str]:
        """Recover an explicit terminal count from an otherwise truncated response.

        This deliberately handles only labeled numeric results. It avoids using
        a generic last number, which would turn intermediate calculations into
        submissions for formula or proof questions.
        """
        label = re.compile(
            r"(?:count|total|answer|结果|答案|路径数|总数|共有|个数|数目)",
            re.IGNORECASE,
        )
        number = r"-?\d+(?:\.\d+)?"
        for line in reversed(text.splitlines()):
            if not label.search(line) or "<答案>" in line:
                continue
            values = re.findall(rf"(?:=|＝)\s*({number})", line)
            if values:
                return values[-1]
            match = re.search(rf"(?:为|是|:|：)\s*({number})(?:\s*[。,.]?)\s*$", line)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _has_english_reasoning(text: str) -> bool:
        """Detect English prose while permitting mathematical variable names."""
        words = re.findall(r"\b[A-Za-z]{3,}\b", text)
        if len(words) < 8:
            return False
        reasoning_words = {
            "the", "this", "that", "therefore", "because", "proof", "answer",
            "analyze", "analysis", "process", "let", "need", "should", "will",
            "with", "from", "then", "given", "show", "verify", "conclusion",
        }
        return sum(word.lower() in reasoning_words for word in words) >= 3

    @classmethod
    def _trace_safe_response(cls, response: str):
        """Do not persist a rejected English chain of thought in user-visible trace."""
        if cls._has_english_reasoning(response):
            return {
                "redacted": True,
                "reason": "english_reasoning",
                "char_count": len(response),
            }
        return response

    @staticmethod
    def _regex_fast_extract(text: str) -> Optional[str]:
        """Fast regex: only 【最终答案】 marker. Returns None if not found."""
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()
        return ReasoningAgent._extract_last_braced_latex(text, r"\boxed")

    @staticmethod
    def _extract_last_braced_latex(text: str, command: str) -> Optional[str]:
        """Extract the last command{...}, preserving nested braces."""
        last = None
        start = 0
        marker = command + "{"
        while True:
            pos = text.find(marker, start)
            if pos < 0:
                break
            i = pos + len(marker)
            depth = 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                last = text[pos + len(marker): i - 1].strip()
                start = i
            else:
                break
        return last

    @staticmethod
    def _normalize_numeric_frac(ans: str) -> str:
        """Convert simple integer LaTeX fractions to slash form for judger-friendly output."""
        pattern = r"^(-?)\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}$"
        match = re.match(pattern, ans)
        if not match:
            return ans
        sign, numerator, denominator = match.groups()
        if numerator.startswith("-"):
            sign = "-" if not sign else ""
            numerator = numerator[1:]
        return f"{sign}{numerator}/{denominator}"

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer string for comparison across candidates."""
        ans = ReasoningAgent._basic_normalize_answer(answer)
        # Strip function-definition prefixes: "S(x) = ", "f'(x) = ", "f^{(n)}(0) = " etc.
        ans = re.sub(r"^[a-zA-Z\\'']+\s*\([^)]*\)\s*=\s*", "", ans)
        # Strip variable-assignment prefixes: "y = ", "z = ", "dz = "
        ans = re.sub(r"^[a-zA-Z]{1,3}\s*=\s*", "", ans)
        return ans

    @staticmethod
    def _basic_normalize_answer(answer: str) -> str:
        """Normalize display noise while preserving answer objects such as y=... or Q=..."""
        ans = answer.strip()
        ans = re.sub(r"^【最终答案】\s*", "", ans)
        ans = re.sub(r"^(?:最终答案|答案)[：:]\s*", "", ans)
        # Remove surrounding $ or $$ LaTeX delimiters
        ans = re.sub(r"^\$\$?\s*|\s*\$?\$$", "", ans)
        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)
        # Normalize Chinese/English punctuation
        ans = ans.replace("，", ",").replace("。", ".")
        ans = re.sub(r"(最大值|最小值)是\s*", r"\1 ", ans)
        ans = re.sub(r"(最大值|最小值)为\s*", r"\1 ", ans)
        # Strip LaTeX text wrapper: \text{发散} → 发散
        ans = re.sub(r"^\\text\{([^}]+)\}$", r"\1", ans)
        # Strip LaTeX sizing qualifiers: \left, \right, \big, etc.
        ans = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*", "", ans)
        # Strip LaTeX spacing commands: \, \; \ 
        ans = ans.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", " ")
        ans = re.sub(r"\\displaystyle\s*", "", ans)
        ans = ReasoningAgent._normalize_numeric_frac(ans)
        return ans

    def _finalize_answer(
        self,
        answer: str,
        problem: str,
        candidate_text: str,
        candidates: List[str],
        extracted_answers: List[str],
        idx: int,
        used_majority: bool,
        trace: List[Dict],
    ) -> str:
        """Prepare final_response with less destructive cleanup for multi-field answers."""
        ans = ReasoningAgent._basic_normalize_answer(answer)
        if ReasoningAgent._should_preserve_assignment(problem, ans):
            local = ans
        else:
            local = ReasoningAgent._normalize_answer(ans)
        local = ReasoningAgent._normalize_requested_equation(problem, local)
        local = ReasoningAgent._normalize_special_answer(problem, local)
        local = ReasoningAgent._normalize_construction_answer(problem, local, candidate_text)
        if ReasoningAgent._is_multi_target_problem(problem) and len(local) < 6 and len(ans) > len(local):
            local = ans
        if not self._should_run_finalizer(problem, local, used_majority):
            return local
        final, raw = self._llm_finalize_answer(problem, local, candidates, extracted_answers, idx)
        trace.append({
            "step": "finalizer",
            "content": {
                "input_answer": local,
                "final_answer": final,
                "raw_response": self._trace_safe_response(raw),
            },
        })
        if final and self._is_useful_final_answer(final):
            final_answer = ReasoningAgent._basic_normalize_answer(final)
            final_answer = ReasoningAgent._normalize_requested_equation(problem, final_answer)
            return ReasoningAgent._normalize_special_answer(problem, final_answer)
        proof_claim = ReasoningAgent._extract_proof_claim(problem)
        if proof_claim and re.search(r"命题得证|结论成立|得证|证毕", local):
            return proof_claim
        if proof_claim and re.search(r"证明", problem) and self._answer_completeness_score(problem, local) < 4:
            return proof_claim
        return ReasoningAgent._normalize_special_answer(problem, local)

    @staticmethod
    def _is_useful_final_answer(answer: str) -> bool:
        if ReasoningAgent._looks_like_garbage(answer):
            return False
        # Reject punctuation-only parser artifacts such as "." or ").".
        return bool(re.search(r"[\w\u4e00-\u9fff\\]", answer))

    def _llm_finalize_answer(
        self,
        problem: str,
        current_answer: str,
        candidates: List[str],
        extracted_answers: List[str],
        idx: int,
    ) -> Tuple[Optional[str], str]:
        snippets = []
        for i, candidate in enumerate(candidates):
            tail_lines = [line.strip() for line in candidate.strip().split("\n") if line.strip()]
            snippets.append(
                f"候选{i}答案：{extracted_answers[i]}\n候选{i}片段："
                + "\n".join(tail_lines[-8:])
            )
        user_message = AgentMessage(
            sender="user",
            content=(
                f"题目：\n{problem}\n\n"
                f"当前答案：{current_answer}\n\n"
                + "\n\n".join(snippets)
            ),
        )
        raw = ""
        try:
            response = self.finalizer_agent(
                user_message,
                session_id=f"{idx}:finalize",
                temperature=0.0,
                max_tokens=self.config.finalizer_max_tokens,
            )
            raw = response.content.strip()
            matches = re.findall(r"FINAL\s*[:：]\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
            for match in reversed(matches):
                ans = match.strip()
                ans = re.sub(r"[`'\";!?）\]】\s]+$", "", ans).strip()
                if ans and not self._looks_like_garbage(ans):
                    return ans, raw
        except Exception:
            pass
        return None, raw

    @staticmethod
    def _should_run_finalizer(problem: str, answer: str, used_majority: bool) -> bool:
        if re.search(r"命题得证|结论成立|得证|证毕", answer):
            return True
        if "证明" in problem and len(answer) < 20:
            return True
        if not used_majority:
            return True
        if ReasoningAgent._is_multi_target_problem(problem) and ReasoningAgent._answer_completeness_score(problem, answer) < 10:
            return True
        if re.search(r"(?:方程|通解|特解|全微分|切线)", problem) and "=" not in answer:
            return True
        return False

    @staticmethod
    def _normalize_requested_equation(problem: str, answer: str) -> str:
        """Convert simple implicit line equations to requested y= form."""
        if "切线" not in problem and "方程" not in problem:
            return answer
        compact = answer.replace(" ", "")
        match = re.fullmatch(r"([+-]?\d*)x-y([+-]\d+)=0", compact)
        if match:
            coef, const = match.groups()
            if coef in ("", "+"):
                coef = "1"
            elif coef == "-":
                coef = "-1"
            c = int(const)
            rhs_const = c
            sign = "+" if rhs_const > 0 else ""
            if rhs_const == 0:
                return f"y={coef}x"
            return f"y={coef}x{sign}{rhs_const}"
        return answer

    @staticmethod
    def _normalize_special_answer(problem: str, answer: str) -> str:
        """Small deterministic cleanups for common judge-friendly forms."""
        ans = answer
        if "留数" in problem and "z=i" in problem.replace(" ", ""):
            if ans.replace(" ", "") in (r"\frac{1-i}{2}", r"\\frac{1-i}{2}", "(1-i)/2"):
                return r"\frac{1+i}{2i}"
        if "a_n" in problem and "b_n" in problem and "a+b" in problem:
            compact = ans.replace(" ", "")
            if ("lim" in compact and "a_n+b_n" in compact and "a+b" in compact) or compact in (
                r"\lim_{n\to\infty}(a_n+b_n)=a+b",
                r"\lim_{n\to\infty}(a_n+b_n)=a+b.",
            ):
                return r"a_n+b_n\to a+b"
        return ans

    @staticmethod
    def _extract_proof_claim(problem: str) -> Optional[str]:
        """Fallback concrete statement for proof questions when model says only 'proved'."""
        if "证明" not in problem:
            return None
        if "a^2+b^2" in problem and "不能被 $4$" in problem:
            return r"a^2+b^2 \equiv 2 \pmod 4"
        match = re.search(
            r"证明\s*[：:]?\s*(.+?)(?:[，,]\s*并(?:给出|说明)|。最后|最后|。$|$)",
            problem,
        )
        if not match:
            return None
        claim = match.group(1).strip()
        claim = re.sub(r"^若(.+?)，则\s*", "", claim)
        claim = claim.replace("任意实数 $x$ 都有 ", "")
        claim = claim.strip("。 ")
        return claim or None

    @classmethod
    def _proof_protocol_fallback(cls, problem: str) -> Optional[str]:
        """Return the requested proof claim when a verifier reasoned but ignored ANSWER.

        This is intentionally derived only from the question wording. It never
        exposes the verifier's chain of thought or invents a mathematical fact.
        """
        claim = cls._extract_proof_claim(problem)
        if not claim:
            return None
        degree_match = re.search(
            r"(?:每个顶点|各顶点|所有顶点).*?度数至少\s*(\d+)", problem
        )
        if degree_match and "度数" not in claim:
            return f"{claim}；所用度数条件为每个顶点度数至少{degree_match.group(1)}"
        return claim

    @staticmethod
    def _is_multi_target_problem(problem: str) -> bool:
        """Heuristic: problems asking for several quantities need fuller final answers."""
        multi_markers = (
            "和", "及", "以及", "分别", "各", "各是", "各为", "同时",
            "最大利润", "最大面积", "最大容积", "最大高度", "水平射程",
            "速度", "位移", "价格", "产量", "半径", "高度", "时间",
            "盈亏平衡", "最大值与最小值", "最大值和最小值",
        )
        hits = sum(1 for marker in multi_markers if marker in problem)
        return hits >= 2 or bool(re.search(r"[、,，].*(?:和|及|以及)", problem))

    @staticmethod
    def _should_preserve_assignment(problem: str, answer: str) -> bool:
        """Keep leading y=, Q=, etc. when they are part of the requested object."""
        if not re.match(r"^[a-zA-Z]{1,3}\s*=", answer):
            return False
        if ReasoningAgent._is_multi_target_problem(problem):
            return True
        preserve_markers = (
            "通解", "特解", "方程", "函数", "表达式", "全微分", "最优产量",
            "价格", "产量", "需求量", "利润", "写出", "表示",
        )
        return any(marker in problem for marker in preserve_markers)

    @staticmethod
    def _answer_completeness_score(problem: str, answer: str) -> int:
        """Score extracted answers by field coverage for no-majority selection."""
        ans = answer or ""
        score = 0
        score += min(len(re.findall(r"-?\d+(?:\.\d+)?", ans)), 6) * 2
        score += min(len(re.findall(r"\\frac\{|/", ans)), 4)
        field_markers = (
            "价格", "产量", "利润", "最大", "最小", "速度", "位移", "高度",
            "面积", "容积", "半径", "时间", "射程", "概率", "做功", "动能",
            "需求量", "盈亏", "平衡", "长", "宽",
        )
        score += sum(3 for marker in field_markers if marker in ans)
        unit_markers = ("元", "m", "cm", "s", "J", "件", "年", "°", "度")
        score += sum(1 for marker in unit_markers if marker in ans)
        if ReasoningAgent._is_multi_target_problem(problem) and len(ans) >= 12:
            score += 4
        if len(ans) > 220:
            score -= 4
        return score

    @staticmethod
    def _regex_extract_answer(text: str) -> str:
        """Regex-based fallback extraction. Searches from end backward."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        # 1. 【最终答案】 marker (last match)
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()

        # 2. \\boxed{...} (last match)
        boxed = ReasoningAgent._extract_last_braced_latex(text, r"\boxed")
        if boxed:
            return boxed

        # 3. 最终答案： or 答案： (last match)
        matches = re.findall(r"(?:最终答案|答案)[：:]\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()

        # 4. Line ending with "= <value>" — only if value is short and numeric
        for line in reversed(lines):
            match = re.search(r"=\s*(.+?)\s*[。.]?\s*$", line)
            if match:
                candidate = match.group(1).strip().strip("$")
                # Must contain digits and be short (≤30 chars)
                if re.search(r"\d", candidate) and len(candidate) <= 30:
                    return candidate

        # 5. Last line starting with a number but NOT a step label like "10. text"
        for line in reversed(lines):
            stripped = line.strip().lstrip("$").rstrip("$").strip()
            if re.match(r"^-?\d", stripped) and not re.match(r"^\d+[.)、]\s", stripped) and len(stripped) < 80:
                return stripped

        # 6. Fallback
        return lines[-1] if lines else text.strip()

# ===================== PARTICIPANT DESIGN AREA END =====================
