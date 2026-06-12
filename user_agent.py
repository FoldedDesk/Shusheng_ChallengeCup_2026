import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from lagent.agents import Agent
from lagent.schema import AgentMessage

from llm_client import InternChatClient


# ==================== PARTICIPANT DESIGN AREA START ====================

POLICY_PROMPT = """你是一个严谨的数学推理智能体。请解决以下数学问题。

要求：
1. 先分析题意，明确已知条件和求解目标。
2. 给出完整、清晰的逐步推导过程。
3. 在最后单独一行，用「【最终答案】<答案>」的格式明确写出最终答案。

注意：最终答案必须放在【最终答案】之后，不要包含额外解释。"""

STAGE_PROMPTS = [
    "请分析以下数学问题，列出已知条件、求解目标、约束条件。不需要计算。",
    "基于前面的分析，提出解题思路和方法。说明解题策略即可，不需要具体计算。",
    "基于前面的分析和策略，逐步推导和计算。展示每一步的推理和计算过程。",
    "基于前面的推导，写出最终答案。严格使用【最终答案】<答案>的格式。",
]

VERIFIER_PROMPT = """你是一个数学答案验证器。请先独立求解下面的数学问题，得出你自己的答案，然后与候选解答的答案进行对比。

用以下格式输出（严格按此格式，每行一个标签）：
SELF_ANSWER: <你的答案>
MATCH: YES 或 NO
CONFIDENCE: <0-10的整数，10为非常确定>"""

EXTRACTION_PROMPT = """从以下数学解答中提取最终答案。用 ANSWER: <答案> 的格式输出，不要输出任何其他内容。找不到答案则输出 ANSWER: NO_ANSWER"""


@dataclass
class AgentConfig:
    policy_sample_times: int = 3
    verifier_voting_times: int = 2
    policy_temperature: float = 0.6
    verifier_temperature: float = 0.0
    max_tokens: int = 8192
    consistency_bonus_weight: float = 0.15
    use_llm_extraction: bool = True
    extraction_max_tokens: int = 1024
    majority_vote_threshold: float = 0.5
    use_staged_reasoning: bool = False
    staged_stages: int = 4


class ReasoningAgent:
    """A generate-verify-select agent with answer extraction and consistency scoring."""

    def __init__(self, client: InternChatClient, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.client = client
        self.policy_agent = Agent(
            llm=client,
            template=POLICY_PROMPT,
            name="policy_agent",
        )
        self.verifier_agent = Agent(
            llm=client,
            template=VERIFIER_PROMPT,
            name="verifier_agent",
        )
        self.extraction_agent = Agent(
            llm=client,
            template=EXTRACTION_PROMPT,
            name="extraction_agent",
        )

    # ---------- public API ----------

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)

        # Phase 1: generate multiple candidate solutions
        candidates, trace = self._generate_candidates(problem, idx)

        # Phase 2: extract answers from each candidate (LLM + regex fallback)
        extracted_answers = []
        for i, c in enumerate(candidates):
            ans, method, raw = self._extract_answer(c, idx, i)
            extracted_answers.append(ans)
            trace.append({
                "step": f"extract_answer_{i}",
                "content": {"method": method, "answer": ans, "raw_response": raw},
            })
        normalized_answers = [self._normalize_answer(a) for a in extracted_answers]

        # Phase 3: try majority voting first
        policy_answer_counts: Dict[str, int] = {}
        for ans in normalized_answers:
            policy_answer_counts[ans] = policy_answer_counts.get(ans, 0) + 1
        max_count = max(policy_answer_counts.values())
        max_fraction = max_count / len(candidates)

        if max_count >= 2 and max_fraction >= self.config.majority_vote_threshold:
            # Majority consensus — skip verifier
            majority_answer = next(a for a, c in policy_answer_counts.items() if c == max_count)
            best_id = next(i for i, a in enumerate(normalized_answers) if a == majority_answer)
            trace.append({
                "step": "majority_vote",
                "content": {
                    "answer": majority_answer,
                    "count": max_count,
                    "total": len(candidates),
                    "selected_candidate": best_id,
                },
            })
        else:
            # Phase 4 (no majority): verify each candidate and score
            scored_candidates = []
            for candidate_id, candidate in enumerate(candidates):
                verifier_score, verify_trace = self._verify_candidate(
                    problem, candidate, idx, candidate_id,
                )
                consistency = policy_answer_counts[normalized_answers[candidate_id]] - 1
                consistency_bonus = consistency * self.config.consistency_bonus_weight
                total_score = verifier_score + consistency_bonus

                scored_candidates.append({
                    "id": candidate_id,
                    "content": candidate,
                    "extracted_answer": extracted_answers[candidate_id],
                    "normalized_answer": normalized_answers[candidate_id],
                    "verifier_score": round(verifier_score, 4),
                    "consistency_bonus": round(consistency_bonus, 4),
                    "total_score": round(total_score, 4),
                })
                trace.extend(verify_trace)

            best = max(scored_candidates, key=lambda item: item["total_score"])
            best_id = best["id"]
            score_lines = " | ".join([
                f"#{s['id']} ans={s['extracted_answer'][:40]} score={s['total_score']:.3f}(v={s['verifier_score']:.3f}+c={s['consistency_bonus']:.3f})"
                for s in scored_candidates
            ])
            trace.append({
                "step": "score_summary",
                "content": f"{score_lines} | selected=#{best_id}",
            })
            trace.append({
                "step": "select_final_response",
                "content": {
                    "method": "verifier_confidence + answer_consistency",
                    "candidates": [
                        {
                            "id": s["id"],
                            "extracted_answer": s["extracted_answer"],
                            "verifier": s["verifier_score"],
                            "consistency": s["consistency_bonus"],
                            "total": s["total_score"],
                        }
                        for s in scored_candidates
                    ],
                    "selected": best_id,
                },
            })

        return {
            "final_response": extracted_answers[best_id],
            "trace": trace,
        }

    # ---------- generation ----------

    def _generate_candidates(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        if self.config.use_staged_reasoning:
            return self._generate_candidates_staged(problem, idx)
        return self._generate_candidates_oneshot(problem, idx)

    def _generate_candidates_oneshot(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        candidates = []
        trace = []
        for sample_id in range(self.config.policy_sample_times):
            user_message = AgentMessage(
                sender="user",
                content=(
                    "题目：\n"
                    f"{problem}\n\n"
                    "请在推理结束后，单独一行用【最终答案】<答案>的格式给出最终答案。"
                ),
            )
            response = self.policy_agent(
                user_message,
                session_id=f"{idx}:policy:{sample_id}",
                temperature=self.config.policy_temperature,
                max_tokens=self.config.max_tokens,
            )
            candidates.append(response.content)
            trace.append({
                "step": f"policy_call_{sample_id}",
                "content": {
                    "message": user_message.content,
                    "response": response.content,
                },
            })
        return candidates, trace

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

    # ---------- verification ----------

    def _verify_candidate(
        self, problem: str, candidate: str, idx: int, candidate_id: int,
    ) -> Tuple[float, List[Dict]]:
        scores = []
        trace = []
        for vote_id in range(self.config.verifier_voting_times):
            user_message = AgentMessage(
                sender="user",
                content=(
                    "题目：\n"
                    f"{problem}\n\n"
                    "候选解答：\n"
                    f"{candidate}"
                ),
            )
            response = self.verifier_agent(
                user_message,
                session_id=f"{idx}:verify:{candidate_id}:{vote_id}",
                temperature=self.config.verifier_temperature,
                max_tokens=1024,
            )
            verdict_text = response.content
            is_match, confidence, self_answer = self._parse_verdict(verdict_text)
            scores.append(confidence if is_match else 0.0)
            trace.append({
                "step": f"verifier_call_{candidate_id}_{vote_id}",
                "content": {
                    "candidate_id": candidate_id,
                    "verdict": verdict_text,
                    "self_answer": self_answer,
                    "parsed_match": is_match,
                    "parsed_confidence": confidence,
                },
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, trace

    # ---------- answer extraction ----------

    def _extract_answer(self, text: str, idx: int, candidate_id: int) -> Tuple[str, str, str | None]:
        """Extract final answer. Returns (answer, method, raw_response).
        raw_response is the LLM extraction agent's raw output, for debugging."""
        if self.config.use_llm_extraction:
            llm_answer, raw = self._llm_extract_answer(text, idx, candidate_id)
            if llm_answer:
                return llm_answer, "llm", raw
            # If LLM gave None, raw might still be useful for debugging
            regex_answer = self._regex_extract_answer(text)
            return regex_answer, "regex", raw
        regex_answer = self._regex_extract_answer(text)
        return regex_answer, "regex", None

    def _llm_extract_answer(self, text: str, idx: int, candidate_id: int) -> Tuple[str | None, str]:
        """Use LLM to extract answer. Matches ANSWER: marker (defense against thinking text).
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
                # Strip trailing punctuation/garbage from answer
                ans = re.sub(r"[`'\".,;:!?）\]】\s]+$", "", ans).strip()
                # Check if it's a NO_ANSWER variant (hyphen, underscore, dot, etc.)
                if ans and not re.match(r"^NO[_\-\s]*ANSWER", ans, re.IGNORECASE):
                    return ans, raw
            # Fallback: try regex extraction on LLM response
            regex_ans = self._regex_extract_answer(raw)
            if regex_ans:
                regex_ans = re.sub(r"[`'\".,;:!?）\]】\s]+$", "", regex_ans).strip()
                if regex_ans:
                    return regex_ans, raw
        except Exception:
            pass
        return None, raw

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Normalize answer string for comparison across candidates."""
        ans = answer.strip()
        # Remove surrounding $ or $$ LaTeX delimiters
        ans = re.sub(r"^\$\$?\s*|\s*\$?\$$", "", ans)
        # Collapse whitespace
        ans = re.sub(r"\s+", " ", ans)
        # Normalize Chinese/English punctuation
        ans = ans.replace("，", ",").replace("。", ".")
        return ans

    @staticmethod
    def _regex_extract_answer(text: str) -> str:
        """Regex-based fallback extraction. Searches from end backward."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        # 1. 【最终答案】 marker (last match)
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()

        # 2. \\boxed{...} (last match)
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].strip()

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

    # ---------- verdict parsing ----------

    @staticmethod
    def _parse_verdict(verdict_text: str) -> Tuple[bool, float, str | None]:
        """Parse verifier output using SELF_ANSWER/MATCH/CONFIDENCE markers.
        Takes LAST match of each marker to skip Intern-S thinking text."""
        self_answer = None
        is_match = False
        confidence = 0.0

        # Parse SELF_ANSWER — take LAST match
        sa_matches = re.findall(r"SELF_ANSWER\s*[:：]\s*(.+?)(?:\n|$)", verdict_text, re.IGNORECASE)
        if sa_matches:
            raw = sa_matches[-1].strip()
            if raw and raw.upper() != "NONE":
                self_answer = raw

        # Parse MATCH — take LAST match
        match_matches = re.findall(r"MATCH\s*[:：]\s*(YES|NO)", verdict_text, re.IGNORECASE)
        if match_matches:
            is_match = match_matches[-1].upper() == "YES"

        # Parse CONFIDENCE — take LAST match
        conf_matches = re.findall(r"CONFIDENCE\s*[:：]\s*(\d+)", verdict_text, re.IGNORECASE)
        if conf_matches:
            confidence = min(int(conf_matches[-1]), 10) / 10.0

        return is_match, confidence, self_answer


# ===================== PARTICIPANT DESIGN AREA END =====================
