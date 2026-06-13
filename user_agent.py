from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lagent.agents import Agent
from lagent.schema import AgentMessage


# ==================== PARTICIPANT DESIGN AREA START ====================

POLICY_PROMPT = """你是一个严谨的数学推理智能体。请解决以下数学问题。

要求：
1. 先分析题意，明确已知条件和求解目标。
2. 给出完整、清晰的逐步推导过程。
3. 在最后单独一行，用「【最终答案】<答案>」的格式明确写出最终答案。

注意：请全程使用中文推导，不要输出英文思考过程。最终答案必须放在【最终答案】之后。"""

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
CONFIDENCE: <0-10的整数，10为非常确定>

注意：不要输出任何思考过程。第一行必须是 SELF_ANSWER:。"""

EXTRACTION_PROMPT = """从以下数学解答中提取最终答案。用 ANSWER: <答案> 的格式输出。如果解答不完整或截断，输出 ANSWER: TRUNCATED。不要输出其他内容。"""

FALLBACK_POLICY_PROMPT = """你是一个数学求解器。直接计算并给出最终答案，跳过分析过程。
严格用一行输出：【最终答案】<答案>
不要输出英文思考。"""


@dataclass
class AgentConfig:
    policy_sample_times: int = 3
    verifier_voting_times: int = 2
    policy_temperature: float = 0.6
    verifier_temperature: float = 0.0
    max_tokens: int = 12288
    consistency_bonus_weight: float = 0.15
    use_llm_extraction: bool = True
    extraction_max_tokens: int = 1024
    majority_vote_threshold: float = 0.5
    use_staged_reasoning: bool = False
    staged_stages: int = 4


class ReasoningAgent:
    """A generate-verify-select agent with answer extraction and consistency scoring."""

    def __init__(self, client, config: Optional[AgentConfig] = None) -> None:
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
        self._fallback_agent = Agent(
            llm=client, template=FALLBACK_POLICY_PROMPT, name="fallback_policy",
        )

    # ---------- public API ----------

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)

        # Phase 1: generate multiple candidate solutions
        candidates, trace = self._generate_candidates(problem, idx)

        # Guard: all candidates truncated after retries
        if not candidates:
            trace.append({"step": "all_truncated", "content": "All candidates truncated after retries"})
            return {"final_response": "TRUNCATED_ALL", "trace": trace}

        # Phase 2: extract answers from each candidate (LLM + regex fallback)
        extracted_answers = []
        is_garbage = []
        for i, c in enumerate(candidates):
            ans, method, raw = self._extract_answer(c, idx, i)
            extracted_answers.append(ans)
            is_garbage.append(ans == "GARBAGE")
            trace.append({
                "step": f"extract_answer_{i}",
                "content": {"method": method, "answer": ans, "raw_response": raw},
            })
        normalized_answers = [self._normalize_answer(a) for a in extracted_answers]

        # Filter to valid (non-garbage) candidates
        valid_ids = [i for i in range(len(candidates)) if not is_garbage[i]]
        if not valid_ids:
            trace.append({"step": "all_garbage", "content": "All extracted answers are garbage"})
            return {"final_response": "ALL_GARBAGE", "trace": trace}

        # Phase 3: try majority voting (only among valid answers)
        valid_counts: Dict[str, int] = {}
        for i in valid_ids:
            ans = normalized_answers[i]
            valid_counts[ans] = valid_counts.get(ans, 0) + 1
        max_count = max(valid_counts.values())
        total_valid = len(valid_ids)

        if max_count >= 2 and max_count / total_valid >= self.config.majority_vote_threshold:
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
        else:
            # Phase 4: only 1 valid candidate → skip verifier, use directly
            if total_valid == 1:
                best_id = valid_ids[0]
                trace.append({
                    "step": "single_valid",
                    "content": {
                        "candidate": best_id,
                        "garbage_skipped": len(candidates) - 1,
                    },
                })
            else:
                # Phase 4b: verify each candidate and score
                scored_candidates = []
                for candidate_id, candidate in enumerate(candidates):
                    if is_garbage[candidate_id]:
                        scored_candidates.append({
                            "id": candidate_id,
                            "content": candidate,
                            "extracted_answer": extracted_answers[candidate_id],
                            "normalized_answer": "GARBAGE",
                            "verifier_score": 0.0,
                            "consistency_bonus": 0.0,
                            "total_score": -1.0,
                        })
                        trace.append({
                            "step": f"verifier_skip_{candidate_id}",
                            "content": {"reason": "extracted answer is garbage"},
                        })
                        continue
                    verifier_score, verify_trace = self._verify_candidate(
                        problem, candidate, idx, candidate_id,
                    )
                    consistency = valid_counts.get(normalized_answers[candidate_id], 1) - 1
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
            "final_response": self._normalize_answer(extracted_answers[best_id]),
            "trace": trace,
        }

    # ---------- generation ----------

    def _generate_candidates(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        if self.config.use_staged_reasoning:
            return self._generate_candidates_staged(problem, idx)
        return self._generate_candidates_oneshot(problem, idx)

    def _generate_candidates_oneshot(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
        """Generate candidates with truncation retry (1 fallback) and 2/2 early exit."""
        candidates = []
        trace = []
        agents = [self.policy_agent, self._fallback_agent]  # 1 normal + 1 fallback
        for sample_id in range(self.config.policy_sample_times):
            for attempt, agent in enumerate(agents):
                user_message = AgentMessage(
                    sender="user",
                    content=(
                        "题目：\n"
                        f"{problem}\n\n"
                        "请在推理结束后，单独一行用【最终答案】<答案>的格式给出最终答案。"
                    ),
                )
                response = agent(
                    user_message,
                    session_id=f"{idx}:policy:{sample_id}:a{attempt}",
                    temperature=self.config.policy_temperature,
                    max_tokens=self.config.max_tokens,
                )
                step_name = f"policy_call_{sample_id}" + (f"_fb" if attempt > 0 else "")
                trace.append({
                    "step": step_name,
                    "content": {
                        "message": user_message.content,
                        "response": response.content,
                        "attempt": attempt,
                    },
                })
                if "【最终答案】" in response.content:
                    candidates.append(response.content)
                    break
            else:
                trace.append({
                    "step": f"policy_call_{sample_id}_failed",
                    "content": {"reason": "all attempts truncated"},
                })

            # Early exit: 2 candidates agree on fast regex → skip 3rd
            if sample_id == 1 and len(candidates) == 2:
                a1 = self._regex_fast_extract(candidates[0])
                a2 = self._regex_fast_extract(candidates[1])
                if a1 is not None and a2 is not None and self._normalize_answer(a1) == self._normalize_answer(a2):
                    trace.append({
                        "step": "early_exit",
                        "content": {"reason": "first 2 candidates agree", "answer": a1},
                    })
                    break
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

    def _llm_extract_answer(self, text: str, idx: int, candidate_id: int) -> Tuple[str | None, str]:
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
        if re.search(r"<答案>|<answer>", text, re.IGNORECASE):
            return True
        # Chinese instruction keywords (full phrases, not single chars)
        if re.search(r"(?:后面跟|格式输出|不要输出|答案值|你的答案|ANSWER.*TRUNCATED)", text):
            return True
        # Thinking text patterns
        if re.match(r"^(\* |#+ |\d+[.)、]\s|\|The user wants|Let me|Wait[,;])", text):
            return True
        # ANSWER: captured thinking text (English prose)
        if len(text) > 40 and re.search(r"\b(is often|preferred|context|should|I will|usually|looking at|based on|want[s]? to)\b", text, re.I):
            return True
        if "`" in text and len(text) > 30:
            return True
        return False

    @staticmethod
    def _regex_fast_extract(text: str) -> str | None:
        """Fast regex: only 【最终答案】 marker. Returns None if not found."""
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].strip()
        return None

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
        # Strip LaTeX text wrapper: \text{发散} → 发散
        ans = re.sub(r"^\\text\{([^}]+)\}$", r"\1", ans)
        # Strip LaTeX sizing qualifiers: \left, \right, \big, etc.
        ans = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*", "", ans)
        # Strip LaTeX spacing commands: \, \; \ 
        ans = ans.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", " ")
        ans = re.sub(r"\\displaystyle\s*", "", ans)
        # Strip function-definition prefixes: "S(x) = ", "f'(x) = ", "f^{(n)}(0) = " etc.
        ans = re.sub(r"^[a-zA-Z\\'']+\s*\([^)]*\)\s*=\s*", "", ans)
        # Strip variable-assignment prefixes: "y = ", "z = ", "dz = "
        ans = re.sub(r"^[a-zA-Z]{1,3}\s*=\s*", "", ans)
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
        """Parse verifier output. Filters template text, takes first valid match."""
        self_answer = None
        is_match = False
        confidence = 0.0

        # Helper: filter out prompt template text
        def _valid_sa(text: str) -> bool:
            text = text.strip()
            if not text or text.upper() == "NONE":
                return False
            if re.search(r"[你的答案]|<|>|格式|输出|MATCH|CONFIDENCE", text):
                return False
            return True

        # Parse SELF_ANSWER — take LAST match that passes validation
        sa_matches = re.findall(r"SELF_ANSWER\s*[:：]\s*(.+?)(?:\n|$)", verdict_text, re.IGNORECASE)
        for raw in reversed(sa_matches):
            if _valid_sa(raw):
                self_answer = raw.strip()
                break

        # Parse MATCH — take LAST match
        match_matches = re.findall(r"MATCH\s*[:：]\s*(YES|NO)", verdict_text, re.IGNORECASE)
        if match_matches:
            is_match = match_matches[-1].upper() == "YES"

        # Parse CONFIDENCE — take LAST match
        conf_matches = re.findall(r"CONFIDENCE\s*[:：]\s*(\d+)", verdict_text, re.IGNORECASE)
        for raw in reversed(conf_matches):
            val = int(raw)
            if 0 <= val <= 10:
                confidence = val / 10.0
                break

        return is_match, confidence, self_answer


# ===================== PARTICIPANT DESIGN AREA END =====================
