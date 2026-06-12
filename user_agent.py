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

VERIFIER_PROMPT = """你是一个数学答案验证器。请判断候选解答是否正确解决了题目。

不要输出解释。严格按以下两行格式输出：
VERDICT: <A或B>
CONFIDENCE: <0到10的整数>

其中：
- VERDICT A 表示解答正确，B 表示解答错误。
- CONFIDENCE 表示你对判断的确信程度，0 完全不确定，10 非常确定。"""


@dataclass
class AgentConfig:
    policy_sample_times: int = 3
    verifier_voting_times: int = 2
    policy_temperature: float = 0.6
    verifier_temperature: float = 0.0
    max_tokens: int = 4096
    consistency_bonus_weight: float = 0.15


class ReasoningAgent:
    """A generate-verify-select agent with answer extraction and consistency scoring."""

    def __init__(self, client: InternChatClient, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
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

    # ---------- public API ----------

    def solve(self, problem: str, metadata: Dict) -> Dict:
        idx = metadata.get("idx", 0)

        # Phase 1: generate multiple candidate solutions
        candidates, trace = self._generate_candidates(problem, idx)

        # Phase 2: extract answers from each candidate
        extracted_answers = [self._extract_answer(c) for c in candidates]
        normalized_answers = [self._normalize_answer(a) for a in extracted_answers]

        # Phase 3: count answer occurrences for consistency bonus
        answer_counts: Dict[str, int] = {}
        for ans in normalized_answers:
            answer_counts[ans] = answer_counts.get(ans, 0) + 1

        # Phase 4: verify each candidate and compute final scores
        scored_candidates = []
        for candidate_id, candidate in enumerate(candidates):
            verifier_score, verify_trace = self._verify_candidate(
                problem, candidate, idx, candidate_id,
            )
            consistency = answer_counts[normalized_answers[candidate_id]] - 1
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

        # Phase 5: select best candidate
        best = max(scored_candidates, key=lambda item: item["total_score"])
        score_lines = " | ".join([
            f"#{s['id']} ans={s['extracted_answer'][:40]} score={s['total_score']:.3f}(v={s['verifier_score']:.3f}+c={s['consistency_bonus']:.3f})"
            for s in scored_candidates
        ])
        trace.append({
            "step": "score_summary",
            "content": f"{score_lines} | selected=#{best['id']}",
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
                "selected": best["id"],
            },
        })
        return {
            "final_response": best["extracted_answer"],
            "trace": trace,
        }

    # ---------- generation ----------

    def _generate_candidates(self, problem: str, idx: int) -> Tuple[List[str], List[Dict]]:
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
            is_correct, confidence = self._parse_verdict(verdict_text)
            scores.append(confidence if is_correct else 0.0)
            trace.append({
                "step": f"verifier_call_{candidate_id}_{vote_id}",
                "content": {
                    "candidate_id": candidate_id,
                    "verdict": verdict_text,
                    "parsed_correct": is_correct,
                    "parsed_confidence": confidence,
                },
            })

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, trace

    # ---------- answer extraction ----------

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract final answer from model output. Searches from end backward."""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

        # 1. 【最终答案】 marker (last match)
        matches = re.findall(r"【最终答案】\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()

        # 2. \boxed{...} (last match)
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].strip()

        # 3. 最终答案： or 答案： (last match)
        matches = re.findall(r"(?:最终答案|答案)[：:]\s*(.+?)(?:\n|$)", text)
        if matches:
            return matches[-1].strip()

        # 4. Line ending with "= <value>" (e.g. "... = 72" or "... = -1")
        for line in reversed(lines):
            match = re.search(r"=\s*(.+?)\s*[。.]?\s*$", line)
            if match:
                candidate = match.group(1).strip().strip("$")
                if re.search(r"\d", candidate):
                    return candidate

        # 5. Last line starting with a number but NOT a step label like "10. text"
        for line in reversed(lines):
            stripped = line.strip().lstrip("$").rstrip("$").strip()
            if re.match(r"^-?\d", stripped) and not re.match(r"^\d+[.)、]\s", stripped) and len(stripped) < 80:
                return stripped

        # 6. Fallback
        return lines[-1] if lines else text.strip()

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

    # ---------- verdict parsing ----------

    @staticmethod
    def _parse_verdict(verdict_text: str) -> Tuple[bool, float]:
        """Parse verifier output. Returns (is_correct: bool, confidence: 0.0~1.0)."""
        # Search for last occurrence to skip thinking text
        all_verdicts = re.findall(
            r"VERDICT\s*[:：]\s*([AB])", verdict_text, re.IGNORECASE
        )
        is_correct = bool(all_verdicts and all_verdicts[-1].upper() == "A")

        all_confs = re.findall(
            r"CONFIDENCE\s*[:：]\s*(\d+)", verdict_text, re.IGNORECASE
        )
        if all_confs:
            confidence = min(int(all_confs[-1]), 10) / 10.0
        else:
            confidence = 1.0 if is_correct else 0.0

        return is_correct, confidence


# ===================== PARTICIPANT DESIGN AREA END =====================
