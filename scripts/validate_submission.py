from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
from pathlib import Path


ROOT = Path(".")
SENSITIVE = re.compile(r"(?:api[_-]?key|x-access-token|ghp_[A-Za-z0-9_]+)", re.IGNORECASE)


class ValidationClient:
    def chat(self, messages, temperature=0.2, max_tokens=4096):
        if "CHOICE:" in messages[0]["content"]:
            return "CHOICE: 0\nREASON: 验证通过"
        return "【最终答案】验证答案"


def main() -> int:
    entry = ROOT / "user_agent.py"
    if not entry.is_file():
        print("missing user_agent.py", file=sys.stderr)
        return 1
    source = entry.read_text(encoding="utf-8")
    if SENSITIVE.search(source):
        print("sensitive token-like text found", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module("user_agent")
    agent_class = module.ReasoningAgent
    init_parameters = list(inspect.signature(agent_class.__init__).parameters.values())
    if [parameter.name for parameter in init_parameters[:2]] != ["self", "client"]:
        print("ReasoningAgent.__init__ must accept client as its first argument", file=sys.stderr)
        return 1
    solve_parameters = list(inspect.signature(agent_class.solve).parameters.values())
    if [parameter.name for parameter in solve_parameters[:3]] != ["self", "problem", "metadata"]:
        print("ReasoningAgent.solve must accept problem and metadata", file=sys.stderr)
        return 1
    result = agent_class(client=ValidationClient()).solve("计算 1+1。", {"idx": 0})
    if not isinstance(result, dict) or not isinstance(result.get("final_response"), str) or not result["final_response"].strip():
        print("invalid response", file=sys.stderr)
        return 1
    json.dumps(result, ensure_ascii=False)
    print("submission validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
