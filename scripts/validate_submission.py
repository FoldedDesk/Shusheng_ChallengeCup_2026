from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    result = module.ReasoningAgent(ValidationClient()).solve("计算 1+1。", {})
    if not isinstance(result, dict) or not isinstance(result.get("final_response"), str):
        print("invalid response", file=sys.stderr)
        return 1
    json.dumps(result, ensure_ascii=False)
    print("submission validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
