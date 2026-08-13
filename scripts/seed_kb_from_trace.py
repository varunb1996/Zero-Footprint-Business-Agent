"""Seed the KB store from an already-generated eval trace, for demoing the
RAG layer against real extracted data without re-running the intake agent.

Usage:
    python scripts/seed_kb_from_trace.py <trace_file> <adaptive|baseline> <business_id>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.kb.loader import save_dialogue_state


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python scripts/seed_kb_from_trace.py <trace_file> <adaptive|baseline> <business_id>")
        sys.exit(1)

    trace_path, agent_type, business_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    fields = data[agent_type]["fields"]

    save_dialogue_state(business_id, agent_type, fields)
    print(f"Seeded '{business_id}' from {trace_path.name} ({agent_type}) -- {len(fields)} fields.")


if __name__ == "__main__":
    main()
