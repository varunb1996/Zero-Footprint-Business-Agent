"""Interactive terminal smoke test for the Groq-backed intake agent.

Run:
    python scripts/chat_cli.py

Each turn, the agent (via Groq) asks a question; type the owner's reply and
press enter. Ends when all fields are confirmed/inferred/uncertain or the
turn budget is hit. Prints the final structured result.

This exercises the exact same graph as Step 1 (tests/test_graph_adaptive.py)
— only the extractor/clarifier are swapped from scripted fakes to real Groq
calls, per the injected Extractor/Clarifier interfaces in src/agent/state.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles/pipes often default to cp1252, which chokes on the
# Unicode punctuation (curly quotes, non-breaking hyphens) LLMs commonly emit.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8")

from src.agent.graph_adaptive import build_adaptive_graph
from src.agent.llm_policy import make_groq_clarifier, make_groq_extractor
from src.agent.state import new_dialogue_state


def cli_owner_responder(field: str, attempts: int, question: str) -> str:
    print(f"\n[agent] {question}")
    try:
        return input("[you]   ")
    except EOFError:
        print("\n(no more input — ending interview early)")
        raise SystemExit(0)


def main() -> None:
    graph = build_adaptive_graph(
        extractor=make_groq_extractor(),
        clarifier=make_groq_clarifier(),
        owner_responder=cli_owner_responder,
    )
    initial = new_dialogue_state(max_clarify_attempts=2, max_turn_budget=30)
    final = graph.invoke(initial, config={"recursion_limit": 200})

    print("\n" + "=" * 60)
    print("FINAL BUSINESS PROFILE")
    print("=" * 60)
    for field, rec in final["fields"].items():
        print(f"\n{field} [{rec['status']}]")
        print(f"  value: {json.dumps(rec['value'], ensure_ascii=False)}")
        if rec["note"]:
            print(f"  note: {rec['note']}")


if __name__ == "__main__":
    main()
