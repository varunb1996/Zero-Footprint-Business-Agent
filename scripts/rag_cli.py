"""Local smoke test for the RAG layer, no WhatsApp needed.

Usage:
    python scripts/rag_cli.py <business_id> "customer question"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.whatsapp.rag import answer_question


def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python scripts/rag_cli.py <business_id> "question"')
        sys.exit(1)

    business_id = sys.argv[1]
    question = " ".join(sys.argv[2:])
    print(answer_question(business_id, question))


if __name__ == "__main__":
    main()
