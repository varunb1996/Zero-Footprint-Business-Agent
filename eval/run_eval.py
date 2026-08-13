"""Evaluation harness (spec Section 6): runs both agents against the eval
set through the real Groq API, logs full per-case decision traces, and
computes Metrics 1-3. Metric 4 (downstream usability) isn't wired into this
harness as a batch-graded run -- the RAG layer it depends on (src/whatsapp/rag.py)
is built and has been verified manually via scripts/rag_cli.py, but the
spec's ~10-15 question correct/incorrect/unanswerable grading pass hasn't
been assembled into an automated suite here.

Usage:
    python eval/run_eval.py                         # full eval set
    python eval/run_eval.py --limit 3                # first 3 cases (smoke test)
    python eval/run_eval.py --cases tailor_01,pharmacy_02
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.case_schema import EvalCase, load_cases
from eval.metrics import classify_field, summarize
from src.agent.graph_adaptive import build_adaptive_graph
from src.agent.graph_baseline import build_baseline_graph
from src.agent.llm_policy import make_groq_clarifier, make_groq_extractor
from src.agent.state import OwnerResponder, new_dialogue_state
from src.schema import FIELD_PRIORITY

RESULTS_DIR = Path(__file__).parent / "results"


def make_case_owner_responder(case: EvalCase) -> OwnerResponder:
    def responder(field: str, attempts: int, question: str) -> str:
        script = case["owner_script"][field]
        return script[min(attempts, len(script) - 1)]

    return responder


def load_trace(run_id: str, case_id: str) -> Optional[dict]:
    trace_path = RESULTS_DIR / run_id / f"{case_id}.json"
    if not trace_path.exists():
        return None
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    return {"adaptive": data["adaptive"], "baseline": data["baseline"]}


def run_case(case: EvalCase, run_id: str) -> dict:
    extractor = make_groq_extractor()
    clarifier = make_groq_clarifier()
    owner_responder = make_case_owner_responder(case)

    adaptive_graph = build_adaptive_graph(extractor, clarifier, owner_responder)
    adaptive_final = adaptive_graph.invoke(
        new_dialogue_state(max_clarify_attempts=2, max_turn_budget=30),
        config={"recursion_limit": 200},
    )

    baseline_graph = build_baseline_graph(extractor, owner_responder)
    baseline_final = baseline_graph.invoke(
        new_dialogue_state(max_clarify_attempts=2, max_turn_budget=30),
        config={"recursion_limit": 200},
    )

    trace_path = RESULTS_DIR / run_id / f"{case['id']}.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "case_id": case["id"],
                "adaptive": {
                    "turn": adaptive_final["turn"],
                    "fields": adaptive_final["fields"],
                    "conversation": adaptive_final["conversation"],
                },
                "baseline": {
                    "turn": baseline_final["turn"],
                    "fields": baseline_final["fields"],
                    "conversation": baseline_final["conversation"],
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return {"adaptive": adaptive_final, "baseline": baseline_final}


def print_metric_1(adaptive_outcomes: list, baseline_outcomes: list) -> None:
    print("\n" + "=" * 70)
    print("METRIC 1 -- FIELD-LEVEL EXTRACTION ACCURACY")
    print("=" * 70)
    for label, outcomes in [("Adaptive", adaptive_outcomes), ("Baseline", baseline_outcomes)]:
        s = summarize(outcomes)
        p = f"{s['precision']:.2f}" if s["precision"] is not None else "n/a"
        r = f"{s['recall']:.2f}" if s["recall"] is not None else "n/a"
        print(
            f"\n{label}: precision={p} recall={r} "
            f"(tp={s['tp']} fp_wrong_value={s['fp_wrong_value']} "
            f"fp_hallucination={s['fp_hallucination']} fn={s['fn']} tn={s['tn']})"
        )


def print_metric_2(adaptive_turns: list[int], baseline_turns: list[int]) -> None:
    print("\n" + "=" * 70)
    print("METRIC 2 -- CLARIFICATION EFFICIENCY (turns to complete)")
    print("=" * 70)
    mean_adaptive = sum(adaptive_turns) / len(adaptive_turns)
    mean_baseline = sum(baseline_turns) / len(baseline_turns)
    print(
        f"\nAdaptive: mean={mean_adaptive:.1f} turns "
        f"(min={min(adaptive_turns)}, max={max(adaptive_turns)})"
    )
    print(
        f"Baseline: mean={mean_baseline:.1f} turns "
        f"(fixed at {len(FIELD_PRIORITY)} by design)"
    )


def print_metric_3(adaptive_outcomes: list, baseline_outcomes: list) -> None:
    print("\n" + "=" * 70)
    print("METRIC 3 -- HALLUCINATION RATE UNDER AMBIGUITY")
    print("=" * 70)
    for label, outcomes in [("Adaptive", adaptive_outcomes), ("Baseline", baseline_outcomes)]:
        rate = summarize(outcomes)["hallucination_rate"]
        text = f"{rate:.1%}" if rate is not None else "n/a (no uncertain ground-truth fields in this run)"
        print(f"\n{label}: {text}")


def print_metric_4() -> None:
    print("\n" + "=" * 70)
    print("METRIC 4 -- DOWNSTREAM USABILITY")
    print("=" * 70)
    print(
        "\nNot batch-computed here -- the RAG layer (src/whatsapp/rag.py) is built "
        "and manually spot-checked via scripts/rag_cli.py. A formal ~10-15 "
        "question correct/incorrect/unanswerable pass isn't wired into this harness."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cases", type=str, default=None, help="comma-separated case IDs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="run_id to resume; cases with an existing trace file are loaded from disk instead of re-calling the API",
    )
    args = parser.parse_args()

    cases = load_cases()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    run_id = args.resume or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.resume:
        print(f"Resuming run_id={run_id} on {len(cases)} case(s)\n")
    else:
        print(f"Running eval on {len(cases)} case(s), run_id={run_id}\n")

    adaptive_outcomes = []
    baseline_outcomes = []
    adaptive_turns = []
    baseline_turns = []

    for i, case in enumerate(cases, 1):
        cached = load_trace(run_id, case["id"]) if args.resume else None
        if cached:
            print(f"[{i}/{len(cases)}] {case['id']} (cached, skipping API calls)", flush=True)
            result = cached
        else:
            print(f"[{i}/{len(cases)}] {case['id']} ...", flush=True)
            result = run_case(case, run_id)
        adaptive_turns.append(result["adaptive"]["turn"])
        baseline_turns.append(result["baseline"]["turn"])

        for field in FIELD_PRIORITY:
            gt = case["ground_truth"][field]
            adaptive_outcomes.append(classify_field(case["id"], field, gt, result["adaptive"]["fields"]))
            baseline_outcomes.append(classify_field(case["id"], field, gt, result["baseline"]["fields"]))

    print_metric_1(adaptive_outcomes, baseline_outcomes)
    print_metric_2(adaptive_turns, baseline_turns)
    print_metric_3(adaptive_outcomes, baseline_outcomes)
    print_metric_4()

    print(f"\nFull per-case decision traces written to eval/results/{run_id}/")


if __name__ == "__main__":
    main()
