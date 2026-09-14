"""
Runs the pipeline over every query in test_queries.json and saves the
results to outputs/test_results.json - the deliverable required by the
assignment ("test queries and corresponding system outputs").

Usage:
    python tests/run_tests.py
"""

import json
import sys
from pathlib import Path

# Allow running this script directly (adds project root to sys.path)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.orchestrator import run_pipeline

QUERIES_PATH = Path(__file__).resolve().parent / "test_queries.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "test_results.json"


def main():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []
    for i, question in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {question}")
        state = run_pipeline(question)
        results.append({
            "question": question,
            "in_scope": state.get("is_in_scope"),
            "answer": state.get("answer"),
            "sources": state.get("sources", []),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
