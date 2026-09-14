"""
CLI entrypoint.

Usage:
    python main.py --query "What are the main steps of an incident response plan?"
"""

import argparse
import json

from src.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Cyber-RAG-Agents CLI")
    parser.add_argument("--query", required=True, help="Cybersecurity question to ask")
    args = parser.parse_args()

    result = run_pipeline(args.query)

    output = {
        "question": result["question"],
        "in_scope": result.get("is_in_scope"),
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
