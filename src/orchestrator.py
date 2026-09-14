"""
Orchestrator - runs the 3 agents in sequence, passing a shared state dict.

This is intentionally plain Python rather than a graph framework: with only
3 agents and a linear flow (no branching loops), an explicit sequence is
easier to read, debug, and explain than a graph abstraction would be.
"""

from src.agents.guard_agent import run_guard_agent
from src.agents.retriever_agent import run_retriever_agent
from src.agents.generator_agent import run_generator_agent

PIPELINE = (
    run_guard_agent,
    run_retriever_agent,
    run_generator_agent,
)


def run_pipeline(question: str) -> dict:
    """Runs the full 3-agent pipeline for a single question.

    Returns the final state dict, containing at least:
        question, is_in_scope, chunks, answer, sources
    """
    state = {"question": question}

    for agent in PIPELINE:
        state = agent(state)

    return state
