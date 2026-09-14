"""
Agent 1 - Guard Agent

Responsibility: decide whether the incoming input, a question OR a short
incident description is within the scope of the cybersecurity knowledge base.
This runs BEFORE retrieval so that clearly out-of-scope input never reaches the LLM with pretrained-knowledge
temptation later in the pipeline.

Input (reads from state):  state["question"]
Output (writes to state):  state["is_in_scope"]  (bool)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL

GUARD_PROMPT = ChatPromptTemplate.from_template(
    """You are a scope-checking agent for a cybersecurity assistant.
The input can be EITHER a direct question OR a short incident description
(e.g. a user reporting suspicious activity, a log observation, a phishing
email they received). Your only job is to decide whether the input relates
to cybersecurity (threats, vulnerabilities, malware, incident response,
security standards/frameworks, network/cloud security, best practices)
regardless of whether it is phrased as a question or as a description of
events.

Input: {question}

Respond with EXACTLY one word, nothing else: IN_SCOPE or OUT_OF_SCOPE."""
)


def run_guard_agent(state: dict) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = GUARD_PROMPT | llm | StrOutputParser()

    verdict = chain.invoke({"question": state["question"]}).strip().upper()
    state["is_in_scope"] = verdict.startswith("IN_SCOPE")

    return state
