"""
Agent 3 - Generator Agent

Responsibility: produce the final answer using ONLY the retrieved chunks.
Must refuse to answer when context is missing or
insufficient, and must list the sources it relied on.

This agent runs a two-step generate-then-verify process internally:
1. Draft the answer from the retrieved context.
2. Re-check that draft against the SAME context and strip/rephrase any
   claim not actually traceable to it.

This matters because a single generation pass can still let pretrained
knowledge leak in, an LLM asked to "answer using only this context" will
often fill an obvious-looking gap (e.g. a partially described list) with
facts it already knows, even when explicitly told not to. The verification
pass catches this by treating grounding as a separate check rather than
trusting the first pass to have followed the instruction perfectly. This
was added after observing exactly this failure mode during testing (see
project report, Failure Cases). It does not guarantee zero hallucination,
but it is a meaningful mitigation, at the cost of one extra LLM call per
query. The pipeline stays at 3 agents, this is two internal calls inside
the Generator's own responsibility, not a separate agent.

Input (reads from state):  state["question"], state["is_in_scope"], state["chunks"]
Output (writes to state):  state["draft_answer"] (str, kept for audit),
                            state["answer"] (str, final verified answer),
                            state["sources"] (list[str])
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import LLM_MODEL

GENERATOR_PROMPT = ChatPromptTemplate.from_template(
    """You are a cybersecurity assistant. The input below is either a direct
question or a short incident description. Respond using ONLY the context
provided. Do not use any outside knowledge, even if it is well known or
commonly true.

- If it is a question, answer it directly.
- If it is an incident description, identify what is relevant in the
  context (e.g. matching attack pattern, recommended response or
  mitigation steps) and explain how it applies to the described incident.

Critical rule: if a specific fact, number, category, list, or example is
not explicitly present in the context, do NOT state it. It is better to
leave a detail out, or say the context does not specify it, than to add
anything not present in the context below, even if you are confident it
is true and commonly known.

If the context does not contain enough information to respond confidently,
say so explicitly instead of guessing or filling gaps with assumptions.

Context:
{context}

Input: {question}

Write a clear, structured response grounded strictly in the context above."""
)

VERIFY_PROMPT = ChatPromptTemplate.from_template(
    """You are a grounding checker. Compare the draft answer below against
the context it was supposed to be based on, and rewrite it so that every
factual statement is verifiably supported by that context.

Context:
{context}

Draft answer:
{draft_answer}

Rewrite the answer, applying these rules strictly:
- Remove or rephrase any statement that introduces facts, numbers,
  categories, or examples that are NOT explicitly present in the context,
  even if they are commonly true or well known in the field.
- Do not fill a removed gap with your own knowledge - if something is
  missing, state plainly that the context does not specify it.
- Keep everything that IS genuinely supported by the context, including
  its structure, level of detail, and citations.

Return ONLY the corrected answer text, with no explanation of what you
changed and no preamble."""
)


def run_generator_agent(state: dict) -> dict:
    if not state.get("is_in_scope"):
        state["answer"] = (
            "This question is outside the scope of this cybersecurity "
            "knowledge base and cannot be answered by this system."
        )
        state["sources"] = []
        return state

    chunks = state.get("chunks", [])
    if not chunks:
        state["answer"] = (
            "No relevant information was found in the knowledge base "
            "for this question."
        )
        state["sources"] = []
        return state

    context = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    sources = sorted({chunk.metadata.get("source", "unknown") for chunk in chunks})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    # Step 1: draft answer from context
    draft_chain = GENERATOR_PROMPT | llm | StrOutputParser()
    draft_answer = draft_chain.invoke({"context": context, "question": state["question"]})

    # Step 2: verify/strip claims not actually supported by that context
    verify_chain = VERIFY_PROMPT | llm | StrOutputParser()
    verified_answer = verify_chain.invoke({"context": context, "draft_answer": draft_answer})

    state["draft_answer"] = draft_answer  # kept for debugging / report evidence
    state["answer"] = verified_answer
    state["sources"] = sources

    return state
