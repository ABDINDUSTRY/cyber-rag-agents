"""
Agent 2 - Retriever Agent

Responsibility: given an in-scope question, search the Chroma vector store
for the most relevant chunks. Does NOT touch the LLM : pure retrieval.

Input (reads from state):  state["question"], state["is_in_scope"]
Output (writes to state):  state["chunks"]  (list[Document])
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR, TOP_K

# Loaded once per process
_embeddings = None
_vectorstore = None


def _get_vectorstore():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=_embeddings,
        )
    return _vectorstore


def run_retriever_agent(state: dict) -> dict:
    if not state.get("is_in_scope"):
        state["chunks"] = []
        return state

    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(state["question"], k=TOP_K)
    state["chunks"] = results

    return state
