"""
Script: loads raw documents, chunks them, embeds them,
and persists the result to a local Chroma vector store.

Run with:
    python -m src.ingestion.build_vectorstore
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR, DATA_DIR
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents


def build_vectorstore():
    print(f"Loading documents from {DATA_DIR} ...")
    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} document(s).")

    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    print(f"Vector store built and persisted at {VECTORSTORE_DIR}")
    print(f"Total chunks indexed: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
