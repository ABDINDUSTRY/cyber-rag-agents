"""
Splits loaded documents into chunks suitable for embedding.

For Markdown sources, splitting happens in two passes:
1. Split by Markdown headers first (MarkdownHeaderTextSplitter), so a
   section (e.g. "## Introduction" together with the intro sentence AND
   the list it introduces) stays together as a single chunk whenever
   possible.
2. Only sections still too large after step 1 are further split by
   RecursiveCharacterTextSplitter, with overlap applied only within that
   oversized section.

This avoids the failure mode where a plain character-count splitter cuts
a sentence like "The five types are as follows:" into a different chunk
than the list it introduces - leaving neither chunk useful for that fact,
since one has the promise and the other has the content, never both.

Non-Markdown sources (PDF, TXT, CSV, JSON) have no header syntax to exploit,
so they fall back to a plain RecursiveCharacterTextSplitter pass. For CSV 
and JSON, since the loaders already split them logically by row/object, 
this pass usually leaves them intact unless they exceed the chunk size.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _chunk_markdown_document(doc) -> list:
    """Header-aware chunking for a single Markdown Document."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep headers in the text, useful context for embedding
    )
    header_sections = md_splitter.split_text(doc.page_content)

    # MarkdownHeaderTextSplitter drops the original document metadata
    for section in header_sections:
        section.metadata.update(doc.metadata)

    # Sections still too large get further split by character count
    return _char_splitter.split_documents(header_sections)


def chunk_documents(documents: list) -> list:
    all_chunks = []

    for doc in documents:
        source = doc.metadata.get("source", "")
        if source.lower().endswith(".md"):
            all_chunks.extend(_chunk_markdown_document(doc))
        else:
            all_chunks.extend(_char_splitter.split_documents([doc]))

    return all_chunks
