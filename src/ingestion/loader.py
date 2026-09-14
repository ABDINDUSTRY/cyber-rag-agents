"""
Loads raw documents (PDF, TXT, MD, CSV, JSON) from a directory.
Each returned Document carries a `metadata["source"]` field used later
for citation in the Generator Agent's answers.
"""

import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader


def load_documents(data_dir: Path) -> list:
    """Load every supported file found under data_dir (recursively)."""
    documents = []

    for file_path in sorted(data_dir.glob("**/*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in (".pdf", ".txt", ".md", ".csv", ".json"):
            continue

        print(f"  loading {file_path.name} ...", flush=True)

        try:
            if file_path.suffix.lower() == ".pdf":
                loaded = PyMuPDFLoader(str(file_path)).load()
            elif file_path.suffix.lower() == ".csv":
                loaded = CSVLoader(str(file_path), encoding="utf-8").load()
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                loaded = []
                if isinstance(data, list):
                    for item in data:
                        loaded.append(Document(page_content=json.dumps(item, ensure_ascii=False)))
                else:
                    loaded.append(Document(page_content=json.dumps(data, ensure_ascii=False)))
            else:
                loaded = TextLoader(str(file_path), encoding="utf-8").load()
        except Exception as e:
            print(f"  WARNING: failed to load {file_path.name} ({e}) - skipping")
            continue

        for doc in loaded:
            doc.metadata["source"] = file_path.name
        documents.extend(loaded)

        print(f"  -> {len(loaded)} page(s)/section(s)/row(s) loaded", flush=True)

    if not documents:
        raise FileNotFoundError(
            f"No supported documents (.pdf, .txt, .md, .csv, .json) found in {data_dir}"
        )

    return documents