# Cyber-RAG-Agents

A 3-agent Retrieval-Augmented Generation system for cybersecurity Q&A.
Answers are grounded strictly in a local document knowledge base.

## Architecture

Three agents run sequentially, passing a shared `state` dict between them:

```
question --> [Guard Agent] --> [Retriever Agent] --> [Generator Agent] --> answer + sources
```

- **Guard Agent** (`src/agents/guard_agent.py`) - classifies whether the
  question is in-scope for cybersecurity. Out-of-scope questions are short-
  circuited before any retrieval happens.
- **Retriever Agent** (`src/agents/retriever_agent.py`) - embeds the question
  and performs a similarity search against the Chroma vector store, returning
  the top-k chunks.
- **Generator Agent** (`src/agents/generator_agent.py`) - produces the final
  answer using ONLY the retrieved chunks, and lists their sources. If no
  chunks were retrieved, it explicitly says the information is not available.

Agents communicate through a plain Python dict (`state`).

## Requirements

- Python 3.10+
- An OpenAI API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your API key:

1. Create a new file named `.env` in the root folder of the project.
2. Open the `.env` file in your text editor and add your API key like this:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## Usage

1. Put your source documents (PDF, TXT, or MD) in `data/raw_docs/`.

2. Build the vector store (run once, or whenever documents change):

```bash
python -m src.ingestion.build_vectorstore
```

3. Ask a question:

```bash
python main.py --query "What are the main steps of an incident response plan?"
```

4. Run the full test suite (10+ queries) and dump results:

```bash
python tests/run_tests.py
```

Results are written to `outputs/test_results.json`.

## Project structure

```
cyber-rag-agents/
├── README.md
├── requirements.txt
├── main.py                          # CLI entrypoint
├── data/
│   └── raw_docs/                    # cybersecurity docs
├── src/
│   ├── config.py                    # paths, model names, chunking params
│   ├── ingestion/
│   │   ├── loader.py                # load documents
│   │   ├── chunker.py               # split documents into chunks
│   │   └── build_vectorstore.py     # embed chunks + persist to Chroma
│   ├── agents/
│   │   ├── guard_agent.py           # Agent 1: scope check
│   │   ├── retriever_agent.py       # Agent 2: vector search
│   │   └── generator_agent.py       # Agent 3: grounded answer generation
│   ├── orchestrator.py              # runs the 3 agents in sequence
│   └── vectorstore/                 # persisted Chroma DB (auto-generated)
├── tests/
│   ├── test_queries.json            # test queries
│   └── run_tests.py                 # batch-runs the pipeline, saves outputs
├── outputs/
│   └── test_results.json            # generated results (auto-generated)
└── notebooks/                       # exploration notebooks
    └── run_and_verify.ipynb         # runs + displays all test query results inline
```

## External tools / libraries used

- [LangChain](https://python.langchain.com/) - document loading, chunking,
  prompt templates, LCEL chains
- [ChromaDB](https://www.trychroma.com/) - local vector store
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) - local,
  free embedding model
- LLM API (OpenAI ) for the Guard and
  Generator agents
- Claude
