# research_rag

A fully local retrieval-augmented generation system over a personal research
library — PDFs, spreadsheets, CSVs, notes — with no cloud calls and no API keys.

Embeddings, storage, and answer synthesis all run on the machine. Ollama serves
`nomic-embed-text` for embeddings and a local chat model for synthesis; the index
is plain SQLite holding chunks plus float32 vectors. Nothing leaves the box.

This repository contains **the machinery only**. The corpus it was built against
is a private library and is not included — point it at your own.

## Why it's built this way

Most RAG examples assume a vector database service and a hosted embedding API.
For a personal research library that is the wrong shape: the corpus is small
enough that SQLite plus brute-force cosine similarity is genuinely fast, and the
documents are exactly the kind of thing you don't want to ship to a third party.
So there is no vector DB, no cloud embedding call, and no API key anywhere in the
pipeline.

A few consequences that turned out to matter more than expected:

- **Ingest is incremental and sha1-keyed**, and it *prunes deleted files*. A
  re-run after dropping ten PDFs in and removing two leaves the index correct.
  Most naive ingest scripts only ever add, and the index silently rots.
- **Answers carry citations back to the source chunk.** An uncited answer from a
  7B model over your own documents is not useful — you cannot tell synthesis from
  invention without the pointer.
- **Retrieval and synthesis are separate commands.** `query.py` will do a pure
  search with no model involved. When an answer looks wrong, the first question
  is always "did retrieval find the right chunks", and you want to be able to ask
  that directly.
- **Mixed formats in one index.** PDFs, xlsx, csv, txt, md and Pine scripts are
  chunked by the same path, so a question can pull a passage from a paper and a
  row from a spreadsheet in the same answer.

## Layout

```
rag/common.py      config, Ollama client, SQLite schema, vector math
rag/ingest.py      incremental sha1 ingest, chunking, prunes deleted files
rag/query.py       one-shot search, optional synthesized answer with citations
rag/chat.py        interactive session over the index
rag/report.py      tabular summary over structured items in the corpus
desktop/           launcher
library/           your corpus goes here (gitignored)
index/             generated SQLite index (gitignored)
```

## Requirements

Python 3.11+ and a local [Ollama](https://ollama.com) with an embedding model
(`nomic-embed-text`) and a chat model (e.g. `qwen2.5:7b`).

```bash
uv sync
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# put documents in library/, then
python -m rag.ingest
python -m rag.query --ask "what does the literature say about backtest overfitting?"
```

## Scope

Research and reference only. It answers questions about documents. It emits no
signals and is not wired into anything that acts.

MIT licensed.
