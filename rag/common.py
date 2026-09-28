"""Shared helpers: Ollama client, SQLite store, vector math.

Research/advisory tool only. Never wire into trade execution.
"""
from __future__ import annotations

import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "rag.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,          -- 'paper' | 'data'
    sha1 TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    page INTEGER,                -- 1-based page, NULL for data cards
    text TEXT NOT NULL,
    embedding BLOB NOT NULL      -- float32 little-endian
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def _post(endpoint: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp_ctx = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), ConnectionRefusedError) or "refused" in str(e).lower():
            raise SystemExit(
                f"\nCan't reach Ollama at {OLLAMA_URL}. Start it, then try again:\n"
                f"  ollama serve\n  ollama pull {CHAT_MODEL}\n  ollama pull {EMBED_MODEL}\n"
            ) from None
        raise
    with resp_ctx as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed with the task prefix nomic-embed-text expects."""
    prefix = "search_query: " if is_query else "search_document: "
    out = _post("/api/embed", {"model": EMBED_MODEL, "input": [prefix + t for t in texts]})
    vecs = np.asarray(out["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def chat(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    out = _post("/api/chat", {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        # default num_ctx silently truncates the front of long RAG prompts
        "options": {"num_ctx": 8192},
    })
    return out["message"]["content"]
