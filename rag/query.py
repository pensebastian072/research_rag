"""Query the research RAG index.

Search only:
    .venv\\Scripts\\python.exe rag\\query.py "currency momentum lookback"

Search + synthesized answer (qwen2.5:7b via Ollama, with citations):
    .venv\\Scripts\\python.exe rag\\query.py --ask "what do the papers say about overfit backtests?"

Options:
    -k N            top-N chunks (default 6)
    --kind papers|data   restrict corpus
    --json          machine-readable output (for HQ advisory integration)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chat, connect, embed_texts

SYSTEM = (
    "You are a quantitative finance research assistant. Answer ONLY from the provided "
    "excerpts of the user's research library. Cite sources as [n] matching the excerpt "
    "numbers. If the excerpts do not contain the answer, say so plainly. "
    "This is research context only, never a trade instruction."
)


def search(query: str, k: int, kind: str | None) -> list[dict]:
    con = connect()
    where = "WHERE d.kind = ?" if kind else ""
    params = (kind,) if kind else ()
    rows = con.execute(
        f"SELECT c.chunk_id, d.path, d.kind, c.page, c.text, c.embedding "
        f"FROM chunks c JOIN docs d ON d.doc_id = c.doc_id {where}", params).fetchall()
    con.close()
    if not rows:
        return []

    mat = np.frombuffer(b"".join(r[5] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    qvec = embed_texts([query], is_query=True)[0]
    scores = mat @ qvec
    order = np.argsort(scores)[::-1][:k]
    return [
        {
            "score": float(scores[i]),
            "source": Path(rows[i][1]).name,
            "kind": rows[i][2],
            "page": rows[i][3],
            "text": rows[i][4],
        }
        for i in order
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--kind", choices=["paper", "data"])
    ap.add_argument("--ask", action="store_true", help="synthesize answer with local LLM")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    hits = search(args.query, args.k, args.kind)
    if not hits:
        print("Index empty or no matches. Run rag\\ingest.py first.")
        return

    if args.ask:
        excerpts = "\n\n".join(
            f"[{i + 1}] ({h['source']}" + (f", p.{h['page']}" if h["page"] else "") + f")\n{h['text']}"
            for i, h in enumerate(hits)
        )
        answer = chat(f"Excerpts:\n\n{excerpts}\n\nQuestion: {args.query}", system=SYSTEM)
        if args.json:
            print(json.dumps({"answer": answer, "sources": hits}, indent=2))
        else:
            print(answer)
            print("\n--- sources ---")
            for i, h in enumerate(hits):
                loc = f" p.{h['page']}" if h["page"] else ""
                print(f"[{i + 1}] {h['source']}{loc} (score {h['score']:.3f})")
        return

    if args.json:
        print(json.dumps(hits, indent=2))
        return
    for i, h in enumerate(hits):
        loc = f" p.{h['page']}" if h["page"] else ""
        print(f"\n[{i + 1}] {h['source']}{loc}  score={h['score']:.3f}")
        print(h["text"][:500])


if __name__ == "__main__":
    main()
