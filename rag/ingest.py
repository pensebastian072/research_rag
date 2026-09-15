"""Ingest library/ into the RAG index.

- library/papers/*.pdf  -> page-aware text chunks
- library/data/*.xlsx   -> one stat card per sheet (TradingView strategy exports)
- library/data/*.csv    -> header + shape summary card
- library/data/*.txt    -> chunked full text

Incremental: files whose sha1 is unchanged are skipped. Re-run any time.

Usage: .venv\\Scripts\\python.exe rag\\ingest.py [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf
import openpyxl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, connect, embed_texts

PAPERS = ROOT / "library" / "papers"
DATA = ROOT / "library" / "data"

CHUNK_CHARS = 1800
OVERLAP_CHARS = 250
EMBED_BATCH = 16
MAX_SHEET_ROWS = 120  # TradingView perf sheets are small; caps runaway trade lists


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, page: int | None) -> list[tuple[int | None, str]]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):
            # break at last sentence/space boundary inside the window
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind(" "))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        chunks.append((page, text[start:end].strip()))
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return chunks


def extract_pdf(path: Path) -> list[tuple[int | None, str]]:
    chunks = []
    with fitz.open(path) as doc:
        for i, pg in enumerate(doc, start=1):
            chunks.extend(chunk_text(pg.get_text(), page=i))
    return chunks


def extract_xlsx(path: Path) -> list[tuple[int | None, str]]:
    """One card per sheet: strategy name from filename + rows as 'key: value' lines."""
    cards = []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            lines = [f"Backtest export: {path.stem}", f"Sheet: {ws.title}"]
            for row in ws.iter_rows(max_row=MAX_SHEET_ROWS, values_only=True):
                cells = [str(c) for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))
            if len(lines) > 2:
                text = "\n".join(lines)
                # sheet cards can exceed one chunk; split but keep header on each
                for _, piece in chunk_text(text, page=None):
                    cards.append((None, piece))
    finally:
        wb.close()
    return cards


def extract_csv(path: Path) -> list[tuple[int | None, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    n = len(rows) - 1
    lines = [f"Data file: {path.name}", f"Columns: {', '.join(rows[0])}", f"Rows: {n}"]
    for r in rows[1: min(len(rows), 8)]:
        lines.append(" | ".join(r))
    if n > 7:
        lines.append("...")
        lines.append(" | ".join(rows[-1]))
    return [(None, "\n".join(lines))]


def extract_txt(path: Path) -> list[tuple[int | None, str]]:
    return chunk_text(path.read_text(encoding="utf-8", errors="replace"), page=None)


EXTRACTORS = {
    ".pdf": extract_pdf, ".xlsx": extract_xlsx, ".csv": extract_csv,
    ".txt": extract_txt, ".md": extract_txt, ".pine": extract_txt,
}


def ingest_file(con, path: Path, kind: str, force: bool) -> str:
    digest = sha1_of(path)
    row = con.execute("SELECT doc_id, sha1 FROM docs WHERE path = ?", (str(path),)).fetchone()
    if row and row[1] == digest and not force:
        return "skip"
    if row:
        con.execute("DELETE FROM docs WHERE doc_id = ?", (row[0],))

    chunks = EXTRACTORS[path.suffix.lower()](path)
    if not chunks:
        return "empty"

    cur = con.execute(
        "INSERT INTO docs (path, kind, sha1, ingested_at) VALUES (?, ?, ?, ?)",
        (str(path), kind, digest, datetime.now(timezone.utc).isoformat()),
    )
    doc_id = cur.lastrowid
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i: i + EMBED_BATCH]
        vecs = embed_texts([t for _, t in batch])
        con.executemany(
            "INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)",
            [(doc_id, pg, t, vecs[j].tobytes()) for j, (pg, t) in enumerate(batch)],
        )
    con.commit()
    return f"{len(chunks)} chunks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    args = ap.parse_args()

    con = connect()
    t0 = time.time()
    targets = [(p, "paper") for p in sorted(PAPERS.glob("*.pdf"))]
    targets += [(p, "data") for p in sorted(DATA.rglob("*"))
                if p.is_file() and p.suffix.lower() in EXTRACTORS]

    # prune docs whose source file was deleted from library/
    live = {str(p) for p, _ in targets}
    for doc_id, path in con.execute("SELECT doc_id, path FROM docs").fetchall():
        if path not in live:
            con.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
            print(f"[prune] {Path(path).name}: removed (file deleted)")
    con.commit()

    for path, kind in targets:
        try:
            result = ingest_file(con, path, kind, args.force)
        except Exception as e:  # keep going; one bad file must not kill the run
            result = f"ERROR: {e}"
        print(f"[{kind}] {path.name}: {result}")

    n_docs, n_chunks = con.execute(
        "SELECT (SELECT COUNT(*) FROM docs), (SELECT COUNT(*) FROM chunks)").fetchone()
    print(f"\nIndex: {n_docs} docs, {n_chunks} chunks ({time.time() - t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
