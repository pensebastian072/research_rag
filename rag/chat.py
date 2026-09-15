"""Interactive chat over the research library. 100% local (Ollama), no API tokens.

Each turn retrieves fresh excerpts (papers + backtest cards) for your question
and answers with qwen2.5:7b, citing sources.

Usage: .venv\\Scripts\\python.exe rag\\chat.py
Commands inside chat:
    /k N          retrieved chunks per turn (default 8)
    /kind paper   restrict to papers   (/kind data | /kind all)
    /sources      show sources of last answer
    /quit         exit
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if not sys.stdin.isatty():  # piped input: decode as UTF-8 and swallow any BOM
    sys.stdin.reconfigure(encoding="utf-8-sig", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHAT_MODEL, chat
from query import search

LEADERBOARD_CSV = Path(__file__).resolve().parent.parent / "index" / "backtests.csv"


def leaderboard_context() -> str:
    """Compact always-in-context summary of the owner's backtests (run report.py to refresh)."""
    if not LEADERBOARD_CSV.exists():
        return ""
    with open(LEADERBOARD_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = ["Owner's TradingView backtest leaderboard "
             "(strategy | symbol | net profit % | max drawdown % | profit factor | sharpe | win % | trades):"]
    for r in rows:
        lines.append(f"- {r['strategy'][:70]} | {r['symbol']} | net {r['net_profit_pct']}% | "
                     f"DD {r['max_dd_pct']}% | PF {r['profit_factor']} | sharpe {r['sharpe']} | "
                     f"win {r['pct_profitable']}% | {r['total_trades']} trades")
    return "\n".join(lines)

SYSTEM = (
    "You are a quantitative finance research assistant chatting with the library owner. "
    "You get two sources each turn: (a) the owner's backtest LEADERBOARD - the "
    "authoritative results of THEIR OWN TradingView strategies; and (b) retrieved "
    "EXCERPTS from academic papers and backtest detail sheets. Questions about 'my "
    "strategies', what worked, or what lost money MUST be answered from the leaderboard "
    "numbers. Paper questions use excerpts, cited as [n]. Do not use outside knowledge. "
    "If neither source covers the question, say so. Research context only - never give "
    "trade instructions."
)
HISTORY_TURNS = 4  # question/answer pairs carried for follow-up context


def main() -> None:
    k, kind = 8, None
    history: list[tuple[str, str]] = []
    last_hits: list[dict] = []
    board = leaderboard_context()
    print(f"research-rag chat ({CHAT_MODEL}, local). /quit to exit, /k N, /kind paper|data|all, /sources")

    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q.startswith("/k "):
            k = max(1, int(q.split()[1]))
            print(f"k = {k}")
            continue
        if q.startswith("/kind"):
            arg = (q.split() + ["all"])[1]
            kind = None if arg == "all" else arg
            print(f"kind = {arg}")
            continue
        if q == "/sources":
            for i, h in enumerate(last_hits):
                loc = f" p.{h['page']}" if h["page"] else ""
                print(f"[{i + 1}] {h['source']}{loc} (score {h['score']:.3f})")
            continue

        last_hits = search(q, k, kind)
        excerpts = "\n\n".join(
            f"[{i + 1}] ({h['source']}" + (f", p.{h['page']}" if h["page"] else "") + f")\n{h['text']}"
            for i, h in enumerate(last_hits)
        )
        convo = "".join(f"\nEarlier Q: {uq}\nEarlier A: {ua}\n" for uq, ua in history[-HISTORY_TURNS:])
        prompt = (f"{convo}\nEXCERPTS:\n\n{excerpts}\n\n=== LEADERBOARD (owner's own strategy results) ===\n"
                  f"{board}\n\nQuestion: {q}")
        answer = chat(prompt, system=SYSTEM)
        print(f"\n{answer}")
        srcs = ", ".join(sorted({h["source"] for h in last_hits}))
        print(f"\n(sources: {srcs} - /sources for detail)")
        history.append((q, answer))


if __name__ == "__main__":
    main()
