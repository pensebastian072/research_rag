"""Strategy leaderboard from TradingView backtest exports in library/data.

Parses Performance / Trades analysis / Risk-adjusted sheets into one row per
export, writes index/backtests.csv, prints a leaderboard sorted by net profit %.
Read-only research reporting; says nothing about what to trade.

Usage: .venv\\Scripts\\python.exe rag\\report.py [--csv-only]
"""
from __future__ import annotations

import argparse
import csv
import sys
from numbers import Number
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "library" / "data"
OUT_CSV = ROOT / "index" / "backtests.csv"

SKIP_SHEETS = {"trades", "list of trades", "properties"}

# metric -> (candidate row labels, which numeric cell holds it: 0=first, 1=second[%])
METRICS = {
    "net_profit_pct": (["net profit"], 1),
    "max_dd_pct": (["max drawdown (close-to-close)", "max equity drawdown", "max drawdown"], 1),
    "cagr_pct": (["annualized return (cagr)"], 0),
    "buy_hold_pct": (["buy and hold % gain"], 0),
    "profit_factor": (["profit factor"], 0),
    "sharpe": (["sharpe ratio"], 0),
    "sortino": (["sortino ratio"], 0),
    "total_trades": (["total trades"], 0),
    "pct_profitable": (["percent profitable"], 0),
}
PROPS = {"symbol": "symbol", "timeframe": "timeframe", "trading range": "range"}


def parse_export(path: Path) -> dict:
    row_map: dict[str, list] = {}
    props: dict[str, str] = {}
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            title = ws.title.lower()
            if title in SKIP_SHEETS and title != "properties":
                continue
            for row in ws.iter_rows(max_row=150, values_only=True):
                if not row or row[0] is None:
                    continue
                label = str(row[0]).strip().lower()
                if title == "properties":
                    if label in PROPS and len(row) > 1 and row[1] is not None:
                        props[PROPS[label]] = str(row[1])
                else:
                    row_map.setdefault(label, [c for c in row[1:] if isinstance(c, Number)])
    finally:
        wb.close()

    rec: dict = {"strategy": path.stem, **{k: "" for k in METRICS}, **{v: props.get(v, "") for v in PROPS.values()}}
    for metric, (labels, idx) in METRICS.items():
        for lab in labels:
            nums = row_map.get(lab)
            if nums:
                rec[metric] = nums[idx] if len(nums) > idx else nums[0]
                break
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-only", action="store_true")
    args = ap.parse_args()

    records = []
    for path in sorted(DATA.rglob("*.xlsx")):
        try:
            records.append(parse_export(path))
        except Exception as e:
            print(f"ERROR {path.name}: {e}")
    records.sort(key=lambda r: r["net_profit_pct"] if isinstance(r["net_profit_pct"], Number) else float("-inf"),
                 reverse=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["strategy", "symbol", "timeframe", "range"] + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {OUT_CSV} ({len(records)} strategies)\n")
    if args.csv_only:
        return

    hdr = f"{'strategy':<58} {'net%':>8} {'maxDD%':>7} {'PF':>6} {'Sharpe':>7} {'win%':>6} {'trades':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        def fmt(v, spec):
            return format(v, spec) if isinstance(v, Number) else "-"
        name = r["strategy"][:57]
        print(f"{name:<58} {fmt(r['net_profit_pct'], '8.2f')} {fmt(r['max_dd_pct'], '7.2f')} "
              f"{fmt(r['profit_factor'], '6.3f')} {fmt(r['sharpe'], '7.3f')} "
              f"{fmt(r['pct_profitable'], '6.2f')} {fmt(r['total_trades'], '7.0f')}")


if __name__ == "__main__":
    main()
