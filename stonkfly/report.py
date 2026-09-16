"""Summarize one or more run directories from their local event logs.

Reads only events.jsonl, provenance.json and the ledger. Baselines are
computed from the same first and last quotes each run actually saw, so runs
on different windows are compared against their own market.
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path

from .config import D


def histogram(values, edges=(-10, -6, -2, 2, 6, 10)):
    """Counts per bin of the decoded difference; the middle bin is the hold band."""
    labels = [f"<{edges[0]}"] + [f"{a}..{b}" for a, b in zip(edges, edges[1:])]
    labels.append(f">={edges[-1]}")
    counts = Counter()
    for v in values:
        i = sum(1 for e in edges if v >= e)
        counts[labels[i]] += 1
    return {k: counts.get(k, 0) for k in labels}


def summarize(run_dir):
    run_dir = Path(run_dir)
    rows = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No events in {run_dir}")
    provenance = json.loads((run_dir / "provenance.json").read_text())
    settings = provenance["settings"]
    db = sqlite3.connect(f"file:{run_dir / 'ledger.sqlite'}?mode=ro", uri=True)
    meta = {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM meta")}
    db.close()
    first, last = rows[0]["quote"], rows[-1]["quote"]
    last_bid = {last["product"]: D(last["bid"])}
    unmarked = [p for p in meta["positions"] if p not in last_bid]
    final = D(meta["cash"]) + sum(
        (D(v) * last_bid[p] for p, v in meta["positions"].items() if p in last_bid),
        D(0),
    )
    equity = [D(r["equity_usdc"]) for r in rows] + [final]
    peak, drawdown = equity[0], D(0)
    for e in equity:
        peak = max(peak, e)
        drawdown = max(drawdown, peak - e)
    sides = Counter(r["neural"]["side"] for r in rows)
    executions = Counter(r["execution"]["status"] for r in rows)
    stimuli = Counter(r["neural"]["stimulus"] for r in rows)
    initial = D(meta["initial_cash"])
    fee = D(settings["paper_fee"])
    stake = D(settings["order_limit"])
    hold_base = stake / D(first["ask"])
    buy_and_hold = initial - stake * (1 + fee) + hold_base * D(last["bid"])
    return {
        "run": run_dir.name,
        "ticks": len(rows),
        "initial": str(initial),
        "final": str(final),
        "change": str(final - initial),
        "max_drawdown": str(drawdown),
        "fills": executions.get("FILLED", 0) + executions.get("SETTLED", 0),
        "vetoes": executions.get("VETO", 0),
        "proposals": {k: sides.get(k, 0) for k in ["BUY", "SELL", "HOLD"]},
        "stimuli": {k: stimuli.get(k, 0) for k in ["reward", "aversive", "none"]},
        "mean_difference_hz": sum(r["neural"]["difference_hz"] for r in rows)
        / len(rows),
        "mean_raw_difference_hz": sum(
            r["neural"].get("raw_difference_hz", r["neural"]["difference_hz"])
            for r in rows
        )
        / len(rows),
        "gate_fraction": sum(1 for r in rows if r["neural"]["gate_spikes"]) / len(rows),
        "difference_hz_histogram": histogram(
            [r["neural"]["difference_hz"] for r in rows]
        ),
        "changed_edges": rows[-1]["neural"]["memory"]["changed_edges"],
        "halted": meta.get("halted"),
        "unmarked_positions": unmarked,
        "baseline_cash": str(initial),
        "baseline_buy_and_hold": str(buy_and_hold),
        "market_change_pct": float((D(last["bid"]) / D(first["ask"]) - 1) * 100),
        "window": [rows[0]["quote"]["timestamp"], last["timestamp"]],
    }


def table(summaries):
    head = [
        "run",
        "ticks",
        "final",
        "change",
        "maxDD",
        "fills",
        "veto",
        "B/S/H",
        "R/A/N",
        "edges",
        "diffHz",
        "gate%",
        "buy&hold",
        "mkt%",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        p, t = s["proposals"], s["stimuli"]
        lines.append(
            "| "
            + " | ".join(
                [
                    s["run"],
                    str(s["ticks"]),
                    f"{D(s['final']):.4f}",
                    f"{D(s['change']):+.4f}",
                    f"{D(s['max_drawdown']):.4f}",
                    str(s["fills"]),
                    str(s["vetoes"]),
                    f"{p['BUY']}/{p['SELL']}/{p['HOLD']}",
                    f"{t['reward']}/{t['aversive']}/{t['none']}",
                    str(s["changed_edges"]),
                    f"{s['mean_difference_hz']:+.2f}",
                    f"{100 * s['gate_fraction']:.0f}",
                    f"{D(s['baseline_buy_and_hold']):.4f}",
                    f"{s['market_change_pct']:+.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
