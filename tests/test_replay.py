"""Offline replay harness: stored candles, candle-time clock, controls, report."""

import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stonkfly.config import D, Settings
from stonkfly.ledger import Ledger
from stonkfly.market import ReplayMarket, fetch_candles
from stonkfly.report import summarize, table
from stonkfly.risk import Guard, Veto

T0 = 1_700_000_040  # arbitrary minute-aligned epoch (divisible by 60)


def candles(n, t0=T0):
    return [
        {
            "start": str(t0 + 60 * i),
            "open": "100",
            "high": "101",
            "low": "99",
            "close": str(100 + i),
            "volume": "1",
        }
        for i in range(n)
    ]


@pytest.fixture
def parquet(tmp_path):
    rows = candles(300)
    path = tmp_path / "BTC-USDC.parquet"
    pq.write_table(
        pa.table(
            {
                "start": pa.array([int(r["start"]) for r in rows], pa.int64()),
                **{
                    k: pa.array([r[k] for r in rows], pa.string())
                    for k in ["open", "high", "low", "close", "volume"]
                },
            }
        ),
        path,
    )
    return path


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def test_replay_seeds_history_only_from_past_candles(parquet):
    m = ReplayMarket(("BTC-USDC",), parquet, iso(T0 + 200 * 60), iso(T0 + 210 * 60))
    assert m.history["BTC-USDC"] == [100.0 + i for i in range(80, 200)]
    q = m.snapshot()["BTC-USDC"]
    assert q.bid < D(300) < q.ask and q.timestamp == T0 + 200 * 60
    assert m.clock() == q.timestamp
    assert m.window == (T0 + 200 * 60, T0 + 209 * 60)


def test_replay_execution_snapshot_is_next_candle(parquet):
    m = ReplayMarket(("BTC-USDC",), parquet, iso(T0 + 200 * 60), iso(T0 + 202 * 60))
    first = m.snapshot()
    m.record(first)
    assert m.history["BTC-USDC"][-1] == 300.0
    execution = m.snapshot()["BTC-USDC"]
    assert execution.timestamp == first["BTC-USDC"].timestamp + 60
    assert m.clock() == execution.timestamp
    assert not m.exhausted
    m.record(m.snapshot())
    assert m.exhausted  # Last candle is reserved for execution only.


@pytest.mark.parametrize(
    "products,start,end",
    [
        (("BTC-USDC", "ETH-USDC"), None, None),
        (("BTC-USDC",), iso(T0), None),  # No history before the window
        (("BTC-USDC",), iso(T0 + 299 * 60), None),  # Fewer than two candles
        (("BTC-USDC",), iso(T0 + 200 * 60), iso(T0 + 100 * 60)),
    ],
)
def test_replay_rejects_invalid_windows(parquet, products, start, end):
    with pytest.raises(ValueError):
        ReplayMarket(products, parquet, start, end)


class PagedSDK:
    """Models the observed public endpoint: candles with start in (start, end],
    newest first, at most 350. A `limit` argument would make the exchange
    ignore the range, so the double refuses it."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get_public_candles(self, product, start, end, granularity, limit=None):
        assert limit is None, "limit makes the exchange ignore start/end"
        self.calls.append((int(start), int(end)))
        assert granularity == "ONE_MINUTE"
        page = [r for r in self.rows if int(start) < int(r["start"]) <= int(end)]
        return {"candles": list(reversed(page))[:350]}


def test_fetch_pages_and_excludes_incomplete_minute(tmp_path):
    rows = candles(800)
    now = T0 + 800 * 60 + 30  # halfway through candle 800 (which does not exist)
    sdk = PagedSDK(rows + candles(1, T0 + 800 * 60))
    out = tmp_path / "c.parquet"
    r = fetch_candles("BTC-USDC", 0.5, sdk, out, now=now)
    stored = pq.read_table(out).to_pydict()
    # (start, end] pages over 720 minutes: the oldest boundary candle is excluded.
    assert r["candles"] == 719 and len(stored["start"]) == 719
    assert stored["start"] == sorted(stored["start"]) and stored["start"][-1] == T0 + 799 * 60
    assert len(sdk.calls) == 3 and all(e - s <= 350 * 60 for s, e in sdk.calls)


def test_guard_uses_injected_clock(tmp_path, parquet):
    s = Settings()
    ledger = Ledger(tmp_path / "ledger.sqlite", s, "paper")
    m = ReplayMarket(("BTC-USDC",), parquet, iso(T0 + 200 * 60))
    guard = Guard(s, ledger, tmp_path / "STOP", m.clock)
    quotes = m.snapshot()
    plan = guard.plan("BTC-USDC", "BUY", quotes)  # Would be stale on wall time.
    plan = ledger.reserve(plan, guard.clock())
    guard.before_submit(plan)
    m.record(quotes)
    with pytest.raises(Veto, match="expired"):
        guard.before_submit(plan)  # Candle clock advanced past the quote age.
    ledger.close()


def test_report_summarizes_events_and_baselines(tmp_path):
    s = Settings()
    run = tmp_path / "run"
    run.mkdir()
    ledger = Ledger(run / "ledger.sqlite", s, "paper")
    ledger.put("cash", "90")
    ledger.put("positions", {"BTC-USDC": "0.1"})
    ledger.close()
    (run / "provenance.json").write_text(
        json.dumps({"settings": {"paper_fee": "0.006", "order_limit": "10"}})
    )
    quote = lambda bid, ask, t: {
        "product": "BTC-USDC",
        "bid": bid,
        "ask": ask,
        "timestamp": t,
    }
    events = [
        {
            "quote": quote("100", "100.1", 1),
            "equity_usdc": "100",
            "neural": {"side": "BUY", "stimulus": "none", "difference_hz": 4.0, "gate_spikes": 1, "memory": {"changed_edges": 1}},
            "execution": {"status": "FILLED"},
        },
        {
            "quote": quote("110", "110.1", 61),
            "equity_usdc": "101",
            "neural": {"side": "HOLD", "stimulus": "reward", "difference_hz": -2.0, "gate_spikes": 0, "memory": {"changed_edges": 5}},
            "execution": {"status": "HOLD"},
        },
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    r = summarize(run)
    assert r["final"] == "101.0" and r["fills"] == 1 and r["changed_edges"] == 5
    assert r["proposals"] == {"BUY": 1, "SELL": 0, "HOLD": 1}
    assert r["stimuli"] == {"reward": 1, "aversive": 0, "none": 1}
    assert D(r["baseline_buy_and_hold"]) == D(100) - D("10.06") + D(10) / D("100.1") * D(110)
    assert round(r["market_change_pct"], 3) == round((110 / 100.1 - 1) * 100, 3)
    assert r["gate_fraction"] == 0.5 and r["difference_hz_histogram"]["2..6"] == 1
    assert "| run |" in table([r]) and "1/0/1" in table([r])
