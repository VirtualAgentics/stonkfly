"""Public Coinbase observations. Synthetic fixtures are explicit test inputs."""

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .config import D, down, up
from .neural.common import DATA


@dataclass(frozen=True)
class Quote:
    product: str
    bid: Decimal
    ask: Decimal
    timestamp: float
    base_increment: Decimal
    quote_increment: Decimal
    price_increment: Decimal
    minimum_quote: Decimal
    minimum_base: Decimal

    def __post_init__(self):
        if (
            not self.bid.is_finite()
            or not self.ask.is_finite()
            or not 0 < self.bid <= self.ask
            or not math.isfinite(self.timestamp)
        ):
            raise ValueError("Invalid quote")
        for k in [
            "base_increment",
            "quote_increment",
            "price_increment",
            "minimum_quote",
            "minimum_base",
        ]:
            if not getattr(self, k).is_finite() or getattr(self, k) <= 0:
                raise ValueError("Invalid market increment")

    def json(self):
        return {
            k: str(v) if isinstance(v, Decimal) else v for k, v in asdict(self).items()
        }


def unwrap(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Exchange timestamp requires a timezone")
    return parsed.timestamp()


class CoinbaseMarket:
    def __init__(self, products, client=None):
        if client is None:
            from coinbase.rest import RESTClient

            # Override SDK environment defaults: public observations never need
            # an account key, even when a live broker exists in this process.
            client = RESTClient(api_key=None, api_secret=None, timeout=10)
        self.client = client
        self.products = products
        self.meta = {}
        self.history = {p: [] for p in products}

    def snapshot(self):
        result = {}
        for product in self.products:
            if not self.history[product]:
                # Seed only completed, past one-minute candles. No future samples.
                end = int(time.time() // 60) * 60
                candles = unwrap(
                    self.client.get_public_candles(
                        product, str(end - 120 * 60), str(end), "ONE_MINUTE", limit=120
                    )
                ).get("candles", [])
                past = sorted(
                    (c for c in candles if int(c["start"]) < end),
                    key=lambda c: int(c["start"]),
                )
                if not past:
                    raise RuntimeError("No historical candles available")
                self.history[product] = [float(D(c["close"])) for c in past]
                if any(not math.isfinite(v) or v <= 0 for v in self.history[product]):
                    raise RuntimeError("Invalid historical price")
            # Refresh tradeability at every observation, not just startup.
            m = unwrap(self.client.get_public_product(product))
            self.meta[product] = m
            if (
                m.get("product_id") != product
                or m.get("product_type") != "SPOT"
                or m.get("quote_currency_id") != "USDC"
            ):
                raise RuntimeError("Unexpected product")
            if any(
                m.get(k, False)
                for k in [
                    "is_disabled",
                    "trading_disabled",
                    "cancel_only",
                    "view_only",
                    "limit_only",
                    "auction_mode",
                ]
            ):
                raise RuntimeError("Product unavailable for immediate spot execution")
            b = unwrap(self.client.get_public_product_book(product, limit=1))[
                "pricebook"
            ]
            if b.get("product_id") != product or not b.get("bids") or not b.get("asks"):
                raise RuntimeError("Empty or mismatched book")
            quote = Quote(
                product,
                D(b["bids"][0]["price"]),
                D(b["asks"][0]["price"]),
                utc_timestamp(b["time"]),
                D(m["base_increment"]),
                D(m["quote_increment"]),
                D(m.get("price_increment", m["quote_increment"])),
                D(m["quote_min_size"]),
                D(m["base_min_size"]),
            )
            result[product] = quote
        return result

    def record(self, quotes):
        for p, q in quotes.items():
            self.history[p].append(float((q.bid + q.ask) / 2))
            self.history[p] = self.history[p][-120:]


class FixtureMarket:
    """Deterministic prices for offline verification, never used in live mode."""

    def __init__(self, products):
        self.products = products
        self.tick = 0
        self.history = {p: [] for p in products}

    def snapshot(self):
        base = {"BTC-USDC": 60000, "ETH-USDC": 2500, "SOL-USDC": 100}
        quotes = {}
        for j, p in enumerate(self.products):
            price = D(base[p]) * D(1 + 0.025 * math.sin(self.tick * 0.6 + j))
            quotes[p] = Quote(
                p,
                price * D(".9995"),
                price * D("1.0005"),
                time.time(),
                D(".00000001"),
                D(".01"),
                D(".01"),
                D("1"),
                D(".00000001"),
            )
            if not self.history[p]:
                self.history[p] = [
                    float(D(base[p]) * D(1 + 0.01 * math.sin(i * 0.4 + j)))
                    for i in range(80)
                ]
        self.tick += 1
        return quotes

    record = CoinbaseMarket.record


CANDLE_PAGE = 350  # Coinbase public candles maximum per request


def fetch_candles(product, days, client=None, out=None, now=None):
    """Store completed public one-minute candles as a chronological parquet file.

    Public endpoint, no key. Pages are merged and de-duplicated; values stay
    exchange strings. The current, incomplete minute is never included.

    The endpoint returns candles with start in (start, end], at most 350. Do
    not pass `limit`: with it, the exchange ignores the requested range and
    returns the newest candles instead.
    """
    if client is None:
        from coinbase.rest import RESTClient

        client = RESTClient(api_key=None, api_secret=None, timeout=10)
    if not math.isfinite(days) or days <= 0:
        raise ValueError("Positive number of days required")
    now = time.time() if now is None else now
    end = int(now // 60) * 60
    start = end - int(days * 86400)
    rows = {}
    cursor = end
    while cursor > start:
        chunk = max(start, cursor - CANDLE_PAGE * 60)
        r = unwrap(
            client.get_public_candles(product, str(chunk), str(cursor), "ONE_MINUTE")
        )
        for c in r.get("candles", []):
            t = int(c["start"])
            if chunk < t <= cursor and t < end:
                rows[t] = c
        cursor = chunk
    if not rows:
        raise RuntimeError("No candles returned")
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys = sorted(rows)
    table = pa.table(
        {
            "start": pa.array(keys, pa.int64()),
            **{
                k: pa.array([str(D(rows[t][k])) for t in keys], pa.string())
                for k in ["open", "high", "low", "close", "volume"]
            },
        }
    )
    out = Path(out) if out else DATA / "candles" / f"{product}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return {
        "product": product,
        "candles": len(keys),
        "first": keys[0],
        "last": keys[-1],
        "path": str(out),
    }


class ReplayMarket:
    """Chronological replay of stored public candles. Paper only.

    Observation t sees candle t's close as the midpoint. The execution
    snapshot taken after an observation is served from candle t+1, so a paper
    fill happens at the next candle, never at the price that produced the
    decision. History is seeded only from candles before the window.
    """

    HALF_SPREAD = D(".0005")

    def __init__(self, products, path, start=None, end=None, seed_history=120):
        if len(products) != 1:
            raise ValueError("Replay supports exactly one product")
        import pyarrow.parquet as pq

        self.products = tuple(products)
        self.product = products[0]
        self.path = Path(path)
        t = pq.read_table(self.path).to_pydict()
        ts, closes = t["start"], t["close"]
        if not ts or any(b <= a for a, b in zip(ts, ts[1:])):
            raise ValueError("Candles must be nonempty and strictly chronological")
        lo = utc_timestamp(start) if start else ts[0]
        hi = utc_timestamp(end) if end else ts[-1] + 1
        if not lo < hi:
            raise ValueError("Empty replay window")
        first = next((i for i, x in enumerate(ts) if x >= lo), len(ts))
        last = next((i for i, x in enumerate(ts) if x >= hi), len(ts))
        # Keep one candle past the window for the final execution snapshot.
        self.candles = [
            (int(ts[i]), D(closes[i])) for i in range(first, min(last + 1, len(ts)))
        ]
        if len(self.candles) < 2:
            raise ValueError("Replay window needs at least two candles")
        if any(c <= 0 for _, c in self.candles):
            raise ValueError("Invalid replay price")
        self.window = (self.candles[0][0], self.candles[-2][0])
        seed = [float(D(closes[i])) for i in range(max(0, first - seed_history), first)]
        if not seed:
            raise ValueError("No history before the replay window")
        self.history = {self.product: seed}
        self.cursor = 0

    @property
    def exhausted(self):
        return self.cursor >= len(self.candles) - 1

    def clock(self):
        return float(self.candles[min(self.cursor, len(self.candles) - 1)][0])

    def snapshot(self):
        if self.cursor >= len(self.candles):
            raise RuntimeError("Replay exhausted")
        t, close = self.candles[self.cursor]
        inc = D(".01")
        return {
            self.product: Quote(
                self.product,
                down(close * (1 - self.HALF_SPREAD), inc),
                up(close * (1 + self.HALF_SPREAD), inc),
                float(t),
                D(".00000001"),
                inc,
                inc,
                D("1"),
                D(".00000001"),
            )
        }

    def record(self, quotes):
        CoinbaseMarket.record(self, quotes)
        self.cursor += 1
