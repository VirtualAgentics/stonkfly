# Running and stopping Stonkfly

Use a dedicated account portfolio. Stonkfly is an experiment capable of losing its entire allocated balance. The funding cap is **100 USDC at initialization**, not an assertion that USDC always equals one dollar.

## Installation and data

Use Python 3.11 and a C++17 compiler (`clang++`/`c++` on macOS, GCC or Clang on Linux). `python -m stonkfly prepare` downloads about 1.1 GB of upstream data, verifies it, and builds the full graph. Allow several additional GB for dependencies, derived data and two checkpoints. `python -m stonkfly verify` independently checks prepared inputs. Set `STONKFLY_DATA` to use another data location.

Existing DOOMFLY researchers can reuse verified local files with `python -m stonkfly prepare --reuse-doomfly /path/to/working-copy`. Stonkfly copies only the three required data artifacts, then checks the same locks. It does not import a Doom environment, run its website, or depend on that checkout afterward.

## Paper modes

```sh
# Real public prices; simulated fills and 0.6% fee per side.
python -m stonkfly run --steps 10

# Explicit synthetic offline market, accelerated development run.
python -m stonkfly run --fixture --fast --steps 10 --out runs/fixture

# Frozen-memory control, always in a separate run directory.
python -m stonkfly run --fixture --fast --frozen --steps 10 --out runs/frozen
```

## Offline replay and controls

Replay runs the same loop over stored public candles at candle time, so the
quote-age, cooldown and daily-limit checks behave as they would live. It is
paper only and never contacts the exchange.

```sh
# Store completed one-minute candles (public endpoint, no key).
python -m stonkfly fetch --product BTC-USDC --days 7

# Chronological window; observation t sees candle t, a fill happens at candle t+1.
python -m stonkfly run --replay data/candles/BTC-USDC.parquet \
  --from 2026-09-14T00:00:00Z --to 2026-09-15T00:00:00Z --out runs/learning

# Controls on the same window.
python -m stonkfly run --replay ... --from ... --to ... --frozen --out runs/frozen
python -m stonkfly run --replay ... --from ... --to ... \
  --reinforcement-file runs/learning/events.jsonl --reinforcement-seed 1 --out runs/shuffled

# Held-out window, starting from the learned brain with fresh cash.
python -m stonkfly run --replay ... --from 2026-09-15T00:00:00Z --to 2026-09-16T00:00:00Z \
  --brain runs/learning/brain-0.npz --out runs/heldout

python -m stonkfly report runs/learning runs/frozen runs/shuffled runs/heldout
```

Timestamps need an explicit zone. History before the window seeds the
chart; nothing after the current candle is visible. Paper fills use the next
candle's close with a 0.05% half spread plus the configured fee; there is no
depth or impact model. `--reinforcement-file` replays another run's stimulus
sequence tick by tick (a yoked control); with `--reinforcement-seed` the
sequence is permuted (a shuffled control). The natural stimulus is still
logged as `natural_stimulus`. `report` reads only local run directories and
compares each run with cash and a single buy-and-hold order on its own
window.

Two optional model variants exist for replay comparisons; both default off
and change the run's settings signature. `--decoder-center ema` subtracts a
running mean of past DNp20 differences before thresholding, so a persistent
circuit bias no longer maps to one-sided exposure. `--reward-mode fill`
delivers a dopamine pulse only on the observation after the fly's own fill,
comparing equity with the mark taken right after that fill; holding through
price drift produces no pulse. Interrupting a replay and rerunning the same command resumes it.

`--fast` skips wall waits only in paper mode. It preserves the 0.1 ms neural timestep and the real 60-second execution cooldown, so an accelerated probe can have many rejected trades. This is a plumbing/neural test, not a backtest of achievable market returns. Paper fills use observed bid/ask plus the configured fee; they do not simulate depth, queue position or all market impact. `--fixture` never claims real market data.

## Coinbase setup, performed by you

1. Create a separate Coinbase Advanced portfolio and put up to 100 USDC in it. Start without other assets or open orders. Do not mix other bots, manual trades or deposits into that portfolio while Stonkfly runs.
2. Create a [Coinbase App API key](https://docs.cdp.coinbase.com/coinbase-app/authentication-authorization/api-key-authentication) with ECDSA, **View and Trade**, **Transfer disabled**, scoped only to that portfolio. The program checks permissions and portfolio scope; an account-wide key is rejected.
3. Save the downloaded key JSON locally as `coinbase-key.json` and restrict its file permissions (`chmod 600 coinbase-key.json`). It typically contains `name` and `privateKey`. Never paste the key into a commit or README.
4. Copy `.env.example` to `.env`, set the key path and `COINBASE_PORTFOLIO_ID`, then set `STONKFLY_LIVE=I_ACCEPT_REAL_TRADES`. The CLI also requires `--live`; paper mode never submits an order even if the environment variable is present.
5. Run `python -m stonkfly run --live --preflight-only`. This reads permissions, account balances and order state, and initializes the local ledger. **It does not submit orders.** Once you have reviewed the configuration, run `python -m stonkfly run --live` yourself.

The key must be allowed to trade the requested pairs in your region. Defaults use BTC-USDC; ETH-USDC and SOL-USDC are optional via `--products`. Only available spot products pass checks. Coinbase preview warnings, unsupported order types or insufficient fee coverage stop the order; there is no fallback to an unbounded market order.

## Execution guarantees and limits

- Maximum initial funding: 100 USDC. Maximum buy commitment: 10 USDC including a 2% fee reserve. Sell quantity is capped by owned inventory and 10 USDC observed notional; a better execution price can yield slightly more proceeds. No borrowing, shorting, transfers or leverage actions are exposed.
- At most 24 order attempts per UTC day and at least 60 seconds between attempts. Rejected previews count. Failed orders do not become new strategy choices.
- Price-bounded fill-or-kill orders use at most 0.5% slippage and 0.5% spread. Quotes must be no older than 15 seconds. A fresh book is fetched after neural integration; a move beyond the observation tolerance vetoes the trade. Preview fees, account balances and the STOP condition are checked before submission.
- At 20 USDC drawdown from starting equity, **stop new orders**. This is not a liquidation order or guaranteed maximum loss. Existing holdings remain exposed; price moves between observations can exceed the threshold. Decide separately how you want to manage those holdings.
- SQLite records a unique client order ID before submission. An uncertain response stays unresolved; the worker searches the exchange for that same ID instead of sending a new order. Missing/ambiguous results stop the worker for manual review. Final fills and fees settle exactly once. Unexpected actual fees are booked, then further orders halt.
- The local process lock prevents two workers using one run directory. It does not coordinate multiple computers or copied ledgers. Run one worker for the dedicated portfolio, and do not delete the live ledger to bypass checks.

## State, recovery and privacy

The worker must stay running on your computer/server. It is not a hosted service. Prevent laptop sleep if you want uninterrupted observations. Closing it preserves committed neural state and memory.

`runs/<name>/` holds a SQLite ledger, two alternating checkpoints, `events.jsonl`, `latest.json`, `latest-input.png`, and provenance with exact code, graph, stimulus and parameter hashes. Each intent binds to the preceding neural observation and checkpoint. The ledger is authoritative if a crash occurs before the human-readable log is written.

To stop: Ctrl-C, or `touch runs/live/STOP` (`runs/paper/STOP` for paper). This stops future decisions/submissions; an already submitted FOK order may still finish. Inspect any uncertain order in Coinbase before taking another action.

For an ordinary clean restart, use the same command and run directory. After reviewing a transient failure and reconciling account state, remove the STOP file if appropriate and pass `--resume-reviewed`. This cannot clear a drawdown or fee-overrun stop, bypass unresolved exchange outcomes, or accept changed source/configuration. A missing unknown order requires manual exchange investigation; do not assume it failed. Source changes require an explicitly reviewed state migration; use a fresh **paper** directory for development.

All runtime state, balances, account IDs, data, `.env` and the default key filenames are git-ignored. Keep custom key paths outside the repository. Tests use doubles and never submit real orders. No live account credentials or real balances are bundled.
