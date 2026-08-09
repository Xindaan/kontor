# Changelog

What changed in Kontor, in plain language — newest first.

Kontor is developed in a private research repository and published here in
curated batches. Entries are grouped by the date a batch landed publicly;
a version number marks the releases that bundle them. Pre-1.0, a minor bump
carries new features and breaking changes alike.

---

## 0.2.0 — 2026-08-09

The first version bump since the initial release. `0.1.0` was never moved while
four batches of features, a breaking change and a pile of fixes landed on top of
it, so this release collects everything published since 2026-07-14. Under
semantic versioning's pre-1.0 rules a minor bump is what carries new features and
breaking changes, which is exactly what accumulated.

**Breaking change in this range** (shipped 2026-07-27, called out again here
because it now has a version attached): `GermanTaxModel` rejects any
`cost_basis_method` other than `"FIFO"` at construction instead of silently
computing FIFO anyway.

### Security

- **Dependency updates closing 27 of 51 known advisories.** All of them are
  transitive — none was a package this project asks for directly:
  `pillow` 12.1.0 → 12.3.0 (13 advisories, the largest single block),
  `protobuf` 6.33.2 → 7.35.1, `idna` 3.11 → 3.18, `soupsieve` 2.8.1 → 2.9.2.
  No constraint in `pyproject.toml` changed; only the resolved lock moved.
- **Still open, deliberately:** `cryptography` (pulled in by `pdfminer-six`) and
  `starlette` / `python-multipart` (pulled in by `fastapi`, and only installed
  with the optional `web` group) need major-version jumps that would drag their
  parents with them. Those are a separate piece of work rather than a quiet
  bundle in a security release. In practical terms the exposure is small: this is
  a locally-run analysis tool, the web UI binds to localhost, and the image
  library is never imported by this codebase — it arrives only as a PDF-parsing
  dependency.

### Documentation

A pass over the README asking the one question the test suite cannot: **does this
work for someone who is not the author?** Several answers were no.

#### Fixed

- **The recommended first step did not work.** The Quick Start pointed newcomers
  at the web UI, but the web dependencies live in an *optional* Poetry group, so
  `poetry install` never installed them and `backtest web` exited with an error.
  The troubleshooting note then sent readers down a `PATH` rabbit hole. Install
  now says `poetry install --with web`, and there is a table of every optional
  group (`web`, `ml`, `sentiment`, `sentiment-finbert`) with what each one buys.
- **Two CLI descriptions were still in German** (`--soxl-proxy`) and did not match
  the actual `--help` text. Both now read as the code does.
- **Example commands referenced files that do not exist in the repository.**
  `--portfolio data/manual/portfolio.json` and friends are your own data or the
  output of an earlier step; nothing said so, which made the examples look broken.
  Now stated explicitly where the portfolio format is documented.
- **API keys had no setup instructions.** `FINNHUB_API_KEY` and `NEWSAPI_API_KEY`
  appeared as end-of-line comments only. There is **no `.env` loading** in this
  project, so a `.env` file would have been silently ignored — the export form is
  now documented, along with which datasets need no key at all.
- **"All off by default" was misleading** in Execution & Cost Realism. It is true
  of the three options in that section, but `slippage_pct` (0.0005) and
  `slippage_bps` (5.0) are non-zero defaults, so no run is frictionless. Scoped
  and corrected.

#### Added

- **A full [Configuration Reference](README.md#configuration-reference)** — all 33
  `BacktestConfig` fields and all 5 `CostConfig` fields with their defaults,
  grouped by topic. Sixteen of them were previously undocumented, including
  `tax_rate`, `tax_partial_exemption`, `risk_free_rate` (which drives every
  Sharpe figure) and `slippage_pct`. Values such as the 26.375 % tax rate were
  quoted in prose, but the field names were not, so there was no way to find out
  how to change them.
- **A [Limitations](README.md#limitations) section**, collecting what the
  framework does not do: the tax model's exclusions, why a backtest is not a
  forecast, that costs are modelled rather than observed, the data caveats, that
  nothing places an order, and that no performance claim is being made.

## 2026-07-27

Execution and cost **realism** — making a simulated result match what actually
reaches a German depot. Every option below is off by default, so existing runs
are unchanged.

### Added

- **Broker commission ceiling (`commission_max`)** — the transaction-cost model
  now brackets the percentage tier with a floor *and* a cap, the way real retail
  fee schedules work. Two ready profiles from the brokers' published tables:
  `TransactionCostModel.deutsche_bank_maxblue()` (0.25%, min €8.90, max €58.90)
  and `TransactionCostModel.trade_republic()` (flat €1.00). Without the cap a
  large order overstated its cost several-fold.
- **No-trade band (`rebalance_min_deviation`)** — rebalance only when the largest
  actual-vs-target weight gap breaches the band, then fully. The old behaviour
  snapped to target at every date down to a one-cent difference, which
  manufactured trades (and tax) that a live drift rule never triggers.
- **Rebalance weekday anchor (`rebalance_weekday`)** — pin weekly rebalancing to a
  fixed weekday (0=Mon … 4=Fri) instead of the ISO-week's last trading day, with
  holiday fallback to the next open day. The weekday barely moves CAGR but shifts
  underwater duration by years, so it should match the day you actually trade.
- See the new **Execution & Cost Realism** section in the README.

### Changed

- **Cost basis is FIFO-only, and says so.** `GermanTaxModel` now **rejects** any
  `cost_basis_method` other than `"FIFO"` at construction. The lot engine has
  always sold FIFO; a non-FIFO value used to be stored and silently ignored, so a
  run asking for AVGCOST quietly computed FIFO. It now fails loud instead of
  returning a wrong-but-plausible tax figure.

### Fixed

- **Empty holdings no longer read as a full-size BUY.** A portfolio with no
  positions loaded returned the same zero weights as one genuinely at 0, so every
  target looked like an urgent full-book drift. `Portfolio.has_holdings()` now
  distinguishes the two and the signal report flags a no-holdings run as
  meaningless rather than emitting a confident wrong signal.

## 2026-07-23

Housekeeping around **what this repository is and what it ships** — no changes to
the backtesting engine.

### Added

- **`NOTICE.md`** — a per-dataset breakdown of everything bundled under `data/`
  and `tests/fixtures/`: where it came from and what terms it carries. The short
  version: the MIT licence covers the code, not the third-party data. The
  point-in-time index constituents are derived from Wikipedia and carry a
  CC BY-SA attribution/share-alike obligation; the total-return test fixtures are
  a frozen Yahoo Finance window, kept because the tests are meaningless without a
  pinned input, and not a redistributable dataset. Everything else is either
  synthetic or original to this project.
- **A "Contributing" section in the README**, stating plainly what was previously
  only implied: Kontor is a published mirror of a private research repository. It
  is maintained as a single squashed commit that is rewritten on every batch, so a
  pull request cannot be merged normally and would be overwritten by the next
  publication. Issues are welcome; forks work as expected.

### Removed

- **`.github/PULL_REQUEST_TEMPLATE.md`** — it invited contributions the publishing
  model cannot absorb. Keeping it would have wasted a contributor's time, which is
  the worst way for this to go wrong.

## 2026-07-20

A large batch focused on **data integrity** — making sure a backtest or a live
signal is never computed on data that is stale, corrupt, or silently wrong.

### Added

- **Stale-price protection** (`freshness.py`) — price data now carries a
  freshness check. If the latest available price is older than expected, the
  run fails loudly instead of quietly computing a signal from yesterday's
  number. This closes a whole class of bug where an outdated quote hides a
  stop that should have triggered.
- **Stop rules and stop monitoring** (`stop_rules.py`, `stop_monitor.py`) —
  explicit, testable rules for when a position's stop is breached, plus a
  monitor that evaluates them against live quotes and reports a verdict.
- **Cache integrity checks** (`cache_integrity.py`) — detects a price cache
  that disagrees with its own manifest (truncated files, gaps, a recorded end
  date that the data doesn't actually reach) instead of trusting it blindly.
- **Atomic JSON storage** (`json_store.py`) — state files are written
  atomically, so an interrupted run can't leave a half-written file behind.
- **Tradegate as a quote source** (`tradegate.py`) — useful for German
  instruments that Yahoo prices unreliably or not at all.
- **Total-return handling**, with recorded fixtures for adjusted close,
  unadjusted close and dividends, so dividend treatment is pinned down by
  tests rather than assumed.
- **A "Data Integrity" section in the README** explaining these guarantees and
  their limits.

### Changed

- **The public API and all error messages are now English.** If you used the
  previous German identifiers, this is a breaking rename — the behaviour is
  unchanged.
- The CLI `--position` help text now shows a neutral example share count.

### Fixed

- Improvements across the backtester, the data loader, portfolio pricing, the
  signal engine and the German tax model that came with the same batch.

---

## 2026-07-15

### Fixed

- **Regime classification is now deterministic.** When every input metric was
  near-constant (for example a perfectly smooth equity curve), the reference
  distribution was degenerate and the percentile ranking fell back on exact
  floating-point equality. The resulting risk bucket could differ between
  machines — the same input classified as `normal` on one CPU and `fragile` on
  another. A degenerate reference now ranks at the median, which is stable
  everywhere. Real (non-degenerate) distributions are unaffected.

### Added

- Continuous integration: the test suite runs on Python 3.11 and 3.12 for
  every push and pull request, with a status badge in the README.
- Issue and pull-request templates.

---

## 2026-07-14

### Added

- **Initial public release.** A backtesting framework built around the parts
  that decide a German private investor's *net* outcome:
  - German capital-gains taxation as part of the simulation —
    Abgeltungssteuer including Solidaritätszuschlag, Teilfreistellung,
    Freistellungsauftrag, the two separate loss pots, on FIFO lot accounting.
  - Survivorship-bias-free backtests on point-in-time index constituents.
  - Plan-only derivation of broker order sheets for Trade Republic and
    Maxblue — the framework never submits an order.
  - A strategy library spanning classic allocations (60/40, All Weather, dual
    momentum, risk parity, trend following) through to leveraged-ETF momentum
    with volatility targeting.
  - A web UI for running, comparing and sweeping strategies.
