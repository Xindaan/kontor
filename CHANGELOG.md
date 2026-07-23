# Changelog

What changed in Kontor, in plain language — newest first.

Kontor is developed in a private research repository and published here in
curated batches, so entries are grouped by the date a batch landed publicly
rather than by semantic version.

---

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
