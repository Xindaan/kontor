# Kontor

[![CI](https://github.com/Xindaan/kontor/actions/workflows/ci.yml/badge.svg)](https://github.com/Xindaan/kontor/actions/workflows/ci.yml)

**Tax-aware backtesting for German investors.** A Python framework where the German
tax treatment of your gains is part of the simulation — and point-in-time signals are turned
into concrete, plan-only broker order sheets for **Trade Republic** and **Maxblue / Deutsche Bank**.

Generic backtesters (backtrader, vectorbt, zipline) optimise gross or lightly-taxed returns.
For a German retail investor that overstates what actually reaches the depot. Kontor
models the things that decide your *net* outcome, per trade, so a `net_liquidation` result reflects
real after-tax money:

- **German capital-gains tax logic** — Abgeltungssteuer 26.375 % (25 % + 5.5 % Solidaritätszuschlag),
  **Teilfreistellung** (30 % partial exemption for equity funds), **Freistellungsauftrag**
  (Sparer-Pauschbetrag), and two loss pots per §20 Abs. 6 EStG (Aktienverlusttopf + allgemeiner
  Verlusttopf), on **FIFO** lot accounting.
- **Live-order derivation** — a point-in-time signal engine that emits broker-specific,
  **plan-only** order sheets for Trade Republic and Maxblue (dry-run / CSV plans; no live order
  submission by design).
- **Survivorship-bias-free** — strategies run on point-in-time index constituents, i.e. the
  universe as it actually was on each date.
- **Data-integrity guards** — a price is a *(value, session date, source)* triple, and the date has
  to be evidenced by the source. Stale prices don't raise errors; they produce confident wrong
  answers. See [Data Integrity](#data-integrity).

> **Scope & honesty:** this is a research and decision-support tool, **not investment advice**.
> The tax model deliberately excludes Vorabpauschale, Kirchensteuer, and the derivatives loss pot
> (see [`src/backtest/tax/de_tax_model.py`](src/backtest/tax/de_tax_model.py) and
> [`docs/german-tax-model.md`](docs/german-tax-model.md)).

**What's new:** Kontor is developed in a private research repository and published here in
curated batches. See the [changelog](CHANGELOG.md) for what each batch changed, in plain language.

---

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/Xindaan/kontor.git
cd kontor
poetry install  # or: python -m pip install -e .
```

> If `pip` isn't on your PATH, use `python -m pip ...` instead.

> **Note:** If you installed with Poetry (without `pip install -e .`), run commands as  
> `poetry run backtest ...` instead of `backtest ...`.

> **If `backtest` is still “command not found”:** it means the console script isn’t on your `PATH`.
> Try one of these:
> - `poetry run backtest web` (if you used Poetry)
> - `python -m backtest.cli web` (uses the module directly)
> - Ensure your Python `bin` directory is on `PATH` (where `python -m pip install -e .` put the script)

### Your First Backtest (2 Minutes)

**Option A: Web UI (Recommended)**
```bash
poetry run backtest web
# Opens at http://localhost:8000
```
Then select a strategy, set dates, and click "Run Backtest".

**Option B: Command Line**
```bash
# Run a simple strategy
poetry run backtest run strategies/dual_momentum.py --start 2015-01-01

# Compare multiple strategies
poetry run backtest compare strategies/dual_momentum.py strategies/classic_60_40.py

# View key metrics only
poetry run backtest metrics strategies/dual_momentum.py
```

---

## Best Practices

### 1. Avoid Survivorship Bias

**Problem:** Using today's S&P 500 stocks for a 2010 backtest is cheating - you're selecting winners with hindsight.

**Solution:** Use Point-in-Time (PIT) strategies:

```bash
# CORRECT: PIT strategies use historical constituent data
backtest run strategies/momentum_topn_pit_sp500.py --start 2010-01-01

# ALSO CORRECT: ETF-based strategies have no stock selection bias
backtest run strategies/sector_rotation_momentum.py --start 2010-01-01
backtest run strategies/dual_momentum.py --start 2010-01-01
```

**Generate PIT data first:**
```bash
# Download historical S&P 500 constituents
python scripts/download_pit_data.py --index sp500

# Download NASDAQ 100 constituents (requires: pip install git+https://github.com/jmccarrell/n100tickers.git)
python scripts/download_pit_data.py --index nasdaq100
```

### 2. Fair Tax Comparison with `--liquidate-at-end`

**Problem:** Buy & Hold pays 0% realized tax (never sells), making it look better than it is.

**Solution:** Use `--liquidate-at-end` to simulate selling at the end:

```bash
# Fair comparison - all strategies show realistic tax impact
backtest compare strategies/*.py --liquidate-at-end

# Robustness testing with fair tax
backtest sweep strategies/*.py --window 10y --liquidate-at-end
```

### 3. Test Robustness, Not Just Performance

**Problem:** A strategy that worked 2010-2020 might fail 2015-2025.

**Solution:** Use rolling window analysis:

```bash
# Test with 10-year rolling windows
backtest sweep strategies/dual_momentum.py --window 10y

# Check consistency across many start dates
backtest sweep strategies/*.py --window 5y --start-grid monthly
```

### 4. Detect Overfitting with Walk-Forward

**Problem:** Optimized parameters might be overfit to your test period.

**Solution:** Use walk-forward optimization:

```bash
backtest optimize strategies/dual_momentum.py \
  --walk-forward \
  --train-years 5 \
  --test-years 1 \
  --param lookback_days=126,189,252

# Good: Degradation < 25%
# Bad: Degradation > 50% (likely overfit)
```

For stronger robustness checks, use nested walk-forward:

```bash
backtest optimize strategies/dual_momentum.py \
  --walk-forward-nested \
  --train-years 5 --test-years 1 \
  --inner-train-years 3 --inner-test-years 1 \
  --param lookback_days=126,189,252
```

Nested mode reports IS/OOS degradation, parameter drift, and a PASS/FAIL robustness gate.

---

## Common Workflows

### Workflow 1: Interactive Analysis (Web UI)

```bash
# Start the web interface
backtest web --port 8000
```

The Web UI provides all backtesting features with an interactive interface:

| Page | Description |
|------|-------------|
| **Run** | Single strategy backtest with parameter customization |
| **Compare** | Side-by-side strategy comparison with charts |
| **Sweep** | Rolling window robustness analysis |
| **Optimize** | Parameter optimization with grid search or walk-forward |
| **Batch Optimize** | Optimize multiple strategies, ranked by metric |
| **Signals** | Live signals, order sizing, drift reconciliation, meta decision, bootstrap start choice |
| **Manual Data** | Manual data provenance registry and verification |

### Workflow 2: Find the Best Strategy (CLI)

```bash
# Step 1: Optimize all strategies
backtest batch-optimize "strategies/[!_]*.py" --out results/opt

# Step 2: Compare with robustness testing
backtest sweep "strategies/[!_]*.py" \
  --params-file results/opt/optimized_params.py \
  --window 10y \
  --liquidate-at-end

# Step 3: Review results
open results/sweep_*/report.html
```

### Workflow 3: Backtest to Live Trading

```bash
# Step 1: Backtest with optimal parameters
backtest run strategies/momentum_topn_pit_sp500.py \
  --param top_n=10 --param lookback_days=189 \
  --start 2010-01-01

# Step 2: Generate today's signals
backtest signals strategies/momentum_topn_pit_sp500.py \
  --param top_n=10 --param lookback_days=189

# Step 3: Compare with your current portfolio
backtest signals strategies/momentum_topn_pit_sp500.py \
  --portfolio portfolio.json \
  --output signals.json

# Step 4: Register manual input provenance (e.g., SeekingAlpha export)
backtest data provenance add data/manual/seekingalpha/factors_2026-02-06.csv \
  --dataset fundamentals_sp500 \
  --seekingalpha \
  --as-of-date 2026-02-06

# Step 5: Enforce provenance checks when loading manual fundamentals
backtest run strategies/factor_value_quality_momentum.py \
  --param fundamentals_provenance_mode=strict \
  --param fundamentals_provenance_dataset=fundamentals_sp500 \
  --param fundamentals_provenance_source=SeekingAlpha
```

**Portfolio JSON format:**
```json
{
  "positions": {"AAPL": 50, "MSFT": 30, "GOOGL": 20},
  "cash": 5000.00,
  "last_rebalance": "2025-12-31"
}
```

### Workflow 4: Parameter Optimization

```bash
# Find best lookback period
backtest optimize strategies/dual_momentum.py \
  --param lookback_days=126,189,252,315 \
  --metric sharpe_ratio

# Multi-parameter optimization
backtest optimize strategies/volatility_targeting.py \
  --param target_vol=0.10,0.12,0.15,0.20 \
  --param lookback_days=20,40,60 \
  --metric sharpe_ratio \
  --output optimization.csv
```

### Workflow 5: Evidence-Gated Meta Switching

```bash
# Step 1: Choose a neutral start strategy
backtest meta-bootstrap \
  --strategy-a strategies/levered_etf_momentum_sticky_adaptive_v2.py \
  --strategy-b strategies/buy_and_hold.py \
  --b-param allocation='{"SPY": 1.0}' \
  --profile ausgewogen

# Step 2: Generate live signals with a challenger
# Candidates with params go into a JSON file, for example:
# [{"strategy":"strategies/buy_and_hold.py","params":{"allocation":{"SPY":1.0}}}]
backtest signals strategies/levered_etf_momentum_sticky_adaptive_v2.py \
  --exposure-policy-enable \
  --exposure-policy-profile us \
  --meta-enable \
  --meta-candidate strategies/buy_and_hold.py \
  --meta-candidates-file meta_candidates.json \
  --meta-evidence-required \
  --meta-evidence-profile ausgewogen \
  --meta-regime-mode strategy_fragility

# Step 3: Refresh / inspect the OOS evidence artifact
backtest meta-evidence \
  --current-strategy strategies/levered_etf_momentum_sticky_adaptive_v2.py \
  --target-strategy strategies/buy_and_hold.py \
  --target-param allocation='{"SPY": 1.0}' \
  --profile ausgewogen
```

Signals now expose the same workflow in the UI:
- **Meta Bootstrap (Startwahl)** decides where to start.
- **Meta Decision & Evidence** decides whether a live switch is allowed.
- **Exposure Policy** keeps the strategy target intact but can de-lever 3x execution to a mapped 1x/Core fallback when fixed shock gates fire.
- The report shows gate-by-gate `switch_checks` including a `Needs next` hint for blocked switches.

---

## Strategy Guide

### Tier-System (Frontend-Dropdown)

Each strategy carries a tier prefix in its name that shows its status at a
glance. The sort order in the frontend follows these tiers (validated winners
at the top, experimental at the bottom):

| Prefix | Meaning | Examples |
|---|---|---|
| `[Production]` | The framework's default reference strategy. | Levered ETF Momentum (Sticky Winner) |
| `[Pilot]` | Candidates with a serious promotion prospect. | Sticky Levered + Vol-Targeting, Cascade |
| `[Research]` | Legitimate alternatives, but not a default. | Sector-Aware VolTarget, Adaptive V2, Tax-Switch, Regime Vol Gate |
| `[Benchmark]` | Classic references — 60/40, All Weather, Dual Momentum, Trend Following, Risk Parity, Buy & Hold. | Buy & Hold, 60/40, All Weather |
| `[Experimental]` | Only with the greatest caution — extreme leverage / fragile data. | Levered 5x Momentum Guard |

Hidden from the frontend (CLI-only): the external-feature demos
(`ml_forecast_tilt`, `sentiment_tilt`, `analyst_momentum_filter`).

### Recommended Strategies (No Survivorship Bias)

| Strategy | File | Description | Best For |
|----------|------|-------------|----------|
| **Dual Momentum** | `dual_momentum.py` | Classic Antonacci dual momentum | Simplicity, low turnover |
| **Sector Rotation** | `sector_rotation_momentum.py` | Top 3 S&P sectors by momentum | Sector exposure |
| **PIT S&P 500 Momentum** | `momentum_topn_pit_sp500.py` | Top-N stocks with historical data | Fair stock momentum |
| **PIT NASDAQ 100** | `momentum_topn_pit_nasdaq100.py` | Top-N from NASDAQ 100 | Tech-focused momentum |
| **PIT Russell 2000** | `momentum_topn_pit_russell2000.py` | Small-cap momentum | Small-cap exposure |
| **Factor VQM (PIT)** | `factor_value_quality_momentum.py` | Value + quality + momentum with PIT fundamentals | Factor investing |
| **60/40 Portfolio** | `classic_60_40.py` | Classic balanced allocation | Low volatility |
| **Volatility Targeting** | `volatility_targeting.py` | Dynamic allocation by vol | Risk control |

### Strategy Categories

**ETF-Based (No PIT needed):**
```bash
# Momentum / Trend
backtest run strategies/dual_momentum.py
backtest run strategies/sector_rotation_momentum.py
backtest run strategies/trend_following_sma.py
backtest run strategies/levered_trend_filter.py

# Volatility Management
backtest run strategies/volatility_targeting.py
backtest run strategies/voltargeting_trendfilter.py

# Risk Parity
backtest run strategies/inverse_vol_risk_parity.py
backtest run strategies/trend_risk_parity.py

# Drawdown Protection
backtest run strategies/drawdown_brake.py
```

**Stock-Picking (Requires PIT data):**
```bash
# Generate PIT data first
python scripts/download_pit_data.py --index sp500

# Then run strategies
backtest run strategies/momentum_topn_pit_sp500.py --start 2000-01-01
backtest run strategies/momentum_topn_pit_nasdaq100.py --start 2016-01-01

# Fundamentals-driven factor strategy
backtest run strategies/factor_value_quality_momentum.py \
  --param csv_path=data/universes/sp500_constituents.csv \
  --param fundamentals_path=data/fundamentals \
  --param top_n=30
```

> Note: the repo ships example CSVs in `data/fundamentals/` (AAPL, MSFT).
> Für echte Backtests bitte die Fundamentals-Daten für alle Universe-Ticker bereitstellen.

### Current Live / Levered Core Set

The Signals UI is intentionally curated around the currently relevant live / leveraged strategies instead of showing every historical file.

| Strategy | File | Role |
|----------|------|------|
| **Buy & Hold** | `buy_and_hold.py` | Unlevered benchmark / robust core |
| **3x Buy & Hold** | `3x_bh.py` | Levered benchmark |
| **60/40**, **All Weather**, **Dual Momentum**, **SMA Trend Filter**, **Volatility Targeting**, **Inverse-Vol Risk Parity**, **Sector Rotation**, **Momentum Top-N (PIT)** | see `strategies/` | Classic reference strategies |
| **Sticky Winner** | `levered_etf_momentum_sticky.py` | Core levered sticky baseline (TQQQ / UPRO / SOXL / TECL, safe: SPY) |
| **Sticky Levered + Vol-Targeting** | `sticky_levered_vol_targeted.py` | Baseline + vol-targeting overlay — the strongest full-cycle result |
| **Sticky Levered + Cascade** | `sticky_levered_cascade.py` | Multi-3x allocation (single_pick / cascade / inverse_vol) |
| **Sticky Adaptive V2** | `levered_etf_momentum_sticky_adaptive_v2.py` | Adaptive research variant |
| **Crash Guard LBH Challenger** | `levered_momentum_crash_guard_lbh_challenger.py` | Alternative challenger with a different regime path |
| **5x Momentum Guard** | `levered_5x_momentum_guard.py` | Experimental real-ticker 5x family |

Notes:
- `levered_etf_momentum_sticky.py` is the Sticky/Core reference baseline. The
  leveraged sleeves default to the liquid US 3x ETFs, which carry the long
  history a meaningful backtest needs.
- If you trade UCITS ETPs at a European broker instead, keep the US tickers as
  the *signal* universe and map them to your broker's tradable line via
  `instrument_mapping` (see [Live Order Derivation](docs/live-order-derivation.md)).
- Adaptive V2 remains available for research and explicit meta/evidence
  comparisons, but should not be treated as a default until it beats the
  Sticky/Core baseline on the same data, costs, and OOS evidence.

Robust Sticky/Core defaults from the 2026-04-13 rolling-window optimization:
- US Sticky/Core: `lookback_days=63`, `switch_buffer=0.04`, `min_hold_periods=2`, `momentum_floor=0.0`, `baseline_drawdown_floor=-1.0`.
- Trade Republic Sticky/Core: `lookback_days=63`, `switch_buffer=0.06`, `min_hold_periods=1`, `momentum_floor=-0.05`, `baseline_drawdown_floor=-1.0`.
- Maxblue Sticky/Core: `lookback_days=63`, `switch_buffer=0.06`, `min_hold_periods=1`, `momentum_floor=-0.05`, `baseline_drawdown_floor=-1.0`.
- These defaults come from a rolling-window robustness optimization (many start dates, degradation-checked), not a single-period fit.

**Sticky Levered + Vol-Targeting** is the first overlay that robustly beats the plain
Sticky-TR baseline: a weekly vol-target overlay scales the 3x position by
`clip(0.40 / 20d-vol, 0, 1)` and parks the remainder in `SXR8.DE`. Backtest
2016-2024 (full backtester incl. German tax): +7.8pp CAGR, +29 % Sharpe,
-11.7pp MaxDD vs plain Sticky Levered. It must be run **weekly**
(`--rebalance-frequency weekly`), otherwise the overlay does not engage.

Broker compatibility: the Trade Republic and Maxblue strategy variants encode
which instruments each broker can actually trade (e.g. Maxblue cannot trade
`3SEM.L`, so it routes via `A2QC5J` / `VVSM.DE`). See
[`docs/live-order-derivation.md`](docs/live-order-derivation.md).


## Web UI

### Starting the Server

```bash
# Default port 8000
backtest web

# Custom port
backtest web --port 8080

# Open browser automatically
backtest web --open
```

### Pages

| Page | URL | Description |
|------|-----|-------------|
| **Home** | `/` | Dashboard with quick links |
| **Run** | `/run` | Single strategy backtest |
| **Compare** | `/compare` | Multi-strategy comparison |
| **Sweep** | `/sweep` | Rolling window robustness analysis |
| **Optimize** | `/optimize` | Parameter grid search & walk-forward |
| **Batch Optimize** | `/batch-optimize` | Multi-strategy optimization |
| **Signals** | `/signals` | Live signals, portfolio drift, meta decision, evidence tuning, bootstrap start decision |
| **Meta-Playbook** | `/playbook` | Strategy promotion governance: baseline vs. candidates, rolling 3Y/5Y, broker-mapping audit, past artifacts + markdown preview |
| **Manual Data** | `/manual-data` | Manual data provenance registry / verification |

### Features

- **Refined Financial Light Theme**: Custom design system with Inter + JetBrains Mono fonts, consistent color palette, KPI cards with positive/negative indicators
- **Dynamic Parameter Forms**: Strategy parameters are auto-detected and rendered as form fields with collapsible advanced options
- **Real-time Results**: HTMX-powered updates without page reload, progress bars with ETA
- **Interactive Charts**: Plotly equity curves, heatmaps, and box plots with unified theme
- **Strategy Comparison**: Sortable table with best-value highlighting (green background on best metric per column)
- **Tax Integration**: Full German tax model support with all metric bases (gross, net realized, net liquidation)
- **Live Trading Signals**: Signal generation with BUY/SELL/HOLD badges, order proposals, and drift reconciliation
- **Meta Bootstrap**: Bilateral OOS evidence check to choose a neutral start strategy
- **Meta Decision & Evidence**: Evidence-gated strategy switching with regime-aware defaults
- **Meta-Playbook (Strategy Promotion Governance)**: Reproducible audit artifact for quarterly reviews — baseline vs. candidates, rolling 3Y/5Y win-rates, broker-mapping check (TR/Maxblue), and an optional SOXL-proxy mode for long-history research. The artifact browser shows past reports with a Markdown preview.
- **Switch Checks**: Gate-by-gate visibility for score, confirmation, cadence, evidence, and conditioned evidence, including a “Needs next” hint for blocked switches
- **Broker-Aware Strategy Set**: Curated Signals dropdown with dedicated Trade Republic and Maxblue variants
- **Manual Data Provenance**: Register, inspect, and verify imported manual datasets (for example SeekingAlpha exports)
- **Responsive Design**: Works on desktop and tablet with scroll indicators on wide tables

### REST API

The Web UI exposes a JSON API for programmatic access:

```bash
# List available strategies
curl http://localhost:8000/api/v1/strategies

# Run a backtest
curl -X POST http://localhost:8000/api/v1/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "strategies/dual_momentum.py",
    "start_date": "2015-01-01",
    "initial_capital": 10000
  }'

# Compare strategies
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": [
      {"strategy": "strategies/dual_momentum.py"},
      {"strategy": "strategies/classic_60_40.py"}
    ],
    "start_date": "2015-01-01"
  }'

# Batch optimize
curl -X POST http://localhost:8000/api/v1/batch-optimize \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["dual_momentum.py", "volatility_targeting.py"],
    "metric": "sharpe_ratio",
    "start_date": "2015-01-01"
  }'

# Live signals with meta decision
curl -X POST http://localhost:8000/api/v1/signals \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "strategies/levered_etf_momentum_sticky_adaptive_v2.py",
    "meta_decision": {
      "enabled": true,
      "candidates": [
        {
          "strategy": "strategies/buy_and_hold.py",
          "params": {"allocation": {"SPY": 1.0}}
        }
      ],
      "evidence_required": true,
      "evidence_profile": "ausgewogen",
      "regime_mode": "strategy_fragility"
    }
  }'

# Explicit meta evidence refresh
curl -X POST http://localhost:8000/api/v1/meta-evidence/run \
  -H "Content-Type: application/json" \
  -d '{
    "current_strategy": "strategies/levered_etf_momentum_sticky_adaptive_v2.py",
    "target_strategy": "strategies/buy_and_hold.py",
    "target_params": {"allocation": {"SPY": 1.0}},
    "evidence_profile": "ausgewogen"
  }'

# Bootstrap start decision
curl -X POST http://localhost:8000/api/v1/meta-bootstrap/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_a": "strategies/levered_etf_momentum_sticky_adaptive_v2.py",
    "strategy_b": "strategies/buy_and_hold.py",
    "strategy_b_params": {"allocation": {"SPY": 1.0}},
    "evidence_profile": "ausgewogen"
  }'
```

Paths in the JSON API can be absolute or workspace-relative. In the Web UI, strategy selection is guided by dropdowns instead of requiring you to type these paths manually.

<details>
<summary>Full API Reference</summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/strategies` | GET | List all available strategies |
| `/api/v1/strategies/{name}/schema` | GET | Get parameter schema for a strategy |
| `/api/v1/run` | POST | Run single backtest |
| `/api/v1/compare` | POST | Compare multiple strategies |
| `/api/v1/sweep` | POST | Run sweep analysis |
| `/api/v1/optimize` | POST | Run parameter optimization |
| `/api/v1/batch-optimize` | POST | Optimize multiple strategies |
| `/api/v1/signals` | POST | Generate live signals and optional meta decision |
| `/api/v1/meta-evidence/run` | POST | Build / refresh a switch evidence artifact |
| `/api/v1/meta-evidence/latest` | GET | Load the latest evidence artifact for a strategy pair |
| `/api/v1/meta-evidence/{artifact_id}` | GET | Load a specific evidence artifact |
| `/api/v1/meta-bootstrap/run` | POST | Bilateral bootstrap start decision between two strategies |
| `/api/v1/data/manual/provenance` | GET/POST | List or register manual provenance entries |
| `/api/v1/data/manual/provenance/{entry_id}` | GET | Load one provenance entry |
| `/api/v1/data/manual/provenance/verify` | GET | Verify registry entries against current files |

</details>

---

## CLI Reference

All commands below assume the `backtest` console script is installed (e.g. via `pip install -e .`).  
If you use Poetry without installing, prefix commands with `poetry run`.

### Essential Commands

```bash
# Start web UI
backtest web [OPTIONS]
  --port PORT                   Server port (default: 8000)
  --host HOST                   Server host (default: 127.0.0.1)
  --open                        Open browser automatically

# Run backtest
backtest run <strategy.py> [OPTIONS]
  -s, --start DATE              Start date (default: 2010-01-01)
  -e, --end DATE                End date (default: today)
  -c, --capital FLOAT           Initial capital (default: 10000)
  -p, --param NAME=VALUE        Override parameter (repeatable)
  --cost-profile-file PATH      JSON profile for per-ticker/asset-class costs
  --execution-lag-days N        Execution delay in trading days (0=same-day, 1=T+1)
  --max-volume-participation X  Max fraction of daily volume per trade (e.g., 0.1)
  --min-daily-dollar-volume USD Skip trades below daily notional threshold
  --no-tax                      Disable German tax model
  --liquidate-at-end            Include unrealized gains in tax
  --drip                        Enable dividend reinvestment (DRIP)
  --no-validate                 Suppress validation warnings
  --skip-failed                 Skip tickers that fail to download (default: enabled)
  --no-skip-failed              Fail fast when ticker download fails

# Compare strategies
backtest compare <strategies...> [OPTIONS]
  --cost-profile-file PATH      JSON profile for per-ticker/asset-class costs
  --execution-lag-days N        Execution delay in trading days
  --max-volume-participation X  Max fraction of daily volume per trade
  --liquidate-at-end            Fair tax comparison
  --allow-misaligned            Allow different date ranges
  --drip                        Enable dividend reinvestment
  --no-validate                 Suppress validation warnings

# Robustness analysis
backtest sweep <strategies...> [OPTIONS]
  --window LENGTH               Window length: 3y, 5y, 10y (default: 10y)
  --mode rolling|end-fixed      Window mode (default: rolling)
  --cost-profile-file PATH      JSON profile for per-ticker/asset-class costs
  --execution-lag-days N        Execution delay in trading days
  --max-volume-participation X  Max fraction of daily volume per trade
  --params-file PATH            Apply optimized params JSON (from batch-optimize)
  --jobs N                      Parallel execution
  --drip                        Enable dividend reinvestment
  --no-validate                 Suppress validation warnings

# Parameter optimization
backtest optimize <strategy.py> [OPTIONS]
  -p, --param NAME=VAL1,VAL2    Parameter to optimize
  -m, --metric METRIC           Target metric (default: sharpe_ratio)
  --cost-profile-file PATH      JSON profile for per-ticker/asset-class costs
  --execution-lag-days N        Execution delay in trading days
  --max-volume-participation X  Max fraction of daily volume per trade
  --walk-forward                Enable walk-forward validation
  --walk-forward-nested         Enable nested walk-forward (inner + outer windows)
  --train-years N               Outer training years (default: 5)
  --test-years N                Outer test years (default: 1)
  --inner-train-years N         Inner training years for nested mode (default: 3)
  --inner-test-years N          Inner test years for nested mode (default: 1)
  --drip                        Enable dividend reinvestment

# Live signals
backtest signals <strategy.py> [OPTIONS]
  --portfolio PATH              Compare with current holdings
  --drift-tolerance FLOAT       Weight drift tolerance (default: 0.005 = 0.5%)
  -f, --format table|json|csv   Output format
  --exposure-policy-enable      Enable optional 3x exposure controller
  --exposure-policy-profile P   trade_republic|tr|maxblue|us
  --exposure-policy-file PATH   JSON config for proxy/core mappings and shock thresholds
  --meta-enable                 Enable evidence-gated meta decisioning
  --meta-candidate PATH         Add a challenger strategy (repeatable)
  --meta-candidates-file PATH   JSON list with candidates incl. params
  --meta-evidence-required      Hard gate: no switch without passing evidence
  --meta-evidence-profile P     defensiv|ausgewogen|aggressiv|custom
  --meta-regime-mode MODE       none|strategy_fragility
  --meta-regime-profile P       defensiv|ausgewogen|aggressiv|custom
  --meta-confirm-points N       Alpha confirmation count
  --meta-switch-margin X        Minimum score margin for alpha-driven switches
  # Includes signal table + order proposals + drift reconciliation

# Meta evidence
backtest meta-evidence [OPTIONS]
  --current-strategy PATH       Current strategy
  --target-strategy PATH        Target strategy
  --current-param NAME=VALUE    Param for current strategy (repeatable)
  --target-param NAME=VALUE     Param for target strategy (repeatable)
  --train-years N               OOS train window
  --test-years N                OOS test window
  --step-months N               OOS step size
  --profile P                   defensiv|ausgewogen|aggressiv|custom

# Meta bootstrap
backtest meta-bootstrap [OPTIONS]
  --strategy-a PATH             Strategy A
  --strategy-b PATH             Strategy B
  --a-param NAME=VALUE          Param for strategy A (repeatable)
  --b-param NAME=VALUE          Param for strategy B (repeatable)
  --train-years N               Bootstrap train window
  --test-years N                Bootstrap test window
  --step-months N               Bootstrap step size
  --profile P                   defensiv|ausgewogen|aggressiv|custom

# Meta-Playbook (Strategy Promotion Governance)
# Reproducible audit artifact for quarterly promotion reviews:
# baseline vs. candidates, rolling 3Y/5Y, broker-mapping audit.
# Default: Sticky/Levered (baseline) vs. Vol-Targeting-Live + Cascade (candidates).
backtest meta-promotion [STRATEGIES...] [OPTIONS]
  --baseline PATH               Incumbent baseline (default: Sticky/Levered Live)
  --soxl-proxy                  Lang-Historien-Research-Mode: SOXL als 3SEM.L-Proxy
  --start DATE                  Backtest start (default: 2016-01-01)
  --end DATE                    Backtest end (default: today)
  --capital FLOAT               Initial capital EUR (default: 10000)
  --costs FLOAT                 Transaction costs fraction (default: 0.001)
  --metric-basis MODE           gross|net_realized|net_liquidation (default: net_liquidation)
  --no-tax                      Disable German tax model
  --broker NAME                 Broker mapping audit (repeatable): trade_republic|maxblue
  --output-dir PATH             Output root (default: results/meta_promotion)
  --align MODE                  intersection|ffill (default: ffill)
  --skip-failed                 Skip tickers that fail to download
  --no-validate                 Disable pre/post-run validation
# Artefakte landen unter results/meta_promotion/<YYYYMMDD>/<artifact_id>/{json,md}.
# Im Web-UI parallel verfuegbar unter /playbook.

# Manual data provenance
backtest data provenance add <file> --dataset NAME --source NAME [OPTIONS]
backtest data provenance list [--dataset NAME] [--source NAME]
backtest data provenance show <entry_id>
backtest data provenance verify

# Optional: strict provenance gates for manual fundamentals
# (set as strategy params, works in CLI and WebUI strategy parameter form)
--param fundamentals_provenance_mode=strict
--param fundamentals_provenance_verify_hash=True
--param fundamentals_provenance_dataset=fundamentals_sp500
--param fundamentals_provenance_source=SeekingAlpha

# External features pipeline (analyst/news/ML)
backtest features pull --dataset mock_analyst --as-of 2026-05-01 [--tickers AAA,BBB]
backtest features list [--dataset NAME] [-f table|json]
backtest features verify [--dataset NAME] [-f table|json]

# Phase B analyst datasets (require explicit --tickers, except mock_analyst):
backtest features pull --dataset yahoo_analyst_current --as-of YYYY-MM-DD --tickers AAPL,MSFT
backtest features pull --dataset finnhub_analyst_current --as-of YYYY-MM-DD --tickers AAPL  # FINNHUB_API_KEY
backtest features pull --dataset finnhub_analyst_actions --as-of YYYY-MM-DD --tickers AAPL  # FINNHUB_API_KEY
backtest features pull --dataset synthetic_analyst_pit --as-of YYYY-MM-DD --tickers AAPL

# Phase C news datasets (Sentiment engine + intraday cutoff are PULL-TIME flags):
backtest features pull --dataset synthetic_news_pit --as-of YYYY-MM-DD --tickers AAPL,MSFT
backtest features pull --dataset yahoo_news --as-of TODAY --tickers AAPL  # current-only
backtest features pull --dataset finnhub_news --as-of YYYY-MM-DD --tickers AAPL \
  --news-engine vader --news-intraday-cutoff 17:00     # FINNHUB_API_KEY
backtest features pull --dataset newsapi_news --as-of YYYY-MM-DD --tickers AAPL \
  --news-engine vader                                  # NEWSAPI_API_KEY

# Activate external features on a run (default disabled, fully backward compatible)
backtest run strategies/analyst_momentum_filter.py \
  --external-features-enable \
  --external-features-dataset finnhub_analyst_actions \
  --external-features-provenance-mode strict
# Same flag trio works on: run, compare, sweep, optimize, batch-optimize, signals

# Runtime news-score component (Phase C). News engine choice and the
# intraday cutoff live on `features pull`; runtime only consumes the
# stored scores:
backtest signals strategies/sentiment_tilt.py \
  --external-features-enable --external-features-dataset synthetic_news_pit \
  --news-dataset synthetic_news_pit --news-score-weight 0.2

# Phase D ML-Forecast: synthetic adapter works without a model bundle.
# Real adapters (`lightgbm_forecast`, `xgboost_forecast`,
# `ml_forecast_ensemble`) need a bundle trained via `backtest ml train`.
backtest features pull --dataset synthetic_ml_forecast --as-of YYYY-MM-DD \
  --tickers AAPL,MSFT
backtest features pull --dataset lightgbm_forecast --as-of YYYY-MM-DD \
  --tickers AAPL --ml-model-bundle data/external_features/ml/models/2024-12-31/lightgbm

# Train a bundle (requires `poetry install --with ml`):
backtest ml train --start 2018-01-01 --end 2024-12-31 \
  --horizons 21,63,252 --models lightgbm --tickers AAPL,MSFT,GOOGL

# Use the ML forecast at runtime:
backtest signals strategies/ml_forecast_tilt.py \
  --external-features-enable --external-features-dataset synthetic_ml_forecast \
  --ml-dataset synthetic_ml_forecast --ml-score-weight 0.2

# Phase E1: Cross-Product-Konsens-Gate über Analyst x News x ML.
# Switch-Gate auf decision_bucket = current_regime_bucket (Codex R3.7).
backtest signals strategies/ml_forecast_tilt.py \
  --external-features-enable --external-features-dataset synthetic_ml_forecast \
  --ml-dataset synthetic_ml_forecast --ml-score-weight 0.2 \
  --cross-product-require [--cross-product-threshold 0.5]

# Phase E4: Risk-Parity ERC im Strategy-Layer.
# strategies/ml_forecast_tilt.py akzeptiert weighting in {equal,
# inverse_vol, erc} plus optional target_sum, max_weight.

# Phase E2: execution-plan pipeline. PHASE E EMITS BROKER-READY
# ORDER PLANS, BUT NO REAL BROKER ORDERS. Real-money submission
# would be Phase F with its own safety spec.
backtest live plan --signals-report results/signals_report.json \
  --broker dry_run --portfolio data/manual/portfolio.json
backtest live status --since 2026-06-01
backtest live reconcile --broker trade_republic_brief \
  --portfolio data/manual/portfolio.json

# Verfuegbare Phase-E2-Broker (alle plan_only=True):
#   dry_run, ibkr_basket_csv (TWS-CSV-Export ohne ib_insync),
#   alpaca_paper_preview, trade_republic_brief, maxblue_brief
```

### Full Command Reference

<details>
<summary>Click to expand full CLI reference</summary>

```
backtest
├── web                                  Start web interface
│   ├── --port PORT                      Server port (default: 8000)
│   ├── --host HOST                      Server host (default: 127.0.0.1)
│   └── --open                           Open browser automatically
│
├── run <strategy.py>                    Run single backtest
│   ├── -s, --start DATE                 Start date (default: 2010-01-01)
│   ├── -e, --end DATE                   End date (default: today)
│   ├── -c, --capital FLOAT              Initial capital (default: 10000)
│   ├── -b, --benchmark NAME             Benchmark name (default: S&P 500)
│   ├── --benchmark-ticker SYMBOL        Override benchmark ticker
│   ├── --costs FLOAT                    Transaction costs (default: 0.001)
│   ├── --cost-profile-file PATH         Per-ticker/asset-class cost profile JSON
│   ├── --execution-lag-days N           Execution delay in trading days (0=same-day, 1=T+1)
│   ├── --t-plus-one                     Shortcut for --execution-lag-days 1
│   ├── --max-volume-participation X     Max fraction of daily volume per trade
│   ├── --min-daily-dollar-volume USD    Skip trades below daily notional threshold
│   ├── --liquidity-on-missing-volume M  allow|skip
│   ├── --rebalance-frequency FREQ       daily|weekly|monthly|quarterly|yearly
│   ├── -p, --param NAME=VALUE           Override strategy parameter (repeatable)
│   ├── -o, --output PATH                Output file path
│   ├── -f, --format FORMAT              html|json
│   ├── --no-tax                         Disable German tax model
│   ├── --tax-exemption FLOAT            Freistellungsauftrag (default: 1000)
│   ├── --metric-basis MODE              gross|net_realized|net_liquidation
│   ├── --liquidate-at-end               Alias for --metric-basis net_liquidation
│   ├── --drip                           Enable dividend reinvestment (DRIP)
│   ├── --no-validate                    Suppress validation warnings
│   ├── --skip-failed                    Skip failed ticker downloads (default: enabled)
│   ├── --no-skip-failed                 Fail fast when ticker download fails
│   └── --allow-universe-lookahead       Allow non-PIT universe for historical backtests
│
├── compare <strategies...>              Compare multiple strategies
│   ├── -s, --start DATE                 Start date
│   ├── -e, --end DATE                   End date
│   ├── -o, --output PATH                Output file path
│   ├── --allow-misaligned               Allow different date ranges
│   ├── --liquidate-at-end               Fair tax comparison
│   ├── --cost-profile-file PATH         Per-ticker/asset-class cost profile JSON
│   ├── --execution-lag-days N           Execution delay in trading days
│   ├── --t-plus-one                     Shortcut for --execution-lag-days 1
│   ├── --max-volume-participation X     Max fraction of daily volume per trade
│   ├── --min-daily-dollar-volume USD    Skip trades below daily notional threshold
│   ├── --liquidity-on-missing-volume M  allow|skip
│   ├── --drip                           Enable dividend reinvestment
│   ├── --no-validate                    Suppress validation warnings
│   └── --allow-universe-lookahead       Allow non-PIT universe
│
├── sweep <strategies...>                Robustness analysis (rolling windows)
│   ├── --mode rolling|end-fixed         Window mode (default: rolling)
│   ├── --window LENGTH                  Window length: 3y, 5y, 10y, 15y
│   ├── --from DATE                      Minimum start date
│   ├── --to DATE                        Maximum start date
│   ├── --start-grid FREQ                weekly|monthly|yearly (default: monthly)
│   ├── --params-file PATH               Apply optimized params JSON (from batch-optimize)
│   ├── --cost-profile-file PATH         Per-ticker/asset-class cost profile JSON
│   ├── --execution-lag-days N           Execution delay in trading days
│   ├── --t-plus-one                     Shortcut for --execution-lag-days 1
│   ├── --max-volume-participation X     Max fraction of daily volume per trade
│   ├── --min-daily-dollar-volume USD    Skip trades below daily notional threshold
│   ├── --liquidity-on-missing-volume M  allow|skip
│   ├── --jobs N                         Parallel jobs
│   ├── --liquidate-at-end               Fair tax comparison
│   ├── --drip                           Enable dividend reinvestment
│   ├── --no-validate                    Suppress validation warnings
│   └── --out DIR                        Output directory
│
├── optimize <strategy.py>               Parameter optimization
│   ├── -p, --param NAME=VAL1,VAL2,...   Parameter to optimize (repeatable)
│   ├── -m, --metric METRIC              Metric to optimize (default: sharpe_ratio)
│   ├── --minimize                       Minimize metric (e.g., max_drawdown)
│   ├── --cost-profile-file PATH         Per-ticker/asset-class cost profile JSON
│   ├── --execution-lag-days N           Execution delay in trading days
│   ├── --t-plus-one                     Shortcut for --execution-lag-days 1
│   ├── --max-volume-participation X     Max fraction of daily volume per trade
│   ├── --min-daily-dollar-volume USD    Skip trades below daily notional threshold
│   ├── --liquidity-on-missing-volume M  allow|skip
│   ├── --walk-forward                   Enable walk-forward validation
│   ├── --walk-forward-nested            Enable nested walk-forward validation
│   ├── --train-years N                  Training window years (default: 5)
│   ├── --test-years N                   Test window years (default: 1)
│   ├── --step-months N                  Outer window step in months (default: 12)
│   ├── --inner-train-years N            Inner training years (nested mode, default: 3)
│   ├── --inner-test-years N             Inner test years (nested mode, default: 1)
│   ├── --inner-step-months N            Inner window step months (nested mode, default: 6)
│   ├── --inner-anchored                 Use anchored inner windows (nested mode)
│   ├── --drip                           Enable dividend reinvestment
│   ├── --no-validate                    Suppress validation warnings
│   └── -o, --output PATH                Output CSV file
│
├── batch-optimize <strategies...>       Optimize all strategies
│   ├── -m, --metric METRIC              Metric to optimize
│   ├── --rebalance-frequency FREQS      Frequencies to test
│   ├── --cost-profile-file PATH         Per-ticker/asset-class cost profile JSON
│   ├── --execution-lag-days N           Execution delay in trading days
│   ├── --t-plus-one                     Shortcut for --execution-lag-days 1
│   ├── --max-volume-participation X     Max fraction of daily volume per trade
│   ├── --min-daily-dollar-volume USD    Skip trades below daily notional threshold
│   ├── --liquidity-on-missing-volume M  allow|skip
│   ├── --drip                           Enable dividend reinvestment
│   ├── --no-validate                    Suppress validation warnings
│   └── --out DIR                        Output directory
│
├── signals <strategy.py>                Generate live trading signals
│   ├── -p, --param NAME=VALUE           Strategy parameter
│   ├── --portfolio PATH                 Portfolio JSON for comparison
│   ├── -d, --date DATE                  Signal date (default: today)
│   ├── --drift-tolerance FLOAT          Weight drift tolerance for reconciliation
│   ├── --meta-enable                    Enable evidence-gated meta switching
│   ├── --meta-candidate PATH            Challenger strategy (repeatable)
│   ├── --meta-candidates-file PATH      JSON list with candidates incl. params
│   ├── --meta-evidence-required         Hard evidence gate
│   ├── --meta-regime-mode MODE          none|strategy_fragility
│   ├── --meta-switch-margin FLOAT       Minimum alpha score margin
│   └── -f, --format table|json|csv      Output format
│
├── meta-evidence                        Build / refresh evidence artifact for a strategy pair
│   ├── --current-strategy PATH          Current strategy path
│   ├── --target-strategy PATH           Target strategy path
│   ├── --current-param NAME=VALUE       Current strategy param (repeatable)
│   ├── --target-param NAME=VALUE        Target strategy param (repeatable)
│   ├── --train-years N                  Train window in years
│   ├── --test-years N                   Test window in years
│   ├── --step-months N                  Step size in months
│   └── --profile PROFILE                defensiv|ausgewogen|aggressiv|custom
│
├── meta-bootstrap                       Bilateral start decision between two strategies
│   ├── --strategy-a PATH                Strategy A path
│   ├── --strategy-b PATH                Strategy B path
│   ├── --a-param NAME=VALUE             Param for strategy A (repeatable)
│   ├── --b-param NAME=VALUE             Param for strategy B (repeatable)
│   ├── --train-years N                  Train window in years
│   ├── --test-years N                   Test window in years
│   ├── --step-months N                  Step size in months
│   └── --profile PROFILE                defensiv|ausgewogen|aggressiv|custom
│
├── meta-promotion [STRATEGIES...]       Strategy-Promotion-Governance-Report
│   ├── --baseline PATH                  Incumbent baseline (default: Sticky/Levered Live)
│   ├── --soxl-proxy                     Lang-Historien-Research-Mode (SOXL als 3SEM.L-Proxy)
│   ├── --start DATE                     Backtest start (default: 2016-01-01)
│   ├── --end DATE                       Backtest end (default: today)
│   ├── --capital FLOAT                  Initial capital EUR (default: 10000)
│   ├── --costs FLOAT                    Transaction costs (default: 0.001)
│   ├── --metric-basis MODE              gross|net_realized|net_liquidation (default: net_liquidation)
│   ├── --no-tax                         Disable German tax model
│   ├── --broker NAME                    Broker mapping audit (repeatable): trade_republic|maxblue
│   ├── --output-dir PATH                Output root (default: results/meta_promotion)
│   ├── --align MODE                     intersection|ffill (default: ffill)
│   ├── --skip-failed                    Skip tickers that fail to download
│   └── --no-validate                    Disable pre/post-run validation
│                                        Artefakte: results/meta_promotion/<YYYYMMDD>/<id>/{json,md}.
│                                        Parallel im Web-UI unter /playbook.
│
├── metrics <strategy.py>                Quick metrics display
├── new <name>                           Generate strategy template
├── assets                               List available assets
├── data                                 Data management
│   ├── download <tickers...>            Download and cache data (one CSV per ticker, reused across date ranges)
│   ├── list                             List cached files
│   ├── clear                            Clear cache (all *.csv + coverage manifest)
│   └── provenance                       Manual data provenance registry
│       ├── add <file>                   Register source/import metadata
│       ├── list                         List provenance entries
│       ├── show <entry_id>              Show a single entry
│       └── verify                       Verify file existence/checksums
│
└── features                             External features pipeline (analyst/news/ML)
    ├── pull                             Pull a snapshot from a registered adapter
    │   ├── --dataset NAME               Dataset id (e.g., mock_analyst)
    │   ├── --as-of YYYY-MM-DD           Snapshot as-of date
    │   ├── --tickers AAA,BBB            Required except for mock_analyst
    │   ├── --root DIR                   Snapshot root (default: data/external_features/snapshots)
    │   ├── --registry PATH              Optional custom provenance registry
    │   ├── --force                      Re-fetch even if cache exists
    │   ├── --news-engine {mock,vader,finbert}   Sentiment engine for news adapters
    │   ├── --news-intraday-cutoff HH:MM         UTC cutoff for headlines aggregated into the snapshot
    │   ├── --ml-model-bundle DIR        Phase D: override path to a specific ML bundle (default: latest with available_from<=as_of)
    │   └── --ml-stacking-only           Phase D: run only Stage-3 stacking head on cached OOF outputs
    ├── list                             List snapshot files and registry status
    └── verify                           Validate snapshot schema and hashes

ml                                   ML-Forecast training and bundle management (Phase D)
└── train                             Train a forecast bundle (walk-forward, label-purge OOF)
    ├── --start YYYY-MM-DD
    ├── --end YYYY-MM-DD
    ├── --horizons 21,63,252           Trading-day forward windows (Stage 3 stacks them)
    ├── --models lightgbm,xgboost      Comma-separated families (requires --with ml extras)
    ├── --tickers AAA,BBB              Required unless --universe-source is set (Codex D4)
    ├── --universe-source PATH         PIT universe CSV (preferred — survivorship protection)
    ├── --output-dir DIR               Bundle root (default: data/external_features/ml/models)
    ├── --inner-train-years N
    ├── --inner-test-months N
    ├── --grid-size N                  Hyperparameter grid size per inner fold
    └── --seed N                       Manifest-pinned seed

Registered analyst datasets (Phase B / B+):
  - mock_analyst              synthetic test data; no --tickers required
  - yahoo_analyst_current     yfinance, current-only (hard-fail on hist as_of)
  - finnhub_analyst_current   Finnhub price_target + trends; FINNHUB_API_KEY required
  - finnhub_analyst_actions   Finnhub upgrade/downgrade events, PIT-safe aggregation
  - finnhub_analyst_trends    Finnhub recommendation_trends, PIT per period (~4M depth on free tier)
  - synthetic_analyst_pit     deterministic from prices; explicit --tickers required

Registered news datasets (Phase C):
  - yahoo_news                yfinance .news (current-only; hard-fail on hist as_of)
  - finnhub_news              Finnhub /company-news (~1y history; release_date=as_of)
  - newsapi_news              NewsAPI /everything (~1 month; 24h delay; 100 req/day)
  - synthetic_news_pit        deterministic from ticker+date; explicit --tickers required

Registered ml-forecast datasets (Phase D):
  - lightgbm_forecast         inference-only; resolves latest bundle with available_from <= as_of
  - xgboost_forecast          same; xgboost-trained Stage 1/2; ensemble pairs with lightgbm
  - ml_forecast_ensemble      stacked mean of lightgbm + xgboost bundles
  - synthetic_ml_forecast     deterministic from ticker+date; no model bundle required

External features flags available on run, compare, sweep, optimize, batch-optimize, signals:
    --external-features-enable
    --external-features-dataset NAME
    --external-features-provenance-mode {off,warn,strict}
    --external-features-root DIR
    --news-dataset NAME           Phase C: dataset id with news_sentiment_score
    --news-score-weight FLOAT     Phase C: blend news_score into meta-decision live_score
    --ml-dataset NAME             Phase D: dataset id with ml_forecast_score
    --ml-score-weight FLOAT       Phase D: blend ml_score into meta-decision live_score
    --cross-product-require       Phase E1: enforce cross-product consensus gate
    --cross-product-threshold X   Phase E1: override profile-default threshold
    --external-features-registry PATH

Meta-decision analyst integration (Phase B):
    REGIME_TOLERANCE_DEFAULTS["<profile>"]["analyst_score_weight"] = 0.0   # default off
    run_meta_decision(... external_features_provider=..., analyst_score_weight=0.25,
                          analyst_datasets=("finnhub_analyst_actions",),
                          allow_synthetic_analyst_evidence=False)
    -> analyst component only contributes when assess_analyst_evidence() PASSes;
       FAIL/MISSING/STALE silently set effective_weight=0 and never abort the run.

Phase E commands:
    backtest live plan --signals-report PATH --broker {dry_run,
        ibkr_basket_csv, alpaca_paper_preview, trade_republic_brief,
        maxblue_brief} [--portfolio PATH] [--new-run TOKEN]
    backtest live status [--since YYYY-MM-DD] [--log-path PATH]
    backtest live reconcile --broker NAME --portfolio PATH

Phase E vocabulary contract (Codex R2.1):
    - NO `submit_order`/`placeOrder`/`cancel_order` calls in
      src/backtest/live/ runtime code.
    - NO `--live-execute` / `--execute` CLI flags. Phase F.
    - All OrderPlanReceipts in Phase E have `plan_only=True`
      (Codex R3.2).

Cross-Product-Evidence (Phase E1):
    REGIME_TOLERANCE_DEFAULTS["<profile>"]["cross_product_consensus_threshold"]
    = 0.7 (defensiv) / 0.5 (ausgewogen) / 0.3 (aggressiv)
    run_meta_decision(... cross_product_require=True,
                          cross_product_threshold=None)
    -> consensus[current_regime_bucket] >= threshold required;
       live_scores unangetastet.

Meta-decision ML integration (Phase D):
    REGIME_TOLERANCE_DEFAULTS["<profile>"]["ml_score_weight"] = 0.0   # default off
    run_meta_decision(... external_features_provider=..., ml_score_weight=0.2,
                          ml_datasets=("lightgbm_forecast",),
                          allow_synthetic_ml_evidence=False,
                          ml_require_conditioned_evidence=False)
    -> ML component only contributes when assess_ml_evidence() PASSes; snapshots
       whose ml_available_from_ordinal > release_date.toordinal() are filtered
       in the evidence engine (Codex D14/D18). Multi-Weight constraint:
       w_analyst + w_news + w_ml must be < 1; ValueError otherwise.
```

</details>

---

## Creating Your Own Strategy

### Generate Template

```bash
backtest new "My Momentum"
```

Creates `strategies/my_momentum.py`:

```python
from datetime import date
import pandas as pd
from backtest.strategy import Strategy, Allocation

class MyMomentum(Strategy):
    name = "My Momentum"
    assets = ["SPY", "BND"]  # Assets needed for this strategy
    rebalance_frequency = "monthly"

    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.params = {"lookback": lookback}

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        # Your strategy logic here
        # data: DataFrame with price history, columns = tickers

        spy_momentum = data["SPY"].iloc[-1] / data["SPY"].iloc[-self.lookback] - 1

        if spy_momentum > 0:
            return Allocation({"SPY": 1.0})
        else:
            return Allocation({"BND": 1.0})

# Required: instantiate for CLI
strategy = MyMomentum(lookback=252)
```

### Using PIT Universe in Custom Strategy

```python
from backtest.universe import CsvPITUniverseProvider

class MyPITStrategy(Strategy):
    def __init__(self):
        self._universe_provider = CsvPITUniverseProvider(
            path="data/universes/sp500_constituents.csv",
            date_col="as_of",
            ticker_col="ticker",
        )

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        # Get universe AS OF the current backtest date (no lookahead)
        snapshot = self._universe_provider.snapshot(current_date)
        universe = snapshot.tickers  # Only stocks that were in index at this date

        # Your logic using the PIT universe...
```

---

## German Tax Model

The framework implements German capital gains taxation (Abgeltungssteuer):

| Feature | Value |
|---------|-------|
| **Tax Rate** | 26.375% (25% + 5.5% Soli) |
| **Teilfreistellung** | 30% tax-free for equity funds |
| **Freistellungsauftrag** | 1,000 EUR (single) / 2,000 EUR (joint) |
| **Cost Basis** | FIFO (First In, First Out) |

### Loss Carry-Forward (Verlustvortrag)

Two separate loss pots per §20 Abs. 6 EStG:

| Loss Pot | Applies To | Can Offset |
|----------|------------|------------|
| **Aktienverlusttopf** | Individual stocks | Only equity gains |
| **Allgemeiner Verlusttopf** | ETFs, Funds | Any gains |

### CLI Tax Options

```bash
# Default: German tax enabled with 1000 EUR exemption
backtest run strategies/dual_momentum.py

# Joint filing (2000 EUR exemption)
backtest run strategies/dual_momentum.py --tax-exemption 2000

# Disable taxes
backtest run strategies/dual_momentum.py --no-tax

# Include unrealized gains (fair Buy&Hold comparison)
backtest run strategies/buy_and_hold.py --liquidate-at-end
```

---

## Dividend Reinvestment (DRIP)

The `--drip` flag enables automatic dividend reinvestment:

```bash
# Run with dividend reinvestment
backtest run strategies/buy_and_hold.py --drip

# Compare strategies with DRIP enabled
backtest compare strategies/*.py --drip --liquidate-at-end
```

**How DRIP works:**
1. Dividends are loaded from Yahoo Finance when `--drip` is enabled
2. When a dividend is paid, tax is deducted (Abgeltungssteuer)
3. The net amount is automatically reinvested in the paying stock
4. Fractional shares are supported

**DRIP is available on all commands:** `run`, `compare`, `sweep`, `optimize`, `batch-optimize`

---

## Point-in-Time (PIT) Constituent Data

### Why PIT Matters

Using today's index constituents for historical backtests introduces **survivorship bias**:
- Failed companies are excluded (they went bankrupt)
- Today's winners are selected based on hindsight

### Available PIT Data Sources

| Index | Source | Data Range | Command |
|-------|--------|------------|---------|
| **S&P 500** | Wikipedia/GitHub | 1996 - present | `python scripts/download_pit_data.py --index sp500` |

> The bundled `data/universes/*_constituents.csv` files are derived from
> Wikipedia index-change articles and carry a CC BY-SA attribution/share-alike
> obligation — see `NOTICE.md`. Regenerating them with the command above is the
> cleaner route if you plan to redistribute.
| **NASDAQ 100** | n100tickers | 2015 - present | `python scripts/download_pit_data.py --index nasdaq100` |

### Generate Constituent Data

```bash
# Using CLI (experimental N-PORT source)
poetry run backtest-constituents download SPY --start 2019-01-01 --source nport
poetry run backtest-constituents generate SPY --start 2019-01-01 --source nport

# If you're not using Poetry, set PYTHONPATH for src-layout projects:
PYTHONPATH=src python -m backtest.constituents download SPY --start 2019-01-01 --source nport
PYTHONPATH=src python -m backtest.constituents generate SPY --start 2019-01-01 --source nport

# Using scripts (recommended)
python scripts/download_pit_data.py --index sp500
python scripts/download_pit_data.py --index nasdaq100
```

### Universe Validation

The framework blocks non-PIT universes for historical backtests:

```bash
# This FAILS (protects you from bias):
backtest run strategies/momentum_topn_yahoo_largecap.py --start 2013-01-01
# Error: Universe look-ahead detected!

# Options to proceed:
# 1. Use PIT strategy
backtest run strategies/momentum_topn_pit_sp500.py --start 2013-01-01

# 2. Explicit bypass (biased results)
backtest run strategies/momentum_topn_yahoo_largecap.py --start 2013-01-01 --allow-universe-lookahead
```

---

## Data Integrity

A backtest — and a live signal far more so — is only as trustworthy as its price series. The failure
mode that matters is not a crash. It is a **stale price silently passing as today's close**, which
produces a confident, wrong answer instead of an error.

### The freshness invariant

A price is a triple: **value, session date, source**. The session date must be *evidenced by the
source* — never inferred from `today`, from the requested date window, or from a page timestamp.

That makes one very common idiom unsafe: `align="ffill"` followed by `iloc[-1]`. Forward-fill
silently re-dates every stale value onto the newest index date, so the series can no longer tell you
that it is old. `backtest.freshness` therefore reads real prints per ticker *without* forward-fill,
and any verdict that cannot prove freshness reports **"no data"** instead of **"hold"**.

The distinction is not academic: a stop-loss check run against a silently-stale quote reports a
comfortable buffer on a position that has actually already breached its stop.

### Cache integrity

`backtest.cache_integrity` (`check_cache`, `manifest_problems`) validates the on-disk `data/` cache
manifest offline, independent of loader behaviour. A manifest whose recorded end date runs ahead of
the data actually present can skew a live signal while every individual component still looks
healthy.

### Total-return prices

When dividends are loaded, `PriceData.prices` is **unadjusted** — split-adjusted, but *not*
dividend-adjusted. Using it alone materially understates distributing instruments.
`total_return_prices()` reinvests distributions at the ex-date and is the canonical path for
research returns:

```python
data = loader.load(tickers, start, end, load_dividends=True)

tr = data.total_return_prices()   # distributions reinvested at ex-date
px = data.prices                  # unadjusted: price-only, understates payers
```

---

## Python API

### Basic Usage

```python
from backtest import DataLoader, Backtester, BacktestConfig
from strategies.dual_momentum import DualMomentum

# Load data
data = DataLoader.yahoo(
    tickers=["SPY", "EFA", "BND"],
    start="2010-01-01",
    currency="EUR"
)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000,
    costs_pct=0.001,
    tax_enabled=True,
    tax_exemption_amount=1000,
)

# Run backtest
strategy = DualMomentum()
result = Backtester(strategy, data, config).run()

# View results
print(result.summary())
result.to_html("report.html")
```

### Using RunConfig

```python
from backtest.config.run_config import RunConfig, TaxConfig, CostConfig

config = RunConfig(
    start_date="2010-01-01",
    end_date="2024-01-01",
    initial_capital=10_000,
    rebalance_frequency="monthly",
    costs=CostConfig.medium(),      # .low(), .high(), .zero()
    tax=TaxConfig.german_single(),  # .german_joint(), .disabled()
)
```

### Tax Model Direct Usage

```python
from datetime import date
from backtest.tax import GermanTaxModel

model = GermanTaxModel(exemption_amount=1000.0)

# Record purchase
model.record_purchase(
    ticker="SPY",
    shares=10,
    price_per_share=100.0,
    purchase_date=date(2024, 1, 1),
    instrument_class="general",  # ETF
)

# Apply sale
result = model.apply_sale(
    ticker="SPY",
    shares_sold=10,
    sale_price=120.0,
    sale_date=date(2024, 12, 1),
    instrument_class="general",
)

print(f"Tax due: {result.tax_due:.2f}")
```

---

## Architecture & Technical Details

<details>
<summary>Click to expand architecture details</summary>

### Project Structure

```
kontor/
├── src/backtest/
│   ├── cli.py              # CLI (argparse)
│   ├── strategy.py         # Strategy ABC, Allocation
│   ├── backtester.py       # Backtester, Portfolio, Trade
│   ├── data.py             # DataLoader, CurrencyConverter
│   ├── metrics.py          # MetricsCalculator
│   ├── sweep.py            # Rolling window analysis
│   ├── universe.py         # Universe providers + PIT guardrail
│   ├── tax/
│   │   └── de_tax_model.py # German tax model
│   ├── constituents/       # PIT constituent data generation
│   │   ├── cli.py          # constituents CLI
│   │   ├── nport.py        # N-PORT SEC filings parser
│   │   └── wikipedia.py    # Wikipedia S&P 500 scraper
│   └── web/                # Web UI (FastAPI)
│       ├── app.py          # FastAPI application
│       ├── routes/         # API and page routes
│       │   ├── api.py      # REST API endpoints
│       │   └── pages.py    # HTML page routes
│       ├── services/       # Business logic
│       │   ├── strategies.py  # Strategy discovery
│       │   └── charts.py   # Plotly chart generation
│       ├── models/         # Pydantic schemas
│       └── templates/      # Jinja2 HTML templates
│
├── strategies/             # Strategy implementations
├── scripts/                # Utility scripts
├── data/
│   ├── <TICKER>.csv        # Price cache: one file per ticker (range-independent)
│   ├── _cache_manifest.json # Tracks the date range each cache file covers
│   └── universes/          # PIT constituent CSVs
└── tests/                  # Unit tests
```

### Daily Data for All Frequencies

All strategies receive daily price data regardless of rebalance frequency:

```
Rebalance Frequency    Data passed to signal()    "126 days" means
─────────────────────────────────────────────────────────────────
daily                  Daily prices               126 trading days
weekly                 Daily prices               126 trading days
monthly                Daily prices               126 trading days
```

### Execution Order

The backtester uses **SELL → Tax → BUY** execution:

1. SELL trades executed first
2. Tax calculated and deducted from cash
3. BUY trades executed with remaining cash

### Metrics Calculated

| Category | Metrics |
|----------|---------|
| **Returns** | Total Return, CAGR |
| **Risk** | Volatility, Max Drawdown, Drawdown Duration |
| **Risk-Adjusted** | Sharpe, Sortino, Calmar Ratio |
| **Trading** | Number of Trades, Turnover, Total Costs |
| **Tax** | Total Tax, Effective Rate, Tax Drag |

### Validation

The framework includes comprehensive validation:

```python
from backtest import BacktestConfig

# Strict mode: raise on validation errors
config = BacktestConfig(
    validate=True,
    strict_validation=True,
)
```

**Validation checks:**
- No negative/NaN prices
- No extreme returns (>50% daily)
- Tax invariants (gross >= net_realized >= net_liquidation)
- Config sanity (positive capital, reasonable costs)

</details>

---

## Running Tests

```bash
poetry run pytest                          # All tests
poetry run pytest --cov=backtest           # With coverage
poetry run pytest tests/test_tax.py -v     # Specific test
poetry run pytest tests/test_invariants.py # Tax invariants
```

---

## License

MIT — see `LICENSE`.

The MIT licence covers the code. Some bundled datasets come from third parties
under their own terms (the point-in-time index constituents are derived from
Wikipedia, CC BY-SA; the total-return test fixtures were captured from Yahoo
Finance). **See `NOTICE.md`** for the per-dataset breakdown before reusing any
of the data.

## Contributing

Kontor is a **published mirror of a private research repository**, not a repo
that is developed here. It is maintained as a single squashed commit that is
rewritten on every batch, so a pull request cannot be merged in the usual way and
would be overwritten by the next publication.

**Issues and discussion are very welcome** — bug reports, questions about the tax
model, and pointers to something that is plainly wrong all land where they should.
If you want to build on the code, fork it; that works exactly as you would expect.

