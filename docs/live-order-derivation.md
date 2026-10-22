# Live-Order Derivation

Kontor closes the loop between *research* and *action*: it turns a point-in-time
strategy signal into a concrete, broker-specific **order plan** — while keeping a hard line
between planning and execution. The code lives in
[`src/backtest/live/`](../src/backtest/live/).

## Plan-only by design

**The system emits broker-ready order plans. It does not place real broker orders.** This is a
deliberate safety boundary, enforced in the code's vocabulary: the live runtime has no
`execute`, `submit`, `place_order`, or `--live-execute` surface. The abstraction is
`ExecutionPlanAdapter.emit_order_plan(...)`, and the CLI subcommand is `backtest live plan`
(never `live execute`).

Shipped adapters:

| Adapter | Real order submission? | Output |
|---|---|---|
| `dry_run` | no | JSON order plan written under `results/`. |
| `ibkr_basket_csv` | no | Interactive Brokers BasketTrader CSV (no `ib_insync` dependency, no API call). |

Real-money submission is intentionally out of scope for this repository.

## Broker awareness

Two things make an order plan *broker-specific*:

1. **Instrument mapping** — a broker can only trade certain products. The mappings in
   [`data/live/instrument_mapping/`](../data/live/instrument_mapping/) (`trade_republic.csv`,
   `maxblue.csv`) resolve a strategy ticker to the WKN/ISIN the broker actually lists. For
   example, Maxblue cannot trade `3SEM.L`, so a Maxblue plan routes the semiconductor sleeve via
   `A2QC5J` / `VVSM.DE`, while Trade Republic uses `3SEM.L` directly.
2. **Broker briefs** — `trade_republic_brief` and `maxblue_brief` render the plan in a form
   suited to each broker's manual order entry.

## Typical flow

```bash
# 1. Generate today's point-in-time signals for your strategy
backtest signals strategies/levered_etf_momentum_sticky.py --date today

# 2. Derive a plan-only order sheet for a broker (dry-run)
backtest live plan --signals-report results/signals_report.json \
  --broker dry_run --portfolio data/manual/portfolio.json
```

The strategy signal always remains the master decision; the live layer only translates it into
what you would enter at the broker. Every plan is auditable via the order-plan receipt log
([`order_plan_log.py`](../src/backtest/live/order_plan_log.py)).

> Order plans are decision support, **not investment advice**, and never an automatic instruction
> to trade.
