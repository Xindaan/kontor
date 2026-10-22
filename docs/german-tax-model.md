# German Tax Model

Kontor applies German capital-gains taxation (Abgeltungssteuer) as part of the
simulation, so that after-tax results reflect what actually reaches a private investor's depot.
The model lives in [`src/backtest/tax/de_tax_model.py`](../src/backtest/tax/de_tax_model.py).

## What is modelled

| Element | Detail |
|---|---|
| **Abgeltungssteuer** | 26.375 % on realized gains (25 % capital-gains tax + 5.5 % Solidaritätszuschlag). |
| **Teilfreistellung** | 30 % of gains are tax-free for equity funds (Aktienfonds). Individual stocks receive **no** Teilfreistellung — this distinction is driven by [`src/backtest/assets.py`](../src/backtest/assets.py) (`equity_fund_map`). |
| **Freistellungsauftrag** | The Sparer-Pauschbetrag: first 1,000 EUR (single) / 2,000 EUR (joint) of gains per year are tax-free, applied **after** loss-pot netting. |
| **Loss pots (§20 Abs. 6 EStG)** | Two carry-forward pots: the **Aktienverlusttopf** (individual-stock losses, can only offset stock gains) and the **allgemeiner Verlusttopf** (ETF/fund losses, can offset any gains). Both carry forward indefinitely. |
| **Cost basis** | **FIFO** — the oldest lots are sold first, as required by German tax law. |

## Metric bases

Every backtest can report results on three bases (`--metric-basis`):

- `gross` — before any tax.
- `net_realized` — tax on gains realized by actual sells during the period.
- `net_liquidation` — as `net_realized`, plus the tax that would be due if the whole portfolio
  were liquidated on the final day. This is the most honest "what is really mine" figure and is
  the recommended basis for comparing strategies.

## Explicitly NOT modelled

The tax model deliberately excludes the following (documented in the module itself):

- **Vorabpauschale** (advance lump-sum taxation on accumulating funds).
- **Kirchensteuer** (church tax).
- **Termingeschäfte-Verlusttopf** (the separate loss pot for derivatives).

These are known limitations, not oversights. Vorabpauschale in particular can matter for
accumulating ETFs held across calendar years; treat `net_liquidation` figures as a lower bound on
lifetime tax for such holdings.

> This document describes a simulation model for research purposes. It is **not tax advice**.
> Consult a Steuerberater for your actual situation.
