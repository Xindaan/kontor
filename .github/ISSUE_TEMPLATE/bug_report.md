---
name: Bug report
about: A backtest result, tax figure, or command that behaves incorrectly
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the incorrect behaviour.

**How to reproduce**
The exact command or steps, e.g.:

```bash
backtest run strategies/dual_momentum.py --start 2015-01-01
```

**Expected vs. actual**
What you expected, and what you got (paste the relevant output / numbers).

**Environment**
- OS:
- Python version:
- Installed via: `poetry install` / `pip install -e .`

**Notes**
Kontor is a research/decision-support tool, not investment advice. The tax
model deliberately excludes Vorabpauschale, Kirchensteuer, and the derivatives
loss pot — please check `docs/german-tax-model.md` before filing a tax-logic bug.
