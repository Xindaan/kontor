"""Phase E2 — Maxblue Order-Brief (T-0369).

Analog Trade-Republic-Brief, andere Mapping-Datei + Output-Pfad.
"""

from __future__ import annotations

from backtest.live.trade_republic_brief import _BriefAdapter


class MaxblueBriefAdapter(_BriefAdapter):
    broker_label_value = "maxblue_brief"
    mapping_broker_key = "maxblue"
    output_dir = "results/live_orders/maxblue"


__all__ = ["MaxblueBriefAdapter"]
