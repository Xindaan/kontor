"""Parser for completed Tradegate EUR closes shown on security pages."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import re
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class TradegateClose:
    price_eur: float
    as_of: datetime


def _field(html: str, field_id: str) -> Optional[str]:
    match = re.search(
        r'id=["\']%s["\'][^>]*>\s*([^<]+?)\s*<' % re.escape(field_id),
        html,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _previous_weekday(day: date) -> date:
    candidate = day - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def parse_tradegate_close(
    html: str, refresh: Mapping[str, Any], today: Optional[date] = None
) -> Optional[TradegateClose]:
    """Parse Tradegate's reference close and attribute it to the session it belongs to.

    ``refresh.php`` exposes ``close`` as the REFERENCE price for the delta of the
    session displayed on the page (``rt_datum``). It is therefore the close of the
    session BEFORE ``rt_datum`` -- never of ``rt_datum`` itself.

    Verified against live payloads: the page showed
    ``rt_datum`` 16.07.2026 22:00 while ``close`` still carried the 15.07. closes
    (MU 800.00 / AMD 470.00 EUR); the live bid/ask on the same payload already
    tracked the 16.07. levels (744.10 / 437.35). The earlier implementation keyed
    the mapping off ``today`` instead of ``rt_datum``, so a completed-session page
    passed through unmapped and stamped the previous session's close as the current
    one -- a silently stale price on the stop check.
    """
    day = _field(html, "rt_datum")
    clock = _field(html, "rt_zeit")
    close = refresh.get("close")
    if close is None or not day or not clock:
        return None
    try:
        if isinstance(close, (int, float)) and not isinstance(close, bool):
            price = float(close)
        else:
            normalized = str(close).replace("\xa0", "").replace("&nbsp;", "").strip()
            price = float(normalized.replace(".", "").replace(",", "."))
        page_stamp = datetime.strptime("%s %s" % (day.strip(), clock.strip()), "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return None
    if price <= 0:
        return None
    as_of = datetime.combine(_previous_weekday(page_stamp.date()), time(22, 0, 0))
    today = today or datetime.today().date()
    if as_of.date() >= today:
        return None
    return TradegateClose(price_eur=price, as_of=as_of)
