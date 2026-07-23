from datetime import date, datetime

from backtest.tradegate import parse_tradegate_close


def _page(day, clock="22:00:00"):
    return '<span id="rt_datum">%s</span><span id="rt_zeit">%s</span>' % (day, clock)


def test_close_belongs_to_the_session_before_the_page_date():
    """``close`` is the delta reference of the displayed session, i.e. its PREDECESSOR."""
    close = parse_tradegate_close(_page("15.07.2026"), {"close": 462.30}, today=date(2026, 7, 16))

    assert close is not None
    assert close.price_eur == 462.30
    assert close.as_of == datetime(2026, 7, 14, 22, 0, 0)


def test_parse_tradegate_close_maps_current_page_to_prior_completed_session():
    close = parse_tradegate_close(_page("16.07.2026", "07:20:00"), {"close": 480.0},
                                  today=date(2026, 7, 16))

    assert close is not None
    assert close.as_of == datetime(2026, 7, 15, 22, 0, 0)


def test_completed_page_close_is_not_stamped_as_that_pages_session():
    """Regression: a real payload from a live production run.

    The page showed the next day's rt_datum at 22:00, but `close` still
    carried the previous day's closing price. The old implementation
    stamped the stale price onto the wrongly dated session, which papered
    over a stop breach.
    """
    close = parse_tradegate_close(_page("16.07.2026"), {"close": 790.6}, today=date(2026, 7, 17))

    assert close is not None
    assert close.price_eur == 790.6
    assert close.as_of.date() == date(2026, 7, 15)
    assert close.as_of.date() != date(2026, 7, 16)


def test_monday_page_maps_back_across_the_weekend():
    close = parse_tradegate_close(_page("20.07.2026", "09:05:00"), {"close": 500.0},
                                  today=date(2026, 7, 20))

    assert close is not None
    assert close.as_of == datetime(2026, 7, 17, 22, 0, 0)


def test_rejects_close_that_would_land_on_or_after_today():
    """A 'completed' close for today/tomorrow cannot exist."""
    assert parse_tradegate_close(_page("18.07.2026"), {"close": 500.0},
                                 today=date(2026, 7, 17)) is None


def test_parse_tradegate_close_rejects_incomplete_page():
    assert parse_tradegate_close('<strong id="last">-</strong>', {"close": 462.30}) is None


def test_parse_tradegate_close_rejects_placeholder_close():
    assert parse_tradegate_close(_page("15.07.2026"), {"close": "./."}) is None


def test_parses_german_decimal_close():
    close = parse_tradegate_close(_page("15.07.2026"), {"close": "1.110,00"},
                                  today=date(2026, 7, 16))

    assert close is not None
    assert close.price_eur == 1110.00
