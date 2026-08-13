"""Tests for the assets module."""

import pytest

from backtest.assets import (
    Asset,
    ASSET_REGISTRY,
    detect_currency,
    resolve_asset,
    get_yahoo_ticker,
    get_benchmark_ticker,
)


class TestAsset:
    """Tests for the Asset dataclass."""

    def test_asset_creation(self):
        """Test creating an asset."""
        asset = Asset(
            wkn="A0AET0",
            name="SPDR S&P 500",
            yahoo_ticker="SPY",
            currency="USD",
            asset_class="equity",
            teilfreistellung=0.30
        )
        assert asset.wkn == "A0AET0"
        assert asset.name == "SPDR S&P 500"
        assert asset.yahoo_ticker == "SPY"
        assert asset.currency == "USD"

    def test_asset_frozen(self):
        """Test that Asset is frozen (immutable)."""
        asset = Asset("A0AET0", "Test", "SPY", "USD", "equity", 0.30)
        with pytest.raises(Exception):  # FrozenInstanceError
            asset.wkn = "NEW"


class TestAssetRegistry:
    """Tests for the asset registry."""

    def test_registry_not_empty(self):
        """Test that registry contains assets."""
        assert len(ASSET_REGISTRY) > 0

    def test_spy_in_registry(self):
        """Test that SPY is in the registry."""
        assert "A0AET0" in ASSET_REGISTRY
        spy = ASSET_REGISTRY["A0AET0"]
        assert spy.yahoo_ticker == "SPY"

    def test_registry_has_required_assets(self):
        """Test that common assets are present."""
        # Check some commonly used assets
        assert any(a.yahoo_ticker == "SPY" for a in ASSET_REGISTRY.values())
        assert any(a.yahoo_ticker == "BND" for a in ASSET_REGISTRY.values())


class TestDetectCurrency:
    """Tests for currency detection."""

    def test_us_ticker(self):
        """Test US ticker (no suffix)."""
        assert detect_currency("SPY") == "USD"
        assert detect_currency("AAPL") == "USD"

    def test_german_ticker(self):
        """Test German ticker."""
        assert detect_currency("EXS1.DE") == "EUR"
        assert detect_currency("VWCE.DE") == "EUR"

    def test_frankfurt_ticker(self):
        """Test Frankfurt ticker."""
        assert detect_currency("SAP.F") == "EUR"

    def test_london_ticker(self):
        """Test London ticker."""
        assert detect_currency("HSBA.L") == "GBP"

    def test_london_usd_line_overrides_suffix(self):
        """LSE ETPs quoted in USD must not be read as GBP (1.34x error)."""
        assert detect_currency("3SEM.L") == "USD"
        assert detect_currency("QQQ3.L") == "USD"

    def test_london_pence_line_overrides_suffix(self):
        """LSE pence line must not be read as GBP (100x error)."""
        assert detect_currency("3LUS.L") == "GBp"

    def test_override_does_not_leak_to_other_lse_tickers(self):
        """Only the listed ETPs are overridden; plain LSE stays GBP."""
        assert detect_currency("VOD.L") == "GBP"


class TestResolveAsset:
    """Tests for asset resolution."""

    def test_resolve_by_wkn(self):
        """Test resolving by WKN."""
        asset = resolve_asset("A0AET0")
        assert asset.yahoo_ticker == "SPY"

    def test_resolve_by_ticker(self):
        """Test resolving by Yahoo ticker."""
        asset = resolve_asset("SPY")
        assert asset.wkn == "A0AET0"

    def test_resolve_by_name(self):
        """Test resolving by partial name."""
        asset = resolve_asset("S&P 500")
        assert asset.yahoo_ticker == "SPY"

    def test_resolve_unknown_raises(self):
        """Test that unknown asset raises error."""
        with pytest.raises(ValueError):
            resolve_asset("UNKNOWN_ASSET_XYZ")


class TestGetYahooTicker:
    """Tests for Yahoo ticker lookup."""

    def test_known_wkn(self):
        """Test with known WKN."""
        ticker = get_yahoo_ticker("A0AET0")
        assert ticker == "SPY"

    def test_maxblue_levered_wkns(self):
        """Test broker-specific levered WKN mappings used by Maxblue strategy."""
        assert get_yahoo_ticker("A3GWDS") == "3XFE.L"
        assert get_yahoo_ticker("A3GWD0") == "3XEE.L"

    def test_known_name(self):
        """Test with known name."""
        ticker = get_yahoo_ticker("SPDR S&P 500")
        assert ticker == "SPY"

    def test_unknown_passthrough(self):
        """Test that unknown identifier is passed through."""
        ticker = get_yahoo_ticker("CUSTOM_TICKER")
        assert ticker == "CUSTOM_TICKER"


class TestGetBenchmarkTicker:
    """Tests for benchmark ticker lookup."""

    def test_sp500(self):
        """Test S&P 500 benchmark."""
        assert get_benchmark_ticker("S&P 500") == "SPY"
        assert get_benchmark_ticker("SP500") == "SPY"

    def test_msci_world(self):
        """Test MSCI World benchmark."""
        assert get_benchmark_ticker("MSCI World") == "URTH"

    def test_unknown_raises(self):
        """Test that unknown benchmark raises error."""
        with pytest.raises(ValueError):
            get_benchmark_ticker("Unknown Benchmark")
