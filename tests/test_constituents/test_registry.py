"""Tests for index registry."""

import pytest
from datetime import date

from backtest.constituents.registry import (
    IndexRegistry,
    INDEX_CONFIGS,
    INDEX_ALIASES,
)
from backtest.constituents.models import DataQuality


class TestIndexRegistry:
    """Tests for IndexRegistry."""

    def test_list_indices(self):
        """List all supported indices."""
        indices = IndexRegistry.list_indices()
        assert "SP500" in indices
        assert "NDX" in indices
        assert "R1000" in indices
        assert "R2000" in indices
        assert "EAFE" in indices
        assert "SX5E" in indices
        assert "UKX" in indices
        assert "N225" in indices

    def test_normalize_index_id(self):
        """Normalize various index ID formats."""
        assert IndexRegistry.normalize_index_id("SP500") == "SP500"
        assert IndexRegistry.normalize_index_id("sp500") == "SP500"
        assert IndexRegistry.normalize_index_id("SPX") == "SP500"
        assert IndexRegistry.normalize_index_id("SPY") == "SP500"
        assert IndexRegistry.normalize_index_id("NASDAQ100") == "NDX"
        assert IndexRegistry.normalize_index_id("NASDAQ-100") == "NDX"
        assert IndexRegistry.normalize_index_id("QQQ") == "NDX"
        assert IndexRegistry.normalize_index_id("FTSE100") == "UKX"
        assert IndexRegistry.normalize_index_id("NIKKEI225") == "N225"

    def test_get_config(self):
        """Get configuration for an index."""
        config = IndexRegistry.get_config("SP500")
        assert config is not None
        assert config.index_id == "SP500"
        assert config.name == "S&P 500"
        assert config.region == "US"
        assert config.currency == "USD"
        assert config.target_count == 500

    def test_get_config_alias(self):
        """Get config using alias."""
        config = IndexRegistry.get_config("SPX")
        assert config is not None
        assert config.index_id == "SP500"

    def test_get_config_unknown(self):
        """Unknown index returns None."""
        config = IndexRegistry.get_config("UNKNOWN_INDEX")
        assert config is None

    def test_list_indices_by_region(self):
        """List indices for a region."""
        us_indices = IndexRegistry.list_indices_by_region("US")
        assert "SP500" in us_indices
        assert "NDX" in us_indices
        assert "R1000" in us_indices
        assert "R2000" in us_indices

        eu_indices = IndexRegistry.list_indices_by_region("EU")
        assert "SX5E" in eu_indices

        uk_indices = IndexRegistry.list_indices_by_region("UK")
        assert "UKX" in uk_indices

        jp_indices = IndexRegistry.list_indices_by_region("JP")
        assert "N225" in jp_indices

    def test_get_provider_unknown(self):
        """Unknown index raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IndexRegistry.get_provider("UNKNOWN_INDEX")
        assert "Unknown index" in str(exc_info.value)

    def test_get_summary(self):
        """Get summary of all indices."""
        summary = IndexRegistry.get_summary()

        assert "total_indices" in summary
        assert summary["total_indices"] > 0

        assert "by_region" in summary
        assert "US" in summary["by_region"]

        assert "by_quality" in summary

        assert "indices" in summary
        assert len(summary["indices"]) == summary["total_indices"]


class TestIndexConfigs:
    """Tests for INDEX_CONFIGS."""

    def test_all_configs_have_required_fields(self):
        """All configs have required fields."""
        for index_id, config in INDEX_CONFIGS.items():
            assert config.index_id == index_id
            assert config.name
            assert config.description
            assert config.region
            assert config.currency
            assert config.default_quality in DataQuality
            assert isinstance(config.available_sources, list)

    def test_sector_etfs_configured(self):
        """S&P 500 sector ETFs are configured."""
        sectors = [
            "SP500_ENERGY",
            "SP500_TECH",
            "SP500_FINANCIALS",
            "SP500_HEALTHCARE",
            "SP500_INDUSTRIALS",
            "SP500_DISCRETIONARY",
            "SP500_STAPLES",
            "SP500_UTILITIES",
            "SP500_MATERIALS",
            "SP500_REALESTATE",
            "SP500_COMMUNICATION",
        ]
        for sector in sectors:
            assert sector in INDEX_CONFIGS


class TestIndexAliases:
    """Tests for INDEX_ALIASES."""

    def test_etf_aliases(self):
        """ETF tickers map to correct indices."""
        assert INDEX_ALIASES["SPY"] == "SP500"
        assert INDEX_ALIASES["QQQ"] == "NDX"
        assert INDEX_ALIASES["IWM"] == "R2000"
        assert INDEX_ALIASES["IWB"] == "R1000"
        assert INDEX_ALIASES["EFA"] == "EAFE"
        assert INDEX_ALIASES["FEZ"] == "SX5E"

    def test_sector_etf_aliases(self):
        """Sector ETF tickers map to correct sector indices."""
        assert INDEX_ALIASES["XLE"] == "SP500_ENERGY"
        assert INDEX_ALIASES["XLK"] == "SP500_TECH"
        assert INDEX_ALIASES["XLF"] == "SP500_FINANCIALS"
