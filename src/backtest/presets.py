"""Reusable execution/risk preset profiles for CLI and UI flows."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


_PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "research": {
        "label": "Research (Fast)",
        "description": "Minimal execution constraints for fast iteration.",
        "settings": {
            "execution_lag_days": 0,
            "max_volume_participation": None,
            "min_daily_dollar_volume": 0.0,
            "liquidity_on_missing_volume": "allow",
            "max_position": None,
            "turnover_budget": None,
            "drawdown_brake_threshold": None,
            "drawdown_brake_cash_target": 1.0,
            "drawdown_brake_release": None,
        },
    },
    "realistic": {
        "label": "Realistic (Core)",
        "description": "Balanced default for realistic execution assumptions.",
        "settings": {
            "execution_lag_days": 1,
            "max_volume_participation": 0.10,
            "min_daily_dollar_volume": 250_000.0,
            "liquidity_on_missing_volume": "skip",
            "max_position": 0.35,
            "turnover_budget": 0.35,
            "drawdown_brake_threshold": None,
            "drawdown_brake_cash_target": 1.0,
            "drawdown_brake_release": None,
        },
    },
    "defensive": {
        "label": "Defensive (Strict)",
        "description": "Tighter liquidity and risk overlays for robustness stress tests.",
        "settings": {
            "execution_lag_days": 1,
            "max_volume_participation": 0.05,
            "min_daily_dollar_volume": 1_000_000.0,
            "liquidity_on_missing_volume": "skip",
            "max_position": 0.25,
            "turnover_budget": 0.20,
            "drawdown_brake_threshold": 0.15,
            "drawdown_brake_cash_target": 0.70,
            "drawdown_brake_release": 0.08,
        },
    },
}


def preset_profile_names() -> list[str]:
    """Return available preset profile names."""
    return list(_PRESET_PROFILES.keys())


def get_preset_profile(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a copy of profile settings by name."""
    if not name:
        return None
    profile = _PRESET_PROFILES.get(str(name).strip().lower())
    if profile is None:
        return None
    return copy.deepcopy(profile.get("settings", {}))


def preset_profile_labels() -> Dict[str, str]:
    """Return name -> label mapping for UIs/help text."""
    return {
        name: str(profile.get("label", name))
        for name, profile in _PRESET_PROFILES.items()
    }
