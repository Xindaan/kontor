"""Configuration for the web frontend."""

from pathlib import Path

# Base paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Default strategies directory (relative to CWD)
DEFAULT_STRATEGIES_DIR = Path("strategies")

# App settings
APP_TITLE = "Kontor"
APP_DESCRIPTION = "A modular Python framework for systematic backtesting of investment strategies"
