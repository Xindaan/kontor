"""
Entry point for running backtest as a module.

Usage:
    python -m backtest run strategies/my_strategy.py
    python -m backtest compare strategies/*.py
    python -m backtest --help
"""

from backtest.cli import app

if __name__ == "__main__":
    app()
