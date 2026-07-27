"""Shared dependencies for the web frontend.

This module contains dependencies that are shared across routes
to avoid circular imports.
"""

from fastapi.templating import Jinja2Templates

from backtest.web.config import TEMPLATES_DIR

# Templates instance for use in routes
templates = Jinja2Templates(directory=TEMPLATES_DIR)
