"""FastAPI application for Kontor web frontend."""

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backtest.web.config import (
    APP_TITLE,
    APP_DESCRIPTION,
    STATIC_DIR,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Import routes here to avoid circular imports
    from backtest.web.routes import pages, api

    application = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version="0.1.0",
    )

    access_logger = logging.getLogger("uvicorn.access")

    class SweepProgressFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            return all(
                endpoint not in message
                for endpoint in (
                    "/api/v1/sweep/progress/",
                    "/api/v1/run/progress/",
                    "/api/v1/compare/progress/",
                    "/api/v1/batch-optimize/progress/",
                    "/api/v1/optimize/progress/",
                )
            )

    access_logger.addFilter(SweepProgressFilter())

    # Mount static files
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Include routers
    application.include_router(pages.router)
    application.include_router(api.router, prefix="/api/v1")

    return application


# Create the app instance
app = create_app()
