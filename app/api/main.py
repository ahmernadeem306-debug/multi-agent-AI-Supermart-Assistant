"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.documents import router as documents_router
from app.api.routes.forecast import router as forecast_router
from app.api.routes.health import router as health_router
from app.api.routes.logs import router as logs_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.query import router as query_router
from app.api.routes.rca import router as rca_router
from app.config import get_settings
from app.core.exceptions import BizAgentError
from app.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="BizAgent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BizAgentError)
    def handle_bizagent_error(request: Request, exc: BizAgentError) -> JSONResponse:
        logger.error("request_failed", error_code=exc.error_code, path=request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.error_code, "message": exc.message, "run_id": None},
        )

    @app.exception_handler(Exception)
    def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error_code": "internal_error", "message": "An unexpected error occurred.", "run_id": None},
        )

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(metrics_router)
    app.include_router(documents_router)
    app.include_router(logs_router)
    app.include_router(forecast_router)
    app.include_router(rca_router)

    return app


app = create_app()
