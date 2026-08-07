"""FastAPI entrypoint for serving the frozen ReturnGuard A3 artifact."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import Settings
from backend.errors import (
    ExplanationFailedError,
    ExplanationsDisabledError,
    InferenceFailedError,
    ModelUnavailableError,
)
from backend.model_service import ModelService
from backend.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)

LOGGER = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unavailable")


def _safe_error(request: Request, status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"request_id": _request_id(request), "code": code},
    )


def create_app(
    settings: Optional[Settings] = None,
    service_factory: Callable[[Settings], ModelService] = ModelService.load,
) -> FastAPI:
    """Build an app whose frozen model is loaded exactly once in lifespan."""
    resolved_settings = settings or Settings.from_environment()
    logging.basicConfig(level=getattr(logging, resolved_settings.log_level, logging.INFO))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = None
        app.state.model_load_failed = False
        try:
            app.state.model_service = service_factory(resolved_settings)
        except Exception:
            app.state.model_load_failed = True
            LOGGER.exception("event=service_startup_failed error_code=model_unavailable")
        yield

    app = FastAPI(
        title="ReturnGuard frozen A3 inference API",
        version="1.0.0",
        summary="Dataset-conditional return-risk scoring for one purchase event.",
        description=(
            "Serves the governed `returnguard-a3-v1` model of record. Scores are "
            "relative risk estimates from a returner-enriched ASOS research sample, "
            "not merchant-wide calibrated probabilities. A request represents one "
            "customer × product-variant purchase event. Unknown categories are accepted "
            "and ignored by the frozen encoder; incomplete product profiles remain scoreable "
            "but have weaker evidence."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = _request_id(request)
        LOGGER.info(
            "event=http_request request_id=%s endpoint=%s status=%s latency_ms=%.2f",
            _request_id(request),
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "message": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"request_id": _request_id(request), "code": "invalid_request", "errors": errors},
        )

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
        return _safe_error(request, 503, "model_unavailable")

    @app.exception_handler(ExplanationsDisabledError)
    async def explanations_disabled_handler(request: Request, exc: ExplanationsDisabledError):
        return _safe_error(request, 503, "explanations_disabled")

    @app.exception_handler(InferenceFailedError)
    async def inference_failed_handler(request: Request, exc: InferenceFailedError):
        return _safe_error(request, 500, "inference_failed")

    @app.exception_handler(ExplanationFailedError)
    async def explanation_failed_handler(request: Request, exc: ExplanationFailedError):
        return _safe_error(request, 500, "explanation_failed")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code = exc.detail.get("code", "request_rejected") if isinstance(exc.detail, dict) else "request_rejected"
        return _safe_error(request, exc.status_code, code)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        LOGGER.exception("event=unexpected_api_error request_id=%s", _request_id(request))
        return _safe_error(request, 500, "internal_error")

    def model_service() -> ModelService:
        service = app.state.model_service
        if service is None:
            raise ModelUnavailableError("Frozen model is unavailable")
        return service

    @app.get("/health", response_model=ReadinessResponse, tags=["operations"])
    async def health() -> ReadinessResponse:
        """Process liveness only; never contacts MLflow."""
        return ReadinessResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse, tags=["operations"])
    async def ready() -> Union[ReadinessResponse, JSONResponse]:
        """Readiness requires successful one-time model and provenance validation."""
        if app.state.model_service is None:
            return JSONResponse(
                status_code=503,
                content=ReadinessResponse(status="not_ready", code="model_unavailable").model_dump(),
            )
        return ReadinessResponse(status="ready")

    @app.get("/model", response_model=ModelInfoResponse, tags=["model"])
    async def model() -> ModelInfoResponse:
        """Safe provenance and model-use constraints; no local paths or MLflow internals."""
        return model_service().model_info()

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["inference"],
        responses={503: {"description": "Frozen model unavailable"}},
    )
    async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
        """Score a single raw purchase event using fitted artifact preprocessing."""
        service = model_service()
        result = service.predict([payload])[0]
        LOGGER.info(
            "event=prediction_completed request_id=%s batch_size=1 customer_profile_imputed=%s "
            "product_profile_complete=%s unknown_count=%s out_of_range_count=%s",
            _request_id(request),
            result["data_context"].customer_profile_imputed,
            result["data_context"].product_profile_complete,
            len(result["data_context"].unknown_categories),
            len(result["data_context"].outside_training_range_fields),
        )
        return PredictionResponse(
            request_id=_request_id(request),
            model=service.concise_provenance(),
            methodology_note=service.methodology_note,
            **result,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    async def predict_batch(request: Request, payload: BatchPredictionRequest) -> BatchPredictionResponse:
        """Score 1–100 events in one vectorized model call, retaining request order."""
        if len(payload.events) > resolved_settings.max_batch_size:
            raise HTTPException(status_code=422, detail={"code": "batch_size_exceeded"})
        service = model_service()
        results = service.predict(payload.events)
        LOGGER.info(
            "event=batch_prediction_completed request_id=%s batch_size=%s",
            _request_id(request),
            len(results),
        )
        return BatchPredictionResponse(
            request_id=_request_id(request),
            results=results,
            model=service.concise_provenance(),
            methodology_note=service.methodology_note,
        )

    @app.post("/explain", response_model=ExplanationResponse, tags=["explanations"])
    async def explain(request: Request, payload: PredictionRequest) -> ExplanationResponse:
        """Provide grouped, non-causal Tree SHAP factors for one purchase event."""
        service = model_service()
        result = service.explain(payload)
        LOGGER.info("event=explanation_completed request_id=%s", _request_id(request))
        return ExplanationResponse(
            request_id=_request_id(request),
            model=service.concise_provenance(),
            methodology_note=service.methodology_note,
            **result,
        )

    return app


app = create_app()
