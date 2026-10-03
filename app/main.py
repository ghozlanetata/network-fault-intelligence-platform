import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.auth import AuthRepository, Role, User
from app.authorization import SESSION_COOKIE, require_any_role, require_authenticated_user
from app.config import get_settings
from app.domain import FaultCause, PredictionStatus
from app.inference import ArtifactInference, InferenceFailed, InferenceUnavailable
from app.investigations import InvestigationRepository
from app.network_health import (
    NetworkHealthRepository,
    NetworkHealthRunInProgress,
    process_dataset,
)
from app.recommendations import cause_name, recommendations_for
from app.repository import HistoryRepository
from app.schemas import (
    AuthenticatedUser,
    BinaryDetection,
    Classification,
    InvestigationCreate,
    InvestigationOutcome,
    InvestigationUpdate,
    InvestigationVerification,
    LoginRequest,
    OperationalPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)

logging.basicConfig(
    level=get_settings().log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("nfi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repository = HistoryRepository(settings.database_path)
    repository.initialize()
    app.state.repository = repository
    app.state.network_health = NetworkHealthRepository(settings.database_path)
    app.state.investigations = InvestigationRepository(settings.database_path)
    auth = AuthRepository(settings.database_path, settings.session_ttl)
    auth.bootstrap(
        settings.bootstrap_username, settings.bootstrap_password, settings.bootstrap_role
    )
    app.state.auth = auth
    app.state.inference = ArtifactInference(settings.model_dir)
    logger.info("application_started inference_available=%s", app.state.inference.available)
    yield


app = FastAPI(
    title="Network Fault Intelligence API",
    description=(
        "Modernized inference API for seven KPI features, using persisted preprocessing, "
        "separate binary and multiclass models, and deterministic recommendations. "
        "The recovered dataset evaluation does not establish independent field performance."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "An internal error occurred."}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    del request
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        error = detail
    else:
        error = {"code": "request_failed", "message": str(detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request did not match the API schema.",
                "details": [
                    {
                        "location": list(error.get("loc", ())),
                        "message": error.get("msg", "Invalid value."),
                        "type": error.get("type", "value_error"),
                    }
                    for error in exc.errors()
                ],
            }
        },
    )


@app.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready", response_model=ReadinessResponse, tags=["health"])
def readiness(request: Request) -> ReadinessResponse:
    repository: HistoryRepository = request.app.state.repository
    database_ok = repository.healthy()
    inference_ok = request.app.state.inference.available
    readiness_status = (
        "ready" if database_ok and inference_ok else "degraded" if database_ok else "not_ready"
    )
    response = ReadinessResponse(
        status=readiness_status,
        database="ready" if database_ok else "unavailable",
        inference="ready" if inference_ok else "artifacts_unavailable",
    )
    if not database_ok:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "service_not_ready",
                "message": "Required application dependencies are unavailable.",
                "readiness": response.model_dump(),
            },
        )
    return response


class RecommendationRequest(BaseModel):
    cause: FaultCause


current_user = require_authenticated_user


def same_origin_request(request: Request) -> None:
    """Reject cross-origin browser writes; same-origin UI requests need no CSRF token."""
    origin = request.headers.get("origin")
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site == "cross-site":
        raise HTTPException(
            status_code=403,
            detail={"code": "csrf_failed", "message": "Cross-origin request rejected."},
        )
    if origin:
        parsed = urlsplit(origin)
        if parsed.netloc.lower() != request.headers.get("host", "").lower():
            raise HTTPException(
                status_code=403,
                detail={"code": "csrf_failed", "message": "Cross-origin request rejected."},
            )


def public_user(user: User) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user.id, username=user.username, is_active=user.is_active, role=user.role
    )


def is_platform_ml_admin(user: User) -> bool:
    return user.role is Role.PLATFORM_ML_ADMIN


def visible_investigation(request: Request, investigation_id: int, user: User) -> dict:
    item = request.app.state.investigations.get(investigation_id)
    if item is None:
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": "Investigation not found."}
        )
    if user.role is Role.OPERATOR and item["submitted_by"] != user.id:
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": "Investigation not found."}
        )
    if (
        user.role is Role.NETWORK_ADMIN
        and item["submitted_by"] != user.id
        and item["investigation_status"] in {"DRAFT", "IN_PROGRESS"}
    ):
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": "Investigation not found."}
        )
    return item


def operator_network_run(run: dict[str, object] | None) -> dict[str, object] | None:
    if run is None:
        return None
    summary = run.get("summary")
    safe_summary = None
    if isinstance(summary, dict):
        fields = (
            "total_cells_analyzed",
            "normal_count",
            "fault_count",
            "review_required_count",
            "normal_percent",
            "fault_percent",
            "review_required_percent",
            "predicted_cause_distribution",
            "located_cell_count",
        )
        safe_summary = {key: summary[key] for key in fields if key in summary}
    response = {
        key: run[key]
        for key in ("run_id", "status", "started_at", "completed_at", "record_count")
        if key in run
    }
    response["summary"] = safe_summary
    if run.get("status") == "FAILED":
        response["error"] = "Network Health processing failed."
    return response


def operator_network_results(results: dict[str, object]) -> dict[str, object]:
    safe_fields = (
        "network_health_result_id",
        "investigation_status",
        "verification_status",
        "cell",
        "latitude",
        "longitude",
        "kpis",
        "status",
        "predicted_cause",
        "recommendations",
    )
    return {key: value for key, value in results.items() if key != "items"} | {
        "items": [
            {key: item[key] for key in safe_fields if key in item}
            for item in results.get("items", [])
        ]
    }


@app.get("/openapi.json", include_in_schema=False)
def openapi_schema(
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
) -> dict[str, object]:
    return app.openapi()


@app.get("/docs", include_in_schema=False)
def swagger_docs(
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
) -> Response:
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")


@app.get("/redoc", include_in_schema=False)
def redoc_docs(
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
) -> Response:
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")


@app.post("/v1/auth/login", response_model=AuthenticatedUser, tags=["authentication"])
def login(body: LoginRequest, request: Request, response: Response) -> AuthenticatedUser:
    same_origin_request(request)
    user = request.app.state.auth.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_credentials", "message": "Username or password is incorrect."},
        )
    token = request.app.state.auth.create_session(user.id)
    settings = get_settings()
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=settings.session_ttl,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )
    return public_user(user)


@app.post("/v1/auth/logout", status_code=204, tags=["authentication"])
def logout(request: Request, response: Response) -> Response:
    same_origin_request(request)
    request.app.state.auth.delete_session(request.cookies.get(SESSION_COOKIE))
    settings = get_settings()
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
    )
    response.status_code = 204
    return response


@app.get("/v1/auth/me", response_model=AuthenticatedUser, tags=["authentication"])
def me(user: Annotated[User, Depends(current_user)]) -> AuthenticatedUser:
    return public_user(user)


@app.post("/v1/recommendations", tags=["fault analysis"])
def recommendations(
    body: RecommendationRequest,
    request: Request,
    _user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict[str, object]:
    same_origin_request(request)
    return {
        "cause": body.cause.value,
        "cause_name": cause_name(body.cause),
        "actions": recommendations_for(body.cause),
        "engine": "deterministic_rules",
    }


@app.post(
    "/v1/predictions",
    response_model=OperationalPredictionResponse | PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["fault analysis"],
    summary="Detect a network fault and classify its likely cause",
    description=(
        "Provide the seven operational KPIs. All roles receive the operational result and input "
        "KPIs. REVIEW_REQUIRED indicates inference disagreement and has no final cause. "
        "Platform/ML Admin also receives internal model outputs."
    ),
    responses={
        422: {"description": "Invalid or incomplete KPI request."},
        503: {"description": "Model or preprocessing artifacts are unavailable."},
        502: {"description": "Loaded inference artifacts failed while predicting."},
    },
)
def create_prediction(
    body: PredictionRequest,
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict[str, object]:
    same_origin_request(request)
    inference = request.app.state.inference
    try:
        result = inference.predict(body)
    except InferenceUnavailable as exc:
        logger.error("prediction_unavailable reason=%s", str(exc))
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_artifacts_unavailable",
                "message": "Prediction inference is unavailable.",
            },
        ) from exc
    except InferenceFailed as exc:
        message = str(exc) if is_platform_ml_admin(user) else "Prediction inference failed."
        raise HTTPException(
            status_code=502,
            detail={"code": "inference_failed", "message": message},
        ) from exc
    cause = FaultCause(result.cause) if result.cause is not None else None
    repository: HistoryRepository = request.app.state.repository
    prediction_id = repository.record(body, result)
    logger.info(
        "prediction_recorded prediction_id=%s model_version=%s",
        prediction_id,
        result.model_version,
    )
    kpis = {
        key: getattr(body, key)
        for key in (
            "retainability",
            "hosr",
            "rsrp",
            "rsrq",
            "sinr",
            "average_throughput",
            "distance",
        )
    }
    response = PredictionResponse(
        prediction_id=prediction_id,
        status=result.status,
        binary_detection=BinaryDetection(
            prediction=result.binary_prediction,
            threshold=result.threshold,
            fault_probability=result.fault_probability,
            normal_probability=result.normal_probability,
        ),
        classification=Classification(
            predicted_class_id=result.class_id,
            predicted_cause=FaultCause(result.predicted_cause),
            confidence=result.confidence,
            probabilities=result.class_probabilities,
        ),
        fault=result.fault,
        cause=cause,
        cause_name=result.cause_name,
        recommendations=recommendations_for(cause) if cause is not None else [],
        model_version=result.model_version,
        created_at=datetime.now(UTC),
        kpis=kpis,
    )
    if is_platform_ml_admin(user):
        return response
    return OperationalPredictionResponse(
        schema_version=response.schema_version,
        prediction_id=response.prediction_id,
        status=response.status,
        fault=response.fault,
        cause=response.cause,
        cause_name=response.cause_name,
        recommendations=response.recommendations,
        created_at=response.created_at,
        kpis=kpis,
    )


@app.get("/v1/history", tags=["history"])
def history(
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict[str, object]]:
    repository: HistoryRepository = request.app.state.repository
    items = repository.list_history(limit)
    if is_platform_ml_admin(user):
        return [item.model_dump(mode="json") for item in items]
    return [
        {
            "prediction_id": item.prediction_id,
            "cell": item.cell,
            "latitude": item.latitude,
            "longitude": item.longitude,
            "fault": item.fault,
            "cause": item.cause.value if item.cause else None,
            "date": item.date.isoformat() if item.date else None,
            "created_at": item.created_at.isoformat(),
            "status": item.status.value,
            "kpis": item.kpis,
            "inference": {
                key: item.inference[key]
                for key in ("status", "cause", "cause_name", "fault", "recommendations")
                if key in item.inference
            },
        }
        for item in items
    ]


@app.post("/v1/network-health/runs", tags=["network health"], status_code=201)
def run_network_health(
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict[str, object]:
    same_origin_request(request)
    inference = request.app.state.inference
    try:
        result = process_dataset(request.app.state.network_health, inference)
    except NetworkHealthRunInProgress as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "network_health_in_progress", "message": str(exc)},
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        message = (
            str(exc)
            if is_platform_ml_admin(user)
            else "Network Health input dataset is unavailable or invalid."
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "network_dataset_invalid",
                "message": message,
            },
        ) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_artifacts_unavailable",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("network_health_run_failed")
        raise HTTPException(
            status_code=502,
            detail={
                "code": "network_health_failed",
                "message": "Network Health inference failed; see the failed run for details.",
            },
        ) from exc
    return result if is_platform_ml_admin(user) else operator_network_run(result) or {}


@app.get("/v1/network-health/latest", tags=["network health"])
def latest_network_health(
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict[str, object]:
    run = request.app.state.network_health.latest() or {"status": "NOT_RUN"}
    return run if is_platform_ml_admin(user) else operator_network_run(run) or {"status": "NOT_RUN"}


@app.get("/v1/network-health/results", tags=["network health"])
def network_health_results(
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
    limit: Annotated[int, Query(ge=1, le=5000)] = 1200,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    cell: Annotated[str | None, Query(max_length=128)] = None,
) -> dict[str, object]:
    run = request.app.state.network_health.latest()
    if run is None or run["status"] != "COMPLETED":
        return {
            "run_id": run["run_id"] if run else None,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "items": [],
        }
    allowed = {item.value for item in PredictionStatus}
    if status_filter and status_filter not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_status", "message": "Unknown Network Health status."},
        )
    results = request.app.state.network_health.results(
        run["run_id"], limit, offset, status_filter, cell
    )
    return results if is_platform_ml_admin(user) else operator_network_results(results)


@app.post(
    "/v1/investigations",
    response_model=InvestigationOutcome,
    status_code=status.HTTP_201_CREATED,
    tags=["investigations"],
)
def create_investigation(
    body: InvestigationCreate,
    request: Request,
    user: Annotated[User, Depends(require_any_role(Role.OPERATOR))],
) -> dict:
    same_origin_request(request)
    repository: InvestigationRepository = request.app.state.investigations
    try:
        return repository.create(
            prediction_id=body.prediction_id,
            network_health_result_id=body.network_health_result_id,
            submitted_by=user.id,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=404, detail={"code": "not_found", "message": str(exc)}
        ) from exc
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409, detail={"code": "investigation_exists", "message": str(exc)}
        ) from exc


@app.get("/v1/investigations", response_model=list[InvestigationOutcome], tags=["investigations"])
def list_investigations(
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    repository: InvestigationRepository = request.app.state.investigations
    if user.role is Role.NETWORK_ADMIN:
        return repository.submitted_queue(limit)
    owner = user.id if user.role is Role.OPERATOR else None
    return repository.list(submitted_by=owner, limit=limit)


@app.get(
    "/v1/investigations/queue",
    response_model=list[InvestigationOutcome],
    tags=["investigations"],
    summary="List submitted investigations for Network Admin review",
)
def investigation_review_queue(
    request: Request,
    _user: Annotated[User, Depends(require_any_role(Role.NETWORK_ADMIN))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    return request.app.state.investigations.submitted_queue(limit)


@app.get(
    "/v1/investigations/evaluatable",
    response_model=list[InvestigationOutcome],
    tags=["investigations"],
)
def evaluatable_investigations(
    request: Request,
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[dict]:
    return request.app.state.investigations.evaluatable(limit)


@app.get(
    "/v1/ml/evaluation",
    response_model=dict[str, object],
    tags=["machine-learning"],
    summary="Evaluate predicted fault causes against verified investigations",
    description=(
        "Cause classification metrics for verified investigations only. This population does "
        "not measure overall production accuracy or binary fault detection performance."
    ),
)
def ml_evaluation_summary(
    request: Request,
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
) -> dict:
    return request.app.state.investigations.evaluation_summary()


@app.get(
    "/v1/ml/verified-outcomes",
    tags=["machine-learning"],
    summary="List operational investigations included in ML evaluation",
    description="Read-only evidence cases using the same eligibility rules as /v1/ml/evaluation.",
)
def ml_verified_outcomes(
    request: Request,
    _user: Annotated[User, Depends(require_any_role(Role.PLATFORM_ML_ADMIN))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, object]:
    repository: InvestigationRepository = request.app.state.investigations
    cases = repository.evaluatable(limit=limit)
    summary = repository.evaluation_summary()
    items = [
        {
            "investigation_id": case["investigation_id"],
            "prediction_id": case["prediction_id"],
            "network_health_result_id": case["network_health_result_id"],
            "cell": case["cell"],
            "predicted_cause": case["predicted_cause"],
            "investigation_cause": case["investigation_cause"],
            "findings": case["findings"],
            "verified_cause": case["verified_cause"],
            "verification_status": case["verification_status"],
            "verified_at": case["verified_at"],
            "model_version": case["model_version"],
            "evaluation_match": case["predicted_cause"] == case["verified_cause"],
        }
        for case in cases
    ]
    return {
        "verified_cases": repository.verified_count(),
        "eligible_cases": summary["eligible_cases"],
        "returned_cases": len(items),
        "truncated": summary["eligible_cases"] > len(items),
        "excluded_cases": summary["excluded_cases"],
        "items": items,
    }


@app.get(
    "/v1/investigations/{investigation_id}",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def get_investigation(
    investigation_id: int,
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict:
    return visible_investigation(request, investigation_id, user)


@app.get(
    "/v1/investigations/{investigation_id}/outcome",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def get_investigation_outcome(
    investigation_id: int,
    request: Request,
    user: Annotated[User, Depends(require_any_role(*Role))],
) -> dict:
    return visible_investigation(request, investigation_id, user)


@app.patch(
    "/v1/investigations/{investigation_id}",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def update_investigation(
    investigation_id: int,
    body: InvestigationUpdate,
    request: Request,
    user: Annotated[User, Depends(require_any_role(Role.OPERATOR))],
) -> dict:
    same_origin_request(request)
    repository: InvestigationRepository = request.app.state.investigations
    visible_investigation(request, investigation_id, user)
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_update", "message": "Provide investigation fields to update."},
        )
    try:
        updated = repository.update(investigation_id, fields=fields)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_update", "message": str(exc)}
        ) from exc
    if updated is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "investigation_locked",
                "message": "Submitted investigations cannot be changed by the operator.",
            },
        )
    return updated


@app.post(
    "/v1/investigations/{investigation_id}/submit",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def submit_investigation(
    investigation_id: int,
    request: Request,
    user: Annotated[User, Depends(require_any_role(Role.OPERATOR))],
) -> dict:
    same_origin_request(request)
    visible_investigation(request, investigation_id, user)
    try:
        result = request.app.state.investigations.submit(investigation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "investigation_incomplete", "message": str(exc)}
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "investigation_locked",
                "message": "Investigation is already submitted.",
            },
        )
    return result


@app.post(
    "/v1/investigations/{investigation_id}/verify",
    response_model=InvestigationOutcome,
    tags=["investigations"],
)
def verify_investigation(
    investigation_id: int,
    body: InvestigationVerification,
    request: Request,
    user: Annotated[User, Depends(require_any_role(Role.NETWORK_ADMIN))],
) -> dict:
    same_origin_request(request)
    try:
        result = request.app.state.investigations.verify(
            investigation_id,
            verification_status=body.verification_status,
            verified_cause=body.verified_cause,
            notes=body.notes,
            verified_by=user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_verification", "message": str(exc)}
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "not_submitted",
                "message": "Only submitted investigations can be verified.",
            },
        )
    return result


# Mount after API routes so the root UI does not shadow health and versioned endpoints.
UI_DIR = Path(__file__).parent / "ui"
app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")
app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
