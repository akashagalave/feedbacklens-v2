import time
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from .schema import QueryRequest, BatchRequest
from .config import settings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from shared.logger import get_logger

logger = get_logger("gateway")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="FeedbackLens Gateway", version="2.0.0")
app.state.limiter = limiter

_review_tracker: dict[str, list[tuple[float, int]]] = {}
MAX_REVIEWS_PER_MINUTE = 200

def _within_review_limit(ip: str, count: int) -> bool:
    now = time.time()
    _review_tracker.setdefault(ip, [])
    _review_tracker[ip] = [(t, c) for t, c in _review_tracker[ip] if now - t < 60]
    if sum(c for _, c in _review_tracker[ip]) + count > MAX_REVIEWS_PER_MINUTE:
        return False
    _review_tracker[ip].append((now, count))
    return True


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded for {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Max 10 requests per minute per IP.",
            "retry_after": "60 seconds"
        }
    )


REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Currently active requests",
    ["endpoint"]
)
ERROR_COUNT = Counter(
    "gateway_errors_total",
    "Total errors",
    ["endpoint", "error_type"]
)
RATE_LIMIT_COUNT = Counter(
    "gateway_rate_limit_total",
    "Total rate limited requests",
    ["endpoint"]
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    endpoint = request.url.path
    start_time = time.time()
    ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        if response.status_code == 429:
            RATE_LIMIT_COUNT.labels(endpoint=endpoint).inc()
        logger.info(f"{endpoint} | {response.status_code} | {duration:.3f}s")
        return response
    except Exception as e:
        duration = time.time() - start_time
        ERROR_COUNT.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        raise
    finally:
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gateway"}


@app.post("/analyze")
@limiter.limit("20/minute")
async def analyze(request: Request, body: QueryRequest):
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.orchestrator_url}/run",
                json=body.model_dump()
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Orchestrator error: {e}")
        ERROR_COUNT.labels(endpoint="/analyze", error_type="orchestrator_error").inc()
        raise HTTPException(status_code=502, detail="Orchestrator unreachable")


@app.post("/batch")
@limiter.limit("10/minute")
async def batch(request: Request, body: BatchRequest):
    # volume check: max 200 total reviews/min per IP (10 req × 20 reviews)
    if not _within_review_limit(request.client.host, len(body.reviews)):
        RATE_LIMIT_COUNT.labels(endpoint="/batch").inc()
        return JSONResponse(
            status_code=429,
            content={
                "error": "Review volume limit exceeded",
                "detail": f"Max {MAX_REVIEWS_PER_MINUTE} reviews/minute. Try again in 60s.",
                "retry_after": "60 seconds"
            }
        )
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{settings.orchestrator_url}/batch",
                json=body.model_dump()
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Orchestrator error: {e}")
        ERROR_COUNT.labels(endpoint="/batch", error_type="orchestrator_error").inc()
        raise HTTPException(status_code=502, detail="Orchestrator unreachable")