# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

router = APIRouter()

logger = init_logger(__name__)


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/responses
    # - /v1/responses/{response_id}
    # - /v1/responses/{response_id}/cancel
    # - /v1/messages
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)

@router.get("/get_last_batch_timing")
async def get_last_batch_timing(request: Request):
    results = await engine_client(request).collective_rpc(
        "get_lmcache_batch_timing",
        kwargs={"mode": "last"},
    )

    per_rank = [r for r in results if r is not None]

    if not per_rank:
        return JSONResponse(content={"available": False, "per_rank": [], "aggregate": None})
    
    def _mean(key: str) -> float:
        xs = [float(r[key]) for r in per_rank if key in r]
        return sum(xs) / len(xs) if xs else 0.0

    aggregate = {
        "mode": "last",
        "forward_ms": _mean("forward_ms"),
        "stall_ms": _mean("stall_ms"),
        "copy_ms": _mean("copy_ms"),
        "compute_ms": _mean("compute_ms"),
    }

    return JSONResponse(content={"available": True, "per_rank": per_rank, "aggregate": aggregate})


@router.get("/get_batch_timing_average")
async def get_batch_timing_average(request: Request, last_n: int = 50):
    results = await engine_client(request).collective_rpc(
        "get_lmcache_batch_timing",
        kwargs={"mode": "avg", "last_n": int(last_n)},
    )

    per_rank = [r for r in results if r is not None]

    if not per_rank:
        return JSONResponse(content={"available": False, "per_rank": [], "aggregate": None})

    def _mean(key: str) -> float:
        xs = [float(r[key]) for r in per_rank if key in r]
        return sum(xs) / len(xs) if xs else 0.0

    aggregate = {
        "mode": "avg",
        "window": int(min(r.get("window", 0) for r in per_rank if isinstance(r.get("window", None), int)) or 0),
        "forward_ms": _mean("forward_ms"),
        "stall_ms": _mean("stall_ms"),
        "copy_ms": _mean("copy_ms"),
        "compute_ms": _mean("compute_ms"),
    }

    return JSONResponse(content={"available": True, "per_rank": per_rank, "aggregate": aggregate})

@router.get("/cache/duplication")
async def get_cache_duplication_stats(
    request: Request,
    per_worker: bool = True,
    include_lmcache_internal: bool = True,
):
    """
    Expensive, on-demand endpoint.
    """
    client = engine_client(request)
    call_utility_async = getattr(client, "call_utility_async", None)
    if callable(call_utility_async):
        stats = await call_utility_async(
            "get_cache_duplication_stats",
            bool(per_worker),
            bool(include_lmcache_internal),
        )
        return JSONResponse(content=stats)

    call_utility = getattr(client, "call_utility", None)
    if callable(call_utility):
        stats = call_utility(
            "get_cache_duplication_stats",
            bool(per_worker),
            bool(include_lmcache_internal),
        )
        return JSONResponse(content=stats)

    return JSONResponse(
        status_code=501,
        content={
            "error": "Engine client does not support utility calls in this configuration.",
            "hint": "This endpoint requires a vLLM v1 EngineCore client that supports call_utility_async/call_utility.",
        },
    )

def register_basic_api_routers(app: FastAPI):
    app.include_router(router)
