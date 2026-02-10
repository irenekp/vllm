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
    stage_map = {}
    for r in per_rank:
        pp_rank = int(r.get("pp_rank", 0))
        stage = stage_map.get(pp_rank)
        if stage is None:
            stage = {
                "pp_rank": pp_rank,
                "pp_size": int(r.get("pp_size", 1)),
                "forward_ms": float(r.get("forward_ms", 0.0)),
                "stall_ms": float(r.get("stall_ms", 0.0)),
                "copy_ms": float(r.get("copy_ms", 0.0)),
                "compute_ms": float(r.get("compute_ms", 0.0)),
            }
            stage_map[pp_rank] = stage
        else:
            stage["forward_ms"] = max(stage["forward_ms"], float(r.get("forward_ms", 0.0)))
            stage["stall_ms"] = max(stage["stall_ms"], float(r.get("stall_ms", 0.0)))
            stage["copy_ms"] = max(stage["copy_ms"], float(r.get("copy_ms", 0.0)))
            stage["compute_ms"] = max(stage["compute_ms"], float(r.get("compute_ms", 0.0)))

    per_stage = [stage_map[k] for k in sorted(stage_map.keys())]

    latency_sum = {
        "mode": "last",
        "view": "latency_sum",
        "forward_ms": float(sum(s["forward_ms"] for s in per_stage)),
        "stall_ms": float(sum(s["stall_ms"] for s in per_stage)),
        "copy_ms": float(sum(s["copy_ms"] for s in per_stage)),
        "compute_ms": float(sum(s["compute_ms"] for s in per_stage)),
    }

    makespan_max = {
        "mode": "last",
        "view": "makespan_max",
        "forward_ms": float(max(s["forward_ms"] for s in per_stage)),
        "stall_ms": float(max(s["stall_ms"] for s in per_stage)),
        "copy_ms": float(max(s["copy_ms"] for s in per_stage)),
        "compute_ms": float(max(s["compute_ms"] for s in per_stage)),
    }

    aggregate = {"mode": "last", "latency_sum": latency_sum, "makespan_max": makespan_max}

    return JSONResponse(
        content={
            "available": True,
            "per_rank": per_rank,
            "per_stage": per_stage,
            "aggregate": aggregate,
        }
    )


@router.get("/get_batch_timing_average")
async def get_batch_timing_average(request: Request, last_n: int = 50):
    results = await engine_client(request).collective_rpc(
        "get_lmcache_batch_timing",
        kwargs={"mode": "avg", "last_n": int(last_n)},
    )

    per_rank = [r for r in results if r is not None]

    if not per_rank:
        return JSONResponse(content={"available": False, "per_rank": [], "aggregate": None})

    windows = [r.get("window", None) for r in per_rank]
    windows = [int(w) for w in windows if isinstance(w, int)]
    common_window = int(min(windows)) if windows else 0

    stage_map = {}
    for r in per_rank:
        pp_rank = int(r.get("pp_rank", 0))
        stage = stage_map.get(pp_rank)
        if stage is None:
            stage = {
                "pp_rank": pp_rank,
                "pp_size": int(r.get("pp_size", 1)),
                "window": int(r.get("window", common_window)),
                "forward_ms": float(r.get("forward_ms", 0.0)),
                "stall_ms": float(r.get("stall_ms", 0.0)),
                "copy_ms": float(r.get("copy_ms", 0.0)),
                "compute_ms": float(r.get("compute_ms", 0.0)),
            }
            stage_map[pp_rank] = stage
        else:
            stage["forward_ms"] = max(stage["forward_ms"], float(r.get("forward_ms", 0.0)))
            stage["stall_ms"] = max(stage["stall_ms"], float(r.get("stall_ms", 0.0)))
            stage["copy_ms"] = max(stage["copy_ms"], float(r.get("copy_ms", 0.0)))
            stage["compute_ms"] = max(stage["compute_ms"], float(r.get("compute_ms", 0.0)))
            stage["window"] = min(stage["window"], int(r.get("window", common_window)))

    per_stage = [stage_map[k] for k in sorted(stage_map.keys())]

    latency_sum = {
        "mode": "avg",
        "view": "latency_sum",
        "window": int(min(s.get("window", common_window) for s in per_stage) if per_stage else common_window),
        "forward_ms": float(sum(s["forward_ms"] for s in per_stage)),
        "stall_ms": float(sum(s["stall_ms"] for s in per_stage)),
        "copy_ms": float(sum(s["copy_ms"] for s in per_stage)),
        "compute_ms": float(sum(s["compute_ms"] for s in per_stage)),
    }

    makespan_max = {
        "mode": "avg",
        "view": "makespan_max",
        "window": int(min(s.get("window", common_window) for s in per_stage) if per_stage else common_window),
        "forward_ms": float(max(s["forward_ms"] for s in per_stage)),
        "stall_ms": float(max(s["stall_ms"] for s in per_stage)),
        "copy_ms": float(max(s["copy_ms"] for s in per_stage)),
        "compute_ms": float(max(s["compute_ms"] for s in per_stage)),
    }

    aggregate = {"mode": "avg", "latency_sum": latency_sum, "makespan_max": makespan_max}

    return JSONResponse(
        content={
            "available": True,
            "per_rank": per_rank,
            "per_stage": per_stage,
            "aggregate": aggregate,
        }
    )


@router.get("/cache/duplication")
async def get_cache_duplication_stats(request: Request):
    stats = await engine_client(request).get_kv_duplication_stats()
    return JSONResponse(content=stats)


@router.post("/flush_batch_timing")
async def flush_batch_timing(request: Request):
    results = await engine_client(request).collective_rpc(
        "flush_lmcache_batch_timing",
        kwargs={},
    )
    per_rank = [r for r in results if r is not None]
    ok = all(bool(r.get("ok", False)) for r in per_rank) if per_rank else False
    return JSONResponse(content={"ok": ok, "per_rank": per_rank})

def register_basic_api_routers(app: FastAPI):
    app.include_router(router)
