# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import threading
import torch


@dataclass
class CudaEventInterval:
    start: torch.cuda.Event
    end: torch.cuda.Event
    layer_id: Optional[int] = None


@dataclass
class BatchTimingEvents:
    """Raw CUDA-event markers collected for one vLLM batch execution."""
    # ---- identity / metadata ----
    batch_id: int = -1
    tp_rank: int = -1
    is_prefill: bool = False
    num_tokens: int = -1
    num_layers: int = -1
    total_cached_tokens: int = 0
    new_prefill_tokens: int = 0
    gpu_resident_tokens: int = 0
    host_fetched_tokens: int = 0
    host_fetched_tokens_by_tier: dict[str, int] = field(default_factory=dict)

    # ---- raw timing events (load / read path) ----
    copy_intervals: List[CudaEventInterval] = field(default_factory=list)
    stall_intervals: List[CudaEventInterval] = field(default_factory=list)

    # ---- raw timing events (store / write path) ----
    store_copy_intervals: List[CudaEventInterval] = field(default_factory=list)
    store_stall_intervals: List[CudaEventInterval] = field(default_factory=list)

    forward_start: Optional[torch.cuda.Event] = None
    forward_end: Optional[torch.cuda.Event] = None
    load_end: Optional[torch.cuda.Event] = None


@dataclass
class BatchTimingRecord:
    # ---- identity / metadata ----
    batch_id: int
    tp_rank: int
    is_prefill: bool
    num_tokens: int
    num_layers: int
    total_cached_tokens: int
    new_prefill_tokens: int
    gpu_resident_tokens: int
    host_fetched_tokens: int
    host_fetched_tokens_by_tier: dict[str, int]

    # ---- validation ----
    valid: bool
    reason: str
    tp_reduced: bool

    reported_layers: int
    missing_layers: List[int]
    unattributed_layer_intervals: int
    out_of_range_layer_intervals: int

    # ---- timings (load / read) ----
    forward_ms: float
    stall_ms: float
    stall_ms_unattributed: float
    stall_ms_by_layer: List[float]

    copy_ms: float
    compute_ms: float

    # ---- timings (store / write) ----
    store_copy_ms: float = 0.0
    store_stall_ms: float = 0.0
    store_stall_ms_by_layer: List[float] = field(default_factory=list)


class VLLMTimingSink:
    """
    Matches LMCache GPUConnectorTimingSink interface.
    LMCache will call:
      - record_copy_interval(start_ev, end_ev, layer_id=?)
      - record_stall_interval(start_ev, end_ev, layer_id=?)
    """

    def __init__(self) -> None:
        self._cur: Optional[BatchTimingEvents] = None

    def start_step(self) -> BatchTimingEvents:
        self._cur = BatchTimingEvents()
        return self._cur

    def current(self) -> Optional[BatchTimingEvents]:
        return self._cur

    def clear_current(self) -> None:
        self._cur = None

    def record_copy_interval(
        self,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        layer_id: Optional[int] = None,
    ) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.copy_intervals.append(
            CudaEventInterval(start_ev, end_ev, layer_id=layer_id)
        )

    def record_stall_interval(
        self,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        layer_id: Optional[int] = None,
    ) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.stall_intervals.append(
            CudaEventInterval(start_ev, end_ev, layer_id=layer_id)
        )

    def record_store_copy_interval(
        self,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        layer_id: Optional[int] = None,
    ) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.store_copy_intervals.append(
            CudaEventInterval(start_ev, end_ev, layer_id=layer_id)
        )

    def record_store_stall_interval(
        self,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        layer_id: Optional[int] = None,
    ) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.store_stall_intervals.append(
            CudaEventInterval(start_ev, end_ev, layer_id=layer_id)
        )


class TimingRingBuffer:
    """Fixed-size ring buffer for raw per-step timing event bundles."""

    def __init__(self, capacity: int = 256) -> None:
        self._buf: Deque[BatchTimingEvents] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push_events(self, ev: BatchTimingEvents) -> None:
        with self._lock:
            self._buf.append(ev)

    def last_events(self) -> Optional[BatchTimingEvents]:
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1]

    def all_events(self) -> List[BatchTimingEvents]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def get_after_batch_id(
        self,
        *,
        after_batch_id: int,
        limit: int,
    ) -> tuple[list[BatchTimingEvents], Optional[int], Optional[int], bool]:
        with self._lock:
            if not self._buf:
                return [], None, None, False

            earliest_batch_id = int(self._buf[0].batch_id)
            latest_batch_id = int(self._buf[-1].batch_id)
            if int(after_batch_id) < earliest_batch_id - 1:
                return [], earliest_batch_id, latest_batch_id, True

            out: list[BatchTimingEvents] = []
            max_items = max(1, min(int(limit), 1000))
            for rec in self._buf:
                if int(rec.batch_id) > int(after_batch_id):
                    out.append(rec)
                if len(out) >= max_items:
                    break
            return out, earliest_batch_id, latest_batch_id, False

    def get_last_where(self, pred):
        with self._lock:
            n = len(self._buf)
            for i in range(1, n + 1):
                r = self._buf[-i]
                if pred(r):
                    return r
        return None

    def latest(self):
        """Return most recent record, or None."""
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1]

    def latest_prefill(self):
        """Return most recent prefill record, or None."""
        with self._lock:
            for r in reversed(self._buf):
                if getattr(r, "is_prefill", False):
                    return r
        return None


def _sum_intervals_ms(intervals: List[CudaEventInterval]) -> float:
    total = 0.0
    for it in intervals:
        total += float(it.start.elapsed_time(it.end))
    return float(total)


def _sum_intervals_ms_by_layer(
    intervals: List[CudaEventInterval],
    num_layers: int,
) -> Tuple[List[float], float, int, int, List[bool]]:
    """
    Returns:
      (stall_ms_by_layer, unattributed_ms, unattributed_cnt, out_of_range_cnt,
       reported_mask)
    reported_mask[l] is True if we saw any interval tagged with that layer id.
    """
    n = max(0, int(num_layers))
    by_layer = [0.0 for _ in range(n)]
    reported = [False for _ in range(n)]

    unattributed_ms = 0.0
    unattributed_cnt = 0
    out_of_range_cnt = 0

    for it in intervals:
        ms = float(it.start.elapsed_time(it.end))
        lid = it.layer_id
        if lid is None:
            unattributed_ms += ms
            unattributed_cnt += 1
            continue
        lid_i = int(lid)
        if 0 <= lid_i < n:
            by_layer[lid_i] += ms
            reported[lid_i] = True
        else:
            unattributed_ms += ms
            out_of_range_cnt += 1

    return (
        by_layer,
        float(unattributed_ms),
        int(unattributed_cnt),
        int(out_of_range_cnt),
        reported,
    )


def _tp_max_reduce_inplace(values: List[float], *, group=None) -> bool:
    """
    In-place MAX reduce over TP ranks using torch.distributed.
    Returns True if reduction occurred; False if dist not initialized or
    world_size==1.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False
    if dist.get_world_size(group=group) <= 1:
        return False

    t = torch.tensor(values, device="cuda", dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    reduced = t.tolist()
    values[:] = [float(x) for x in reduced]
    return True


def finalize_step_timing(
    events: BatchTimingEvents,
    *,
    block: bool,
    tp_group=None,
    tp_reduce_max: bool = True,
) -> Optional[BatchTimingRecord]:
    """
    Finalize one BatchTimingEvents bundle into a numeric timing record with
    validation.

    Validation philosophy:
      - If invariants are missing, we still return a record when possible,
        but mark valid=False with an explicit reason.
      - We never silently "estimate" missing semantics.
    """
    # Basic availability checks
    if events.forward_start is None or events.forward_end is None:
        return None

    if block:
        torch.cuda.synchronize()

    forward_ms = float(events.forward_start.elapsed_time(events.forward_end))
    copy_ms = float(_sum_intervals_ms(events.copy_intervals))
    host_fetched_tokens_by_tier = {
        str(k): int(v)
        for k, v in (events.host_fetched_tokens_by_tier or {}).items()
        if int(v) > 0
    }
    host_fetched_tokens = int(events.host_fetched_tokens)
    gpu_resident_tokens = int(events.gpu_resident_tokens)
    total_cached_tokens = int(events.total_cached_tokens)
    new_prefill_tokens = int(events.new_prefill_tokens)
    host_fetched_tokens_sum = int(sum(host_fetched_tokens_by_tier.values()))

    # Compute per-layer stalls + attribution stats
    num_layers = int(events.num_layers)
    if num_layers <= 0:
        # Can't build per-layer vectors; return a clearly-invalid record.
        stall_ms = float(_sum_intervals_ms(events.stall_intervals))
        return BatchTimingRecord(
            batch_id=int(events.batch_id),
            tp_rank=int(events.tp_rank),
            is_prefill=bool(events.is_prefill),
            num_tokens=int(events.num_tokens),
            num_layers=num_layers,
            total_cached_tokens=total_cached_tokens,
            new_prefill_tokens=new_prefill_tokens,
            gpu_resident_tokens=gpu_resident_tokens,
            host_fetched_tokens=host_fetched_tokens,
            host_fetched_tokens_by_tier=host_fetched_tokens_by_tier,
            valid=False,
            reason="num_layers_not_set",
            tp_reduced=False,
            reported_layers=0,
            missing_layers=[],
            unattributed_layer_intervals=len([
                x for x in events.stall_intervals if x.layer_id is None
            ]),
            out_of_range_layer_intervals=0,
            forward_ms=forward_ms,
            stall_ms=stall_ms,
            stall_ms_unattributed=stall_ms,
            stall_ms_by_layer=[],
            copy_ms=copy_ms,
            compute_ms=float(forward_ms - stall_ms),
            store_copy_ms=float(_sum_intervals_ms(
                events.store_copy_intervals)),
            store_stall_ms=float(_sum_intervals_ms(
                events.store_stall_intervals)),
            store_stall_ms_by_layer=[],
        )

    (stall_by_layer, stall_unattributed, unattributed_cnt, out_of_range_cnt,
     reported_mask) = _sum_intervals_ms_by_layer(
         events.stall_intervals, num_layers=num_layers)
    has_layer_tags = any(it.layer_id is not None for it in events.stall_intervals)

    # TP reduction (MAX) for "true stall" semantics
    tp_reduced = False
    if tp_reduce_max:
        tp_reduced = _tp_max_reduce_inplace(stall_by_layer, group=tp_group)
        # Reduce unattributed bucket as MAX as well
        bucket = [float(stall_unattributed)]
        tp_reduced_bucket = _tp_max_reduce_inplace(bucket, group=tp_group)
        stall_unattributed = float(bucket[0])
        tp_reduced = tp_reduced or tp_reduced_bucket

    # Missing layers: any layer with no reported interval tag at all
    missing_layers = [i for i, seen in enumerate(reported_mask) if not seen]
    reported_layers = num_layers - len(missing_layers)
    if not has_layer_tags:
        missing_layers = []
        reported_layers = 0

    stall_ms = float(sum(stall_by_layer) + stall_unattributed)
    compute_ms = float(forward_ms - stall_ms)

    # Store (write-back) timing
    store_copy_ms = float(_sum_intervals_ms(events.store_copy_intervals))
    if events.store_stall_intervals:
        (store_stall_by_layer, _store_unattr, _, _, _) = \
            _sum_intervals_ms_by_layer(
                events.store_stall_intervals, num_layers=num_layers)
        store_stall_ms = float(sum(store_stall_by_layer) + _store_unattr)
    else:
        store_stall_by_layer = []
        store_stall_ms = 0.0

    # Validity rules (explicit + conservative)
    valid = True
    reasons = []

    if events.batch_id < 0:
        valid = False
        reasons.append("batch_id_unset")
    if events.tp_rank < 0:
        valid = False
        reasons.append("tp_rank_unset")
    tp_reduce_required = False
    if tp_reduce_max:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            try:
                tp_reduce_required = dist.get_world_size(group=tp_group) > 1
            except Exception:
                tp_reduce_required = True

    if tp_reduce_required and not tp_reduced:
        valid = False
        reasons.append("tp_reduce_not_applied")
    if has_layer_tags and len(missing_layers) > 0:
        valid = False
        reasons.append(f"missing_layer_tags:{len(missing_layers)}")
    if has_layer_tags and unattributed_cnt > 0:
        valid = False
        reasons.append(f"unattributed_intervals:{unattributed_cnt}")
    if has_layer_tags and out_of_range_cnt > 0:
        valid = False
        reasons.append(f"out_of_range_intervals:{out_of_range_cnt}")
    if host_fetched_tokens != host_fetched_tokens_sum:
        valid = False
        reasons.append("host_fetched_tokens_mismatch")
    if total_cached_tokens != gpu_resident_tokens + host_fetched_tokens:
        valid = False
        reasons.append("total_cached_tokens_mismatch")
    if abs(compute_ms - (forward_ms - stall_ms)) > 1e-3:
        valid = False
        reasons.append("compute_ms_mismatch")

    reason = "ok" if valid else ";".join(reasons)

    return BatchTimingRecord(
        batch_id=int(events.batch_id),
        tp_rank=int(events.tp_rank),
        is_prefill=bool(events.is_prefill),
        num_tokens=int(events.num_tokens),
        num_layers=num_layers,
        total_cached_tokens=total_cached_tokens,
        new_prefill_tokens=new_prefill_tokens,
        gpu_resident_tokens=gpu_resident_tokens,
        host_fetched_tokens=host_fetched_tokens,
        host_fetched_tokens_by_tier=host_fetched_tokens_by_tier,
        valid=bool(valid),
        reason=str(reason),
        tp_reduced=bool(tp_reduced),
        reported_layers=int(reported_layers),
        missing_layers=[int(x) for x in missing_layers],
        unattributed_layer_intervals=int(unattributed_cnt),
        out_of_range_layer_intervals=int(out_of_range_cnt),
        forward_ms=forward_ms,
        stall_ms=float(stall_ms),
        stall_ms_unattributed=float(stall_unattributed),
        stall_ms_by_layer=[float(x) for x in stall_by_layer],
        copy_ms=float(copy_ms),
        compute_ms=float(compute_ms),
        store_copy_ms=float(store_copy_ms),
        store_stall_ms=float(store_stall_ms),
        store_stall_ms_by_layer=[float(x) for x in store_stall_by_layer],
    )
