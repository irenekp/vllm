from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

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

    # ---- raw timing events ----
    copy_intervals: List[CudaEventInterval] = field(default_factory=list)
    stall_intervals: List[CudaEventInterval] = field(default_factory=list)

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

    # ---- validation ----
    valid: bool
    reason: str
    tp_reduced: bool

    reported_layers: int
    missing_layers: List[int]
    unattributed_layer_intervals: int
    out_of_range_layer_intervals: int

    # ---- timings ----
    forward_ms: float
    stall_ms: float
    stall_ms_unattributed: float
    stall_ms_by_layer: List[float]

    copy_ms: float
    compute_ms: float


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
        cur.copy_intervals.append(CudaEventInterval(start_ev, end_ev, layer_id=layer_id))

    def record_stall_interval(
        self,
        start_ev: torch.cuda.Event,
        end_ev: torch.cuda.Event,
        layer_id: Optional[int] = None,
    ) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.stall_intervals.append(CudaEventInterval(start_ev, end_ev, layer_id=layer_id))


class TimingRingBuffer:
    """Fixed-size ring buffer for raw per-step timing event bundles."""
    def __init__(self, capacity: int = 256) -> None:
        self._buf: Deque[BatchTimingEvents] = deque(maxlen=capacity)

    def push_events(self, ev: BatchTimingEvents) -> None:
        self._buf.append(ev)

    def last_events(self) -> Optional[BatchTimingEvents]:
        if not self._buf:
            return None
        return self._buf[-1]

    def all_events(self) -> List[BatchTimingEvents]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

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
      (stall_ms_by_layer, unattributed_ms, unattributed_cnt, out_of_range_cnt, reported_mask)
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

    return by_layer, float(unattributed_ms), int(unattributed_cnt), int(out_of_range_cnt), reported


def _tp_max_reduce_inplace(values: List[float], *, group=None) -> bool:
    """
    In-place MAX reduce over TP ranks using torch.distributed.
    Returns True if reduction occurred; False if dist not initialized or world_size==1.
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
    Finalize one BatchTimingEvents bundle into a numeric timing record with validation.

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
            valid=False,
            reason="num_layers_not_set",
            tp_reduced=False,
            reported_layers=0,
            missing_layers=[],
            unattributed_layer_intervals=len([x for x in events.stall_intervals if x.layer_id is None]),
            out_of_range_layer_intervals=0,
            forward_ms=forward_ms,
            stall_ms=stall_ms,
            stall_ms_unattributed=stall_ms,
            stall_ms_by_layer=[],
            copy_ms=copy_ms,
            compute_ms=float(forward_ms - stall_ms),
        )

    stall_by_layer, stall_unattributed, unattributed_cnt, out_of_range_cnt, reported_mask = \
        _sum_intervals_ms_by_layer(events.stall_intervals, num_layers=num_layers)

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

    stall_ms = float(sum(stall_by_layer) + stall_unattributed)
    compute_ms = float(forward_ms - stall_ms)

    # Validity rules (explicit + conservative)
    valid = True
    reasons = []

    if events.batch_id < 0:
        valid = False
        reasons.append("batch_id_unset")
    if events.tp_rank < 0:
        valid = False
        reasons.append("tp_rank_unset")
    if tp_reduce_max and not tp_reduced:
        valid = False
        reasons.append("tp_reduce_not_applied")
    if len(missing_layers) > 0:
        valid = False
        reasons.append(f"missing_layer_tags:{len(missing_layers)}")
    if unattributed_cnt > 0:
        valid = False
        reasons.append(f"unattributed_intervals:{unattributed_cnt}")
    if out_of_range_cnt > 0:
        valid = False
        reasons.append(f"out_of_range_intervals:{out_of_range_cnt}")

    reason = "ok" if valid else ";".join(reasons)

    return BatchTimingRecord(
        batch_id=int(events.batch_id),
        tp_rank=int(events.tp_rank),
        is_prefill=bool(events.is_prefill),
        num_tokens=int(events.num_tokens),
        num_layers=num_layers,
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
    )
