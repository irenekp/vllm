from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import torch


@dataclass
class CudaEventInterval:
    start: torch.cuda.Event
    end: torch.cuda.Event


@dataclass
class BatchTimingEvents:
    """Raw CUDA-event markers collected for one vLLM prefill step."""
    copy_intervals: List[CudaEventInterval] = field(default_factory=list)
    stall_intervals: List[CudaEventInterval] = field(default_factory=list)
    forward_start: Optional[torch.cuda.Event] = None
    forward_end: Optional[torch.cuda.Event] = None
    load_end: Optional[torch.cuda.Event] = None


@dataclass
class BatchTimingRecord:
    """Finalized (ms) timing record for one step."""
    forward_ms: float
    stall_ms: float
    copy_ms: float
    compute_ms: float


class VLLMTimingSink:
    """matches LMCache GPUConnectorTimingSink.

    LMCache will call:
      - record_copy_interval(start_ev, end_ev)
      - record_stall_interval(start_ev, end_ev)
    """

    def __init__(self) -> None:
        self._cur: Optional[BatchTimingEvents] = None

    def start_step(self) -> BatchTimingEvents:
        """Begin collecting events for a new step."""
        self._cur = BatchTimingEvents()
        return self._cur

    def current(self) -> Optional[BatchTimingEvents]:
        return self._cur

    def clear_current(self) -> None:
        self._cur = None

    def record_copy_interval(self, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.copy_intervals.append(CudaEventInterval(start_ev, end_ev))

    def record_stall_interval(self, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event) -> None:
        cur = self._cur
        if cur is None:
            return
        cur.stall_intervals.append(CudaEventInterval(start_ev, end_ev))


class TimingRingBuffer:
    """Fixed-size ring buffer for raw per-step timing event bundles.

    We store CUDA events (not finalized ms numbers) so we do not block the hot path.
    Finalization (elapsed_time / one-shot wait) happens later (e.g., at query time).
    """
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




def _sum_intervals_ms(intervals: List[CudaEventInterval]) -> float:
    total = 0.0
    for it in intervals:
        total += it.start.elapsed_time(it.end)
    return total


def finalize_step_timing(
    events: BatchTimingEvents,
    *,
    block: bool,
) -> Optional[BatchTimingRecord]:
    if events.forward_start is None or events.forward_end is None:
        return None

    if block:
        torch.cuda.synchronize()

    forward_ms = events.forward_start.elapsed_time(events.forward_end)
    stall_ms = _sum_intervals_ms(events.stall_intervals)
    copy_ms = _sum_intervals_ms(events.copy_intervals)

    compute_ms = forward_ms - stall_ms

    return BatchTimingRecord(
        forward_ms=float(forward_ms),
        stall_ms=float(stall_ms),
        copy_ms=float(copy_ms),
        compute_ms=float(compute_ms),
    )
