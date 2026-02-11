from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    MEDIUM_GPU,
)


def _normalize_block_hash(block_hash: Any) -> str:
    if isinstance(block_hash, bytes):
        return block_hash.hex()
    if isinstance(block_hash, int):
        return f"{block_hash:x}"
    return str(block_hash)


def _get_medium_name(medium: str | None) -> str:
    return medium or "unknown"


@dataclass
class DuplicationCounts:
    gpu_only: int = 0
    cpu_only_by_tier: Dict[str, int] = field(default_factory=dict)
    both_by_tier: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_only": self.gpu_only,
            "cpu_only_by_tier": dict(self.cpu_only_by_tier),
            "both_by_tier": dict(self.both_by_tier),
        }


class KVDuplicationTracker:
    """Tracks GPU/CPU duplication using KV cache events."""

    def __init__(self) -> None:
        self._gpu_resident: dict[str, int] = {}
        self._cpu_resident: dict[str, dict[str, int]] = {}
        self._counts = DuplicationCounts()

    def reset(self) -> None:
        self._gpu_resident.clear()
        self._cpu_resident.clear()
        self._counts = DuplicationCounts()

    def counts(self) -> DuplicationCounts:
        return self._counts

    def update(self, events: Iterable[KVCacheEvent]) -> None:
        for event in events:
            if isinstance(event, BlockStored):
                self._handle_store(event)
            elif isinstance(event, BlockRemoved):
                self._handle_remove(event)
            elif isinstance(event, AllBlocksCleared):
                self._clear_gpu_only()

    def _handle_store(self, event: BlockStored) -> None:
        medium = _get_medium_name(event.medium)
        if medium == MEDIUM_GPU:
            for bh in event.block_hashes:
                self._gpu_add(_normalize_block_hash(bh), event.block_size)
        else:
            for bh in event.block_hashes:
                self._cpu_add(_normalize_block_hash(bh), event.block_size, medium)

    def _handle_remove(self, event: BlockRemoved) -> None:
        medium = _get_medium_name(event.medium)
        if medium == MEDIUM_GPU:
            for bh in event.block_hashes:
                self._gpu_remove(_normalize_block_hash(bh))
        else:
            for bh in event.block_hashes:
                self._cpu_remove(_normalize_block_hash(bh), medium)

    def _gpu_add(self, key: str, num_tokens: int) -> None:
        if key in self._gpu_resident:
            return
        self._gpu_resident[key] = num_tokens
        cpu_tiers = self._cpu_resident.get(key)
        if cpu_tiers:
            for tier in cpu_tiers:
                self._counts.both_by_tier[tier] = (
                    self._counts.both_by_tier.get(tier, 0) + num_tokens
                )
                self._counts.cpu_only_by_tier[tier] = max(
                    0, self._counts.cpu_only_by_tier.get(tier, 0) - num_tokens
                )
        else:
            self._counts.gpu_only += num_tokens

    def _cpu_add(self, key: str, num_tokens: int, tier: str) -> None:
        tiers = self._cpu_resident.setdefault(key, {})
        if tier in tiers:
            return
        had_cpu_tiers = bool(tiers)
        tiers[tier] = num_tokens
        if key in self._gpu_resident:
            self._counts.both_by_tier[tier] = (
                self._counts.both_by_tier.get(tier, 0) + num_tokens
            )
            if not had_cpu_tiers:
                self._counts.gpu_only = max(0, self._counts.gpu_only - num_tokens)
        else:
            self._counts.cpu_only_by_tier[tier] = (
                self._counts.cpu_only_by_tier.get(tier, 0) + num_tokens
            )

    def _gpu_remove(self, key: str) -> None:
        num_tokens = self._gpu_resident.pop(key, None)
        if num_tokens is None:
            return
        cpu_tiers = self._cpu_resident.get(key)
        if cpu_tiers:
            for tier in cpu_tiers:
                self._counts.both_by_tier[tier] = max(
                    0, self._counts.both_by_tier.get(tier, 0) - num_tokens
                )
                self._counts.cpu_only_by_tier[tier] = (
                    self._counts.cpu_only_by_tier.get(tier, 0) + num_tokens
                )
        else:
            self._counts.gpu_only = max(0, self._counts.gpu_only - num_tokens)

    def _clear_gpu_only(self) -> None:
        keys = list(self._gpu_resident.keys())
        for key in keys:
            self._gpu_remove(key)

    def _cpu_remove(self, key: str, tier: str) -> None:
        tiers = self._cpu_resident.get(key)
        if not tiers or tier not in tiers:
            return
        had_other_tiers = len(tiers) > 1
        num_tokens = tiers.pop(tier)
        if not tiers:
            self._cpu_resident.pop(key, None)
        if key in self._gpu_resident:
            self._counts.both_by_tier[tier] = max(
                0, self._counts.both_by_tier.get(tier, 0) - num_tokens
            )
            if not had_other_tiers:
                self._counts.gpu_only += num_tokens
        else:
            self._counts.cpu_only_by_tier[tier] = max(
                0, self._counts.cpu_only_by_tier.get(tier, 0) - num_tokens
            )
