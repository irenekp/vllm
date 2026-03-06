from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable

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


def _ordered_residence(combo: FrozenSet[str]) -> list[str]:
    if not combo:
        return []
    if MEDIUM_GPU in combo:
        return [MEDIUM_GPU] + sorted(t for t in combo if t != MEDIUM_GPU)
    return sorted(combo)


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
        self._residence_combo_tokens: dict[FrozenSet[str], int] = {}
        self._integrity_errors_by_key: dict[str, str] = {}
        self._counts = DuplicationCounts()

    def reset(self) -> None:
        self._gpu_resident.clear()
        self._cpu_resident.clear()
        self._residence_combo_tokens.clear()
        self._integrity_errors_by_key.clear()
        self._counts = DuplicationCounts()

    def counts(self) -> DuplicationCounts:
        return self._counts

    def _get_combo_for_key(self, key: str) -> FrozenSet[str]:
        combo: set[str] = set()
        if key in self._gpu_resident:
            combo.add(MEDIUM_GPU)
        tiers = self._cpu_resident.get(key)
        if tiers:
            combo.update(tiers.keys())
        return frozenset(combo)

    def _get_token_count_for_key(self, key: str) -> int | None:
        gpu_tokens = self._gpu_resident.get(key)
        if gpu_tokens is not None:
            return int(gpu_tokens)
        tiers = self._cpu_resident.get(key)
        if not tiers:
            return None
        first = next(iter(tiers.values()), None)
        return int(first) if first is not None else None

    def _apply_combo_transition(
        self,
        old_combo: FrozenSet[str],
        old_tokens: int | None,
        new_combo: FrozenSet[str],
        new_tokens: int | None,
    ) -> None:
        if old_combo == new_combo and old_tokens == new_tokens:
            return

        if old_combo and old_tokens is not None and old_tokens > 0:
            updated = self._residence_combo_tokens.get(old_combo, 0) - int(old_tokens)
            if updated > 0:
                self._residence_combo_tokens[old_combo] = int(updated)
            else:
                self._residence_combo_tokens.pop(old_combo, None)

        if new_combo and new_tokens is not None and new_tokens > 0:
            self._residence_combo_tokens[new_combo] = int(
                self._residence_combo_tokens.get(new_combo, 0) + int(new_tokens)
            )

    def _refresh_key_integrity(self, key: str) -> None:
        observed: dict[str, int] = {}
        gpu_tokens = self._gpu_resident.get(key)
        if gpu_tokens is not None:
            observed[MEDIUM_GPU] = int(gpu_tokens)

        tiers = self._cpu_resident.get(key, {})
        for tier, tokens in tiers.items():
            observed[str(tier)] = int(tokens)

        if len(observed) <= 1:
            self._integrity_errors_by_key.pop(key, None)
            return

        token_values = set(observed.values())
        if len(token_values) == 1:
            self._integrity_errors_by_key.pop(key, None)
            return

        ordered = _ordered_residence(frozenset(observed))
        details = ", ".join(f"{name}={observed[name]}" for name in ordered)
        self._integrity_errors_by_key[key] = (
            f"Token-count mismatch for block {key}: {details}"
        )

    def _raise_if_integrity_violated(self) -> None:
        if not self._integrity_errors_by_key:
            return
        first_key = sorted(self._integrity_errors_by_key.keys())[0]
        raise RuntimeError(self._integrity_errors_by_key[first_key])

    def _serialize_residence_tokens(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for combo, tokens in self._residence_combo_tokens.items():
            if not combo or int(tokens) <= 0:
                continue
            rows.append(
                {
                    "residence": _ordered_residence(combo),
                    "tokens": int(tokens),
                }
            )
        rows.sort(key=lambda row: (len(row["residence"]), row["residence"]))
        return rows

    def duplicated_tokens(self) -> int:
        """Return unique duplicated tokens across GPU and any host tier."""
        self._raise_if_integrity_violated()
        total = 0
        for combo, tokens in self._residence_combo_tokens.items():
            if MEDIUM_GPU not in combo or len(combo) <= 1:
                continue
            total += int(tokens)
        return int(total)

    def duplication_stats(self) -> dict[str, Any]:
        self._raise_if_integrity_violated()
        duplicated_tokens = 0
        for combo, tokens in self._residence_combo_tokens.items():
            if MEDIUM_GPU not in combo or len(combo) <= 1:
                continue
            duplicated_tokens += int(tokens)
        return {
            "duplicated_tokens": int(duplicated_tokens),
            "residence_tokens": self._serialize_residence_tokens(),
        }

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
        old_combo = self._get_combo_for_key(key)
        old_tokens = self._get_token_count_for_key(key)

        if key not in self._gpu_resident:
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

        new_combo = self._get_combo_for_key(key)
        new_tokens = self._get_token_count_for_key(key)
        self._apply_combo_transition(old_combo, old_tokens, new_combo, new_tokens)
        self._refresh_key_integrity(key)

    def _cpu_add(self, key: str, num_tokens: int, tier: str) -> None:
        old_combo = self._get_combo_for_key(key)
        old_tokens = self._get_token_count_for_key(key)

        tiers = self._cpu_resident.setdefault(key, {})
        if tier not in tiers:
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

        new_combo = self._get_combo_for_key(key)
        new_tokens = self._get_token_count_for_key(key)
        self._apply_combo_transition(old_combo, old_tokens, new_combo, new_tokens)
        self._refresh_key_integrity(key)

    def _gpu_remove(self, key: str) -> None:
        old_combo = self._get_combo_for_key(key)
        old_tokens = self._get_token_count_for_key(key)

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

        new_combo = self._get_combo_for_key(key)
        new_tokens = self._get_token_count_for_key(key)
        self._apply_combo_transition(old_combo, old_tokens, new_combo, new_tokens)
        self._refresh_key_integrity(key)

    def _clear_gpu_only(self) -> None:
        keys = list(self._gpu_resident.keys())
        for key in keys:
            self._gpu_remove(key)

    def _cpu_remove(self, key: str, tier: str) -> None:
        old_combo = self._get_combo_for_key(key)
        old_tokens = self._get_token_count_for_key(key)

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

        new_combo = self._get_combo_for_key(key)
        new_tokens = self._get_token_count_for_key(key)
        self._apply_combo_transition(old_combo, old_tokens, new_combo, new_tokens)
        self._refresh_key_integrity(key)
