"""Resource arbitration: readers share, writers are exclusive, ties break by plan order."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .models import Claim

WaitCallback = Callable[[str, list[Claim], list[str]], None]


@dataclass(eq=False)
class Lease:
    step: str
    rank: int
    seq: int
    claims: tuple[Claim, ...]
    owner: str = ""

    @property
    def key(self) -> tuple[int, int]:
        return (self.rank, self.seq)


def _clash(a: Lease, b: Lease) -> bool:
    return any(x.conflicts_with(y) for x in a.claims for y in b.claims)


@dataclass
class ResourceManager:
    """Thread-safe lock manager for one run.

    All of a call's claims are granted atomically (no deadlock). A request waits for current
    holders and for clashing requests of *lower plan rank*, so conflicting steps run in plan
    order no matter which thread reaches the manager first. The scheduler announces each wave
    with :meth:`expect`; a claim-holding step also waits for lower-ranked wave members that have
    not yet declared their claims, which is what makes the order deterministic.
    """

    on_wait: WaitCallback | None = None
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _holders: list[Lease] = field(default_factory=list)
    _queue: list[Lease] = field(default_factory=list)
    _unarrived: dict[str, int] = field(default_factory=dict)
    _seq: itertools.count = field(default_factory=itertools.count)

    def expect(self, ranks: dict[str, int]) -> None:
        """Announce the steps about to run together (``step id -> plan rank``)."""
        with self._cond:
            self._unarrived.update(ranks)

    def arrived(self, step: str) -> None:
        """Mark a step as having declared its claims (or as never going to)."""
        with self._cond:
            if self._unarrived.pop(step, None) is not None:
                self._cond.notify_all()

    def _blockers(self, lease: Lease, *, real_only: bool = False) -> list[str]:
        blockers = [h.owner for h in self._holders if _clash(lease, h)]
        blockers += [q.owner for q in self._queue if q.key < lease.key and _clash(lease, q)]
        if not real_only:  # steps that have not declared claims yet might still clash
            blockers += [f"{s}/?" for s, rank in self._unarrived.items() if rank < lease.rank and s != lease.step]
        return blockers

    def acquire(self, step: str, rank: int, owner: str, claims: list[Claim]) -> Lease:
        lease = Lease(step, rank, next(self._seq), tuple(claims), owner)
        if not lease.claims:
            self.arrived(step)
            return lease
        with self._cond:
            self._unarrived.pop(step, None)
            self._queue.append(lease)
            self._queue.sort(key=lambda q: q.key)
            self._cond.notify_all()
            reported = False

            def clear() -> bool:
                nonlocal reported
                real = self._blockers(lease, real_only=True)
                if real and not reported and self.on_wait:
                    reported = True
                    self.on_wait(owner, list(lease.claims), real)
                return not self._blockers(lease)

            try:
                self._cond.wait_for(clear)
            finally:
                self._queue.remove(lease)
            self._holders.append(lease)
        return lease

    def release(self, lease: Lease) -> None:
        if not lease.claims:
            return
        with self._cond:
            self._holders.remove(lease)
            self._cond.notify_all()
