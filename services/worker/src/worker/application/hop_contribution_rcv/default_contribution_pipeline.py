import asyncio
from dataclasses import dataclass, field
from typing import override

from worker.application.hop_contribution_rcv.abc.contribution_gathering_policy import (
    ContributionGatheringPolicy,
    HopContributionCollection,
    HopContributionCollectionKey,
)
from worker.application.hop_contribution_rcv.abc.contribution_pipeline import (
    CollectionHandle,
    ContributionPipeline,
)
from worker.application.ports.outbound.contribution.contribution_store import (
    ContributionStore,
)
from worker.application.postbox.abc.outbox import Outbox
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


@dataclass(slots=True)
class _LockSlot:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class DefaultContributionPipeline(ContributionPipeline):
    def __init__(
        self,
        gathering_policy: ContributionGatheringPolicy,
        contribution_store: ContributionStore[
            HopContributionCollectionKey, HopInvocationContribution
        ],
        outbox: Outbox[HopContributionCollection],
    ) -> None:
        self._gathering_policy = gathering_policy
        self._outbox = outbox

        self._contribution_store = contribution_store
        self._handles: dict[HopContributionCollectionKey, CollectionHandle] = {}
        self._lock_slots: dict[HopContributionCollectionKey, _LockSlot] = {}

    @override
    async def submit(self, contribution: HopInvocationContribution) -> CollectionHandle:
        key = await self._gathering_policy.key_of(contribution)
        lock_slot = self._acquire_lock_slot(key)

        try:
            async with lock_slot.lock:
                handle = self._handles.get(key)
                if handle is not None and handle.future.done():
                    return handle

                old_contributions = list(await self._contribution_store.get(key))
                await self._gathering_policy.validate(old_contributions, contribution)

                candidate = [*old_contributions, contribution]

                if not await self._gathering_policy.is_ready(candidate):
                    await self._contribution_store.put(key, contribution)
                    return self._get_or_create_handle(key)

                collected = await self._gathering_policy.gather(candidate)
                await self._outbox.deposit(collected)
                await self._contribution_store.discard(key)

                handle = self._get_or_create_handle(key)
                handle.future.set_result(None)
                return handle
        finally:
            self._release_lock_slot(key, lock_slot)

    def _get_or_create_handle(
        self, key: HopContributionCollectionKey
    ) -> CollectionHandle:
        handle = self._handles.get(key)
        if handle is None:
            handle = CollectionHandle(
                future=asyncio.get_running_loop().create_future(),
            )
            self._handles[key] = handle

        return handle

    def _acquire_lock_slot(self, key: HopContributionCollectionKey) -> _LockSlot:
        lock_slot = self._lock_slots.get(key)
        if lock_slot is None:
            lock_slot = _LockSlot()
            self._lock_slots[key] = lock_slot

        lock_slot.users += 1
        return lock_slot

    def _release_lock_slot(
        self, key: HopContributionCollectionKey, lock_slot: _LockSlot
    ) -> None:
        lock_slot.users -= 1

        if lock_slot.users == 0:
            self._lock_slots.pop(key, None)

            handle = self._handles.get(key)
            if handle is not None and handle.future.done():
                self._handles.pop(key, None)
