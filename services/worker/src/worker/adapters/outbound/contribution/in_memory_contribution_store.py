from typing import override

from worker.application.ports.outbound.contribution.contribution_store import (
    ContributionStore,
)


class InMemoryContributionStore[KeyT, ConT](ContributionStore[KeyT, ConT]):
    def __init__(
        self,
    ) -> None:
        self._in_memory_store: dict[KeyT, list[ConT]] = {}

    @override
    async def put(self, key: KeyT, contribution: ConT) -> None:
        self._in_memory_store.setdefault(key, [])
        self._in_memory_store[key].append(contribution)

    @override
    async def get(self, key: KeyT) -> tuple[ConT, ...]:
        contributions = self._in_memory_store.get(key)
        if contributions is None:
            return ()
        return tuple(contributions)

    @override
    async def discard(self, key: KeyT) -> None:
        self._in_memory_store.pop(key, None)
