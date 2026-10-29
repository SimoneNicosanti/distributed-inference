from abc import ABC, abstractmethod


class ContributionStore[KeyT, ConT](ABC):
    @abstractmethod
    async def put(self, key: KeyT, contribution: ConT) -> None: ...

    @abstractmethod
    async def get(self, key: KeyT) -> tuple[ConT, ...]: ...

    @abstractmethod
    async def discard(self, key: KeyT) -> None: ...
