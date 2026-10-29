from abc import ABC, abstractmethod


class ContributionGatheringPolicy[KeyT, ConT, CollT](ABC):
    @abstractmethod
    async def key_of(self, contribution: ConT) -> KeyT: ...

    @abstractmethod
    async def is_ready(self, contributions: list[ConT]) -> bool: ...

    @abstractmethod
    async def gather(self, contributions: list[ConT]) -> CollT: ...

    @abstractmethod
    async def validate(
        self,
        other_contributions: list[ConT],
        new_contribution: ConT,
    ) -> None: ...
