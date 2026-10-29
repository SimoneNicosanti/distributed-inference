from abc import ABC, abstractmethod


class ContributionAssembler[ConT, GathT](ABC):
    @abstractmethod
    async def assemble(self, contributions: list[ConT]) -> GathT: ...
