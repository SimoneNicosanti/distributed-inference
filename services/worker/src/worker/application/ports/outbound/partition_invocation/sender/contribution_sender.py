from abc import ABC, abstractmethod


class ContributionSender[ConT](ABC):
    @abstractmethod
    async def send(
        self,
        contribution: ConT,
    ) -> None: ...
