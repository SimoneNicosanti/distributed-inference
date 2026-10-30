from abc import ABC, abstractmethod


class Outbox[ValT](ABC):
    @abstractmethod
    async def deposit(self, value: ValT) -> None: ...
