from abc import ABC, abstractmethod


class InvalidOutboxDepositError(Exception):
    pass


class Outbox[ValT](ABC):
    @abstractmethod
    async def deposit(self, value: ValT) -> None: ...
