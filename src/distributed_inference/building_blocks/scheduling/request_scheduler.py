from abc import ABC, abstractmethod
from asyncio import Future
from dataclasses import dataclass
from typing import Any, Tuple


class RequestScheduler(ABC):
    @dataclass
    class QueueRequest:
        request: Any
        future: Future[Any]
        timestamp: float

    @abstractmethod
    async def enqueue(self, request: Any, future: Future[Any]) -> None: ...

    @abstractmethod
    async def dequeue(self) -> Tuple[Any, Future[Any]]: ...

    @abstractmethod
    async def length(self) -> int: ...
