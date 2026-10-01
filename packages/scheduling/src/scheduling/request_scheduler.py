from abc import ABC, abstractmethod
from asyncio.futures import Future

# @dataclass
# class QueueRequest[ReqT, ResT]:
#     request: ReqT
#     future: Future[ResT]
#     timestamp: float


class RequestScheduler[ReqT, ResT](ABC):
    @abstractmethod
    async def enqueue(self, request: ReqT, future: Future[ResT]) -> None: ...

    @abstractmethod
    async def dequeue(self) -> tuple[ReqT, Future[ResT]]: ...

    @abstractmethod
    async def length(self) -> int: ...
