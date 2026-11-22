from asyncio import Future
from dataclasses import dataclass


@dataclass
class QueueRequest[ReqT, ResT]:
    request: ReqT
    future: Future[ResT]
    timestamp: float
