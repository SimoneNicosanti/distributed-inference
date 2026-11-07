import asyncio
from collections.abc import Callable
from typing import override

from worker.application.postbox.abc.inbox import OrderedInbox
from worker.application.postbox.abc.outbox import InvalidOutboxDepositError, Outbox


class OrderedPostbox[KeyT, ValT](
    OrderedInbox[ValT],
    Outbox[ValT],
):
    def __init__(self, key_compute_callable: Callable[[ValT], KeyT]) -> None:
        self._key_compute_callable = key_compute_callable

        self._queue: asyncio.Queue[ValT] = asyncio.Queue()
        self._all_keys: set[KeyT] = set()

    @override
    async def deposit(self, value: ValT) -> None:
        key = self._key_compute_callable(value)

        if key in self._all_keys:
            raise InvalidOutboxDepositError(f"Duplicate key: {key}")

        self._all_keys.add(key)
        await self._queue.put(value)

    @override
    async def extract_next(self) -> ValT:
        return await self._queue.get()
