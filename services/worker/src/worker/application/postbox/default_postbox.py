import asyncio
from collections import OrderedDict
from collections.abc import Callable
from typing import override

from worker.application.postbox.abc.inbox import Inbox
from worker.application.postbox.abc.outbox import Outbox


class DefaultPostbox[KeyT, ValT](
    Inbox[KeyT, ValT],
    Outbox[ValT],
):
    def __init__(self, key_compute_callable: Callable[[ValT], KeyT]) -> None:
        self._key_compute_callable = key_compute_callable

        self._available: OrderedDict[KeyT, ValT] = OrderedDict()
        self._extracted: set[KeyT] = set()
        self._condition = asyncio.Condition()

    @override
    async def deposit(self, value: ValT) -> None:
        key = self._key_compute_callable(value)

        async with self._condition:
            if key in self._available or key in self._extracted:
                raise ValueError(f"Duplicate result: {key}")

            self._available[key] = value
            self._condition.notify_all()

    @override
    async def extract_by_key(self, key: KeyT) -> ValT:
        async with self._condition:
            await self._condition.wait_for(
                lambda: key in self._available or key in self._extracted
            )

            if key in self._extracted:
                raise ValueError(f"Result already extracted: {key}")

            value = self._available.pop(key)
            self._extracted.add(key)
            self._condition.notify_all()

            return value

    @override
    async def extract_next(self) -> ValT:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._available))

            key, value = self._available.popitem(last=False)
            self._extracted.add(key)
            self._condition.notify_all()

            return value
