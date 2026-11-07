import asyncio
from collections.abc import Callable
from typing import override

from worker.application.postbox.abc.inbox import InvalidInboxExtractionError, KeyedInbox
from worker.application.postbox.abc.outbox import InvalidOutboxDepositError, Outbox


class KeyedPostbox[KeyT, ValT](
    KeyedInbox[KeyT, ValT],
    Outbox[ValT],
):
    def __init__(self, key_compute_callable: Callable[[ValT], KeyT]) -> None:
        self._key_compute_callable = key_compute_callable

        self._available: dict[KeyT, ValT] = {}
        self._extracted: set[KeyT] = set()
        self._condition = asyncio.Condition()

    @override
    async def deposit(self, value: ValT) -> None:
        key = self._key_compute_callable(value)

        async with self._condition:
            if key in self._available or key in self._extracted:
                raise InvalidOutboxDepositError(f"Duplicate key: {key}")

            self._available[key] = value
            self._condition.notify_all()

    @override
    async def extract_by_key(self, key: KeyT) -> ValT:
        async with self._condition:
            await self._condition.wait_for(
                lambda: key in self._available or key in self._extracted
            )

            if key in self._extracted:
                raise InvalidInboxExtractionError(f"Key {key} already extracted")

            value = self._available.pop(key)
            self._extracted.add(key)

            return value
