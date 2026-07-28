from typing import Protocol


class AsyncLifecycle(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...
