import asyncio
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


## This is used in such a way that if the coroutine getting the lease of a resource is cancelled
## Then it must wait for thread execution to finish.
## Otherwise the thread might continue working on resources whose lease has been releaseds


## TODO: We should use it but right now it is not so important
async def run_sync_non_abandoning(
    function: Callable[..., T],
    *args: object,
) -> T:
    worker = asyncio.create_task(asyncio.to_thread(function, *args))

    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        raise
