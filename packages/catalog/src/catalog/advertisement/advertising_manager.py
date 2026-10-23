import asyncio
from contextlib import suppress
from typing import override

from catalog.advertisement.lifechecker import Lifechecker
from catalog.api.registrar import Registrar
from integration.service_instance.service_instance import ServiceInstance
from lifecycle.async_lifecycle import AsyncLifecycle


class AdvertisingManager(AsyncLifecycle):
    def __init__(
        self,
        registrar: Registrar,
        service_instance: ServiceInstance,
        lifechecker: Lifechecker,
        heartbeat_interval_s: float,
    ) -> None:
        self._registrar = registrar
        self._service_instance = service_instance
        self._heartbeat_interval_s = heartbeat_interval_s
        self._lifechecker = lifechecker

        self._heartbeat_task: asyncio.Task[None] | None = None
        pass

    async def _heartbeat_loop(self) -> None:
        while True:
            if await self._lifechecker.check_life():
                await self._registrar.register_or_renew(self._service_instance)
            await asyncio.sleep(self._heartbeat_interval_s)

    @override
    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    @override
    async def stop(self) -> None:
        if self._heartbeat_task is None:
            return

        self._heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._heartbeat_task

        self._heartbeat_task = None
