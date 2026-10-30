import asyncio
import logging
from contextlib import suppress
from typing import override

from advertisement.capability_life_checker import (
    CapabilityLifeChecker,
)
from catalog.api.registrar import Registrar
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityType,
    ServiceInstance,
)
from lifecycle.async_lifecycle import AsyncLifecycle
from shared.service.service import ServiceId

logger = logging.getLogger(__name__)


class AdvertisingManager(AsyncLifecycle):
    def __init__(
        self,
        registrar: Registrar,
        service_id: ServiceId,
        capability_life_checkers: list[CapabilityLifeChecker],
        heartbeat_interval_s: float,
    ) -> None:
        self._registrar = registrar
        self._service_id = service_id
        self._life_checkers = capability_life_checkers
        if heartbeat_interval_s <= 0:
            raise ValueError("Heartbeat interval must be positive")
        self._heartbeat_interval_s = heartbeat_interval_s

        self._heartbeat_task: asyncio.Task[None] | None = None

    async def _advertise(self) -> None:
        capability_couples = await asyncio.gather(
            *(checker.get_if_alive() for checker in self._life_checkers)
        )

        interfaces_by_capability_type: dict[
            CapabilityType, list[CapabilityInterface]
        ] = {}
        for capability_couple in capability_couples:
            if capability_couple is None:
                continue
            interfaces_by_capability_type.setdefault(
                capability_couple.type, []
            ).append(capability_couple.interface)

        capabilities = [
            Capability(type=capability_type, interfaces=interfaces)
            for capability_type, interfaces in interfaces_by_capability_type.items()
        ]
        service_instance = ServiceInstance(
            service_id=self._service_id, capabilities=capabilities
        )

        await self._registrar.register_or_renew(service_instance)

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await self._advertise()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to publish service advertisement",
                    extra={"service_id": str(self._service_id)},
                )

            await asyncio.sleep(self._heartbeat_interval_s)

    @override
    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"service-advertisement-{self._service_id}",
        )

    @override
    async def stop(self) -> None:
        if self._heartbeat_task is None:
            return

        heartbeat_task = self._heartbeat_task
        self._heartbeat_task = None
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task
