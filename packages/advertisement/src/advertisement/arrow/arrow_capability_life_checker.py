import asyncio
from typing import override

from pyarrow import flight

from advertisement.capability_life_checker import (
    CapabilityLifeChecker,
)
from integration.service_instance.service_instance import (
    CapabilityCouple,
    CapabilityProtocol,
    ServiceInstance,
)


class ArrowCapabilityLifeChecker(CapabilityLifeChecker):
    def __init__(
        self,
        probe_endpoint: str,
        capability: CapabilityCouple,
        timeout_s: float = 1.0,
    ) -> None:
        self._probe_endpoint = probe_endpoint
        self._capability = capability
        if timeout_s <= 0:
            raise ValueError("Timeout must be positive")
        self._timeout_s = timeout_s

    def _probe(self) -> None:
        client = flight.connect(self._probe_endpoint)
        try:
            client.wait_for_available(timeout=self._timeout_s)
        finally:
            client.close()

    @override
    async def get_if_alive(self) -> CapabilityCouple | None:
        try:
            await asyncio.to_thread(self._probe)
        except flight.FlightError, OSError:
            return None

        return self._capability


def build_arrow_capability_life_checkers(
    service_instance: ServiceInstance,
    probe_host: str = "127.0.0.1",
) -> list[CapabilityLifeChecker]:
    checkers: list[CapabilityLifeChecker] = []
    for capability in service_instance.capabilities:
        for interface in capability.interfaces:
            if interface.protocol != CapabilityProtocol.ARROW:
                continue
            port = interface.endpoint.port
            if port is None:
                raise ValueError("Arrow capability endpoint must include a port")
            checkers.append(
                ArrowCapabilityLifeChecker(
                    probe_endpoint=(
                        f"{interface.endpoint.scheme}://{probe_host}:{port}"
                    ),
                    capability=CapabilityCouple(
                        type=capability.type,
                        interface=interface,
                    ),
                )
            )

    return checkers
