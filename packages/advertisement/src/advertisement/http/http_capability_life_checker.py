from typing import override

import httpx

from advertisement.capability_life_checker import (
    CapabilityLifeChecker,
)
from integration.service_instance.service_instance import (
    CapabilityCouple,
    CapabilityProtocol,
    ServiceInstance,
)

DEFAULT_HTTP_READINESS_PATH = "/health/ready"


class HttpCapabilityLifeChecker(CapabilityLifeChecker):
    def __init__(
        self,
        client: httpx.AsyncClient,
        probe_url: str,
        capability: CapabilityCouple,
        timeout_s: float = 1.0,
    ) -> None:
        self._client = client
        self._probe_url = probe_url
        self._capability = capability
        if timeout_s <= 0:
            raise ValueError("Timeout must be positive")
        self._timeout_s = timeout_s

    @override
    async def get_if_alive(self) -> CapabilityCouple | None:
        try:
            response = await self._client.get(
                self._probe_url,
                timeout=self._timeout_s,
            )
        except httpx.RequestError:
            return None

        if response.status_code != 204:
            return None

        return self._capability


def build_http_capability_life_checkers(
    service_instance: ServiceInstance,
    client: httpx.AsyncClient,
    probe_host: str = "127.0.0.1",
    readiness_path: str = DEFAULT_HTTP_READINESS_PATH,
) -> list[CapabilityLifeChecker]:
    if not readiness_path.startswith("/"):
        raise ValueError("Readiness path must start with '/'")

    checkers: list[CapabilityLifeChecker] = []
    for capability in service_instance.capabilities:
        for interface in capability.interfaces:
            if interface.protocol != CapabilityProtocol.HTTP:
                continue
            port = interface.endpoint.port
            if port is None:
                raise ValueError("HTTP capability endpoint must include a port")
            checkers.append(
                HttpCapabilityLifeChecker(
                    client=client,
                    probe_url=(
                        f"{interface.endpoint.scheme}://{probe_host}:{port}"
                        f"{readiness_path}"
                    ),
                    capability=CapabilityCouple(
                        type=capability.type,
                        interface=interface,
                    ),
                )
            )

    return checkers
