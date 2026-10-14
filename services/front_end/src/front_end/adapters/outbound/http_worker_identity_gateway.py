from typing import override

import httpx

from front_end.application.ports.outbound.worker_gateway import (
    WorkerIdentityGateway,
)
from integration.worker.http_worker_constants import (
    DEFAULT_LOCAL_WORKER_HTTP_BASE_URL,
)
from integration.worker.worker_identity.http_worker_identity_query_paths import (
    WORKER_IDENTITY_PATH,
)
from shared.service.service import ServiceId, WorkerId


class HttpWorkerIdentityGateway(WorkerIdentityGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str = DEFAULT_LOCAL_WORKER_HTTP_BASE_URL,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    @override
    async def get_worker_identity(self) -> WorkerId:
        response = await self._http_client.get(
            f"{self._base_url}{WORKER_IDENTITY_PATH}"
        )
        response.raise_for_status()
        return ServiceId.model_validate(response.json())
