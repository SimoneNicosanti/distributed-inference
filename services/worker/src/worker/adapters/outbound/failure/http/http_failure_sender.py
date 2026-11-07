from typing import override

import httpx

from catalog.api.service.service_directory import ServiceInstanceReadError
from catalog.api.worker.worker_directory import WorkerDirectory
from integration.dto.model_pass_error.model_pass_error_dto import ModelPassErrorDto
from integration.service_instance.service_instance import (
    CapabilityProtocol,
    CapabilityType,
)
from integration.worker.failure.http_failure_schema import (
    NOTIFY_FAILURE_PATH,
)
from shared.service.service import WorkerId
from worker.adapters.context_mapper import model_pass_context_to_dto
from worker.application.ports.outbound.failure_snd.failure_sender import (
    FailureSender,
    FailureSendError,
)
from worker.domain.model.model_pass import ModelPassError


class HttpFailureSender(FailureSender):
    def __init__(
        self,
        directory: WorkerDirectory,
        client: httpx.AsyncClient,
        timeout_s: float = 10.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("Timeout must be positive")
        self._directory = directory
        self._client = client
        self._timeout_s = timeout_s

    @override
    async def send_failure(self, error: ModelPassError, worker_id: WorkerId) -> None:
        try:
            worker = await self._directory.get_worker_instance_by_id(worker_id)
            interfaces = worker.get_capability_interfaces_by_type_and_protocol(
                CapabilityType.FAILURE_RECEIVE, CapabilityProtocol.HTTP
            )
            if len(interfaces) != 1:
                raise ValueError(
                    f"Worker {worker_id} must expose exactly one HTTP failure interface"
                )
            path = NOTIFY_FAILURE_PATH.format(model_pass_id=error.context.model_pass_id)
            url = f"{str(interfaces[0].endpoint).rstrip('/')}{path}"
            dto = ModelPassErrorDto(
                context=model_pass_context_to_dto(error.context),
                message=error.message,
            )
            response = await self._client.post(
                url, json=dto.model_dump(mode="json"), timeout=self._timeout_s
            )
            response.raise_for_status()
            if response.status_code != 204:
                raise ValueError("Unexpected failure notification acknowledgement")
        except (httpx.HTTPError, ServiceInstanceReadError, ValueError) as cause:
            raise FailureSendError(
                f"Unable to notify worker {worker_id} of model pass failure"
            ) from cause
