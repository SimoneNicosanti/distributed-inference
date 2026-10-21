import asyncio
from collections.abc import Iterable
from typing import override

import httpx

from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from integration.model_manager.http.model_manager_http_paths import (
    GENERATE_MODEL_PARTITION_PATH,
)
from integration.model_manager.http.model_manager_http_schema import (
    GenerateModelPartitionRequest,
    GenerateModelPartitionResponse,
)
from shared.model.model_partition import ModelPartitionId


class HttpPartitionBuilderGateway(PartitionBuilderGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    @override
    async def build_all_partitions_by_ids(
        self,
        partition_ids: Iterable[ModelPartitionId],
    ) -> None:
        await asyncio.gather(
            *(self._build_partition(partition_id) for partition_id in partition_ids)
        )

    async def _build_partition(self, partition_id: ModelPartitionId) -> None:
        request = GenerateModelPartitionRequest(
            model_variant_id=partition_id.model_variant_id,
            layers=list(partition_id.layers),
        )
        response = await self._http_client.post(
            f"{self._base_url}{GENERATE_MODEL_PARTITION_PATH}",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        response_dto = GenerateModelPartitionResponse.model_validate(response.json())
        if response_dto.model_partition_id != partition_id:
            raise ValueError(
                "Model manager returned a different model partition identifier"
            )
