import asyncio
from collections.abc import Iterable
from typing import override

import httpx

from catalog.api.directory import Directory
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
from integration.service_instance.service_instance import (
    CapabilityProtocol,
    CapabilityType,
)
from shared.model.model_partition import ModelPartitionId


class HttpPartitionBuilderGateway(PartitionBuilderGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        directory: Directory,
    ) -> None:
        self._http_client = http_client
        self._directory = directory

    @override
    async def build_all_partitions_by_ids(
        self,
        partition_ids: Iterable[ModelPartitionId],
    ) -> None:
        partition_ids = tuple(partition_ids)
        if not partition_ids:
            return

        base_url = await self._get_partition_builder_base_url()
        await asyncio.gather(
            *(
                self._build_partition(base_url, partition_id)
                for partition_id in partition_ids
            )
        )

    async def _build_partition(
        self,
        base_url: str,
        partition_id: ModelPartitionId,
    ) -> None:
        request = GenerateModelPartitionRequest(
            model_variant_id=partition_id.model_variant_id,
            layers=list(partition_id.layers),
        )
        response = await self._http_client.post(
            f"{base_url}{GENERATE_MODEL_PARTITION_PATH}",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        response_dto = GenerateModelPartitionResponse.model_validate(response.json())
        if response_dto.model_partition_id != partition_id:
            raise ValueError(
                "Model manager returned a different model partition identifier"
            )

    async def _get_partition_builder_base_url(self) -> str:
        service_instances = await self._directory.get_all_service_instances()
        interfaces = [
            interface
            for service_instance in service_instances
            for interface in service_instance.get_capability_interfaces_by_type_and_protocol(
                CapabilityType.MODEL_PARTITION,
                CapabilityProtocol.HTTP,
            )
        ]

        if not interfaces:
            raise ValueError("No HTTP model partition interface is registered")
        if len(interfaces) > 1:
            raise ValueError("Multiple HTTP model partition interfaces are registered")

        return str(interfaces[0].endpoint).rstrip("/")
