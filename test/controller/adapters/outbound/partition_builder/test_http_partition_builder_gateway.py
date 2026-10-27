from typing import override

import httpx
import pytest
from pydantic import HttpUrl

from catalog.api.directory import Directory
from controller.adapters.outbound.partition_builder.http_partition_builder_gateway import (
    HttpPartitionBuilderGateway,
)
from integration.model_manager.http.model_manager_http_schema import (
    GenerateModelPartitionRequest,
    GenerateModelPartitionResponse,
)
from integration.service_instance.service_instance import (
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId


class StubServiceDirectory(Directory):
    def __init__(self, service_instances: list[ServiceInstance]) -> None:
        self._service_instances = service_instances
        self.get_all_calls = 0

    @override
    async def get_service_instance_by_service_id(
        self,
        service_id: ServiceId,
    ) -> ServiceInstance:
        for service_instance in self._service_instances:
            if service_instance.service_id == service_id:
                return service_instance
        raise KeyError(service_id)

    @override
    async def get_all_service_instances(self) -> list[ServiceInstance]:
        self.get_all_calls += 1
        return self._service_instances


def _model_manager_instance(endpoint: str) -> ServiceInstance:
    return ServiceInstance(
        service_id=ServiceId(),
        capabilities=[
            Capability(
                type=CapabilityType.MODEL_PARTITION,
                interfaces=[
                    CapabilityInterface(
                        protocol=CapabilityProtocol.HTTP,
                        endpoint=HttpUrl(endpoint),
                    )
                ],
            )
        ],
    )


def _partition_ids() -> tuple[ModelPartitionId, ModelPartitionId]:
    model_variant_id = ModelVariantId(
        model_id=ModelId(model_name="model"),
        variant_tag="variant",
    )
    return (
        ModelPartitionId(
            model_variant_id=model_variant_id,
            layers=("first",),
        ),
        ModelPartitionId(
            model_variant_id=model_variant_id,
            layers=("second",),
        ),
    )


@pytest.mark.asyncio
async def test_builds_all_requested_partitions() -> None:
    requests: list[httpx.Request] = []
    directory = StubServiceDirectory(
        [_model_manager_instance("http://model-manager.local:8000/")]
    )

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        request_dto = GenerateModelPartitionRequest.model_validate_json(request.content)
        partition_id = ModelPartitionId(
            model_variant_id=request_dto.model_variant_id,
            layers=tuple(request_dto.layers),
        )
        response_dto = GenerateModelPartitionResponse(
            model_partition_id=partition_id,
            artifact_ref=ArtifactRef(value="artifact"),
        )
        return httpx.Response(
            status_code=200,
            json=response_dto.model_dump(mode="json"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as http_client:
        gateway = HttpPartitionBuilderGateway(
            http_client,
            directory,
        )
        partition_ids = _partition_ids()

        await gateway.build_all_partitions_by_ids(partition_ids)

    assert len(requests) == 2
    assert directory.get_all_calls == 1
    assert {request.method for request in requests} == {"POST"}
    assert {request.url.host for request in requests} == {"model-manager.local"}
    assert {request.url.path for request in requests} == {"/model-manager/partitions"}
    received_ids = {
        ModelPartitionId(
            model_variant_id=request_dto.model_variant_id,
            layers=tuple(request_dto.layers),
        )
        for request_dto in (
            GenerateModelPartitionRequest.model_validate_json(request.content)
            for request in requests
        )
    }
    assert received_ids == set(partition_ids)


@pytest.mark.asyncio
async def test_rejects_response_for_a_different_partition() -> None:
    requested_partition, different_partition = _partition_ids()
    directory = StubServiceDirectory(
        [_model_manager_instance("http://model-manager.local:8000")]
    )

    def handle_request(_request: httpx.Request) -> httpx.Response:
        response_dto = GenerateModelPartitionResponse(
            model_partition_id=different_partition,
            artifact_ref=ArtifactRef(value="artifact"),
        )
        return httpx.Response(
            status_code=200,
            json=response_dto.model_dump(mode="json"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as http_client:
        gateway = HttpPartitionBuilderGateway(
            http_client,
            directory,
        )

        with pytest.raises(
            ValueError,
            match="returned a different model partition identifier",
        ):
            await gateway.build_all_partitions_by_ids((requested_partition,))


@pytest.mark.asyncio
async def test_rejects_missing_model_partition_provider() -> None:
    directory = StubServiceDirectory([])

    async with httpx.AsyncClient() as http_client:
        gateway = HttpPartitionBuilderGateway(http_client, directory)

        with pytest.raises(
            ValueError,
            match="No HTTP model partition interface is registered",
        ):
            await gateway.build_all_partitions_by_ids((_partition_ids()[0],))


@pytest.mark.asyncio
async def test_rejects_multiple_model_partition_providers() -> None:
    directory = StubServiceDirectory(
        [
            _model_manager_instance("http://model-manager-a.local:8000"),
            _model_manager_instance("http://model-manager-b.local:8000"),
        ]
    )

    async with httpx.AsyncClient() as http_client:
        gateway = HttpPartitionBuilderGateway(http_client, directory)

        with pytest.raises(
            ValueError,
            match="Multiple HTTP model partition interfaces are registered",
        ):
            await gateway.build_all_partitions_by_ids((_partition_ids()[0],))


@pytest.mark.asyncio
async def test_skips_discovery_for_an_empty_partition_batch() -> None:
    directory = StubServiceDirectory([])

    async with httpx.AsyncClient() as http_client:
        gateway = HttpPartitionBuilderGateway(http_client, directory)
        await gateway.build_all_partitions_by_ids(())

    assert directory.get_all_calls == 0
