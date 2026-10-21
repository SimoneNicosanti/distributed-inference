import httpx
import pytest

from controller.adapters.outbound.partition_builder.http_partition_builder_gateway import (
    HttpPartitionBuilderGateway,
)
from integration.model_manager.http.model_manager_http_schema import (
    GenerateModelPartitionRequest,
    GenerateModelPartitionResponse,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.user.user import UserId


def _partition_ids() -> tuple[ModelPartitionId, ModelPartitionId]:
    user_id = UserId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(owner_id=user_id, model_name="model"),
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
            "http://model-manager.local:8000/",
        )
        partition_ids = _partition_ids()

        await gateway.build_all_partitions_by_ids(partition_ids)

    assert len(requests) == 2
    assert {request.method for request in requests} == {"POST"}
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
            "http://model-manager.local:8000",
        )

        with pytest.raises(
            ValueError,
            match="returned a different model partition identifier",
        ):
            await gateway.build_all_partitions_by_ids((requested_partition,))
