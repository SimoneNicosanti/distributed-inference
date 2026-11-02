# mypy: disable-error-code="attr-defined,name-defined"

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pytest
from front_end.domain.context.model_invocation_context import ModelInvocationContext
from front_end.domain.context.model_pass_context import ModelPassContext, ModelPassType
from shared.plan.hop_deployment_id import HopDeploymentId

from catalog.api.service.service_directory import ServiceDirectory
from front_end.adapters.outbound.inference.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from front_end.adapters.outbound.inference.arrow_worker_sink_gateway import (
    ArrowWorkerSinkGateway,
)
from front_end.adapters.outbound.mappers.context_dto_mapper import (
    external_source_context_to_dto,
    model_pass_context_to_dto,
)
from front_end.adapters.outbound.worker_endpoint_resolver import (
    get_arrow_worker_endpoint,
)
from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.tensor_map import TensorMap
from integration.dto.context.context_dto import (
    HopInvocationContributionAckDto,
    HopInvocationContributionMetadataDto,
    IngressHopDto,
    ModelPassContextDto,
    ModelPassResultMetadataDto,
)
from integration.service_instance.service_instance import (
    ArrowUrl,
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    RETRIEVE_RESULT_COMMAND,
    build_result_payload_schema,
)
from integration.worker.hop_contribution.arrow.arrow_hop_invocation_contribution_schema import (
    CONTEXT_FIELD_NAME,
    SUBMIT_CONTRIBUTION_COMMAND,
    validate_contribution_schema,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


def _context() -> ModelPassContext:
    return ModelPassContext(
        model_invocation_context=ModelInvocationContext(
            flow_id=FlowId(user_id=UserId())
        ),
        plan_version=PlanVersion(version_number=5),
        model_pass_type=ModelPassType.FORWARD,
    )


def _directory(worker_id: ServiceId, capability: CapabilityType) -> AsyncMock:
    directory = AsyncMock(spec=ServiceDirectory)
    directory.get_service_instance_by_service_id.return_value = ServiceInstance(
        service_id=worker_id,
        capabilities=[
            Capability(
                type=capability,
                interfaces=[
                    CapabilityInterface(
                        protocol=CapabilityProtocol.ARROW,
                        endpoint=ArrowUrl("grpc+tcp://remote-worker:8811"),
                    )
                ],
            )
        ],
    )
    return directory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ack_mode", ["valid", "missing", "extra", "wrong-source", "wrong-destination"]
)
async def test_ingress_uses_discovery_and_validates_ack(ack_mode: str) -> None:
    context = ExternalSourceInvocationContext(_context())
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    worker_id = ServiceId()
    ingress = PlannedIngressReplicaId(variant, replica_idx=2)
    directory = _directory(worker_id, CapabilityType.HOP_INVOCATION)
    metadata = HopInvocationContributionMetadataDto(
        source_context=external_source_context_to_dto(context),
        destination_hop=IngressHopDto(ingress_replica_id=ingress),
    )
    ack = HopInvocationContributionAckDto(
        source_context=metadata.source_context, destination_hop=metadata.destination_hop
    )
    if ack_mode == "wrong-source":
        ack = ack.model_copy(
            update={
                "source_context": ack.source_context.model_copy(
                    update={"hop_invocation_id": uuid4()}
                )
            }
        )
    if ack_mode == "wrong-destination":
        ack = ack.model_copy(
            update={
                "destination_hop": IngressHopDto(
                    ingress_replica_id=PlannedIngressReplicaId(variant, 3)
                )
            }
        )
    buffer = pa.py_buffer(ack.model_dump_json().encode())
    writer, reader, client = MagicMock(), MagicMock(), MagicMock()
    reader.read.side_effect = [
        None if ack_mode == "missing" else buffer,
        buffer if ack_mode == "extra" else None,
    ]
    client.do_put.return_value = (writer, reader)
    gateway = ArrowWorkerIngressGateway(directory)
    bundle = {"input": np.arange(6, dtype=np.float32).reshape(2, 3)}

    with patch(
        "front_end.adapters.outbound.arrow_worker_ingress_gateway.flight.connect",
        return_value=client,
    ) as connect:
        if ack_mode == "valid":
            await gateway.submit_model_pass(
                context, bundle, HopDeploymentId(ingress, worker_id)
            )
        else:
            with pytest.raises(RuntimeError):
                await gateway.submit_model_pass(
                    context, bundle, HopDeploymentId(ingress, worker_id)
                )
        connect.assert_called_once_with("grpc+tcp://remote-worker:8811")

    directory.get_service_instance_by_service_id.assert_awaited_once_with(worker_id)
    assert client.do_put.call_args.args[0].command == SUBMIT_CONTRIBUTION_COMMAND
    batch = writer.write_batch.call_args.args[0]
    validate_contribution_schema(batch.schema)
    decoded = HopInvocationContributionMetadataDto.model_validate_json(
        batch.column(batch.schema.get_field_index(CONTEXT_FIELD_NAME))[0].as_py()
    )
    assert decoded.source_context.hop.kind == "external"
    assert (
        decoded.source_context.hop_invocation_id == context.external_src_invocation_id
    )
    assert (
        decoded.source_context.model_pass_context.model_pass_id
        == context.model_pass_context.model_pass_id
    )
    assert decoded.destination_hop == metadata.destination_hop
    writer.done_writing.assert_called_once()
    writer.close.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize("scalar", [False, True])
def test_ingress_encoding_preserves_shapes_dtypes_and_noncontiguous_inputs(
    scalar: bool,
) -> None:
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    value = (
        np.asarray(2, dtype=np.float32)
        if scalar
        else np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    )
    metadata = HopInvocationContributionMetadataDto(
        source_context=external_source_context_to_dto(
            ExternalSourceInvocationContext(_context())
        ),
        destination_hop=IngressHopDto(
            ingress_replica_id=PlannedIngressReplicaId(variant, 0)
        ),
    )
    batch = ArrowWorkerIngressGateway._encode_contribution(metadata, {"input": value})
    restored = batch.column(batch.schema.get_field_index("input"))[0].to_numpy()
    np.testing.assert_array_equal(restored, value)
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype


def _result_batch(context: ModelPassContextDto, bundle: TensorMap) -> pa.RecordBatch:
    metadata = ModelPassResultMetadataDto(model_pass_context=context)
    arrays = [pa.array([metadata.model_dump_json()], type=pa.large_string())]
    arrays.extend(
        pa.FixedShapeTensorArray.from_numpy_ndarray(np.expand_dims(value, 0))
        for value in bundle.values()
    )
    return pa.RecordBatch.from_arrays(
        arrays, schema=build_result_payload_schema(bundle)
    )


@pytest.mark.asyncio
async def test_sink_resolves_result_capability_and_retrieves_correlated_tensors() -> (
    None
):
    context = _context()
    dto = model_pass_context_to_dto(context)
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    worker_id = ServiceId()
    directory = _directory(worker_id, CapabilityType.RESULT_RETRIEVE)
    bundle = {"result": np.arange(4, dtype=np.float32)}
    writer, reader, client = MagicMock(), MagicMock(), MagicMock()
    reader.read_all.return_value = pa.Table.from_batches([_result_batch(dto, bundle)])
    client.do_exchange.return_value = (writer, reader)

    with patch(
        "front_end.adapters.outbound.arrow_worker_sink_gateway.flight.connect",
        return_value=client,
    ) as connect:
        result = await ArrowWorkerSinkGateway(directory).retrieve_result(
            context, HopDeploymentId(PlannedSinkReplicaId(variant, 1), worker_id)
        )
        connect.assert_called_once_with("grpc+tcp://remote-worker:8811")
    np.testing.assert_array_equal(result["result"], bundle["result"])
    assert client.do_exchange.call_args.args[0].command == RETRIEVE_RESULT_COMMAND
    assert writer.write_batch.call_args.args[0].column(0)[0].as_py() == str(
        context.model_pass_id
    )
    writer.done_writing.assert_called_once()
    client.close.assert_called_once()


def test_sink_rejects_a_result_for_another_pass() -> None:
    expected = model_pass_context_to_dto(_context())
    other = expected.model_copy(update={"model_pass_id": uuid4()})
    batch = _result_batch(other, {"result": np.ones(1, dtype=np.float32)})
    with pytest.raises(ValueError, match="context does not match"):
        ArrowWorkerSinkGateway._decode_result(batch, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("interface_count", [0, 2])
async def test_discovery_rejects_missing_or_ambiguous_capability(
    interface_count: int,
) -> None:
    worker_id = ServiceId()
    directory = _directory(worker_id, CapabilityType.HOP_INVOCATION)
    instance = directory.get_service_instance_by_service_id.return_value
    interface = instance.capabilities[0].interfaces[0]
    instance.capabilities[0].interfaces[:] = [interface] * interface_count
    with pytest.raises(ValueError, match="exactly one Arrow"):
        await get_arrow_worker_endpoint(
            directory, worker_id, CapabilityType.HOP_INVOCATION
        )
