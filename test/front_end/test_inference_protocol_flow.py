# mypy: disable-error-code="attr-defined,name-defined"

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pyarrow as pa
import pytest
from front_end.adapters.outbound.redis_hop_deployment_directory import (
    RedisHopDeploymentDirectory,
)
from redis import asyncio as redis_asyncio
from shared.plan.hop_deployment_id import HopDeploymentId

from catalog.redis.hop.keys_factory import StringHopDeploymentKeyFactory
from catalog.redis.service.keys_factory import build_catalog_key, build_service_hset_key
from catalog.redis.service.redis_service_directory import RedisServiceDirectory
from front_end.adapters.outbound.inference.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from front_end.adapters.outbound.inference.arrow_worker_sink_gateway import (
    ArrowWorkerSinkGateway,
)
from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from front_end.application.inference.default_inference_runner import (
    DefaultInferenceRunner,
)
from front_end.application.ports.inbound.model_loader import ModelLoader
from front_end.application.ports.outbound.inference.local_worker_identity_gateway import (
    LocalWorkerIdentityGateway,
)
from front_end.application.ports.outbound.optimization.optimization_gateway import (
    OptimizationGateway,
)
from front_end.composition.composition_root import FrontEndComposition
from front_end.main import build_http_app
from integration.dto.context.context_dto import (
    HopInvocationContributionAckDto,
    HopInvocationContributionMetadataDto,
    ModelPassResultMetadataDto,
)
from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.front_end.http.front_end_http_paths import FRONT_END_RUN_INFERENCE_PATH
from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from integration.service_instance.service_instance import (
    ArrowUrl,
    Capability,
    CapabilityInterface,
    CapabilityProtocol,
    CapabilityType,
    ServiceInstance,
)
from integration.shared.id_json import (
    hop_deployment_id_to_json,
    planned_invocable_hop_id_to_json,
)
from integration.sink.arrow.model_pass_result.arrow_model_pass_result_schema import (
    build_result_payload_schema,
)
from integration.worker.hop_contribution.arrow.arrow_hop_invocation_contribution_schema import (
    CONTEXT_FIELD_NAME,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


@pytest.mark.asyncio
async def test_http_to_ingress_and_sink_using_versioned_redis_contracts() -> None:
    flow_id = FlowId(user_id=UserId())
    version = PlanVersion(version_number=5)
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    ingress, sink = (
        PlannedIngressReplicaId(variant, 2),
        PlannedSinkReplicaId(variant, 3),
    )
    ingress_worker, sink_worker = ServiceId(), ServiceId()
    plan = FlowPlanDto(
        plan_version=version,
        model_specs=ModelSpecsDto(
            model_variant_id=variant,
            inputs=(InputSpecDto(name="input", shape=(2,), dtype="float32"),),
        ),
        ingress_replica_id=ingress,
        sink_replica_id=sink,
    )
    values = {
        (
            await StringFlowPlanKeyFactory.build_flow_plans_dict_key(flow_id),
            await StringFlowPlanKeyFactory.build_flow_plan_elem_key(version),
        ): plan.model_dump_json(),
        (
            await StringHopDeploymentKeyFactory.build_hop_deployments_key(version),
            planned_invocable_hop_id_to_json(ingress),
        ): hop_deployment_id_to_json(HopDeploymentId(ingress, ingress_worker)),
        (
            await StringHopDeploymentKeyFactory.build_hop_deployments_key(version),
            planned_invocable_hop_id_to_json(sink),
        ): hop_deployment_id_to_json(HopDeploymentId(sink, sink_worker)),
    }
    for worker, capability, location in (
        (ingress_worker, CapabilityType.HOP_INVOCATION, "grpc+tcp://ingress:8811"),
        (sink_worker, CapabilityType.RESULT_RETRIEVE, "grpc+tcp://sink:8820"),
    ):
        values[build_catalog_key(), build_service_hset_key(worker)] = ServiceInstance(
            service_id=worker,
            capabilities=[
                Capability(
                    type=capability,
                    interfaces=[
                        CapabilityInterface(
                            protocol=CapabilityProtocol.ARROW,
                            endpoint=ArrowUrl(location),
                        )
                    ],
                )
            ],
        ).model_dump_json()
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.get.return_value = str(version.version_number)

    async def hget(key: str, field: str) -> str | None:
        return values.get((key, field))

    redis.hget.side_effect = hget
    directory = RedisServiceDirectory(redis)
    runner = DefaultInferenceRunner(
        flow_plan_reader=RedisFlowPlanReader(redis, StringFlowPlanKeyFactory()),
        hop_deployment_directory=RedisHopDeploymentDirectory(
            redis, StringHopDeploymentKeyFactory
        ),
        ingress_gateway=ArrowWorkerIngressGateway(directory),
        sink_gateway=ArrowWorkerSinkGateway(directory),
    )
    composition = FrontEndComposition(
        inference_runner=runner,
        worker_identity_gateway=AsyncMock(spec=LocalWorkerIdentityGateway),
        model_loader=AsyncMock(spec=ModelLoader),
        optimization_gateway=AsyncMock(spec=OptimizationGateway),
    )
    ingress_client, ingress_writer, ack_reader = MagicMock(), MagicMock(), MagicMock()
    ingress_client.do_put.return_value = (ingress_writer, ack_reader)

    def sent_metadata() -> HopInvocationContributionMetadataDto:
        batch = ingress_writer.write_batch.call_args.args[0]
        return HopInvocationContributionMetadataDto.model_validate_json(
            batch.column(batch.schema.get_field_index(CONTEXT_FIELD_NAME))[0].as_py()
        )

    def ack() -> pa.Buffer:
        metadata = sent_metadata()
        return pa.py_buffer(
            HopInvocationContributionAckDto(
                source_context=metadata.source_context,
                destination_hop=metadata.destination_hop,
            )
            .model_dump_json()
            .encode()
        )

    # Build the acknowledgement only after the gateway has written the request.
    ack_reader.read.side_effect = lambda: (
        ack() if ack_reader.read.call_count == 1 else None
    )
    sink_client, sink_writer, result_reader = MagicMock(), MagicMock(), MagicMock()
    sink_client.do_exchange.return_value = (sink_writer, result_reader)
    expected = np.array([10, 20], dtype=np.float32)

    def result() -> pa.Table:
        metadata = sent_metadata()
        context = ModelPassResultMetadataDto(
            model_pass_context=metadata.source_context.model_pass_context
        )
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([context.model_dump_json()], type=pa.large_string()),
                pa.FixedShapeTensorArray.from_numpy_ndarray(
                    np.expand_dims(expected, 0)
                ),
            ],
            schema=build_result_payload_schema({"result": expected}),
        )
        return pa.Table.from_batches([batch])

    result_reader.read_all.side_effect = result

    def connect(location: str) -> MagicMock:
        return {
            "grpc+tcp://ingress:8811": ingress_client,
            "grpc+tcp://sink:8820": sink_client,
        }[location]

    request = RunInferenceRequest(
        flow_id=flow_id,
        bundle=TensorBundleDto.from_ndarrays({"input": np.ones(2, dtype=np.float32)}),
    )
    with patch(
        "front_end.adapters.outbound.arrow_worker_ingress_gateway.flight.connect",
        side_effect=connect,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=build_http_app(composition)),
            base_url="http://front-end.local",
        ) as client:
            response = await client.post(
                FRONT_END_RUN_INFERENCE_PATH, json=request.model_dump(mode="json")
            )

    assert response.status_code == 200, response.text
    decoded = RunInferenceResponse.model_validate(response.json())
    np.testing.assert_array_equal(decoded.bundle.to_ndarrays()["result"], expected)
    metadata = sent_metadata()
    assert metadata.source_context.hop.kind == "external"
    assert metadata.source_context.model_pass_context.plan_version == version
    assert (
        metadata.source_context.model_pass_context.model_invocation_context.flow_id
        == flow_id
    )
    assert sink_writer.write_batch.call_args.args[0].column(0)[0].as_py() == str(
        metadata.source_context.model_pass_context.model_pass_id
    )
