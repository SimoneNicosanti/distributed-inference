from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from redis import asyncio as redis_asyncio

from artifacts.storage.artifact_store import ArtifactStore
from front_end.adapters.outbound.inference.arrow_worker_ingress_gateway import (
    ArrowWorkerIngressGateway,
)
from front_end.adapters.outbound.inference.arrow_worker_sink_gateway import (
    ArrowWorkerSinkGateway,
)
from front_end.adapters.outbound.optimization.http_optimization_gateway import (
    HttpOptimizationGateway,
)
from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from front_end.application.inference.default_inference_runner import (
    DefaultInferenceRunner,
)
from front_end.application.ports.inbound.inference_runner import InferenceCoordinator
from front_end.application.ports.inbound.model_loader import ModelLoader
from front_end.application.ports.outbound.optimization.optimization_gateway import (
    OptimizationGateway,
)
from front_end.application.ports.outbound.inference.local_worker_identity_gateway import (
    LocalWorkerIdentityGateway,
)
from front_end.composition.composition_root import FrontEndComposition
from front_end.composition.default_composition_root import DefaultCompositionRoot
from front_end.domain.inference_result import InferenceResult
from front_end.domain.model_contract import ModelContract
from front_end.domain.model_specs import InputSpec, ModelSpecs
from front_end.main import build_http_app
from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
    FRONT_END_RUN_INFERENCE_PATH,
)
from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId


@pytest.mark.asyncio
async def test_composition_wires_all_gateways_and_closes_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://test-redis:6379/1")
    monkeypatch.setenv("CONTROLLER_HTTP_BASE_URL", "http://controller.local:9000")
    redis = AsyncMock(spec=redis_asyncio.Redis)
    root = DefaultCompositionRoot()

    @asynccontextmanager
    async def artifact_context() -> AsyncGenerator[ArtifactStore]:
        yield AsyncMock(spec=ArtifactStore)

    with (
        patch.object(root, "build_artifact_store_context", artifact_context),
        patch(
            "front_end.composition.default_composition_root.redis_asyncio.Redis.from_url",
            return_value=redis,
        ) as from_url,
    ):
        async with root.compose() as composition:
            assert isinstance(composition.inference_runner, DefaultInferenceRunner)
            assert isinstance(
                composition.inference_runner._flow_plan_reader, RedisFlowPlanReader
            )
            assert isinstance(
                composition.inference_runner._ingress_gateway, ArrowWorkerIngressGateway
            )
            assert isinstance(
                composition.inference_runner._sink_gateway, ArrowWorkerSinkGateway
            )
            assert isinstance(composition.optimization_gateway, HttpOptimizationGateway)
            assert (
                composition.optimization_gateway._base_url
                == "http://controller.local:9000"
            )
            from_url.assert_called_once_with("redis://test-redis:6379/1")
        redis.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_inference_builds_versioned_request_and_keeps_local_identity_separate() -> (
    None
):
    worker_id = ServiceId()
    flow_id = FlowId(user_id=UserId())
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    runner = AsyncMock(spec=InferenceCoordinator)
    runner.get_current_contract_for_flow.return_value = ModelContract(
        model_specs=ModelSpecs(variant, (InputSpec("input", (2,), "float32"),)),
        plan_version=PlanVersion(version_number=5),
    )
    runner.run_inference_for_request.return_value = InferenceResult(
        {"result": np.ones(1, dtype=np.float32)}
    )
    identity = AsyncMock(spec=LocalWorkerIdentityGateway)
    identity.get_worker_identity.return_value = worker_id
    composition = FrontEndComposition(
        inference_runner=runner,
        worker_identity_gateway=identity,
        model_loader=AsyncMock(spec=ModelLoader),
        optimization_gateway=AsyncMock(spec=OptimizationGateway),
    )
    request = RunInferenceRequest(
        flow_id=flow_id,
        bundle=TensorBundleDto.from_ndarrays({"input": np.ones(2, dtype=np.float32)}),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_http_app(composition)),
        base_url="http://front-end.local",
    ) as client:
        response = await client.post(
            FRONT_END_RUN_INFERENCE_PATH, json=request.model_dump(mode="json")
        )
        identity_response = await client.get(FRONT_END_LOCAL_WORKER_IDENTITY_PATH)

    assert response.status_code == 200
    decoded = RunInferenceResponse.model_validate(response.json())
    np.testing.assert_array_equal(
        decoded.bundle.to_ndarrays()["result"], np.ones(1, dtype=np.float32)
    )
    runner.get_current_contract_for_flow.assert_awaited_once_with(flow_id)
    forwarded = runner.run_inference_for_request.await_args.args[0]
    assert forwarded.flow_id == flow_id
    assert forwarded.plan_version == PlanVersion(version_number=5)
    np.testing.assert_array_equal(
        forwarded.bundle["input"], np.ones(2, dtype=np.float32)
    )
    assert identity_response.json() == {"service_id": str(worker_id.id)}
    identity.get_worker_identity.assert_awaited_once()
