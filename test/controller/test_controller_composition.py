from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from redis import asyncio as redis_asyncio
from test.controller.application.optimization.test_compact_optimization_problem_factory import (
    _context,
    _profile,
    _request,
)

from advertisement.advertising_manager import AdvertisingManager
from controller.adapters.outbound.store.in_memory_optimization_solution_store import (
    InMemoryOptimizationSolutionStore,
)
from controller.application.deployment.default_actuation_manager import (
    DefaultActuationManager,
)
from controller.application.deployment.default_solution_to_plan_converter import (
    DefaultSolutionToPlanConverter,
)
from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.ports.outbound.directory.worker_directory_gateway import (
    WorkerDirectoryGateway,
)
from controller.composition.default_composition_root import DefaultCompositionRoot
from controller.domain.backend import Backend
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.main import build_service_instance
from integration.optimizer.http.optimizer_http_schema import (
    OptimizeRequest,
    OptimizeResponse,
)
from shared.plan.plan_version import PlanVersion


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "http", "timeout"])
async def test_composed_manager_sends_problem_then_enriches_and_builds_local_plans(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    monkeypatch.setenv("OPTIMIZER_HTTP_BASE_URL", "http://optimizer.test:8100/")
    monkeypatch.setenv("OPTIMIZER_TIMEOUT_S", "240")
    profile = _profile()
    profile = replace(profile, shape_profiles=(profile.shape_profiles[0],))
    context = _context(profile)
    context = OptimizationContext(
        static_profiles_by_variant=context.static_profiles_by_variant,
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker={
            worker: replace(
                profiles,
                execution_profiles={
                    variant: replace(
                        measured,
                        shape_execution_profiles=tuple(
                            shape
                            for shape in measured.shape_execution_profiles
                            if shape.input_shapes
                            == profile.shape_profiles[0].input_shapes
                        ),
                    )
                    for variant, measured in profiles.execution_profiles.items()
                },
            )
            for worker, profiles in context.execution_profiles_by_worker.items()
        },
    )
    context_builder = AsyncMock(spec=OptimizationContextBuilder)
    context_builder.build_optimization_context.return_value = context
    request = _request()
    store = InMemoryOptimizationSolutionStore()

    def respond(http_request: httpx.Request) -> httpx.Response:
        assert (
            str(http_request.url) == "http://optimizer.test:8100/optimizer/optimization"
        )
        assert http_request.extensions["timeout"]["read"] == 240
        sent = OptimizeRequest.model_validate_json(http_request.content).problem
        assert sent.policy.max_components == 5
        assert sent.policy.priority_count == 3
        assert sent.policy.compute_utilization_limit == 0.8
        assert sent.policy.network_utilization_limit == 0.8
        assert sent.policy.objective.value == "arrival_weighted_response_time"
        if failure == "http":
            return httpx.Response(503)
        if failure == "timeout":
            raise httpx.ReadTimeout("optimizer timed out", request=http_request)
        flow = sent.flows[0]
        ingress, compute = flow.workers[:2]
        variant = sent.variants[0]
        response = OptimizeResponse.model_validate(
            {
                "solution": {
                    "flow_solutions": [
                        {
                            "flow_id": flow.flow_id,
                            "ingress_worker_id": ingress,
                            "sink_worker_id": compute,
                            "model_variant_id": variant.model_variant_id,
                            "components": [
                                {
                                    "component_index": index,
                                    "worker_id": worker,
                                    "backend": backend,
                                    "layer_keys": layers,
                                    "input_layer_keys": inputs,
                                    "output_layer_keys": outputs,
                                    "tensor_contract": {
                                        "input_tensor_keys": tensor_inputs,
                                        "output_tensor_keys": tensor_outputs,
                                    },
                                    "priority": 1,
                                    "mean_execution_time_s": 0,
                                    "mean_execution_queue_time_s": 0,
                                    "mean_transmission_time_s": 0,
                                    "transmission_latency_s": 0,
                                }
                                for index, worker, backend, layers, inputs, outputs, tensor_inputs, tensor_outputs in (
                                    (
                                        1,
                                        ingress,
                                        "cpu",
                                        ("input",),
                                        ("input",),
                                        (),
                                        ("features", "mask"),
                                        ("features", "mask"),
                                    ),
                                    (
                                        2,
                                        compute,
                                        "gpu",
                                        ("a", "b", "output"),
                                        (),
                                        ("output",),
                                        ("features", "mask"),
                                        ("prediction",),
                                    ),
                                )
                            ],
                            "cuts": [
                                {
                                    "source_component_index": 1,
                                    "destination_component_index": 2,
                                    "tensor_keys": ("features", "mask"),
                                }
                            ],
                            "mean_batch_fill_time_s": 0,
                            "mean_response_time_s": 0,
                            "input_shapes": sent.configurations[0].key.input_shapes,
                        }
                    ],
                    "objective_value": 0,
                    "is_optimal": True,
                }
            }
        )
        return httpx.Response(200, json=response.model_dump(mode="json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        root = DefaultCompositionRoot()
        with patch.object(
            root, "build_optimization_context_builder", return_value=context_builder
        ):
            manager = root.build_optimization_manager(
                store,
                AsyncMock(spec=redis_asyncio.Redis),
                AsyncMock(spec=WorkerDirectoryGateway),
                client,
            )
        if failure:
            error = httpx.HTTPStatusError if failure == "http" else httpx.ReadTimeout
            with pytest.raises(error):
                await manager.optimize_inference_distribution(
                    {request.flow_id: request}
                )
            assert await store.get_current_solution() is None
            return
        await manager.optimize_inference_distribution({request.flow_id: request})

    enriched = await store.get_current_solution()
    assert enriched is not None
    assert enriched.solution.flow_solutions[0].components[1].backend is Backend.GPU
    assert set(
        enriched.input_tensor_properties_by_variant[profile.model_variant_id]
    ) == {"features", "mask"}
    (
        worker_plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(1), enriched
    )
    assert len(worker_plans) == 2
    assert set(flow_plans) == {request.flow_id}
    assert flow_plans[request.flow_id].plan_version == PlanVersion(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_composition_closes_clients_and_stops_background_managers(
    fail: bool,
) -> None:
    redis = AsyncMock(spec=redis_asyncio.Redis)
    redis.__aenter__.return_value = redis
    redis.ping = AsyncMock(return_value=True)
    advertising = AsyncMock(spec=AdvertisingManager)
    actuation = AsyncMock(spec=DefaultActuationManager)
    clients: list[httpx.AsyncClient] = []
    client_factory = httpx.AsyncClient

    def create_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = client_factory(*args, **kwargs)
        clients.append(client)
        return client

    with (
        patch(
            "controller.composition.default_composition_root.redis_asyncio.Redis",
            return_value=redis,
        ),
        patch(
            "controller.composition.default_composition_root.AdvertisingManager",
            return_value=advertising,
        ),
        patch(
            "controller.composition.default_composition_root.DefaultActuationManager",
            return_value=actuation,
        ),
        patch(
            "controller.composition.default_composition_root.httpx.AsyncClient",
            side_effect=create_client,
        ),
    ):
        with (
            pytest.raises(RuntimeError, match="request failed")
            if fail
            else nullcontext()
        ):
            async with DefaultCompositionRoot().compose(
                build_service_instance()
            ) as manager:
                assert manager._optimizer_gateway._http_client in clients
                advertising.start.assert_awaited_once()
                actuation.start.assert_awaited_once()
                if fail:
                    raise RuntimeError("request failed")
        advertising.stop.assert_awaited_once()
        actuation.stop.assert_awaited_once()
        redis.__aexit__.assert_awaited_once()
    assert clients and all(client.is_closed for client in clients)
