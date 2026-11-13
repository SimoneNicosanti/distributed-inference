from dataclasses import replace
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from test.controller.application.optimization.test_compact_optimization_problem_factory import (
    _context,
    _factory,
    _profile,
    _request,
)

from controller.adapters.outbound.optimizer.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as ControllerMapper,
)
from controller.application.optimization.formulation.compact_optimization_problem_factory import (
    CompactOptimizationProblemBuildError,
)
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelInputConcreteShapes as ControllerShapes,
)
from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from integration.dto.optimizer.optimization_problem_dto import (
    CompactOptimizationProblemDto,
)
from integration.optimizer.http.optimizer_http_paths import OPTIMIZE_PATH
from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.adapters.inbound.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper,
)
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    FlowData,
    ModelConfigurationKey,
    ObjectiveKind,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.local_worker_config import NoLocalWorkerConfig
from optimizer.domain.problem.other.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import ServiceId
from shared.user.user import UserId


def single_configuration_problem(variant_count=1):
    flow = FlowId(UserId(UUID(int=1)), UUID(int=2))
    worker = ServiceId(UUID(int=3))
    metric = PerformanceMetric(PerformanceMetricType.ACCURACY, "dataset", 0.9)
    graph = VariantData(
        accuracies=(metric,),
        layers=("input", "compute", "output"),
        input_layers=("input",),
        output_layers=("output",),
        edges=(
            EdgeDescription("input", "compute", frozenset({"x"})),
            EdgeDescription("compute", "output", frozenset({"y"})),
        ),
        weight_bytes_by_layer={"input": 0, "compute": 10, "output": 0},
    )
    variants, configurations, costs = {}, {}, {}
    for index in range(variant_count):
        variant = ModelVariantId(ModelId("model"), f"variant-{index + 1}")
        batch, cost = (1, 0.1) if index == 0 else (4, 0.2)
        key = ModelConfigurationKey(
            variant, ModelInputConcreteShapes({"x": (batch, 3)})
        )
        variants[variant] = graph
        configurations[key] = ConfigurationData(
            batch, {"x": batch * 12, "y": batch * 4}, batch * 12
        )
        costs[key, worker] = {"input": 0.0, "compute": cost, "output": 0.0}
    return CompactOptimizationProblem(
        variants=variants,
        configurations=configurations,
        execution_times_s=costs,
        flows={
            flow: FlowData(
                1.0,
                10.0,
                metric,
                1000.0,
                tuple(configurations),
                (worker,),
                FlowEndpointPlacement(AnyPlacement(), AnyPlacement()),
                NoLocalWorkerConfig(),
            )
        },
        workers={worker: WorkerData(Backend.CPU, 100_000_000)},
        connections=(),
        policy=OptimizationPolicy(
            1, 1, 0.8, 0.7, ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
        ),
    )


def single_shape_context():
    profile = _profile()
    profile = replace(profile, shape_profiles=(profile.shape_profiles[0],))
    context = _context(profile)
    shapes = profile.shape_profiles[0].input_shapes
    execution = {
        worker: replace(
            profiles,
            execution_profiles={
                variant: replace(
                    measured,
                    shape_execution_profiles=tuple(
                        shape
                        for shape in measured.shape_execution_profiles
                        if shape.input_shapes == shapes
                    ),
                )
                for variant, measured in profiles.execution_profiles.items()
            },
        )
        for worker, profiles in context.execution_profiles_by_worker.items()
    }
    return OptimizationContext(
        static_profiles_by_variant=context.static_profiles_by_variant,
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=execution,
    )


async def controller_problem():
    request = _request()
    return await _factory().build_optimization_problem(
        {request.flow_id: request}, single_shape_context()
    )


def test_distinct_variants_and_flows_can_share_their_single_configuration():
    problem = single_configuration_problem(2)
    assert len(problem.variants) == len(problem.configurations) == 2
    flow = next(iter(problem.flows))
    other = replace(flow, flow_id=UUID(int=9))
    shared = replace(
        problem, flows={flow: problem.flows[flow], other: problem.flows[flow]}
    )
    assert shared.flows[flow].configurations == shared.flows[other].configurations


@pytest.mark.parametrize(
    "invalid",
    ["two_shapes", "missing_configuration", "unconfigured_variant", "unknown_variant"],
)
def test_optimizer_domain_enforces_exactly_one_configuration_per_variant(invalid):
    problem = single_configuration_problem()
    variants, configurations = dict(problem.variants), dict(problem.configurations)
    key = next(iter(configurations))
    other_variant = ModelVariantId(ModelId("other"), "fp32")
    if invalid == "two_shapes":
        configurations[
            replace(key, input_shapes=ModelInputConcreteShapes({"x": (4, 3)}))
        ] = configurations[key]
    elif invalid == "missing_configuration":
        configurations.clear()
    elif invalid == "unconfigured_variant":
        variants[other_variant] = variants[key.model_variant_id]
    else:
        configurations = {
            replace(key, model_variant_id=other_variant): configurations[key]
        }
    with pytest.raises(ValueError, match="exactly one configuration per variant"):
        replace(problem, variants=variants, configurations=configurations)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", ["two_shapes", "missing_configuration", "unconfigured_variant"]
)
async def test_controller_domain_enforces_the_same_cardinality(invalid):
    problem = await controller_problem()
    key = next(iter(problem.configurations))
    configurations, variants = dict(problem.configurations), dict(problem.variants)
    if invalid == "two_shapes":
        configurations[
            replace(
                key, input_shapes=ControllerShapes({"features": (4, 3), "mask": (4,)})
            )
        ] = configurations[key]
    elif invalid == "missing_configuration":
        configurations.clear()
    else:
        variants[ModelVariantId(ModelId("other"), "fp32")] = variants[
            key.model_variant_id
        ]
    with pytest.raises(ValueError, match="exactly one configuration per variant"):
        replace(problem, variants=variants, configurations=configurations)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape_count", [0, 2])
async def test_factory_rejects_profiles_without_one_unambiguous_shape(shape_count):
    profile = _profile()
    profile = replace(profile, shape_profiles=profile.shape_profiles[:shape_count])
    request = _request()
    with pytest.raises(
        CompactOptimizationProblemBuildError,
        match="exactly one configuration per variant",
    ):
        await _factory().build_optimization_problem(
            {request.flow_id: request}, _context(profile)
        )


@pytest.mark.asyncio
async def test_factory_and_dto_mappers_preserve_one_complete_shape_per_variant():
    original = await controller_problem()
    dto = ControllerMapper.to_dto(original)
    dto = CompactOptimizationProblemDto.model_validate_json(dto.model_dump_json())
    mapped = OptimizationProblemMapper.from_dto(dto)
    assert len(original.variants) == len(original.configurations) == 1
    assert len(mapped.variants) == len(mapped.configurations) == 1
    key = next(iter(mapped.configurations))
    assert key.input_shapes.input_shapes == {"features": (1, 3), "mask": (1,)}
    assert all(pair[0] == key for pair in mapped.execution_times_s)


def _duplicate_configuration(dto):
    original = dto.configurations[0]
    extra = original.model_copy(
        update={
            "key": original.key.model_copy(
                update={
                    "input_shapes": ModelInputConcreteShapesDto(
                        input_shapes={
                            name: (4, *shape[1:])
                            for name, shape in original.key.input_shapes.input_shapes.items()
                        }
                    )
                }
            ),
            "batch_size": 4,
            "tensor_bytes": {
                tensor: size * 4 for tensor, size in original.tensor_bytes.items()
            },
            "max_cut_size_bytes": original.max_cut_size_bytes * 4,
        }
    )
    payload = dto.model_dump(mode="json")
    payload["configurations"].append(extra.model_dump(mode="json"))
    return payload


@pytest.mark.asyncio
async def test_dto_rejects_two_shapes_of_the_same_variant():
    dto = ControllerMapper.to_dto(await controller_problem())
    with pytest.raises(ValidationError, match="exactly one configuration per variant"):
        CompactOptimizationProblemDto.model_validate(_duplicate_configuration(dto))


@pytest.mark.asyncio
async def test_dto_rejects_a_declared_variant_without_configuration():
    dto = ControllerMapper.to_dto(await controller_problem())
    extra = dto.variants[0].model_copy(
        update={"model_variant_id": ModelVariantId(ModelId("other"), "fp32")}
    )
    payload = dto.model_dump(mode="json")
    payload["variants"].append(extra.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="exactly one configuration per variant"):
        CompactOptimizationProblemDto.model_validate(payload)


@pytest.mark.asyncio
async def test_http_rejects_multiple_configurations_before_optimizer_invocation():
    class NeverOptimize(OptimizerController):
        calls = 0

        async def optimize(self, problem):
            self.calls += 1
            raise AssertionError("Invalid requests must not reach the optimizer")

    backend = NeverOptimize()
    app = FastAPI()
    app.include_router(build_optimizer_router(backend))
    dto = ControllerMapper.to_dto(await controller_problem())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        result = await client.post(
            f"http://optimizer.test{OPTIMIZE_PATH}",
            json={"problem": _duplicate_configuration(dto)},
        )
    assert result.status_code == 422
    assert "exactly one configuration per variant" in result.text
    assert backend.calls == 0
