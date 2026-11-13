from collections.abc import Mapping
from dataclasses import replace
from typing import cast
from uuid import UUID

import pytest

from controller.application.optimization.formulation.compact_optimization_problem_factory import (
    CompactOptimizationProblemBuildError,
    CompactOptimizationProblemFactory,
)
from controller.domain.backend import Backend
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.problem.compact_optimization_problem import (
    CompactOptimizationProblem,
    ModelConfigurationKey,
    ObjectiveKind,
    OptimizationPolicy,
    VariantData,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
)
from controller.domain.optimization.request.qos_constraints import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from controller.domain.profile.model_variant_execution_profile import (
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelVariantExecutionProfile,
    ShapeExecutionProfile,
    WorkerExecutionProfiles,
)
from controller.domain.profile.model_variant_static_profile import (
    LayerProperty,
    ModelInputConcreteShapes,
    ModelVariantInfo,
    ModelVariantStaticProfile,
    ShapeProfile,
    TensorProperty,
    TensorShapeProperty,
    WeightProperty,
)
from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from controller.domain.profile.network_topology import (
    NetworkConnection,
    NetworkTopology,
)
from controller.domain.profile.resource_profile import (
    CpuProfile,
    GpuProfile,
    RamProfile,
    WorkerResourceProfile,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId


def _worker(index: int) -> WorkerId:
    return ServiceId(UUID(int=index))


def _metric(value: float = 0.9) -> PerformanceMetric:
    return PerformanceMetric(PerformanceMetricType.ACCURACY, "dataset", value)


def _shapes(batch: int) -> ModelInputConcreteShapes:
    return ModelInputConcreteShapes({"features": (batch, 3), "mask": (batch,)})


def _profile() -> ModelVariantStaticProfile:
    return ModelVariantStaticProfile(
        model_variant_id=ModelVariantId(ModelId("model"), "fp32"),
        info=ModelVariantInfo(ModelTask.CLASSIFICATION, (_metric(),)),
        topology=ModelVariantTopology(
            layers={
                key: LayerDescription(key, "Compute", key == "input", key == "output")
                for key in ("input", "a", "b", "output")
            },
            edges=(
                EdgeDescription("input", "a", frozenset({"features", "mask"})),
                EdgeDescription("input", "b", frozenset({"features"})),
                EdgeDescription("a", "b", frozenset({"hidden"})),
                EdgeDescription("b", "output", frozenset({"prediction"})),
            ),
        ),
        invariant_layer_properties={
            "a": LayerProperty({"weight": WeightProperty("float32", (2, 3), 4)})
        },
        invariant_tensor_properties={
            "features": TensorProperty("float32", 4, ("batch", 3)),
            "mask": TensorProperty("bool", 1, ("batch",)),
            "hidden": TensorProperty("float32", 4, ("batch", 2)),
            "prediction": TensorProperty("float32", 4, ("batch", 1)),
        },
        shape_profiles=tuple(
            ShapeProfile(
                input_shapes=_shapes(batch),
                layer_properties={},
                tensor_properties={
                    "features": TensorShapeProperty((batch, 3)),
                    "mask": TensorShapeProperty((batch,)),
                    "hidden": TensorShapeProperty((batch, 2)),
                    "prediction": TensorShapeProperty((batch, 1)),
                },
                max_cut_size_mib=(25 * batch + 0.25) / 1024**2,
            )
            for batch in (1, 4)
        ),
        optimization_contractions=(("a", "b"),),
    )


def _resource(worker: WorkerId, *, gpu: bool = False) -> WorkerResourceProfile:
    return WorkerResourceProfile(
        worker_id=worker,
        cpu_profile=CpuProfile(4, 4),
        ram_profile=RamProfile(4096, 1024, 3072, 128),
        gpu_profiles=(GpuProfile(0, "gpu-id", "GPU", 128, 32, 96),) if gpu else (),
    )


def _context(
    profile: ModelVariantStaticProfile | None = None,
    execution: Mapping[WorkerId, WorkerExecutionProfiles] | None = None,
) -> OptimizationContext:
    profile = profile or _profile()
    if execution is None:
        execution = {
            _worker(index): WorkerExecutionProfiles(
                _worker(index),
                {
                    profile.model_variant_id: ModelVariantExecutionProfile(
                        tuple(
                            ShapeExecutionProfile(
                                # Match static profiles independently of input order.
                                ModelInputConcreteShapes(
                                    dict(
                                        reversed(
                                            tuple(_shapes(batch).input_shapes.items())
                                        )
                                    )
                                ),
                                tuple(
                                    BackendExecutionProfile(
                                        backend,
                                        {
                                            "a": LayerExecutionProfile(
                                                batch * factor, 999999
                                            ),
                                            "b": LayerExecutionProfile(
                                                batch * factor * 2
                                            ),
                                        },
                                    )
                                    for backend, factor in (
                                        (Backend.CPU, 0.1),
                                        (Backend.GPU, 0.01),
                                    )
                                ),
                            )
                            for batch in (1, 4)
                        )
                    )
                },
            )
            for index in (1, 2)
        }
    return OptimizationContext(
        static_profiles_by_variant={profile.model_variant_id: profile},
        network_topology=NetworkTopology(
            workers=tuple(_worker(index) for index in (1, 2, 3, 4)),
            connections=(
                NetworkConnection(_worker(1), _worker(2), 1000, 0.02),
                NetworkConnection(_worker(1), _worker(4), 2000, 0.04),
            ),
        ),
        resource_profiles_by_worker={
            _worker(index): _resource(_worker(index), gpu=index == 2)
            for index in (1, 2, 3)
        },
        execution_profiles_by_worker=execution,
    )


def _request() -> OptimizationRequest:
    return OptimizationRequest(
        flow_id=FlowId(UserId()),
        task=ModelTask.CLASSIFICATION,
        stream_info=StreamInfo(2),
        qos_requirements=QoSRequirements(10, _metric(0.8), 50),
        flow_endpoint_placement=FlowEndpointPlacement(
            FixedPlacement(_worker(1)), AnyPlacement()
        ),
        local_worker_config=ValidLocalWorkerConfig(_worker(1), False),
    )


def _factory() -> CompactOptimizationProblemFactory:
    return CompactOptimizationProblemFactory(
        OptimizationPolicy(5, 3, 0.8, 0.8, ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    )


@pytest.mark.asyncio
async def test_projects_all_shapes_with_shared_graph_and_backend_specific_costs() -> (
    None
):
    request = _request()
    problem = await _factory().build_optimization_problem(
        {request.flow_id: request}, _context()
    )
    variant_id = _profile().model_variant_id
    assert isinstance(problem, CompactOptimizationProblem)
    assert set(problem.variants) == {variant_id}
    assert len(problem.configurations) == 2
    assert len(problem.flows[request.flow_id].configurations) == 2
    variant = problem.variants[variant_id]
    assert variant.weight_bytes_by_layer == {"input": 0, "a": 24, "b": 0, "output": 0}
    assert len([edge for edge in variant.edges if "features" in edge.tensors]) == 2
    for batch in (1, 4):
        key = ModelConfigurationKey(variant_id, _shapes(batch))
        configuration = problem.configurations[key]
        assert configuration.batch_size == batch
        assert configuration.tensor_bytes == {
            "features": 12 * batch,
            "mask": batch,
            "hidden": 8 * batch,
            "prediction": 4 * batch,
        }
        assert configuration.max_cut_size_bytes == 25 * batch + 1
        assert problem.execution_times_s[key, _worker(1)]["a"] == pytest.approx(
            0.1 * batch
        )
        assert problem.execution_times_s[key, _worker(2)]["a"] == pytest.approx(
            0.01 * batch
        )
        assert problem.execution_times_s[key, _worker(3)] == {"input": 0, "output": 0}
    assert problem.workers[_worker(1)].available_inference_memory_bytes == 3072
    assert problem.workers[_worker(2)].available_inference_memory_bytes == 96
    assert problem.workers[_worker(2)].backend is Backend.GPU
    assert _worker(4) not in problem.workers
    assert problem.connections == (
        NetworkConnection(_worker(1), _worker(2), 1000, 0.02),
    )
    flow = problem.flows[request.flow_id]
    assert flow.arrival_rate_per_s == 2
    assert flow.max_response_time_s == 10
    assert flow.max_energy_consumption == 50
    assert flow.endpoint_placement == request.flow_endpoint_placement
    assert flow.local_worker_config == request.local_worker_config


@pytest.mark.asyncio
async def test_each_flow_gets_all_shapes_only_of_quality_eligible_variants() -> None:
    context = _context()
    first_profile = _profile()
    other_id = ModelVariantId(ModelId("other"), "fp32")
    other = replace(
        first_profile,
        model_variant_id=other_id,
        info=replace(first_profile.info, accuracies=(_metric(0.85),)),
    )
    context = OptimizationContext(
        static_profiles_by_variant={
            first_profile.model_variant_id: first_profile,
            other_id: other,
        },
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=context.execution_profiles_by_worker,
    )
    relaxed = _request()
    strict = replace(
        _request(), qos_requirements=QoSRequirements(10, _metric(0.88), 50)
    )
    problem = await _factory().build_optimization_problem(
        {relaxed.flow_id: relaxed, strict.flow_id: strict}, context
    )
    assert len(problem.variants) == 2
    assert len(problem.configurations) == 4
    assert len(problem.flows[relaxed.flow_id].configurations) == 4
    assert len(problem.flows[strict.flow_id].configurations) == 2
    assert {
        key.model_variant_id for key in problem.flows[strict.flow_id].configurations
    } == {first_profile.model_variant_id}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", [PerformanceMetricType.MAE, PerformanceMetricType.RMSE]
)
async def test_error_metrics_use_the_lower_is_better_direction(
    kind: PerformanceMetricType,
) -> None:
    profile = _profile()
    profile = replace(
        profile,
        info=replace(
            profile.info, accuracies=(PerformanceMetric(kind, "dataset", 0.1),)
        ),
    )
    request = replace(
        _request(),
        qos_requirements=QoSRequirements(
            10, PerformanceMetric(kind, "dataset", 0.2), 50
        ),
    )
    problem = await _factory().build_optimization_problem(
        {request.flow_id: request}, _context(profile)
    )
    assert len(problem.configurations) == 2


@pytest.mark.asyncio
async def test_missing_gpu_profile_does_not_fall_back_to_cpu_and_partial_layers_stay_absent() -> (
    None
):
    context = _context()
    variant_id = _profile().model_variant_id
    cpu = BackendExecutionProfile(Backend.CPU, {"a": LayerExecutionProfile(0.0)})
    execution = {
        worker: WorkerExecutionProfiles(
            worker,
            {
                variant_id: ModelVariantExecutionProfile(
                    (ShapeExecutionProfile(_shapes(1), (cpu,)),)
                )
            },
        )
        for worker in (_worker(1), _worker(2))
    }
    request = _request()
    problem = await _factory().build_optimization_problem(
        {request.flow_id: request}, _context(execution=execution)
    )
    key = ModelConfigurationKey(variant_id, _shapes(1))
    assert problem.execution_times_s[key, _worker(1)] == {
        "a": 0,
        "input": 0,
        "output": 0,
    }
    assert problem.execution_times_s[key, _worker(2)] == {"input": 0, "output": 0}
    assert (
        context.execution_profiles_by_worker[_worker(2)]
        .execution_profiles[variant_id]
        .shape_execution_profiles
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "missing_cut",
        "missing_tensor",
        "input_mismatch",
        "batch_mismatch",
        "no_shapes",
    ],
)
async def test_rejects_invalid_static_configurations(failure: str) -> None:
    profile = _profile()
    shape = profile.shape_profiles[0]
    if failure == "duplicate":
        shapes = (
            shape,
            replace(
                shape,
                input_shapes=ModelInputConcreteShapes(
                    dict(reversed(tuple(shape.input_shapes.input_shapes.items())))
                ),
            ),
        )
    elif failure == "missing_cut":
        shapes = (replace(shape, max_cut_size_mib=None),)
    elif failure == "missing_tensor":
        shapes = (replace(shape, tensor_properties={}),)
    elif failure == "input_mismatch":
        shapes = (
            replace(
                shape,
                input_shapes=ModelInputConcreteShapes(
                    {"features": (1, 4), "mask": (1,)}
                ),
            ),
        )
    elif failure == "batch_mismatch":
        shapes = (
            replace(
                shape,
                input_shapes=ModelInputConcreteShapes(
                    {"features": (1, 3), "mask": (2,)}
                ),
            ),
        )
    else:
        shapes = ()
    profile = replace(profile, shape_profiles=shapes)
    request = _request()
    with pytest.raises(CompactOptimizationProblemBuildError):
        await _factory().build_optimization_problem(
            {request.flow_id: request}, _context(profile, execution={})
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["duplicate_shape", "duplicate_backend", "unknown_shape", "unknown_layer"],
)
async def test_rejects_ambiguous_or_unmatched_execution_profiles(failure: str) -> None:
    profile = _profile()
    backend = BackendExecutionProfile(Backend.CPU, {"a": LayerExecutionProfile(0.1)})
    shape = ShapeExecutionProfile(_shapes(1), (backend,))
    if failure == "duplicate_shape":
        shapes = (shape, shape)
    elif failure == "duplicate_backend":
        shapes = (replace(shape, backend_profiles=(backend, backend)),)
    elif failure == "unknown_shape":
        shapes = (replace(shape, input_shapes=_shapes(2)),)
    else:
        shapes = (
            replace(
                shape,
                backend_profiles=(
                    replace(
                        backend, layer_profiles={"unknown": LayerExecutionProfile(0.1)}
                    ),
                ),
            ),
        )
    execution = {
        _worker(1): WorkerExecutionProfiles(
            _worker(1), {profile.model_variant_id: ModelVariantExecutionProfile(shapes)}
        )
    }
    request = _request()
    with pytest.raises(CompactOptimizationProblemBuildError):
        await _factory().build_optimization_problem(
            {request.flow_id: request}, _context(execution=execution)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "empty_requests",
        "wrong_flow_key",
        "quality",
        "dataset",
        "endpoint",
        "no_workers",
    ],
)
async def test_rejects_invalid_requests_or_unavailable_candidates(failure: str) -> None:
    request = _request()
    context = _context()
    if failure in {"quality", "dataset"}:
        metric = (
            _metric(0.99)
            if failure == "quality"
            else replace(_metric(), dataset="other")
        )
        request = replace(request, qos_requirements=QoSRequirements(10, metric, 50))
    elif failure == "endpoint":
        request = replace(
            request,
            flow_endpoint_placement=FlowEndpointPlacement(
                FixedPlacement(_worker(4)), AnyPlacement()
            ),
        )
    elif failure == "no_workers":
        context = OptimizationContext(
            static_profiles_by_variant=context.static_profiles_by_variant,
            network_topology=context.network_topology,
            resource_profiles_by_worker={},
            execution_profiles_by_worker={},
        )
    requests = {request.flow_id: request}
    if failure == "empty_requests":
        requests = {}
    elif failure == "wrong_flow_key":
        requests = {_request().flow_id: request}
    with pytest.raises(CompactOptimizationProblemBuildError):
        await _factory().build_optimization_problem(requests, context)


@pytest.mark.asyncio
async def test_problem_owns_immutable_copies_of_nested_cost_mappings() -> None:
    request = _request()
    problem = await _factory().build_optimization_problem(
        {request.flow_id: request}, _context()
    )
    variants = dict(problem.variants)
    times = {key: dict(values) for key, values in problem.execution_times_s.items()}
    copied = replace(problem, variants=variants, execution_times_s=times)
    variants.clear()
    key = next(iter(times))
    times[key].clear()
    assert copied.variants
    assert copied.execution_times_s[key]
    with pytest.raises(TypeError):
        variant_id, variant = next(iter(copied.variants.items()))
        cast(dict[ModelVariantId, VariantData], copied.variants)[variant_id] = variant
    with pytest.raises(TypeError):
        cast(dict[str, float], copied.execution_times_s[key])["a"] = 1.0
    assert isinstance(copied.execution_times_s, Mapping)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_components", 0),
        ("priority_count", True),
        ("compute_utilization_limit", 1.0),
        ("network_utilization_limit", float("nan")),
    ],
)
def test_rejects_invalid_optimization_policy(field: str, value: int | float) -> None:
    policy = OptimizationPolicy(
        5, 3, 0.8, 0.8, ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
    )
    with pytest.raises(ValueError):
        replace(policy, **{field: value})
