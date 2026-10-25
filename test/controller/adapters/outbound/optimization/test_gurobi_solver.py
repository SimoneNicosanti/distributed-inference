from dataclasses import replace
from uuid import UUID

import pytest

from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_execution_profiles,
)
from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    OptimizationInfeasibleError,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import OptimizationRequest
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.solver_config import (
    ObjectiveKind,
    ProblemConfig,
    SolverConfig,
)
from controller.domain.optimization.stream_info import StreamInfo
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
    ModelVariantExecutionProfile,
    ShapeExecutionProfile,
    WorkerExecutionProfiles,
)
from controller.domain.profile.model_variant_static_profile import (
    InputShapePoint,
    LayerProperty,
    LayerShapeProperty,
    ModelVariantInfo,
    ModelVariantStaticProfile,
    ShapePoint,
    ShapeProfile,
    TensorProperty,
    TensorShapeProperty,
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
    RamProfile,
    WorkerResourceProfile,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model import ModelId, ModelTask, ModelVisibility
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId

_INPUT = "input"
_COMPUTE = "compute"
_OUTPUT = "output"
_INPUT_TENSOR = "input_tensor"
_OUTPUT_TENSOR = "output_tensor"

_BRANCHED_INPUT = "branched_input"
_STEM = "stem"
_BRANCH_A_1 = "branch_a_1"
_BRANCH_A_2 = "branch_a_2"
_BRANCH_B = "branch_b"
_MERGE = "merge"
_HEAD = "head"
_BRANCHED_OUTPUT = "branched_output"
_BRANCHED_LAYERS = (
    _BRANCHED_INPUT,
    _STEM,
    _BRANCH_A_1,
    _BRANCH_A_2,
    _BRANCH_B,
    _MERGE,
    _HEAD,
    _BRANCHED_OUTPUT,
)
_BRANCHED_EDGES = (
    (_BRANCHED_INPUT, _STEM, "input_to_stem"),
    (_STEM, _BRANCH_A_1, "stem_fanout"),
    (_STEM, _BRANCH_B, "stem_fanout"),
    (_BRANCH_A_1, _BRANCH_A_2, "branch_a_1_output"),
    (_BRANCH_A_2, _MERGE, "branch_a_2_output"),
    (_BRANCH_B, _MERGE, "branch_b_output"),
    (_MERGE, _HEAD, "merge_output"),
    (_HEAD, _BRANCHED_OUTPUT, "head_output"),
)

_USER_ID = UserId(id=UUID("00000000-0000-0000-0000-000000000001"))
_MODEL_VARIANT_ID = ModelVariantId(
    model_id=ModelId(owner_id=_USER_ID, model_name="test-model"),
    variant_tag="batch-1",
)
_ALTERNATIVE_MODEL_VARIANT_ID = ModelVariantId(
    model_id=ModelId(owner_id=_USER_ID, model_name="test-model"),
    variant_tag="batch-1-fast",
)
_BRANCHED_MODEL_VARIANT_ID = ModelVariantId(
    model_id=ModelId(owner_id=_USER_ID, model_name="branched-test-model"),
    variant_tag="batch-1",
)
_PERFORMANCE = PerformanceMetric(
    type=PerformanceMetricType.MAP_AT_50,
    dataset="test",
    value=0.9,
)
_SHAPE_POINT = ShapePoint(
    input_shape_points=(InputShapePoint(name="input", shape=(1, 1)),)
)


def _worker(number: int) -> WorkerId:
    return ServiceId(id=UUID(int=number))


def _flow(number: int) -> FlowId:
    return FlowId(user_id=_USER_ID, flow_id=UUID(int=number))


def _static_profile(
    model_variant_id: ModelVariantId = _MODEL_VARIANT_ID,
) -> ModelVariantStaticProfile:
    layers = {
        _INPUT: LayerDescription(
            name=_INPUT,
            type="Input",
            is_input=True,
            is_output=False,
        ),
        _COMPUTE: LayerDescription(
            name=_COMPUTE,
            type="Compute",
            is_input=False,
            is_output=False,
        ),
        _OUTPUT: LayerDescription(
            name=_OUTPUT,
            type="Output",
            is_input=False,
            is_output=True,
        ),
    }
    return ModelVariantStaticProfile(
        model_variant_id=model_variant_id,
        info=ModelVariantInfo(
            visibility=ModelVisibility.PUBLIC,
            task=ModelTask.CLASSIFICATION,
            accuracies=(_PERFORMANCE,),
        ),
        topology=ModelVariantTopology(
            layers=layers,
            edges=(
                EdgeDescription(
                    source=_INPUT,
                    target=_COMPUTE,
                    tensors=frozenset({_INPUT_TENSOR}),
                ),
                EdgeDescription(
                    source=_COMPUTE,
                    target=_OUTPUT,
                    tensors=frozenset({_OUTPUT_TENSOR}),
                ),
            ),
        ),
        invariant_layer_properties={
            layer: LayerProperty(weight_properties={}) for layer in layers
        },
        invariant_tensor_properties={
            tensor: TensorProperty(
                data_type="float32",
                data_type_size_in_bytes=4,
                shape_expression=(1,),
            )
            for tensor in (_INPUT_TENSOR, _OUTPUT_TENSOR)
        },
        shape_profiles=(
            ShapeProfile(
                shape_point=_SHAPE_POINT,
                layer_properties={
                    layer: LayerShapeProperty(flops=0) for layer in layers
                },
                tensor_properties={
                    tensor: TensorShapeProperty(shape=(1,))
                    for tensor in (_INPUT_TENSOR, _OUTPUT_TENSOR)
                },
            ),
        ),
        optimization_contractions=(),
    )


def _branched_static_profile() -> ModelVariantStaticProfile:
    layers = {
        layer: LayerDescription(
            name=layer,
            type="Input"
            if layer == _BRANCHED_INPUT
            else "Output"
            if layer == _BRANCHED_OUTPUT
            else "Compute",
            is_input=layer == _BRANCHED_INPUT,
            is_output=layer == _BRANCHED_OUTPUT,
        )
        for layer in _BRANCHED_LAYERS
    }
    tensors = frozenset(tensor for _, _, tensor in _BRANCHED_EDGES)
    return ModelVariantStaticProfile(
        model_variant_id=_BRANCHED_MODEL_VARIANT_ID,
        info=ModelVariantInfo(
            visibility=ModelVisibility.PUBLIC,
            task=ModelTask.CLASSIFICATION,
            accuracies=(_PERFORMANCE,),
        ),
        topology=ModelVariantTopology(
            layers=layers,
            edges=tuple(
                EdgeDescription(
                    source=source,
                    target=target,
                    tensors=frozenset({tensor}),
                )
                for source, target, tensor in _BRANCHED_EDGES
            ),
        ),
        invariant_layer_properties={
            layer: LayerProperty(weight_properties={}) for layer in layers
        },
        invariant_tensor_properties={
            tensor: TensorProperty(
                data_type="float32",
                data_type_size_in_bytes=4,
                shape_expression=(1,),
            )
            for tensor in tensors
        },
        shape_profiles=(
            ShapeProfile(
                shape_point=_SHAPE_POINT,
                layer_properties={
                    layer: LayerShapeProperty(flops=0) for layer in layers
                },
                tensor_properties={
                    tensor: TensorShapeProperty(shape=(1,)) for tensor in tensors
                },
            ),
        ),
        optimization_contractions=(),
    )


def _variant_execution_profile(
    compute_time_s: float | None,
) -> ModelVariantExecutionProfile:
    layer_profiles: dict[LayerKey, LayerExecutionProfile] = {}
    if compute_time_s is not None:
        layer_profiles[_COMPUTE] = LayerExecutionProfile(
            execution_time_s=compute_time_s
        )
    return ModelVariantExecutionProfile(
        shape_execution_profiles=(
            ShapeExecutionProfile(
                shape_point=_SHAPE_POINT,
                layer_profiles=layer_profiles,
            ),
        )
    )


def _branched_execution_profile(
    layer_execution_times: dict[str, float],
) -> ModelVariantExecutionProfile:
    return ModelVariantExecutionProfile(
        shape_execution_profiles=(
            ShapeExecutionProfile(
                shape_point=_SHAPE_POINT,
                layer_profiles={
                    layer: LayerExecutionProfile(execution_time_s=execution_time)
                    for layer, execution_time in layer_execution_times.items()
                },
            ),
        )
    )


def _execution_profiles(
    worker: WorkerId,
    compute_time_s: float | None,
) -> WorkerExecutionProfiles:
    return WorkerExecutionProfiles(
        worker_id=worker,
        execution_profiles={
            _MODEL_VARIANT_ID: _variant_execution_profile(compute_time_s)
        },
    )


def _resource_profile(worker: WorkerId) -> WorkerResourceProfile:
    return WorkerResourceProfile(
        worker_id=worker,
        gpu_profiles=(),
        cpu_profile=CpuProfile(logical_cpu_count=1, capacity_cores=1),
        ram_profile=RamProfile(
            capacity_bytes=1_000_000,
            used_bytes=0,
            available_bytes=1_000_000,
            worker_rss_bytes=0,
        ),
    )


def _context(
    execution_times: dict[WorkerId, float | None],
    *,
    throughput_bytes_per_s: float = 1_000_000,
) -> OptimizationContext:
    workers = tuple(execution_times)
    return OptimizationContext(
        static_profiles_by_variant={_MODEL_VARIANT_ID: _static_profile()},
        network_topology=NetworkTopology(
            workers=workers,
            connections=tuple(
                NetworkConnection(
                    source=source,
                    target=target,
                    throughput_bytes_per_sec=throughput_bytes_per_s,
                    round_trip_time_s=0.002,
                )
                for source in workers
                for target in workers
                if source != target
            ),
        ),
        resource_profiles_by_worker={
            worker: _resource_profile(worker) for worker in workers
        },
        execution_profiles_by_worker={
            worker: _execution_profiles(worker, execution_time)
            for worker, execution_time in execution_times.items()
        },
    )


def _multi_variant_context(
    execution_times: dict[
        WorkerId,
        dict[ModelVariantId, float | None],
    ],
    *,
    throughput_bytes_per_s: float = 1_000_000,
) -> OptimizationContext:
    workers = tuple(execution_times)
    variants = {
        model_variant
        for worker_execution_times in execution_times.values()
        for model_variant in worker_execution_times
    }
    return OptimizationContext(
        static_profiles_by_variant={
            model_variant: _static_profile(model_variant) for model_variant in variants
        },
        network_topology=NetworkTopology(
            workers=workers,
            connections=tuple(
                NetworkConnection(
                    source=source,
                    target=target,
                    throughput_bytes_per_sec=throughput_bytes_per_s,
                    round_trip_time_s=0.002,
                )
                for source in workers
                for target in workers
                if source != target
            ),
        ),
        resource_profiles_by_worker={
            worker: _resource_profile(worker) for worker in workers
        },
        execution_profiles_by_worker={
            worker: WorkerExecutionProfiles(
                worker_id=worker,
                execution_profiles={
                    model_variant: _variant_execution_profile(execution_time)
                    for model_variant, execution_time in worker_execution_times.items()
                },
            )
            for worker, worker_execution_times in execution_times.items()
        },
    )


def _branched_context(
    layer_execution_times_by_worker: dict[WorkerId, dict[str, float]],
    connections: tuple[NetworkConnection, ...],
) -> OptimizationContext:
    workers = tuple(layer_execution_times_by_worker)
    return OptimizationContext(
        static_profiles_by_variant={
            _BRANCHED_MODEL_VARIANT_ID: _branched_static_profile()
        },
        network_topology=NetworkTopology(
            workers=workers,
            connections=connections,
        ),
        resource_profiles_by_worker={
            worker: _resource_profile(worker) for worker in workers
        },
        execution_profiles_by_worker={
            worker: WorkerExecutionProfiles(
                worker_id=worker,
                execution_profiles={
                    _BRANCHED_MODEL_VARIANT_ID: _branched_execution_profile(
                        layer_execution_times
                    )
                },
            )
            for worker, layer_execution_times in (
                layer_execution_times_by_worker.items()
            )
        },
    )


def _request(
    ingress: WorkerId,
    *,
    arrival_rate: float,
    max_response_time_s: float,
    flow_id: FlowId | None = None,
) -> OptimizationRequest:
    return OptimizationRequest(
        flow_id=flow_id or FlowId(user_id=_USER_ID),
        task=ModelTask.CLASSIFICATION,
        stream_info=StreamInfo(arrival_rate=arrival_rate),
        qos_requirements=QoSRequirements(
            max_mean_response_time=max_response_time_s,
            min_model_performance=PerformanceMetric(
                type=PerformanceMetricType.MAP_AT_50,
                dataset="test",
                value=0.8,
            ),
            max_energy_consumption=1,
        ),
        ingress_id=ingress,
    )


def _problem_config(
    max_components: int,
    *,
    priority_count: int = 1,
) -> ProblemConfig:
    return ProblemConfig(
        max_components=max_components,
        priority_count=priority_count,
        compute_utilization_limit=0.9,
        network_utilization_limit=0.9,
        relative_error=1.0,
    )


def _solver() -> GurobiSolver:
    return GurobiSolver(
        SolverConfig(
            objective=(ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_SLACK),
        )
    )


async def _solve(
    request: OptimizationRequest,
    context: OptimizationContext,
    *,
    max_components: int,
    priority_count: int = 1,
) -> OptimizationSolution:
    return await _solve_requests(
        (request,),
        context,
        max_components=max_components,
        priority_count=priority_count,
    )


async def _solve_requests(
    requests: tuple[OptimizationRequest, ...],
    context: OptimizationContext,
    *,
    max_components: int,
    priority_count: int,
) -> OptimizationSolution:
    problem_config = _problem_config(
        max_components,
        priority_count=priority_count,
    )
    requests_by_flow = {request.flow_id: request for request in requests}
    return await _solver().solve(
        requests_by_flow,
        context,
        problem_config,
    )


def test_static_boundary_layers_have_zero_execution_time_on_every_worker() -> None:
    worker = _worker(1)

    profiles = layer_execution_profiles(
        _context({worker: None}),
        _MODEL_VARIANT_ID,
        worker,
    )

    assert profiles is not None
    assert profiles == {
        _INPUT: LayerExecutionProfile(execution_time_s=0.0),
        _OUTPUT: LayerExecutionProfile(execution_time_s=0.0),
    }


@pytest.mark.asyncio
async def test_solves_feasible_single_worker_instance() -> None:
    device = _worker(1)

    solution = await _solve(
        _request(device, arrival_rate=0.1, max_response_time_s=1),
        _context({device: 0.1}),
        max_components=1,
    )

    assert solution.is_optimal
    assert len(solution.flow_solutions) == 1
    flow_solution = solution.flow_solutions[0]
    assert flow_solution.ingress_id == device
    assert flow_solution.mean_response_time_s <= 1
    assert len(flow_solution.components) == 1
    assert flow_solution.components[0].worker_id == device
    assert flow_solution.components[0].layer_keys == (_INPUT, _COMPUTE, _OUTPUT)
    assert flow_solution.components[0].input_layer_keys == (_INPUT,)
    assert flow_solution.components[0].output_layer_keys == (_OUTPUT,)
    assert flow_solution.components[0].executable_layer_keys == (_COMPUTE,)
    assert flow_solution.components[0].tensor_contract.input_tensor_keys == (
        _INPUT_TENSOR,
    )
    assert flow_solution.components[0].tensor_contract.output_tensor_keys == (
        _OUTPUT_TENSOR,
    )
    assert flow_solution.cuts == ()
    assert flow_solution.components[0].mean_execution_time_s == pytest.approx(0.1)

    with pytest.raises(
        ValueError,
        match="The first component must be placed on the ingress worker",
    ):
        replace(flow_solution, ingress_id=_worker(2))


@pytest.mark.asyncio
async def test_solves_with_multiple_priority_classes() -> None:
    device = _worker(1)

    solution = await _solve(
        _request(device, arrival_rate=0.1, max_response_time_s=1),
        _context({device: 0.1}),
        max_components=1,
        priority_count=2,
    )

    assert solution.is_optimal
    assert solution.flow_solutions[0].components[0].priority == 1


@pytest.mark.asyncio
async def test_solves_multi_variant_two_worker_two_priority_instance() -> None:
    device_a = _worker(1)
    edge = _worker(2)
    context = _multi_variant_context(
        {
            device_a: {
                _MODEL_VARIANT_ID: 0.25,
                _ALTERNATIVE_MODEL_VARIANT_ID: None,
            },
            edge: {
                _MODEL_VARIANT_ID: 0.15,
                _ALTERNATIVE_MODEL_VARIANT_ID: 0.04,
            },
        }
    )
    request = _request(
        device_a,
        arrival_rate=0.05,
        max_response_time_s=1,
        flow_id=_flow(101),
    )

    solution = await _solve_requests(
        (request,),
        context,
        max_components=3,
        priority_count=2,
    )

    assert solution.is_optimal
    assert len(solution.flow_solutions) == 1
    flow_solution = solution.flow_solutions[0]
    assert flow_solution.model_variant_id == _ALTERNATIVE_MODEL_VARIANT_ID
    assert tuple(component.worker_id for component in flow_solution.components) == (
        device_a,
        edge,
        device_a,
    )
    assert tuple(
        component.mean_execution_time_s for component in flow_solution.components
    ) == pytest.approx((0, 0.04, 0))


@pytest.mark.asyncio
async def test_solves_two_flows_sharing_a_worker() -> None:
    device = _worker(1)
    edge = _worker(2)
    requests = (
        _request(
            device,
            arrival_rate=0.05,
            max_response_time_s=1,
            flow_id=_flow(201),
        ),
        _request(
            device,
            arrival_rate=0.05,
            max_response_time_s=1,
            flow_id=_flow(202),
        ),
    )

    solution = await _solve_requests(
        requests,
        _context({device: 2.0, edge: 0.04}),
        max_components=3,
        priority_count=1,
    )

    assert solution.is_optimal
    assert len(solution.flow_solutions) == 2
    solutions_by_flow = {item.flow_id: item for item in solution.flow_solutions}
    assert tuple(
        component.worker_id for component in solutions_by_flow[_flow(201)].components
    ) == (device, edge, device)
    assert tuple(
        component.worker_id for component in solutions_by_flow[_flow(202)].components
    ) == (device, edge, device)


@pytest.mark.asyncio
async def test_solves_two_flows_two_variants_two_priorities() -> None:
    device = _worker(1)
    edge = _worker(2)
    context = _multi_variant_context(
        {
            device: {
                _MODEL_VARIANT_ID: 0.25,
                _ALTERNATIVE_MODEL_VARIANT_ID: None,
            },
            edge: {
                _MODEL_VARIANT_ID: 0.15,
                _ALTERNATIVE_MODEL_VARIANT_ID: 0.04,
            },
        }
    )
    requests = tuple(
        _request(
            device,
            arrival_rate=0.05,
            max_response_time_s=1,
            flow_id=_flow(number),
        )
        for number in (301, 302)
    )

    solution = await _solve_requests(
        requests,
        context,
        max_components=3,
        priority_count=2,
    )

    assert solution.is_optimal
    assert len(solution.flow_solutions) == 2
    for flow_solution in solution.flow_solutions:
        assert flow_solution.model_variant_id == _ALTERNATIVE_MODEL_VARIANT_ID
        assert tuple(component.worker_id for component in flow_solution.components) == (
            device,
            edge,
            device,
        )


@pytest.mark.asyncio
async def test_solves_three_flows_two_variants_three_workers_three_priorities() -> None:
    device = _worker(1)
    edge_a = _worker(2)
    edge_b = _worker(3)
    context = _multi_variant_context(
        {
            device: {
                _MODEL_VARIANT_ID: 0.25,
                _ALTERNATIVE_MODEL_VARIANT_ID: None,
            },
            edge_a: {
                _MODEL_VARIANT_ID: 0.15,
                _ALTERNATIVE_MODEL_VARIANT_ID: 0.04,
            },
            edge_b: {
                _MODEL_VARIANT_ID: 0.16,
                _ALTERNATIVE_MODEL_VARIANT_ID: 0.05,
            },
        }
    )
    requests = tuple(
        _request(
            device,
            arrival_rate=0.05,
            max_response_time_s=1,
            flow_id=_flow(number),
        )
        for number in (401, 402, 403)
    )

    solution = await _solve_requests(
        requests,
        context,
        max_components=3,
        priority_count=3,
    )

    assert solution.is_optimal
    assert len(solution.flow_solutions) == 3
    for flow_solution in solution.flow_solutions:
        assert flow_solution.model_variant_id == _ALTERNATIVE_MODEL_VARIANT_ID
        component_workers = tuple(
            component.worker_id for component in flow_solution.components
        )
        assert component_workers[0] == device
        assert component_workers[1] in {edge_a, edge_b}
        assert component_workers[2] == device


@pytest.mark.asyncio
async def test_solves_branched_dag_on_sparse_asymmetric_network() -> None:
    device = _worker(1)
    edge_a = _worker(2)
    edge_b = _worker(3)
    context = _branched_context(
        {
            device: {
                _BRANCHED_INPUT: 0,
                _BRANCHED_OUTPUT: 0,
            },
            edge_a: {
                _STEM: 0.03,
                _BRANCH_A_1: 0.03,
                _BRANCH_A_2: 0.03,
            },
            edge_b: {
                _BRANCH_B: 0.02,
                _MERGE: 0.02,
                _HEAD: 0.02,
            },
        },
        connections=(
            NetworkConnection(
                source=device,
                target=edge_a,
                throughput_bytes_per_sec=1_000_000,
                round_trip_time_s=0.002,
            ),
            NetworkConnection(
                source=edge_a,
                target=edge_b,
                throughput_bytes_per_sec=800_000,
                round_trip_time_s=0.004,
            ),
            NetworkConnection(
                source=edge_b,
                target=device,
                throughput_bytes_per_sec=600_000,
                round_trip_time_s=0.006,
            ),
        ),
    )

    solution = await _solve(
        _request(device, arrival_rate=0.05, max_response_time_s=1),
        context,
        max_components=4,
        priority_count=2,
    )

    assert solution.is_optimal
    flow_solution = solution.flow_solutions[0]
    assert flow_solution.ingress_id == device
    assert flow_solution.model_variant_id == _BRANCHED_MODEL_VARIANT_ID
    assert tuple(component.worker_id for component in flow_solution.components) == (
        device,
        edge_a,
        edge_b,
        device,
    )
    assert tuple(
        frozenset(component.layer_keys) for component in flow_solution.components
    ) == (
        frozenset({_BRANCHED_INPUT}),
        frozenset({_STEM, _BRANCH_A_1, _BRANCH_A_2}),
        frozenset({_BRANCH_B, _MERGE, _HEAD}),
        frozenset({_BRANCHED_OUTPUT}),
    )
    assert tuple(
        component.tensor_contract.input_tensor_keys
        for component in flow_solution.components
    ) == (
        ("input_to_stem",),
        ("input_to_stem",),
        ("branch_a_2_output", "stem_fanout"),
        ("head_output",),
    )
    assert tuple(
        component.tensor_contract.output_tensor_keys
        for component in flow_solution.components
    ) == (
        ("input_to_stem",),
        ("branch_a_2_output", "stem_fanout"),
        ("head_output",),
        ("head_output",),
    )
    assert tuple(
        (
            cut.source_component_index,
            cut.destination_component_index,
            cut.tensor_keys,
        )
        for cut in flow_solution.cuts
    ) == (
        (1, 2, ("input_to_stem",)),
        (2, 3, ("branch_a_2_output", "stem_fanout")),
        (3, 4, ("head_output",)),
    )


@pytest.mark.asyncio
async def test_rejects_placement_across_missing_network_connection() -> None:
    device = _worker(1)
    edge_a = _worker(2)
    edge_b = _worker(3)
    context = _branched_context(
        {
            device: {
                _BRANCHED_INPUT: 0,
                _BRANCHED_OUTPUT: 0,
            },
            edge_a: {
                _STEM: 0.03,
                _BRANCH_A_1: 0.03,
                _BRANCH_A_2: 0.03,
            },
            edge_b: {
                _BRANCH_B: 0.02,
                _MERGE: 0.02,
                _HEAD: 0.02,
            },
        },
        connections=(
            NetworkConnection(
                source=device,
                target=edge_a,
                throughput_bytes_per_sec=1_000_000,
                round_trip_time_s=0.002,
            ),
            NetworkConnection(
                source=edge_b,
                target=device,
                throughput_bytes_per_sec=600_000,
                round_trip_time_s=0.006,
            ),
        ),
    )

    with pytest.raises(OptimizationInfeasibleError):
        await _solve(
            _request(device, arrival_rate=0.05, max_response_time_s=1),
            context,
            max_components=4,
            priority_count=2,
        )


@pytest.mark.asyncio
async def test_partitions_and_places_compute_on_faster_worker() -> None:
    device = _worker(1)
    edge = _worker(2)

    solution = await _solve(
        _request(device, arrival_rate=0.1, max_response_time_s=0.5),
        _context({device: 2.0, edge: 0.1}),
        max_components=3,
    )

    components = solution.flow_solutions[0].components
    assert tuple(component.worker_id for component in components) == (
        device,
        edge,
        device,
    )
    assert tuple(component.layer_keys for component in components) == (
        (_INPUT,),
        (_COMPUTE,),
        (_OUTPUT,),
    )
    assert tuple(component.input_layer_keys for component in components) == (
        (_INPUT,),
        (),
        (),
    )
    assert tuple(component.output_layer_keys for component in components) == (
        (),
        (),
        (_OUTPUT,),
    )
    assert tuple(component.executable_layer_keys for component in components) == (
        (),
        (_COMPUTE,),
        (),
    )
    assert tuple(
        component.tensor_contract.input_tensor_keys for component in components
    ) == ((_INPUT_TENSOR,), (_INPUT_TENSOR,), (_OUTPUT_TENSOR,))
    assert tuple(
        component.tensor_contract.output_tensor_keys for component in components
    ) == ((_INPUT_TENSOR,), (_OUTPUT_TENSOR,), (_OUTPUT_TENSOR,))
    assert tuple(
        (
            cut.source_component_index,
            cut.destination_component_index,
            cut.tensor_keys,
        )
        for cut in solution.flow_solutions[0].cuts
    ) == (
        (1, 2, (_INPUT_TENSOR,)),
        (2, 3, (_OUTPUT_TENSOR,)),
    )


@pytest.mark.asyncio
async def test_rejects_unstable_compute_queue() -> None:
    device = _worker(1)

    with pytest.raises(OptimizationInfeasibleError):
        await _solve(
            _request(device, arrival_rate=1.0, max_response_time_s=100),
            _context({device: 1.0}),
            max_components=1,
        )


@pytest.mark.asyncio
async def test_rejects_unstable_network_queue() -> None:
    device = _worker(1)
    edge = _worker(2)

    with pytest.raises(OptimizationInfeasibleError):
        await _solve(
            _request(device, arrival_rate=1.0, max_response_time_s=100),
            _context(
                {device: None, edge: 0.1},
                throughput_bytes_per_s=1,
            ),
            max_components=3,
        )
