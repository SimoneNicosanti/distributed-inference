from dataclasses import replace

import pytest
from controller.adapters.outbound.optimization.gurobi_solver import (
    GurobiSolver,
    OptimizationInfeasibleError,
)
from test.controller.adapters.outbound.optimization.test_gurobi_solver import (
    _MODEL_VARIANT_ID,
    _context,
    _flow,
    _problem_config,
    _request,
    _worker,
)

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from controller.application.optimization.formulation.compact.compact_formulation_data_builder import (
    CompactFormulationDataBuilder,
)
from controller.application.optimization.formulation.compact.profile_data import (
    ProfileDataError,
    layer_execution_profiles,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_execution_profile import (
    BackendExecutionProfile,
    LayerExecutionProfile,
)
from controller.domain.profile.resource_profile import GpuProfile
from shared.service.service import WorkerId


def _gpu(*, available_bytes: int = 2_000_000) -> GpuProfile:
    return GpuProfile(
        device_index=0,
        device_uuid="gpu-0",
        name="Test GPU",
        total_memory_bytes=available_bytes,
        used_memory_bytes=0,
        available_memory_bytes=available_bytes,
    )


def _with_gpu(
    context: OptimizationContext,
    worker: WorkerId,
    *,
    gpu_time_s: float | None = 0.1,
    available_bytes: int = 2_000_000,
) -> OptimizationContext:
    resources = dict(context.resource_profiles_by_worker)
    resources[worker] = replace(
        resources[worker], gpu_profiles=(_gpu(available_bytes=available_bytes),)
    )
    executions = dict(context.execution_profiles_by_worker)
    variants = {}
    for variant, profile in executions[worker].execution_profiles.items():
        variants[variant] = replace(
            profile,
            shape_execution_profiles=tuple(
                replace(
                    shape,
                    backend_profiles=shape.backend_profiles
                    + (
                        (
                            BackendExecutionProfile(
                                backend=Backend.GPU,
                                layer_profiles={
                                    "compute": LayerExecutionProfile(gpu_time_s)
                                },
                            ),
                        )
                        if gpu_time_s is not None
                        else ()
                    ),
                )
                for shape in profile.shape_execution_profiles
            ),
        )
    executions[worker] = replace(executions[worker], execution_profiles=variants)
    return OptimizationContext(
        static_profiles_by_variant=context.static_profiles_by_variant,
        network_topology=context.network_topology,
        resource_profiles_by_worker=resources,
        execution_profiles_by_worker=executions,
    )


def test_worker_backend_and_memory_follow_the_single_gpu_policy() -> None:
    worker = _worker(1)
    context = _context({worker: 0.9})
    cpu = context.resource_profiles_by_worker[worker]
    gpu = _with_gpu(context, worker).resource_profiles_by_worker[worker]

    assert cpu.execution_backend is Backend.CPU
    assert cpu.available_inference_memory_bytes == 1_000_000
    assert gpu.execution_backend is Backend.GPU
    assert gpu.available_inference_memory_bytes == 2_000_000
    with pytest.raises(ValueError, match="Only one GPU"):
        replace(
            cpu,
            gpu_profiles=(
                _gpu(),
                replace(_gpu(), device_index=1, device_uuid="gpu-1"),
            ),
        )


def test_gpu_profiles_are_selected_even_when_cpu_is_faster() -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: 0.01}), worker, gpu_time_s=0.2)
    profiles = layer_execution_profiles(context, _MODEL_VARIANT_ID, worker)

    assert profiles is not None
    assert profiles["compute"].execution_time_s == 0.2
    assert profiles["input"].execution_time_s == 0
    assert profiles["output"].execution_time_s == 0


def test_cpu_worker_ignores_gpu_execution_profiles() -> None:
    worker = _worker(1)
    gpu_context = _with_gpu(_context({worker: 0.2}), worker, gpu_time_s=0.01)
    context = OptimizationContext(
        static_profiles_by_variant=gpu_context.static_profiles_by_variant,
        network_topology=gpu_context.network_topology,
        resource_profiles_by_worker={
            worker: replace(
                gpu_context.resource_profiles_by_worker[worker], gpu_profiles=()
            )
        },
        execution_profiles_by_worker=gpu_context.execution_profiles_by_worker,
    )
    profiles = layer_execution_profiles(context, _MODEL_VARIANT_ID, worker)

    assert profiles is not None
    assert profiles["compute"].execution_time_s == 0.2


def test_missing_gpu_profile_does_not_fall_back_to_cpu() -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: 0.01}), worker, gpu_time_s=None)
    profiles = layer_execution_profiles(context, _MODEL_VARIANT_ID, worker)

    assert profiles is not None
    assert set(profiles) == {"input", "output"}


def test_problem_queue_bounds_use_gpu_execution_times() -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: 0.9}), worker)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )
    problem = CompactFormulationDataBuilder().build_formulation_data(
        {request.flow_id: request}, context, _problem_config(1)
    )

    assert problem.max_queue_wait_s_by_worker[worker] == pytest.approx(0.05)


def test_duplicate_gpu_profile_is_rejected() -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: 0.9}), worker)
    executions = dict(context.execution_profiles_by_worker)
    profile = executions[worker].execution_profiles[_MODEL_VARIANT_ID]
    shape = profile.shape_execution_profiles[0]
    duplicate = replace(
        profile,
        shape_execution_profiles=(
            replace(
                shape,
                backend_profiles=(*shape.backend_profiles, shape.backend_profiles[-1]),
            ),
        ),
    )
    executions[worker] = replace(
        executions[worker], execution_profiles={_MODEL_VARIANT_ID: duplicate}
    )
    context = OptimizationContext(
        static_profiles_by_variant=context.static_profiles_by_variant,
        network_topology=context.network_topology,
        resource_profiles_by_worker=context.resource_profiles_by_worker,
        execution_profiles_by_worker=executions,
    )

    with pytest.raises(ProfileDataError, match="Duplicate GPU"):
        layer_execution_profiles(context, _MODEL_VARIANT_ID, worker)


@pytest.mark.parametrize(
    ("ram_bytes", "vram_bytes", "feasible"),
    ((100_000, 2_100_000, True), (10_000_000, 1_900_000, False)),
)
def test_gpu_memory_uses_vram_instead_of_ram(
    ram_bytes: int, vram_bytes: int, feasible: bool
) -> None:
    from test.controller.adapters.outbound.optimization.test_compact_memory_and_profiles import (
        context_with_memory,
    )

    worker = _worker(1)
    context = _with_gpu(
        context_with_memory(capacity=ram_bytes, weights=True),
        worker,
        available_bytes=vram_bytes,
    )
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )
    solver = GurobiSolver(
        GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    )
    if not feasible:
        with pytest.raises(OptimizationInfeasibleError):
            solver._solve({request.flow_id: request}, context, _problem_config(1))
    else:
        assert solver._solve(
            {request.flow_id: request}, context, _problem_config(1)
        ).is_optimal


def test_gpu_worker_without_gpu_execution_profile_is_incompatible() -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: 0.01}), worker, gpu_time_s=None)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )

    with pytest.raises(OptimizationInfeasibleError):
        GurobiSolver(
            GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
        )._solve({request.flow_id: request}, context, _problem_config(1))


@pytest.mark.parametrize(
    ("cpu_time_s", "gpu_time_s", "feasible"),
    ((10.0, 0.1, True), (0.01, 10.0, False)),
)
def test_gpu_queue_stability_uses_gpu_service_time(
    cpu_time_s: float, gpu_time_s: float, feasible: bool
) -> None:
    worker = _worker(1)
    context = _with_gpu(_context({worker: cpu_time_s}), worker, gpu_time_s=gpu_time_s)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=100, flow_id=_flow(1)
    )
    solver = GurobiSolver(
        GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    )
    if not feasible:
        with pytest.raises(OptimizationInfeasibleError):
            solver._solve({request.flow_id: request}, context, _problem_config(1))
    else:
        component = (
            solver._solve({request.flow_id: request}, context, _problem_config(1))
            .flow_solutions[0]
            .components[0]
        )
        assert component.backend is Backend.GPU
        assert component.mean_execution_time_s == pytest.approx(gpu_time_s)


def test_mixed_cpu_gpu_solution_survives_worker_plan_transport() -> None:
    from controller.adapters.outbound.plan_share.mappers.worker_plan_mapper import (
        WorkerPlanMapper,
    )
    from controller.application.deployment.builder.endpoint_replica_id_builder import (
        EndpointReplicaIdBuilder,
    )
    from controller.application.deployment.builder.partition_replica_id_builder import (
        PartitionReplicaIdBuilder,
    )
    from controller.application.deployment.builder.worker_plan_builder import (
        WorkerPlanBuilder,
    )
    from integration.worker.deployment.http.service_plan_http_schema import (
        PrepareWorkerPlanRequest,
    )
    from shared.plan.plan_version import PlanVersion
    from worker.adapters.inbound.deployment.http.service_plan_mapper import (
        worker_plan_from_dto,
    )

    cpu, gpu = _worker(1), _worker(2)
    context = _with_gpu(_context({cpu: 0.6, gpu: 0.001}), gpu, gpu_time_s=0.1)
    request = _request(cpu, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1))
    solution = GurobiSolver(
        GurobiSolverConfig(CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME)
    )._solve({request.flow_id: request}, context, _problem_config(3))

    components = solution.flow_solutions[0].components
    assert tuple(component.worker_id for component in components) == (cpu, gpu, cpu)
    assert tuple(component.backend for component in components) == (
        Backend.CPU,
        Backend.GPU,
        Backend.CPU,
    )
    assert components[1].mean_execution_time_s == pytest.approx(0.1)

    plans = WorkerPlanBuilder().build_all(
        PlanVersion(version_number=1),
        solution,
        PartitionReplicaIdBuilder().build_all(solution),
        EndpointReplicaIdBuilder().build_all(solution),
    )
    assert plans[cpu].deployment_plan.allocations == ()
    assert tuple(
        allocation.use_gpu for allocation in plans[gpu].deployment_plan.allocations
    ) == (True,)
    payload = PrepareWorkerPlanRequest(
        worker_plan=WorkerPlanMapper.to_worker_plan_dto(plans[gpu])
    ).model_dump_json()
    received = worker_plan_from_dto(
        PrepareWorkerPlanRequest.model_validate_json(payload).worker_plan
    )
    assert tuple(
        allocation.use_gpu for allocation in received.deployment_plan.allocations
    ) == (True,)
