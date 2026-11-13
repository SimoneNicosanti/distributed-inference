from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import gurobipy as gp
from gurobipy import GRB

import optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.constraints as structure_constraints
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.request.local_worker_config import (
    NoLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.model_variant_topology import (
    LayerDescription,
    ModelVariantTopology,
)
from controller.domain.profile.network_topology import NetworkTopology
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_builder import (
    _add_assignment_variables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_builder import (
    CompactExecutionBuilder,
    _add_execution_variables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from optimizer.application.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId

_LAYERS = ("a", "b", "c")
_USER = UserId(id=UUID(int=1))
_FLOW = FlowId(user_id=_USER, flow_id=UUID(int=2))
_MODEL_A = ModelVariantId(
    model_id=ModelId(model_name="model"),
    variant_tag="a",
)
_MODEL_B = ModelVariantId(
    model_id=ModelId(model_name="model"),
    variant_tag="b",
)
_MODELS = (_MODEL_A, _MODEL_B)
_WORKER_A = cast(WorkerId, ServiceId(id=UUID(int=3)))
_WORKER_B = cast(WorkerId, ServiceId(id=UUID(int=4)))
_WORKERS = (_WORKER_A, _WORKER_B)
_MAX_COMPONENTS = 2


def _requests_by_flow() -> OptimizationRequestsByFlow:
    return {
        _FLOW: cast(
            OptimizationRequest,
            SimpleNamespace(local_worker_config=NoLocalWorkerConfig()),
        )
    }


def _profile() -> ModelVariantStaticProfile:
    topology = ModelVariantTopology(
        layers={
            layer: LayerDescription(
                name=layer,
                type="Compute",
                is_input=False,
                is_output=False,
            )
            for layer in _LAYERS
        },
        edges=(),
    )
    return cast(ModelVariantStaticProfile, SimpleNamespace(topology=topology))


def _problem() -> CompactFormulationData:
    return CompactFormulationData(
        max_components=_MAX_COMPONENTS,
        priority_count=1,
        compute_utilization_limit=0.9,
        network_utilization_limit=0.9,
        objective_kind=ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        static_profiles_by_variant={model: _profile() for model in _MODELS},
        resource_profiles_by_worker={},
        network_topology=NetworkTopology(workers=_WORKERS, connections=()),
        layer_execution_profiles_by_model_worker={},
        arrival_rate_per_s={},
        max_response_time_s_by_flow={},
        performance_requirement_by_flow={},
        endpoint_placement_by_flow={},
        local_worker_config_by_flow={_FLOW: NoLocalWorkerConfig()},
        flows=(_FLOW,),
        workers=_WORKERS,
        workers_by_flow={_FLOW: _WORKERS},
        models_by_flow={_FLOW: _MODELS},
        service_arrival_rate_per_s={},
        batch_fill_time_s={},
        compute_bin_limits=(0.0, 0.9),
        network_bin_limits=(0.0, 0.9),
        max_queue_wait_s_by_worker={},
        max_queue_wait_s_by_flow={},
        flow_index={_FLOW: 0},
        worker_index={worker: index for index, worker in enumerate(_WORKERS)},
        model_index={(_FLOW, model): index for index, model in enumerate(_MODELS)},
        layer_index={
            (model, layer): index
            for model in _MODELS
            for index, layer in enumerate(_LAYERS)
        },
        tensor_index={},
    )


def _context(*, available_bytes: int = 1_000) -> OptimizationContext:
    return cast(
        OptimizationContext,
        SimpleNamespace(
            static_profiles_by_variant={model: _profile() for model in _MODELS},
            resource_profiles_by_worker={
                worker: SimpleNamespace(
                    ram_profile=SimpleNamespace(available_bytes=available_bytes)
                )
                for worker in _WORKERS
            },
        ),
    )


def _variables(
    model: gp.Model,
    problem: CompactFormulationData,
    context: OptimizationContext,
) -> tuple[AssignmentVariables, ExecutionVariables]:
    assignment = _add_assignment_variables(
        model,
        problem,
        max_components=_MAX_COMPONENTS,
    )
    execution = _add_execution_variables(
        model,
        problem,
        max_components=_MAX_COMPONENTS,
    )
    model.update()

    for candidate in _MODELS:
        selected = candidate == _MODEL_A
        assignment.model_selected[_FLOW, candidate].LB = int(selected)
        assignment.model_selected[_FLOW, candidate].UB = int(selected)
        for layer in _LAYERS:
            for component in range(1, _MAX_COMPONENTS + 1):
                value = int(selected and component == 1)
                variable = assignment.layer_component[
                    _FLOW, candidate, layer, component
                ]
                variable.LB = value
                variable.UB = value

    for component in range(1, _MAX_COMPONENTS + 1):
        for worker in _WORKERS:
            value = int(component == 1 and worker == _WORKER_A)
            variable = assignment.component_worker[_FLOW, component, worker]
            variable.LB = value
            variable.UB = value

    return assignment, execution


def test_missing_layer_profile_still_forbids_placement() -> None:
    problem = _problem()
    context = _context()

    def profiles_without_last_layer(
        _context: OptimizationContext,
        _model: ModelVariantId,
        _worker: WorkerId,
    ) -> dict[str, LayerExecutionProfile]:
        return {
            layer: LayerExecutionProfile(execution_time_s=1) for layer in _LAYERS[:-1]
        }

    problem = replace(
        problem,
        layer_execution_profiles_by_model_worker={
            (variant, worker): profiles_without_last_layer(context, variant, worker)
            for variant in _MODELS
            for worker in _WORKERS
        },
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, execution = _variables(model, problem, context)
        CompactExecutionBuilder().add_constraints(
            model, problem, assignment=assignment, execution=execution
        )
        model.optimize()

        assert model.Status == GRB.INFEASIBLE


def test_minimum_execution_time_uses_only_compatible_workers() -> None:
    profiles_by_worker = {
        _WORKER_A: {
            "a": LayerExecutionProfile(execution_time_s=1),
            "b": LayerExecutionProfile(execution_time_s=4),
        },
        _WORKER_B: {
            "a": LayerExecutionProfile(execution_time_s=3),
            "b": LayerExecutionProfile(execution_time_s=2),
            "c": LayerExecutionProfile(execution_time_s=6),
        },
    }

    assert structure_constraints._minimum_execution_time_by_layer(
        ("a", "b", "c", "unsupported"),
        profiles_by_worker,
    ) == {"a": 1, "b": 2, "c": 6}
    assert structure_constraints._minimum_execution_time_by_layer(
        ("a",),
        {_WORKER_A: profiles_by_worker[_WORKER_A]},
    ) == {"a": 1}


def test_execution_minimum_bound_supports_multiple_components_and_workers() -> None:
    problem = _problem()
    context = _context()

    def heterogeneous_profiles(
        _context: OptimizationContext,
        _model: ModelVariantId,
        worker: WorkerId,
    ) -> dict[str, LayerExecutionProfile]:
        if worker == _WORKER_A:
            return {
                "a": LayerExecutionProfile(execution_time_s=1),
                "b": LayerExecutionProfile(execution_time_s=4),
            }
        return {
            "a": LayerExecutionProfile(execution_time_s=3),
            "b": LayerExecutionProfile(execution_time_s=2),
            "c": LayerExecutionProfile(execution_time_s=6),
        }

    problem = replace(
        problem,
        layer_execution_profiles_by_model_worker={
            (variant, worker): heterogeneous_profiles(context, variant, worker)
            for variant in _MODELS
            for worker in _WORKERS
        },
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, execution = _variables(model, problem, context)

        for component, value in ((1, 0), (2, 1)):
            layer_component = assignment.layer_component[
                _FLOW, _MODEL_A, "c", component
            ]
            layer_component.LB = value
            layer_component.UB = value
        for worker, value in ((_WORKER_A, 0), (_WORKER_B, 1)):
            component_worker = assignment.component_worker[_FLOW, 2, worker]
            component_worker.LB = value
            component_worker.UB = value

        CompactExecutionBuilder().add_constraints(
            model, problem, assignment=assignment, execution=execution
        )
        model.setObjective(gp.quicksum(execution.execution_time_s.values()))
        model.optimize()

        assert model.Status == GRB.OPTIMAL
        assert execution.execution_time_s[_FLOW, _MODEL_A, 1, _WORKER_A].X == 5
        assert execution.execution_time_s[_FLOW, _MODEL_A, 2, _WORKER_B].X == 6
        assert model.getConstrByName("execution_minimum[0,0,1]") is not None
        assert model.getConstrByName("execution_minimum[0,0,2]") is not None
