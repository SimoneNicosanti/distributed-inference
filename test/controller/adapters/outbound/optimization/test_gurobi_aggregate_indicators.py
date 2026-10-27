from types import SimpleNamespace
from typing import cast
from uuid import UUID

import gurobipy as gp
import pytest
from gurobipy import GRB

import controller.adapters.outbound.optimization.gurobi_memory_constraints as memory_constraints
import controller.adapters.outbound.optimization.gurobi_structure_constraints as structure_constraints
from controller.adapters.outbound.optimization.gurobi_memory_constraints import (
    add_memory_constraints,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    add_execution_constraints,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ExecutionVariables,
    MemoryVariables,
    _add_assignment_variables,
    _add_execution_variables,
    _add_memory_variables,
)
from controller.domain.optimization.optimization_context import OptimizationContext
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
_EXECUTION_TIMES = (1, 2, 3)
_WEIGHT_BYTES = (10, 20, 30)
_ACTIVATION_BYTES = (4, 5, 6)


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


def _problem() -> GurobiProblem:
    return GurobiProblem(
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
    problem: GurobiProblem,
    context: OptimizationContext,
) -> tuple[AssignmentVariables, ExecutionVariables, MemoryVariables]:
    assignment = _add_assignment_variables(
        model,
        problem,
        context,
        max_components=_MAX_COMPONENTS,
    )
    execution = _add_execution_variables(
        model,
        problem,
        max_components=_MAX_COMPONENTS,
    )
    memory = _add_memory_variables(
        model,
        problem,
        context,
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

    return assignment, execution, memory


def _execution_profiles(
    _context: OptimizationContext,
    _model: ModelVariantId,
    _worker: WorkerId,
) -> dict[str, LayerExecutionProfile]:
    return {
        layer: LayerExecutionProfile(execution_time_s=execution_time)
        for layer, execution_time in zip(_LAYERS, _EXECUTION_TIMES, strict=True)
    }


def _weight_size_bytes(_profile: object, layer: str) -> int:
    return _WEIGHT_BYTES[_LAYERS.index(layer)]


def _activation_sizes_bytes(_profile: object) -> dict[str, int]:
    return dict(zip(_LAYERS, _ACTIVATION_BYTES, strict=True))


def _legacy_aggregate_values(
    *,
    layer_selected: bool,
    worker_selected: bool,
) -> tuple[float, float, float]:
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        x = {
            layer: model.addVar(
                lb=int(layer_selected),
                ub=int(layer_selected),
                vtype=GRB.BINARY,
            )
            for layer in _LAYERS
        }
        e = model.addVar(
            lb=int(worker_selected),
            ub=int(worker_selected),
        )
        z = {layer: model.addVar(lb=0.0, ub=1.0) for layer in _LAYERS}
        for layer in _LAYERS:
            model.addConstr(z[layer] <= x[layer])
            model.addConstr(z[layer] <= e)
            model.addConstr(z[layer] >= x[layer] + e - 1)

        aggregates = tuple(model.addVar(lb=0.0) for _ in range(3))
        for aggregate, coefficients in zip(
            aggregates,
            (_EXECUTION_TIMES, _WEIGHT_BYTES, _ACTIVATION_BYTES),
            strict=True,
        ):
            model.addConstr(
                aggregate
                == gp.quicksum(
                    coefficient * z[layer]
                    for layer, coefficient in zip(_LAYERS, coefficients, strict=True)
                )
            )
        model.optimize()
        assert model.Status == GRB.OPTIMAL
        return tuple(aggregate.X for aggregate in aggregates)


def test_indicator_aggregates_replace_all_layer_worker_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    context = _context()
    monkeypatch.setattr(
        structure_constraints,
        "layer_execution_profiles",
        _execution_profiles,
    )
    monkeypatch.setattr(
        memory_constraints,
        "layer_weight_size_bytes",
        _weight_size_bytes,
    )
    monkeypatch.setattr(
        memory_constraints,
        "layer_activation_sizes_bytes",
        _activation_sizes_bytes,
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, execution, memory = _variables(model, problem, context)
        add_execution_constraints(
            model,
            problem,
            assignment,
            execution,
            context=context,
            max_components=_MAX_COMPONENTS,
        )
        add_memory_constraints(
            model,
            problem,
            assignment,
            memory,
            context=context,
            max_components=_MAX_COMPONENTS,
        )
        model.setObjective(gp.quicksum(memory.max_activation_mb.values()))
        model.update()

        aggregate_count = len(_MODELS) * _MAX_COMPONENTS * len(_WORKERS)
        eliminated_z_count = len(_LAYERS) * aggregate_count
        assert eliminated_z_count == 24
        assert model.NumVars == 48
        assert model.NumBinVars == 16
        assert model.NumConstrs == 10
        assert model.NumGenConstrs == 6 * aggregate_count == 48
        assert all(
            not variable.VarName.startswith("z[") for variable in model.getVars()
        )
        assert all(
            variable.VType == GRB.BINARY
            for variable in assignment.component_worker.values()
        )

        model.optimize()

        assert model.Status == GRB.OPTIMAL
        active = (_FLOW, _MODEL_A, 1, _WORKER_A)
        assert pytest.approx(6) == execution.execution_time_s[active].X
        assert pytest.approx(60e-6) == memory.component_weight_mb[active].X
        assert pytest.approx(15e-6) == memory.component_activation_mb[active].X
        assert pytest.approx(15e-6) == memory.max_activation_mb[_WORKER_A].X
        assert pytest.approx(0) == memory.max_activation_mb[_WORKER_B].X

        active_values = (
            execution.execution_time_s[active].X,
            memory.component_weight_mb[active].X * 1_000_000,
            memory.component_activation_mb[active].X * 1_000_000,
        )
        assert (
            pytest.approx(
                _legacy_aggregate_values(layer_selected=True, worker_selected=True)
            )
            == active_values
        )
        assert _legacy_aggregate_values(layer_selected=True, worker_selected=False) == (
            0,
            0,
            0,
        )
        assert _legacy_aggregate_values(layer_selected=False, worker_selected=True) == (
            0,
            0,
            0,
        )

        for key, execution_time in execution.execution_time_s.items():
            if key != active:
                assert pytest.approx(0) == execution_time.X
                assert pytest.approx(0) == memory.component_weight_mb[key].X
                assert pytest.approx(0) == memory.component_activation_mb[key].X


def test_memory_capacity_uses_conditioned_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    context = _context(available_bytes=74)
    monkeypatch.setattr(
        memory_constraints,
        "layer_weight_size_bytes",
        _weight_size_bytes,
    )
    monkeypatch.setattr(
        memory_constraints,
        "layer_activation_sizes_bytes",
        _activation_sizes_bytes,
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, _, memory = _variables(model, problem, context)
        add_memory_constraints(
            model,
            problem,
            assignment,
            memory,
            context=context,
            max_components=_MAX_COMPONENTS,
        )
        model.optimize()

        assert model.Status == GRB.INFEASIBLE


def test_missing_layer_profile_still_forbids_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        structure_constraints,
        "layer_execution_profiles",
        profiles_without_last_layer,
    )

    with gp.Model() as model:
        model.Params.OutputFlag = 0
        assignment, execution, _ = _variables(model, problem, context)
        add_execution_constraints(
            model,
            problem,
            assignment,
            execution,
            context=context,
            max_components=_MAX_COMPONENTS,
        )
        model.optimize()

        assert model.Status == GRB.INFEASIBLE
