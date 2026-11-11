from typing import cast
from uuid import UUID

import gurobipy as gp
import pytest
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.constraints import (
    _successive_components_use_different_workers,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.profile.network_topology import NetworkTopology
from shared.flow.flow import FlowId
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId

_FLOW = FlowId(user_id=UserId(id=UUID(int=1)), flow_id=UUID(int=2))
_WORKER_A = cast(WorkerId, ServiceId(id=UUID(int=3)))
_WORKER_B = cast(WorkerId, ServiceId(id=UUID(int=4)))
_WORKERS = (_WORKER_A, _WORKER_B)


def _problem() -> CompactFormulationData:
    return CompactFormulationData(
        max_components=2,
        priority_count=1,
        compute_utilization_limit=0.9,
        network_utilization_limit=0.9,
        objective_kind=CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        static_profiles_by_variant={},
        resource_profiles_by_worker={},
        network_topology=NetworkTopology(workers=_WORKERS, connections=()),
        layer_execution_profiles_by_model_worker={},
        arrival_rate_per_s={},
        max_response_time_s_by_flow={},
        performance_requirement_by_flow={},
        endpoint_placement_by_flow={},
        local_worker_config_by_flow={},
        flows=(_FLOW,),
        workers=_WORKERS,
        workers_by_flow={_FLOW: _WORKERS},
        models_by_flow={_FLOW: ()},
        service_arrival_rate_per_s={},
        batch_fill_time_s={},
        compute_bin_limits=(),
        network_bin_limits=(),
        max_queue_wait_s_by_worker={},
        max_queue_wait_s_by_flow={},
        flow_index={_FLOW: 0},
        worker_index={_WORKER_A: 0, _WORKER_B: 1},
        model_index={},
        layer_index={},
        tensor_index={},
    )


@pytest.mark.parametrize(
    ("placements", "expected_status"),
    (
        ((_WORKER_A, _WORKER_A), GRB.INFEASIBLE),
        ((_WORKER_A, _WORKER_B), GRB.OPTIMAL),
        ((_WORKER_A, None), GRB.OPTIMAL),
    ),
)
def test_successive_components_cannot_use_the_same_worker(
    placements: tuple[WorkerId | None, WorkerId | None],
    expected_status: int,
) -> None:
    problem = _problem()
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        component_worker = {
            (_FLOW, component, worker): model.addVar(
                lb=int(placements[component - 1] == worker),
                ub=int(placements[component - 1] == worker),
                vtype=GRB.BINARY,
            )
            for component in (1, 2)
            for worker in _WORKERS
        }
        assignment = AssignmentVariables(
            max_components=2,
            model_selected={},
            layer_component={},
            component_worker=component_worker,
            component_active={},
            component_last={},
        )

        _successive_components_use_different_workers(
            model,
            assignment,
            problem,
            _FLOW,
            0,
            1,
        )
        model.update()

        assert model.NumConstrs == len(_WORKERS)
        assert all(
            model.getConstrByName(
                f"successive_components_different_workers[0,1,{worker_index}]"
            )
            is not None
            for worker_index in range(len(_WORKERS))
        )

        model.optimize()

        assert model.Status == expected_status
