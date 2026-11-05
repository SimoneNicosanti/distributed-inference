from typing import cast
from uuid import UUID

import gurobipy as gp
import pytest
from gurobipy import GRB

from controller.adapters.outbound.optimization.compact_network_queue import (
    CompactNetworkVariables,
    add_compact_network_queue_constraints,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId, WorkerId
from shared.user.user import UserId

_FLOW = FlowId(user_id=UserId(id=UUID(int=1)), flow_id=UUID(int=2))
_MODEL_A = ModelVariantId(ModelId("model"), "a")
_MODEL_B = ModelVariantId(ModelId("model"), "b")
_MODELS = (_MODEL_A, _MODEL_B)
_WORKER_A = cast(WorkerId, ServiceId(id=UUID(int=3)))
_WORKER_B = cast(WorkerId, ServiceId(id=UUID(int=4)))
_WORKERS = (_WORKER_A, _WORKER_B)
_MAX_COMPONENTS = 3


def _problem() -> GurobiProblem:
    return GurobiProblem(
        flows=(_FLOW,),
        workers=_WORKERS,
        workers_by_flow={_FLOW: _WORKERS},
        models_by_flow={_FLOW: _MODELS},
        service_arrival_rate_per_s={
            (_FLOW, model_variant): 0.1 for model_variant in _MODELS
        },
        batch_fill_time_s={},
        compute_bin_limits=(0.0, 0.9),
        network_bin_limits=(0.0, 0.1),
        max_queue_wait_s_by_worker={},
        max_queue_wait_s_by_flow={},
        flow_index={_FLOW: 0},
        worker_index={worker: index for index, worker in enumerate(_WORKERS)},
        model_index={
            (_FLOW, model_variant): index for index, model_variant in enumerate(_MODELS)
        },
        layer_index={},
        tensor_index={},
    )


def _network_variables(model: gp.Model) -> CompactNetworkVariables:
    nominal_values = {
        (_MODEL_A, 1, _WORKER_A): 0.2,
        (_MODEL_B, 1, _WORKER_B): 0.2,
    }
    transmission_time_s = {
        (_FLOW, model_variant, cut, worker): model.addVar(
            lb=nominal_values.get((model_variant, cut, worker), 0.0),
            ub=nominal_values.get((model_variant, cut, worker), 0.0),
        )
        for model_variant in _MODELS
        for cut in range(1, _MAX_COMPONENTS)
        for worker in _WORKERS
    }
    return CompactNetworkVariables(
        tensor_crossing_auxiliary={},
        adjacent_component_workers={},
        transmission_time_s=transmission_time_s,
        network_utilization={worker: model.addVar(lb=0.0) for worker in _WORKERS},
        network_bin_selected={
            (worker, 1): model.addVar(vtype=GRB.BINARY) for worker in _WORKERS
        },
        component_transmission_response_s={
            (_FLOW, cut): model.addVar(lb=0.0) for cut in range(1, _MAX_COMPONENTS)
        },
        nominal_upper_s={(_FLOW, worker): 0.2 for worker in _WORKERS},
        source_response_s={
            (_FLOW, cut, worker): model.addVar(lb=0.0)
            for cut in range(1, _MAX_COMPONENTS)
            for worker in _WORKERS
        },
    )


def test_projected_network_queue_sums_fractional_source_responses() -> None:
    problem = _problem()
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        network = _network_variables(model)
        model.update()
        variable_count = model.NumVars

        add_compact_network_queue_constraints(
            model,
            problem,
            network,
            max_components=_MAX_COMPONENTS,
            network_utilization_limit=0.1,
        )
        model.setObjective(
            gp.quicksum(network.component_transmission_response_s.values())
        )
        model.update()

        assert model.NumVars == variable_count
        assert model.NumGenConstrs == 0
        assert model.getConstrByName("compact_ps_sum[0,1]") is not None
        assert model.getConstrByName("compact_ps_sum[0,2]") is not None

        model.optimize()

        assert model.Status == GRB.OPTIMAL
        assert pytest.approx(0.02) == network.network_utilization[_WORKER_A].X
        assert pytest.approx(0.02) == network.network_utilization[_WORKER_B].X
        assert (
            pytest.approx(0.4 / (1 - problem.network_bin_limits[-1]))
            == network.component_transmission_response_s[_FLOW, 1].X
        )
        assert (
            pytest.approx(0.0) == network.component_transmission_response_s[_FLOW, 2].X
        )
