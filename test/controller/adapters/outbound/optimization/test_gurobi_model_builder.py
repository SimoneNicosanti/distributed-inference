from contextlib import contextmanager
from typing import override

import gurobipy as gp
import pytest
from test.controller.adapters.outbound.optimization.test_gurobi_solver import (
    _context,
    _flow,
    _problem_config,
    _request,
    _worker,
)

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.gurobi_compact_formulation_compiler import (
    GurobiCompactFormulationCompiler,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.gurobi_formulation_model_builder import (
    GurobiFormulationModelBuilder,
)
from controller.application.optimization.formulation.compact.compact_formulation_data_builder import (
    CompactFormulationDataBuilder,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)


def _inputs():
    worker = _worker(1)
    request = _request(
        worker, arrival_rate=0.1, max_response_time_s=10, flow_id=_flow(1)
    )
    return {request.flow_id: request}, _context({worker: 0.1}), _problem_config(2)


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(CompactObjectiveKind))
async def test_builds_unsolved_model_with_matching_deadline_policy(objective):
    requests, context, config = _inputs()
    data = CompactFormulationDataBuilder(objective).build_formulation_data(
        requests, context, config
    )
    builder = GurobiFormulationModelBuilder(GurobiCompactFormulationCompiler())

    async with builder.build_formulation_model(data) as compiled:
        model = compiled.native_model
        assert model.NumVars > 0
        assert model.NumConstrs > 0
        assert model.Status == gp.GRB.LOADED
        assert model.SolCount == 0
        assert compiled.data is data
        assert compiled.data.flows == tuple(requests)
        assert compiled.variables.qos.response_time_s.keys() == requests.keys()
        assert compiled.construction_time_s >= 0
        assert model.ModelSense == (
            gp.GRB.MAXIMIZE
            if objective == CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
            else gp.GRB.MINIMIZE
        )
        hard_deadline = objective in {
            CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
            CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
        }
        assert (model.getConstrByName("response_requirement[0]") is not None) == (
            hard_deadline
        )
        assert model.NumGenConstrs == (
            len(requests)
            if objective == CompactObjectiveKind.DEADLINE_VIOLATIONS
            else 0
        )


class _BuildError(Exception):
    pass


class _ClientError(Exception):
    pass


@pytest.mark.asyncio
async def test_accepts_another_formulation_data_and_variable_container():
    class ScalarFormulation(GurobiFormulationCompiler[float, dict[str, gp.Var]]):
        @override
        def compile_formulation(
            self, model: gp.Model, data: float
        ) -> dict[str, gp.Var]:
            return {"value": model.addVar(lb=data, name="value")}

    builder = GurobiFormulationModelBuilder(ScalarFormulation())
    async with builder.build_formulation_model(2.0) as compiled:
        assert compiled.data == 2.0
        assert compiled.variables["value"].LB == 2.0
        assert compiled.native_model.Status == gp.GRB.LOADED


@pytest.mark.asyncio
async def test_prepared_data_does_not_need_the_original_requests():
    requests, context, config = _inputs()
    data = CompactFormulationDataBuilder().build_formulation_data(
        requests, context, config
    )
    requests.clear()
    async with GurobiFormulationModelBuilder(
        GurobiCompactFormulationCompiler()
    ).build_formulation_model(data) as compiled:
        assert compiled.native_model.NumVars > 0
        assert set(compiled.variables.qos.response_time_s) == set(data.flows)


class _FailingFormulation(GurobiCompactFormulationCompiler):
    @override
    def compile_formulation(self, *args, **kwargs):
        raise _BuildError


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "build", "client"])
async def test_releases_model_before_environment_on_success_and_failure(
    monkeypatch, failure
):
    closed = []
    native_environment = gp.Env
    native_model = gp.Model

    @contextmanager
    def tracked_environment(*args, **kwargs):
        try:
            with native_environment(*args, **kwargs) as resource:
                yield resource
        finally:
            closed.append("environment")

    @contextmanager
    def tracked_model(*args, **kwargs):
        try:
            with native_model(*args, **kwargs) as resource:
                yield resource
        finally:
            closed.append("model")

    monkeypatch.setattr(gp, "Env", tracked_environment)
    monkeypatch.setattr(gp, "Model", tracked_model)
    formulation = (
        _FailingFormulation()
        if failure == "build"
        else GurobiCompactFormulationCompiler()
    )
    builder = GurobiFormulationModelBuilder(formulation)
    data = CompactFormulationDataBuilder().build_formulation_data(*_inputs())

    async def use_model():
        async with builder.build_formulation_model(data) as compiled:
            assert not closed
            assert compiled.native_model.NumVars > 0
            if failure == "client":
                raise _ClientError

    if failure is None:
        await use_model()
    else:
        with pytest.raises(_BuildError if failure == "build" else _ClientError):
            await use_model()

    assert closed == ["model", "environment"]
