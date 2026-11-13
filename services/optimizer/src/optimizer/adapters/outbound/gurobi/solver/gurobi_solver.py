import asyncio
import math
from threading import Event
from typing import override

import gurobipy as gp
from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.trackers.gurobi_incumbent_tracker import (
    GurobiIncumbentTracker,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.ports.outbound.solver_result import (
    SolverOutput,
    SolverTerminationKind,
)


class GurobiSolver[FormulationT, VariablesT](
    Solver[
        GurobiCompiledFormulation[FormulationT, VariablesT],
        GurobiSolverConfig,
        GurobiCompiledFormulation[FormulationT, VariablesT],
        GurobiRunStatistics,
    ]
):
    """Execute one prepared model, leaving its lifecycle to the model builder."""

    @override
    async def solve_compiled_formulation(
        self,
        compiled_formulation: GurobiCompiledFormulation[FormulationT, VariablesT],
        solver_config: GurobiSolverConfig,
    ) -> SolverOutput[
        GurobiCompiledFormulation[FormulationT, VariablesT], GurobiRunStatistics
    ]:
        model = compiled_formulation
        config = solver_config
        cancel_requested = Event()
        optimization = asyncio.create_task(
            asyncio.to_thread(self._solve_model, model, config, cancel_requested)
        )
        try:
            return await asyncio.shield(optimization)
        except asyncio.CancelledError:
            cancel_requested.set()
            try:
                model.native_model.terminate()
            finally:
                # The builder must not close native resources while this thread runs.
                await self._wait_for_optimization(optimization)
            raise

    @staticmethod
    async def _wait_for_optimization[OutputT](
        optimization: asyncio.Task[OutputT],
    ) -> None:
        while not optimization.done():
            try:
                await asyncio.shield(optimization)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not optimization.cancelled():
            # Retrieve worker failures while preserving the caller's cancellation.
            optimization.exception()

    def _solve_model(
        self,
        model: GurobiCompiledFormulation[FormulationT, VariablesT],
        config: GurobiSolverConfig,
        cancel_requested: Event,
    ) -> SolverOutput[
        GurobiCompiledFormulation[FormulationT, VariablesT], GurobiRunStatistics
    ]:
        if cancel_requested.is_set():
            raise asyncio.CancelledError

        native_model = model.native_model
        self._configure_model(native_model, config)
        tracker = (
            GurobiIncumbentTracker(minimize=native_model.ModelSense == GRB.MINIMIZE)
            if config.record_incumbent_history
            else None
        )

        def callback(native: gp.Model, where: int) -> None:
            if cancel_requested.is_set():
                native.terminate()
            elif tracker is not None:
                tracker(native, where)

        # Covers cancellation before optimize, including queued worker execution.
        if cancel_requested.is_set():
            raise asyncio.CancelledError
        native_model.optimize(callback)
        result = self._collect_result(model, tracker)
        if config.observer is not None:
            config.observer(result.statistics)
        return result

    def _configure_model(self, model: gp.Model, config: GurobiSolverConfig) -> None:
        model.Params.OutputFlag = int(config.output_flag)
        for name, value in (
            ("TimeLimit", config.time_limit_s),
            ("MIPGap", config.mip_gap),
            ("Threads", config.threads),
            ("Seed", config.seed),
        ):
            if value is not None:
                model.setParam(name, value)

    @staticmethod
    def _finite_attribute(model: gp.Model, name: str) -> float | None:
        try:
            value = float(model.getAttr(name))
        except AttributeError:
            return None
        except gp.GurobiError as error:
            if error.errno != GRB.Error.DATA_NOT_AVAILABLE:
                raise
            return None
        return value if math.isfinite(value) and abs(value) < GRB.INFINITY else None

    @staticmethod
    def _termination_kind(status: int) -> SolverTerminationKind:
        # Local-only or unrecognized statuses carry no global proof.
        return {
            GRB.OPTIMAL: SolverTerminationKind.OPTIMAL,
            GRB.INFEASIBLE: SolverTerminationKind.INFEASIBLE,
            GRB.UNBOUNDED: SolverTerminationKind.UNBOUNDED,
            GRB.INF_OR_UNBD: SolverTerminationKind.INFEASIBLE_OR_UNBOUNDED,
            GRB.CUTOFF: SolverTerminationKind.OBJECTIVE_LIMIT,
            GRB.ITERATION_LIMIT: SolverTerminationKind.ITERATION_LIMIT,
            GRB.NODE_LIMIT: SolverTerminationKind.NODE_LIMIT,
            GRB.TIME_LIMIT: SolverTerminationKind.TIME_LIMIT,
            GRB.SOLUTION_LIMIT: SolverTerminationKind.SOLUTION_LIMIT,
            GRB.INTERRUPTED: SolverTerminationKind.INTERRUPTED,
            GRB.NUMERIC: SolverTerminationKind.NUMERICAL_FAILURE,
            GRB.SUBOPTIMAL: SolverTerminationKind.SUBOPTIMAL,
            GRB.USER_OBJ_LIMIT: SolverTerminationKind.OBJECTIVE_LIMIT,
            GRB.WORK_LIMIT: SolverTerminationKind.WORK_LIMIT,
            GRB.MEM_LIMIT: SolverTerminationKind.MEMORY_LIMIT,
        }.get(status, SolverTerminationKind.UNKNOWN)

    def _collect_result(
        self,
        model: GurobiCompiledFormulation[FormulationT, VariablesT],
        tracker: GurobiIncumbentTracker | None,
    ) -> SolverOutput[
        GurobiCompiledFormulation[FormulationT, VariablesT], GurobiRunStatistics
    ]:
        native_model = model.native_model
        status = int(native_model.Status)
        solution_count = int(native_model.SolCount)
        runtime_s = self._finite_attribute(native_model, GRB.Attr.Runtime)
        objective_value = (
            self._finite_attribute(native_model, GRB.Attr.ObjVal)
            if solution_count > 0
            else None
        )
        statistics = GurobiRunStatistics(
            native_status=status,
            solution_count=solution_count,
            relative_gap=(
                self._finite_attribute(native_model, GRB.Attr.MIPGap)
                if solution_count > 0
                else None
            ),
            construction_time_s=model.construction_time_s,
            time_limit_s=float(native_model.Params.TimeLimit),
            mip_gap_target=float(native_model.Params.MIPGap),
            threads=int(native_model.Params.Threads),
            seed=int(native_model.Params.Seed),
            continuous_variables=(
                int(native_model.NumVars) - int(native_model.NumIntVars)
            ),
            binary_variables=int(native_model.NumBinVars),
            integer_variables=(
                int(native_model.NumIntVars) - int(native_model.NumBinVars)
            ),
            linear_constraints=int(native_model.NumConstrs),
            quadratic_constraints=int(native_model.NumQConstrs),
            general_constraints=int(native_model.NumGenConstrs),
            node_count=self._finite_attribute(native_model, GRB.Attr.NodeCount),
            time_to_first_incumbent_s=(
                tracker.time_to_first_incumbent_s if tracker is not None else None
            ),
            incumbent_history=(
                tracker.samples(runtime_s, objective_value)
                if tracker is not None
                else ()
            ),
        )
        return SolverOutput(
            termination=self._termination_kind(status),
            solved_model=model if solution_count > 0 else None,
            objective_value=objective_value,
            objective_bound=self._finite_attribute(native_model, GRB.Attr.ObjBound),
            runtime_s=runtime_s,
            statistics=statistics,
        )
