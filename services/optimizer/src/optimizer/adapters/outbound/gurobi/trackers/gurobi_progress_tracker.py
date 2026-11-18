"""Observe presolve removals and a bounded-rate best-bound trace."""

import math

import gurobipy as gp
from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.entities.gurobi_bound_sample import (
    GurobiBoundSample,
)


class GurobiProgressTracker:
    def __init__(self, *, minimize: bool) -> None:
        self._minimize = minimize
        self._last_second = -1
        self.bound_history: list[GurobiBoundSample] = []
        self.presolve_columns_removed: int | None = None
        self.presolve_rows_removed: int | None = None
        self.time_to_zero_objective_s: float | None = None

    def record_bound(self, runtime: float, bound: float, nodes: float) -> None:
        if all(
            math.isfinite(value) and abs(value) < GRB.INFINITY
            for value in (runtime, bound, nodes)
        ):
            self.bound_history.append(GurobiBoundSample(runtime, bound, nodes))

    def __call__(self, model: gp.Model, where: int) -> None:
        if where == GRB.Callback.PRESOLVE:
            self.presolve_columns_removed = max(
                self.presolve_columns_removed or 0,
                int(model.cbGet(GRB.Callback.PRE_COLDEL)),
            )
            self.presolve_rows_removed = max(
                self.presolve_rows_removed or 0,
                int(model.cbGet(GRB.Callback.PRE_ROWDEL)),
            )
        elif where == GRB.Callback.MIP:
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            second = math.floor(runtime)
            if second > self._last_second:
                self._last_second = second
                self.record_bound(
                    runtime,
                    float(model.cbGet(GRB.Callback.MIP_OBJBND)),
                    float(model.cbGet(GRB.Callback.MIP_NODCNT)),
                )
        elif (
            where == GRB.Callback.MIPSOL
            and self._minimize
            and self.time_to_zero_objective_s is None
        ):
            value = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if math.isfinite(value) and abs(value) <= 1e-9:
                self.time_to_zero_objective_s = float(model.cbGet(GRB.Callback.RUNTIME))
